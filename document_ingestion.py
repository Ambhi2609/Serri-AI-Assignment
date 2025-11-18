import os
import json
import logging
from typing import List, Dict, Optional, Any
import numpy as np
from datetime import datetime
from pathlib import Path
import faiss
import httpx # Add this import

# Pydantic for data validation
from pydantic import BaseModel, Field, ConfigDict, field_validator

# Document Processing
import fitz  # PyMuPDF for PDF extraction
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('support_bot_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class DocumentChunk(BaseModel):
    """Represents a chunk of text from the document."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_id: int = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="Text content of the chunk")
    embedding: np.ndarray = Field(..., description="Vector embedding of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (excluding embeddings)."""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'metadata': self.metadata
        }


class RetrievalResult(BaseModel):
    """Represents a retrieval result with relevance score."""
    chunk_id: int = Field(..., description="ID of the retrieved chunk")
    text: str = Field(..., description="Text content of the chunk")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")

    @field_validator('similarity_score')
    @classmethod
    def score_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError('Similarity score must be between 0 and 1')
        return v

class ExpandedRetrievalResult(BaseModel):
    """Extended retrieval result with context expansion."""
    chunk_id: int
    primary_chunk: str
    full_context: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    context_before: List[str] = Field(default_factory=list)
    context_after: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Configuration for the ingestion pipeline."""
    chunk_size: int = Field(default=500, gt=0, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=50, ge=0, description="Overlap between chunks")
    embedding_model: str = Field(
        default='sentence-transformers/all-MiniLM-L6-v2',
        description="HuggingFace model name"
    )
    similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity for retrieval"
    )

    @field_validator('chunk_overlap')
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        chunk_size = info.data.get('chunk_size', 500)
        if v >= chunk_size:
            raise ValueError('Overlap must be less than chunk_size')
        return v


# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    """Handles document extraction and text processing."""

    def __init__(self):
        logger.info("Initializing DocumentProcessor")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
      if not Path(pdf_path).exists():
          raise FileNotFoundError(f"PDF file not found: {pdf_path}")

      logger.info(f"Extracting text from PDF: {pdf_path}")

      try:
          doc = fitz.open(pdf_path)
          text_parts = []

          for page_num, page in enumerate(doc, 1):
              text = page.get_text()
              text_parts.append(text)
              logger.debug(f"Extracted {len(text)} characters from page {page_num}")

          full_text = "\n\n".join(text_parts)

          # ✅ Log BEFORE closing
          logger.info(f"Total text extracted: {len(full_text)} characters from {len(doc)} pages")

          doc.close()  # Now safe to close

          return full_text

      except Exception as e:
          logger.error(f"Error extracting text from PDF: {e}")
          raise

    def recursive_character_split(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
        separators: Optional[List[str]] = None
    ) -> List[str]:
        """
        Split text recursively using multiple separators.

        This method preserves natural text boundaries by trying different
        separators in order of priority: paragraphs, sentences, words, characters.

        Args:
            text: Input text to split
            chunk_size: Target size for each chunk in characters
            overlap: Number of characters to overlap between chunks
            separators: List of separators in priority order

        Returns:
            List of text chunks with overlap
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        logger.info(f"Chunking text with size={chunk_size}, overlap={overlap}")

        def split_text(text: str, separator_idx: int) -> List[str]:
            """Recursively split text using separators."""
            if separator_idx >= len(separators):
                # Base case: character-level split
                return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

            separator = separators[separator_idx]
            if not separator:
                # Empty separator means character split
                return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

            splits = text.split(separator)
            result_chunks = []
            current_chunk = ""

            for split in splits:
                test_chunk = current_chunk + split + separator

                if len(test_chunk) <= chunk_size:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        result_chunks.append(current_chunk.strip())

                    if len(split) > chunk_size:
                        # Recursively split with next separator
                        result_chunks.extend(split_text(split, separator_idx + 1))
                        current_chunk = ""
                    else:
                        current_chunk = split + separator

            if current_chunk:
                result_chunks.append(current_chunk.strip())

            return result_chunks

        chunks = split_text(text, 0)

        # Add overlap between chunks
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0 and overlap > 0 and len(chunks[i-1]) >= overlap:
                # Add overlap from previous chunk
                prev_overlap = chunks[i-1][-overlap:]
                overlapped_chunks.append(prev_overlap + " " + chunk)
            else:
                overlapped_chunks.append(chunk)

        # Filter out very small chunks
        overlapped_chunks = [c for c in overlapped_chunks if len(c.strip()) > 10]

        logger.info(f"Created {len(overlapped_chunks)} chunks")
        return overlapped_chunks


# ============================================================================
# VECTOR STORE
# ============================================================================

class VectorStore:
    """Manages document embeddings and retrieval using FAISS."""

    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        similarity_threshold: float = 0.3
    ):
        logger.info(f"Initializing VectorStore with model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.chunks: List[DocumentChunk] = []

        # Initialize FAISS index for cosine similarity (Inner Product on normalized vectors)
        embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner Product index

        logger.info(f"Embedding dimension: {embedding_dim}")

    def add_documents(
      self,
      texts: List[str],
      metadata: Optional[List[Dict[str, Any]]] = None,
      window_size: int = 2  # Number of chunks before/after
  ) -> None:
      if not texts:
          raise ValueError("Cannot add empty text list")

      logger.info(f"Adding {len(texts)} documents with window_size={window_size}")

      if metadata is None:
          metadata = [{}] * len(texts)

      # Generate embeddings
      embeddings = self.model.encode(
          texts,
          show_progress_bar=True,
          batch_size=32,
          convert_to_numpy=True
      ).astype('float32')

      faiss.normalize_L2(embeddings)
      self.index.add(embeddings)

      # Create chunks with neighbor metadata
      for i, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadata)):
          # Calculate neighboring chunk IDs
          chunk_neighbors = {
              'prev_chunks': list(range(max(0, i - window_size), i)),
              'next_chunks': list(range(i + 1, min(len(texts), i + window_size + 1))),
              'position': i,  # Original position in document
              'total_chunks': len(texts)
          }

          # Merge with existing metadata
          enriched_meta = {**meta, **chunk_neighbors}

          chunk = DocumentChunk(
              chunk_id=len(self.chunks),
              text=text,
              embedding=embedding,
              metadata=enriched_meta
          )
          self.chunks.append(chunk)

      logger.info(f"Successfully added {len(texts)} documents with context windows")

    def retrieve(
      self,
      query: str,
      top_k: int = 3
  ) -> List[RetrievalResult]:
      """
      Basic retrieval WITHOUT context expansion.
      This is the original method - do NOT modify it.
      """
      if not query or not query.strip():
          raise ValueError("Query cannot be empty")

      if top_k <= 0:
          raise ValueError("top_k must be positive")

      logger.info(f"Retrieving documents for query: '{query[:50]}...'")

      if not self.chunks:
          logger.warning("No documents in vector store")
          return []

      # Generate and normalize query embedding
      query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
      faiss.normalize_L2(query_embedding)

      # Search using FAISS
      similarities, indices = self.index.search(query_embedding, top_k)

      # Convert to results
      results = []
      for score, idx in zip(similarities[0], indices[0]):
          if idx == -1:
              break

          score = float(score)

          if score >= self.similarity_threshold:
              chunk = self.chunks[idx]
              result = RetrievalResult(
                  chunk_id=chunk.chunk_id,
                  text=chunk.text,
                  similarity_score=score,
                  metadata=chunk.metadata
              )
              results.append(result)

      logger.info(f"Retrieved {len(results)} relevant documents")
      return results

    def retrieve_with_context(
      self,
      query: str,
      top_k: int = 3,
      expand_window: int = 2
  ) -> List[ExpandedRetrievalResult]:  # ✅ Return Pydantic models
      """Retrieve chunks and expand with surrounding context."""
      initial_results = self.retrieve(query, top_k)

      expanded_results = []

      for result in initial_results:
          chunk_id = result.chunk_id
          chunk_meta = result.metadata

          # Get position information
          position = chunk_meta.get('position', chunk_id)

          # Calculate neighbor indices
          start_idx = max(0, position - expand_window)
          end_idx = min(len(self.chunks), position + expand_window + 1)

          # Build context
          context_before = []
          for i in range(start_idx, position):
              if i < len(self.chunks):
                  context_before.append(self.chunks[i].text)

          context_after = []
          for i in range(position + 1, end_idx):
              if i < len(self.chunks):
                  context_after.append(self.chunks[i].text)

          # Combine context
          full_context = (
              "\n\n".join(context_before) +
              "\n\n" + result.text +
              "\n\n" + "\n\n".join(context_after)
          ).strip()

          # Create Pydantic model instead of dict
          expanded = ExpandedRetrievalResult(
              chunk_id=chunk_id,
              primary_chunk=result.text,
              full_context=full_context,
              similarity_score=result.similarity_score,
              context_before=context_before,
              context_after=context_after,
              metadata=chunk_meta
          )
          expanded_results.append(expanded)

          logger.info(
              f"Expanded chunk {chunk_id}: "
              f"{len(context_before)} chunks before, {len(context_after)} chunks after"
          )

      return expanded_results

    def save(self, save_dir: str) -> None:
        logger.info(f"Saving vector store to {save_dir}")
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index (single file)
        index_path = save_path / "faiss.index"
        faiss.write_index(self.index, str(index_path))

        # Save chunks metadata (without embeddings)
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        chunks_path = save_path / "chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        # Save config
        config = {
            'model_name': self.model_name,
            'similarity_threshold': self.similarity_threshold,
            'num_chunks': len(self.chunks),
            'embedding_dim': self.index.d
        }
        config_path = save_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info("Vector store saved successfully")

    def load(self, load_dir: str) -> None:
      """Load vector store from disk."""
      logger.info(f"Loading vector store from {load_dir}")
      load_path = Path(load_dir)

      if not load_path.exists():
          raise FileNotFoundError(f"Directory not found: {load_dir}")

      # Load config
      config_path = load_path / "config.json"
      with open(config_path, 'r') as f:
          config = json.load(f)

      loaded_model_name = config['model_name']
      self.similarity_threshold = config['similarity_threshold']

      # Reload model only if different from current
      if loaded_model_name != self.model_name:
          logger.info(f"Loading different model: {loaded_model_name}")
          self.model_name = loaded_model_name
          self.model = SentenceTransformer(self.model_name)
      else:
          logger.info(f"Using existing model: {self.model_name}")

      # Load FAISS index
      index_path = load_path / "faiss.index"
      self.index = faiss.read_index(str(index_path))

      # Load chunks
      chunks_path = load_path / "chunks.json"
      with open(chunks_path, 'r', encoding='utf-8') as f:
          chunks_data = json.load(f)

      # Reconstruct chunks (embeddings are in FAISS)
      self.chunks = []
      dummy_embedding = np.zeros(self.index.d, dtype=np.float32)
      for chunk_dict in chunks_data:
          chunk = DocumentChunk(
              chunk_id=chunk_dict['chunk_id'],
              text=chunk_dict['text'],
              embedding=dummy_embedding,
              metadata=chunk_dict['metadata']
          )
          self.chunks.append(chunk)

      logger.info(f"Loaded {len(self.chunks)} chunks from disk")

# ============================================================================
# DOCUMENT INGESTION PIPELINE
# ============================================================================

class DocumentIngestionPipeline:
    """Complete pipeline for document ingestion, storage, and retrieval."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the ingestion pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        if config is None:
            config = PipelineConfig()

        self.config = config

        logger.info("="*80)
        logger.info("Initializing DocumentIngestionPipeline")
        logger.info("="*80)
        logger.info(f"Configuration: {self.config.model_dump()}")

        self.processor = DocumentProcessor()
        self.vector_store = VectorStore(
            model_name=self.config.embedding_model,
            similarity_threshold=self.config.similarity_threshold
        )

    def ingest_document(
        self,
        pdf_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        window_size:int = 0
    ) -> int:
        """
        Ingest a PDF document into the system.

        Args:
            pdf_path: Path to PDF file
            metadata: Optional metadata to attach to all chunks

        Returns:
            Number of chunks created

        Raises:
            FileNotFoundError: If PDF file doesn't exist
        """
        logger.info(f"Starting document ingestion for: {pdf_path}")
        start_time = datetime.now()

        # Extract text
        text = self.processor.extract_text_from_pdf(pdf_path)

        # Create chunks
        chunks = self.processor.recursive_character_split(
            text=text,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap
        )

        # Prepare metadata
        if metadata is None:
            metadata = {}

        chunk_metadata = [
            {**metadata, 'source': pdf_path, 'chunk_index': i}
            for i in range(len(chunks))
        ]

        # Add to vector store
        self.vector_store.add_documents(chunks, chunk_metadata, window_size)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Document ingestion completed in {elapsed:.2f}s")
        logger.info(f"Created {len(chunks)} chunks from {pdf_path}")

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
        expand_window: int = 0
    ) -> List[RetrievalResult]:
        """
        Search for relevant document chunks.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant chunks with scores
        """
        logger.info("-"*80)
        logger.info(f"SEARCH QUERY: {query}")
        logger.info("-"*80)

        if expand_window > 0:
            results = self.vector_store.retrieve_with_context(
                query,
                top_k=top_k,
                expand_window=expand_window
            )
            logger.info(f"Found {len(results)} results with context expansion")
        else:
            results = self.vector_store.retrieve(query, top_k=top_k)
            logger.info(f"Found {len(results)} results (basic retrieval)")

        return results

        if not results:
            logger.warning("No relevant documents found above threshold")
        else:
            logger.info(f"Found {len(results)} relevant chunks")
            for i, result in enumerate(results, 1):
                logger.info(
                    f"  Result {i}: Score={result.similarity_score:.4f}, "
                    f"Text preview: {result.text[:100]}..."
                )

        return results

    def save_pipeline(self, save_dir: str) -> None:
        """Save the entire pipeline to disk."""
        self.vector_store.save(save_dir)

        # Also save config
        config_path = Path(save_dir) / "pipeline_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.model_dump(), f, indent=2)

        logger.info(f"Pipeline configuration saved to {config_path}")

    def load_pipeline(self, load_dir: str) -> None:
        """Load the entire pipeline from disk."""
        # Load config
        config_path = Path(load_dir) / "pipeline_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = PipelineConfig(**config_dict)
            logger.info(f"Loaded pipeline configuration: {self.config.model_dump()}")

        # Load vector store
        self.vector_store.load(load_dir)
