import faiss
import numpy as np
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class SemanticCache:
    """
    Lightweight semantic cache using FAISS for fast similarity search.
    Stores query embeddings and their corresponding responses.
    """
    
    def __init__(
        self,
        entries = 0,  # Total unique queries stored
        hits = 0,     # Queries found in cache
        misses = 0,  # Queries not found (need generation)
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        similarity_threshold: float = 0.90,
        cache_file: str = 'semantic_cache.json',
        max_cache_size: int = 1000
    ):
        """
        Initialize semantic cache.
        
        Args:
            embedding_model: Model for generating query embeddings
            similarity_threshold: Minimum similarity score for cache hit (0.80-0.95 recommended)
            cache_file: JSON file to persist cache data
            max_cache_size: Maximum number of cached queries
        """
        self.similarity_threshold = similarity_threshold
        self.cache_file = Path(cache_file)
        self.max_cache_size = max_cache_size
        
        # Load embedding model (reuse existing model)
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Initialize FAISS index (IndexFlatIP for cosine similarity)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Storage for cached responses
        self.cache_data: List[Dict[str, Any]] = []
        
        # Stats
        self.hits = 0
        self.misses = 0
        
        # Load existing cache if available
        self._load_cache()
        
        logger.info(f"Semantic cache initialized with {len(self.cache_data)} entries")
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity."""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def _load_cache(self):
        """Load cache from JSON file."""
        if not self.cache_file.exists():
            logger.info("No existing cache file found, starting fresh")
            return
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self.cache_data = json.load(f)
            
            # Rebuild FAISS index from stored embeddings
            if self.cache_data:
                embeddings = np.array([
                    entry['embedding'] for entry in self.cache_data
                ], dtype=np.float32)
                
                # Normalize embeddings for cosine similarity
                embeddings = np.array([
                    self._normalize_embedding(emb) for emb in embeddings
                ])
                
                self.index.add(embeddings)
                logger.info(f"Loaded {len(self.cache_data)} cached entries")
        
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            self.cache_data = []
    
    def _save_cache(self):
        """Persist cache to JSON file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Cache saved with {len(self.cache_data)} entries")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Check if query exists in cache.
        
        Args:
            query: User query string
        
        Returns:
            Cached response dict if hit, None if miss
        """
        if len(self.cache_data) == 0:
            self.misses += 1
            logger.info(f"Cache MISS (empty cache): {query[:50]}...")
            return None
        
        # Generate query embedding
        query_embedding = self.encoder.encode(query, convert_to_numpy=True)
        query_embedding = self._normalize_embedding(query_embedding).reshape(1, -1).astype(np.float32)
        
        # Search in FAISS index (k=1 for top match)
        distances, indices = self.index.search(query_embedding, k=1)
        
        similarity_score = float(distances[0][0])
        best_match_idx = int(indices[0][0])
        
        logger.info(f"Cache lookup - Query: '{query[:50]}...' | Similarity: {similarity_score:.4f} | Threshold: {self.similarity_threshold}")
        
        # Check if similarity exceeds threshold
        if similarity_score >= self.similarity_threshold:
            self.hits += 1
            cached_entry = self.cache_data[best_match_idx]
            logger.info(f"✅ Cache HIT! Matched query: '{cached_entry['query'][:50]}...'")
            
            return {
                'query': cached_entry['query'],
                'answer': cached_entry['answer'],
                'iteration_details': cached_entry['iteration_details'],
                'similarity_score': similarity_score,
                'cached_at': cached_entry['timestamp']
            }
        else:
            self.misses += 1
            logger.info(f"❌ Cache MISS - Similarity {similarity_score:.4f} below threshold {self.similarity_threshold}")
            return None
    
    def set(
        self,
        query: str,
        answer: str,
        iteration_details: List[Dict[str, Any]]
    ):
        """
        Add query and response to cache.
        
        Args:
            query: User query
            answer: Final answer
            iteration_details: List of all iteration details with answers and feedbacks
        """
        # Generate and normalize embedding
        query_embedding = self.encoder.encode(query, convert_to_numpy=True)
        query_embedding = self._normalize_embedding(query_embedding).astype(np.float32)
        
        # Create cache entry
        cache_entry = {
            'query': query,
            'answer': answer,
            'iteration_details': iteration_details,
            'embedding': query_embedding.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to cache data
        self.cache_data.append(cache_entry)
        
        # Add to FAISS index
        self.index.add(query_embedding.reshape(1, -1))
        
        logger.info(f"Added to cache: '{query[:50]}...' | Total cached: {len(self.cache_data)}")
        
        # Enforce max cache size (FIFO eviction)
        if len(self.cache_data) > self.max_cache_size:
            logger.info(f"Cache size exceeded {self.max_cache_size}, removing oldest entry")
            self.cache_data.pop(0)
            self._rebuild_index()
        
        # Save to disk
        self._save_cache()
    
    def _rebuild_index(self):
        """Rebuild FAISS index from current cache_data."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        if self.cache_data:
            embeddings = np.array([
                entry['embedding'] for entry in self.cache_data
            ], dtype=np.float32)
            
            self.index.add(embeddings)
            logger.info("FAISS index rebuilt")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            'total_cached': len(self.cache_data),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'threshold': self.similarity_threshold
        }
    
    def clear(self):
        """Clear all cache data."""
        self.cache_data = []
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.hits = 0
        self.misses = 0
        
        if self.cache_file.exists():
            self.cache_file.unlink()
        
        logger.info("Cache cleared")
