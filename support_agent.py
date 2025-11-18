from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Optional, Any, Literal, Union
from datetime import datetime
from openai import AsyncOpenAI
import httpx
import json
import os
import logging
import random
from semantic_cache import SemanticCache
# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('support_bot_agent.txt'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS FOR AGENT
# ============================================================================

class QueryInput(BaseModel):
    """Validated input for customer queries."""
    query: str = Field(..., min_length=3, max_length=500, description="Customer query text")
    max_iterations: int = Field(default=2, ge=1, le=5, description="Maximum refinement iterations")
    
    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()


class FeedbackType(BaseModel):
    """Structured feedback from simulated evaluation."""
    type: Literal['good', 'too_vague', 'not_helpful'] = Field(
        ..., 
        description="Type of feedback"
    )
    comment: str = Field(..., min_length=5, description="Explanation of feedback")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in feedback")
    
    @field_validator('comment')
    @classmethod
    def comment_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Comment cannot be empty')
        return v.strip()


class IterationDetails(BaseModel):
    """Details about a single answer generation iteration."""
    iteration_number: int = Field(..., ge=1, description="Iteration number")
    answer: str = Field(..., description="Generated answer")
    feedback: FeedbackType = Field(..., description="Feedback received")
    retrieval_count: int = Field(..., ge=0, description="Number of chunks retrieved")
    timestamp: datetime = Field(default_factory=datetime.now)
    tokens_used: Optional[int] = Field(None, ge=0, description="Tokens consumed")


class QueryResponse(BaseModel):
    """Complete response for a customer query."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., min_length=10, description="Final answer")
    in_scope: bool = Field(..., description="Whether query was in scope")
    iterations: int = Field(..., ge=0, description="Number of iterations performed")
    retrieval_scores: List[float] = Field(default_factory=list, description="Similarity scores")
    feedback_history: List[FeedbackType] = Field(default_factory=list)
    iteration_details: List[IterationDetails] = Field(default_factory=list)
    processing_time: float = Field(..., ge=0.0, description="Total processing time in seconds")
    reason: Optional[str] = Field(None, description="Reason if out of scope")
    
    @field_validator('retrieval_scores')
    @classmethod
    def validate_scores(cls, v: List[float]) -> List[float]:
        """Ensure all scores are between 0 and 1."""
        for score in v:
            if not 0.0 <= score <= 1.0:
                raise ValueError(f'Retrieval score {score} must be between 0 and 1')
        return v
    
    @model_validator(mode='after')
    def check_consistency(self) -> 'QueryResponse':
        """Validate logical consistency across fields."""
        if not self.in_scope and self.iterations > 0:
            raise ValueError('Out-of-scope queries should have 0 iterations')
        
        if self.in_scope and not self.retrieval_scores:
            raise ValueError('In-scope queries must have retrieval scores')
        
        if len(self.feedback_history) != self.iterations:
            raise ValueError('Feedback history length must match iterations')
        
        return self


class AgentConfig(BaseModel):
    """Configuration for SupportBotAgent."""
    answer_model: str = Field(
        default="openai/gpt-oss-20b:free",
        description="Model for answer generation"
    )
    similarity_threshold: float = Field(
        default=0.3, 
        ge=0.0, 
        le=1.0,
        description="Minimum similarity for in-scope"
    )
    max_context_length: int = Field(
        default=15000, 
        gt=0, 
        le=20000,
        description="Maximum context length in characters"
    )
    app_name: str = Field(default="CustomerSupportBot", min_length=3)
    site_url: str = Field(default="http://localhost:5000")


class AgentStats(BaseModel):
    """Statistics about the agent."""
    total_queries_processed: int = Field(default=0, ge=0)
    in_scope_count: int = Field(default=0, ge=0)
    out_of_scope_count: int = Field(default=0, ge=0)
    average_processing_time: float = Field(default=0.0, ge=0.0)
    average_iterations: float = Field(default=0.0, ge=0.0)
    total_tokens_used: int = Field(default=0, ge=0)
    
    @model_validator(mode='after')
    def validate_counts(self) -> 'AgentStats':
        """Ensure counts are consistent."""
        if self.in_scope_count + self.out_of_scope_count != self.total_queries_processed:
            raise ValueError('Scope counts must sum to total queries')
        return self


# ============================================================================
# SIMPLIFIED SUPPORT BOT AGENT
# ============================================================================

from semantic_cache import SemanticCache

class SupportBotAgent:
    """Simplified agentic customer support bot with semantic caching."""
    
    def __init__(
        self,
        pipeline: 'DocumentIngestionPipeline',
        openrouter_api_key: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        enable_cache: bool = True,  # ✅ NEW
        cache_threshold: float = 0.90  # ✅ NEW
    ):
        """Initialize the Support Bot Agent."""
        self.config = config if config else AgentConfig()
        self.pipeline = pipeline
        
        # Get API key from environment or parameter
        self.api_key = openrouter_api_key 
        if not self.api_key:
            raise ValueError("OpenRouter API key must be provided or set in OPENROUTER_API_KEY env var")
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": self.config.site_url,
                "X-Title": self.config.app_name,
            },
            timeout=httpx.Timeout(30.0, connect=10.0)
        )
        
        # ✅ Initialize semantic cache
        self.cache = None
        if enable_cache:
            self.cache = SemanticCache(
                embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                similarity_threshold=cache_threshold,
                cache_file='semantic_cache.json',
                max_cache_size=1000
            )
            logger.info("Semantic caching ENABLED")
        
        # Stats tracking
        self._stats = {
            'total_queries': 0,
            'in_scope': 0,
            'out_of_scope': 0,
            'total_time': 0.0,
            'total_iterations': 0,
            'total_tokens': 0,
            'cache_hits': 0  # ✅ NEW
        }
        
        logger.info(f"Initialized SupportBotAgent with config: {self.config.model_dump()}")
    
    async def answer_query(
        self,
        query_input: Union[str, QueryInput],
        **kwargs
    ) -> QueryResponse:
        """
        Answer a customer query with semantic caching and feedback-based refinement.
        
        Args:
            query_input: Either a string query or QueryInput instance
            **kwargs: Additional parameters
        
        Returns:
            QueryResponse with validated fields
        """
        # Convert string to QueryInput if needed
        if isinstance(query_input, str):
            query_input = QueryInput(query=query_input, **kwargs)
        
        logger.info("="*80)
        logger.info(f"NEW QUERY: {query_input.query}")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # ✅ CHECK CACHE FIRST
        if self.cache:
            cached_response = self.cache.get(query_input.query)
            if cached_response:
                elapsed = (datetime.now() - start_time).total_seconds()
                self._stats['total_queries'] += 1
                self._stats['cache_hits'] += 1
                self._stats['in_scope'] += 1
                self._stats['total_time'] += elapsed
                
                logger.info(f"🚀 Returning CACHED response (similarity: {cached_response['similarity_score']:.4f})")
                
                return QueryResponse(
                    query=query_input.query,
                    answer=cached_response['answer'],
                    in_scope=True,
                    iterations=len(cached_response['iteration_details']),
                    retrieval_scores=[],
                    feedback_history=[],
                    iteration_details=[],  # Will be populated from cached data
                    processing_time=elapsed
                )
        
        # Continue with normal flow if no cache hit
        iteration = 0
        current_answer = None
        previous_answer = None
        feedback_history = []
        iteration_details = []
        
        # Step 1: Retrieve relevant chunks
        logger.info("Step 1: Retrieving relevant documents...")
        retrieval_results = self.pipeline.search(
            query=query_input.query,
            top_k=2,
            expand_window=3
        )
        
        self.print_retrieved_contexts(retrieval_results, query_input.query)
        
        retrieval_scores = [r.similarity_score for r in retrieval_results] if retrieval_results else []
        
        # Step 2: Check if query is in scope
        logger.info("Step 2: Checking if query is in scope...")
        if not retrieval_results or retrieval_results[0].similarity_score < self.config.similarity_threshold:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.warning(f"Query OUT OF SCOPE. Time: {elapsed:.2f}s")
            
            self._update_stats(in_scope=False, time=elapsed, iterations=0)
            
            return QueryResponse(
                query=query_input.query,
                answer=self._get_fallback_message(),
                in_scope=False,
                iterations=0,
                retrieval_scores=retrieval_scores,
                feedback_history=[],
                iteration_details=[],
                processing_time=elapsed,
                reason="No relevant information found in documentation"
            )
        
        # Step 3: Extract and prepare context
        contexts = self._extract_contexts(retrieval_results)
        logger.info(f"Extracted {len(contexts)} contexts for generation")
        
        # Step 4: Generate answer with feedback loop
        while iteration <= query_input.max_iterations:
            iteration += 1
            logger.info(f"\n--- Iteration {iteration}/{query_input.max_iterations} ---")
            
            current_answer, tokens = await self._generate_answer(
                query=query_input.query,
                contexts=contexts,
                previous_answer=previous_answer,
                previous_feedback=feedback_history[-1] if feedback_history else None,
                iteration=iteration
            )
            
            # Generate feedback for THIS answer
            feedback = self._simulate_feedback(current_answer, iteration)
            feedback_history.append(feedback)
            
            # Store the answer for this iteration
            iter_detail = IterationDetails(
                iteration_number=iteration,
                answer=current_answer,
                feedback=feedback,
                retrieval_count=len(retrieval_results),
                tokens_used=tokens
            )
            iteration_details.append(iter_detail)
            
            logger.info(f"Answer {iteration}: {current_answer[:100]}...")
            logger.info(f"Feedback: {feedback.type} - {feedback.comment}")
            
            # Update previous answer for next iteration
            previous_answer = current_answer
            
                
        
        elapsed = (datetime.now() - start_time).total_seconds()
        total_tokens = sum(d.tokens_used or 0 for d in iteration_details)
        
        self._update_stats(
            in_scope=True,
            time=elapsed,
            iterations=iteration,
            tokens=total_tokens
        )
        
        # ✅ ADD TO CACHE
        if self.cache:
            iteration_details_for_cache = [
                {
                    'iteration_number': d.iteration_number,
                    'answer': d.answer,
                    'feedback': {
                        'type': d.feedback.type,
                        'comment': d.feedback.comment,
                        'confidence': d.feedback.confidence
                    }
                }
                for d in iteration_details
            ]
            
            self.cache.set(
                query=query_input.query,
                answer=current_answer,
                iteration_details=iteration_details_for_cache
            )
            logger.info("✅ Response added to cache")
        
        # Return validated response
        return QueryResponse(
            query=query_input.query,
            answer=current_answer,
            in_scope=True,
            iterations=iteration,
            retrieval_scores=retrieval_scores,
            feedback_history=feedback_history,
            iteration_details=iteration_details,
            processing_time=elapsed
        )


    
    def print_retrieved_contexts(self, results: List, query: str) -> None:
      """
      Pretty print the retrieved contexts for debugging.
      
      Args:
          results: List of retrieval results from pipeline
          query: The original query
      """
      print("\n" + "="*80)
      print(f"RETRIEVED CONTEXTS FOR QUERY: '{query}'")
      print("="*80)
      
      if not results:
          print("❌ NO CONTEXTS RETRIEVED")
          return
      
      for i, result in enumerate(results, 1):
          print(f"\n--- Context {i} ---")
          print(f"Similarity Score: {result.similarity_score:.4f}")
          print(f"Chunk ID: {result.chunk_id}")
          
          # Get the actual text content
          if hasattr(result, 'full_context'):
              text = result.full_context
              print(f"Type: Expanded Context (with neighboring chunks)")
              print(f"Primary Chunk: {result}...")
              if hasattr(result, 'context_before') and result.context_before:
                  print(f"Context Before: {len(result.context_before)} chunks")
              if hasattr(result, 'context_after') and result.context_after:
                  print(f"Context After: {len(result.context_after)} chunks")
          else:
              text = result.text
              print(f"Type: Basic Chunk")
          
          print(f"\nText Length: {len(text)} characters")
          print(f"Text Preview (first 300 chars):")
          print("-" * 40)
          print(text[:300])
          if len(text) > 300:
              print(f"... (truncated, {len(text) - 300} more characters)")
          print("-" * 40)
          
          # Print metadata if available
          if hasattr(result, 'metadata') and result.metadata:
              print(f"\nMetadata: {result.metadata}")
      
      print("\n" + "="*80 + "\n")
    
    async def _generate_answer(
        self,
        query: str,
        contexts: List[str],
        previous_answer: Optional[str],  # ✅ NEW: Accept previous answer
        previous_feedback: Optional[FeedbackType],
        iteration: int
    ) -> tuple[str, int]:
        """
        Generate or refine an answer using LLM.
        
        Args:
            query: User query
            contexts: List of relevant contexts
            previous_answer: The answer from the previous iteration
            previous_feedback: Feedback from previous iteration
            iteration: Current iteration number
        
        Returns:
            Tuple of (answer, token_count)
        """
        # Build prompt based on iteration
        prompt = self._build_prompt(query, contexts, previous_answer, previous_feedback, iteration)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.answer_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful customer support assistant. "
                                "Answer questions clearly and concisely using only "
                                "the provided documentation context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=700,
                temperature=0.3,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            usage = response.usage
            
            logger.debug(
                f"Tokens used - Prompt: {usage.prompt_tokens}, "
                f"Completion: {usage.completion_tokens}, "
                f"Total: {usage.total_tokens}"
            )
            
            return answer, usage.total_tokens
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            error_answer = (
                "I apologize, but I encountered an error generating a response. "
                "Please try rephrasing your question or contact support."
            )
            return error_answer, 0

    
    def _build_prompt(
        self,
        query: str,
        contexts: List[str],
        previous_answer: Optional[str],  # ✅ NEW
        previous_feedback: Optional[FeedbackType],
        iteration: int
    ) -> str:
        """
        Build prompt that uses previous answer + feedback for refinement.
        
        Args:
            query: User query
            contexts: List of contexts
            previous_answer: Answer from previous iteration
            previous_feedback: Previous feedback
            iteration: Current iteration number
        
        Returns:
            Formatted prompt string
        """
        combined_context = "\n\n---\n\n".join(contexts[:3])
        
        # ITERATION 1: Generate initial answer
        if iteration == 1 or previous_answer is None:
            prompt = f"""Based on the following documentation, answer the customer's question.

    DOCUMENTATION:
    {combined_context}

    QUESTION: {query}

    INSTRUCTIONS:
    - Answer clearly and concisely
    - Use specific information from the documentation
    - Include examples if available
    - Keep your answer under 150 words
    - Be helpful and professional

    ANSWER:"""
        
        # ITERATION 2+: Refine based on previous answer + feedback
        else:
            prompt = f"""Your previous answer needs improvement based on the feedback below.

    DOCUMENTATION:
    {combined_context}

    QUESTION: {query}

    YOUR PREVIOUS ANSWER:
    {previous_answer}

    FEEDBACK RECEIVED:
    Type: {previous_feedback.type}
    Comment: {previous_feedback.comment}

    INSTRUCTIONS FOR IMPROVEMENT:
    """
            
            if previous_feedback.type == 'too_vague':
                prompt += """- Add MORE SPECIFIC details from the documentation
    - Include CONCRETE EXAMPLES or step-by-step instructions
    - Use bullet points or numbered lists if applicable
    - Explain the "how" and "why", not just the "what"
    - Aim for 100-150 words

    IMPROVED ANSWER:"""
            
            elif previous_feedback.type == 'not_helpful':
                prompt += """- Focus on directly answering what the user asked
    - Provide actionable information
    - Be more specific about what the user should do
    - Avoid generic statements
    - Keep it practical and clear

    REVISED ANSWER:"""
            
            else:
                prompt += """- Enhance clarity and completeness
    - Add any missing details
    - Ensure the answer fully addresses the question

    REVISED ANSWER:"""
        
        return prompt

    
    

    def _simulate_feedback(self, answer: str, iteration: int) -> FeedbackType:
        """
        Generate RANDOM feedback to test refinement loop.
        
        Args:
            answer: Generated answer
            iteration: Current iteration number
        
        Returns:
            Validated FeedbackType instance
        """
        answer_lower = answer.lower()
        
        # Check for error messages (always flag these)
        if "error" in answer_lower or "encountered an error" in answer_lower:
            return FeedbackType(
                type='not_helpful',
                comment='Error in response generation',
                confidence=0.95
            )
        
        # RANDOM FEEDBACK GENERATION
        feedback_options = [
            {
                'type': 'good',
                'comment': 'Clear and specific answer with good structure',
                'confidence': random.uniform(0.85, 0.95)
            },
            {
                'type': 'too_vague',
                'comment': 'Answer could use more specific details and examples',
                'confidence': random.uniform(0.75, 0.90)
            },
            {
                'type': 'too_vague',
                'comment': 'Could benefit from step-by-step instructions or concrete examples',
                'confidence': random.uniform(0.70, 0.85)
            },
            {
                'type': 'not_helpful',
                'comment': 'Answer does not directly address the specific question asked',
                'confidence': random.uniform(0.65, 0.85)
            },
            {
                'type': 'not_helpful',
                'comment': 'Could be more actionable and provide clearer guidance',
                'confidence': random.uniform(0.70, 0.90)
            },
            {
                'type': 'good',
                'comment': 'Comprehensive answer with relevant details',
                'confidence': random.uniform(0.80, 0.95)
            }
        ]
        
        # On first iteration: 50% chance of refinement needed
        if iteration == 1:
            # Weight towards refinement
            weights = [0.3, 0.2, 0.2, 0.15, 0.15, 0.0]  # Last 'good' has 0 weight
            chosen = random.choices(feedback_options, weights=weights, k=1)[0]
        
        # On second iteration: 70% chance of accepting
        elif iteration >= 2:
            weights = [0.4, 0.1, 0.1, 0.1, 0.1, 0.2]
            chosen = random.choices(feedback_options, weights=weights, k=1)[0]
        
        else:
            # Equal probability
            chosen = random.choice(feedback_options)
        
        logger.info(f"🎲 Random feedback generated: {chosen['type']} - {chosen['comment']}")
        
        return FeedbackType(
            type=chosen['type'],
            comment=chosen['comment'],
            confidence=chosen['confidence']
        )

    
    def _extract_contexts(self, results: List) -> List[str]:
        """
        Extract text contexts from retrieval results.
        
        Args:
            results: List of retrieval results
        
        Returns:
            List of context strings
        """
        contexts = []
        
        for result in results:
            # Use full_context if available (from expand_window), else use text
            if hasattr(result, 'full_context'):
                text = result.full_context
            else:
                text = result.text
            
            # Truncate if too long
            if len(text) > self.config.max_context_length:
                text = text[:self.config.max_context_length] + "..."
            
            contexts.append(text)
        
        logger.debug(f"Extracted {len(contexts)} contexts")
        return contexts
    
    def _get_fallback_message(self) -> str:
        """Generate graceful fallback response for out-of-scope queries."""
        return (
            "I apologize, but I don't have information about that in my current documentation. "
            "My knowledge is limited to the customer support materials I've been trained on.\n\n"
            "Would you like to:\n"
            "1. Rephrase your question to focus on the topics in the document you provided?\n"
            "2. Connect with a human support agent who can better assist you?"
        )
    
    def _update_stats(self, in_scope: bool, time: float, iterations: int, tokens: int = 0):
        """Update internal statistics."""
        self._stats['total_queries'] += 1
        self._stats['total_time'] += time
        self._stats['total_iterations'] += iterations
        self._stats['total_tokens'] += tokens
        
        if in_scope:
            self._stats['in_scope'] += 1
        else:
            self._stats['out_of_scope'] += 1
    
    def get_stats(self) -> AgentStats:
        """Get validated statistics."""
        avg_time = (
            self._stats['total_time'] / self._stats['total_queries']
            if self._stats['total_queries'] > 0 else 0.0
        )
        avg_iterations = (
            self._stats['total_iterations'] / self._stats['total_queries']
            if self._stats['total_queries'] > 0 else 0.0
        )
        
        return AgentStats(
            total_queries_processed=self._stats['total_queries'],
            in_scope_count=self._stats['in_scope'],
            out_of_scope_count=self._stats['out_of_scope'],
            average_processing_time=avg_time,
            average_iterations=avg_iterations,
            total_tokens_used=self._stats['total_tokens']
        )
