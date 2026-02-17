"""Cross-encoder reranker for relevance scoring."""

import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks retrieved chunks using a cross-encoder model."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name for the cross-encoder
            
        Raises:
            Exception: If model loading fails
        """
        try:
            logger.info(f"Loading cross-encoder model: {model_name}")
            self.model = CrossEncoder(model_name)
            self.model_name = model_name
            logger.info(f"Successfully loaded cross-encoder model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model {model_name}: {e}")
            raise Exception(f"Failed to initialize CrossEncoderReranker: {e}") from e

    def rerank(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        top_k: int = 20,
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks based on relevance to the query.
        
        Args:
            query: User query text
            chunks: List of candidate chunks (up to 50) with 'text' field
            top_k: Number of top chunks to return (default: 20)
            batch_size: Batch size for processing (default: 32)
            
        Returns:
            List of top-k chunks sorted by relevance score (descending)
            
        Note:
            If reranking fails, returns the original list with a warning logged
        """
        if not chunks:
            logger.warning("Empty chunks list provided for reranking")
            return []
        
        if len(chunks) > 50:
            logger.warning(f"Received {len(chunks)} chunks, truncating to 50")
            chunks = chunks[:50]
        
        try:
            # Prepare query-chunk pairs for scoring
            pairs = [(query, chunk.get("text", "")) for chunk in chunks]
            
            logger.info(f"Reranking {len(chunks)} chunks with batch_size={batch_size}")
            
            # Compute relevance scores in batches
            scores = self.model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False
            )
            
            # Attach scores to chunks
            scored_chunks = []
            for chunk, score in zip(chunks, scores):
                chunk_copy = chunk.copy()
                chunk_copy["rerank_score"] = float(score)
                scored_chunks.append(chunk_copy)
            
            # Sort by score descending and return top-k
            scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            result = scored_chunks[:top_k]
            
            logger.info(f"Reranking complete. Returning top {len(result)} chunks")
            return result
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Returning original list")
            return chunks[:top_k]
