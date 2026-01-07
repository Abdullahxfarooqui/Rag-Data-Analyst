"""
Re-ranking for improved retrieval precision.

Provides cross-encoder re-ranking for top-k candidates.
"""
from typing import List, Dict, Any, Optional, Protocol, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Reranker(Protocol):
    """Protocol for re-rankers."""
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Re-rank candidates based on query relevance.
        
        Args:
            query: The search query
            candidates: List of candidate results with 'text' field
            top_k: Number of results to return
            
        Returns:
            Re-ranked results with updated scores
        """
        ...


class CrossEncoderReranker:
    """
    Cross-encoder based re-ranker using sentence-transformers.
    
    Cross-encoders provide more accurate relevance scores by
    jointly encoding query and document, at the cost of speed.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize cross-encoder re-ranker.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        self._model_name = model_name
        self._device = device
        self._model = None
        self._is_loaded = False
    
    def _load_model(self) -> None:
        """Lazy load the cross-encoder model."""
        if self._is_loaded:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name, device=self._device)
            self._is_loaded = True
        except ImportError:
            raise ImportError(
                "sentence-transformers required for cross-encoder reranking. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load cross-encoder model: {e}")
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Re-rank candidates using cross-encoder scores.
        
        Args:
            query: The search query
            candidates: List of candidate results with 'text' field
            top_k: Number of results to return
            
        Returns:
            Re-ranked results sorted by cross-encoder score
        """
        if not candidates:
            return []
        
        self._load_model()
        
        # Prepare query-document pairs
        pairs = [(query, c.get("text", "")) for c in candidates]
        
        # Get cross-encoder scores
        scores = self._model.predict(pairs)
        
        # Add scores to candidates
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)
            candidates[i]["original_score"] = candidates[i].get("score", 0)
        
        # Sort by rerank score
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked[:top_k]
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded


class NoOpReranker:
    """
    Pass-through re-ranker that preserves original order.
    
    Useful as a default when re-ranking is disabled.
    """
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Return candidates unchanged."""
        return candidates[:top_k]


def create_reranker(
    enabled: bool = True,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> Reranker:
    """
    Factory function to create a re-ranker.
    
    Args:
        enabled: Whether to use actual re-ranking
        model_name: Cross-encoder model name
        
    Returns:
        Reranker instance
    """
    if enabled:
        return CrossEncoderReranker(model_name)
    return NoOpReranker()
