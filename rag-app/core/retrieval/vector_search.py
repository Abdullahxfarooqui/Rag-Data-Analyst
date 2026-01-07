"""
Vector Search using FAISS.

Provides semantic search over document chunks using embeddings.
Wraps the existing VectorStore with a cleaner interface.
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from core.vector_store import VectorStore, get_vector_store


@dataclass
class SearchResult:
    """A single search result with metadata."""
    text: str
    score: float
    doc_hash: str
    filename: str
    chunk_id: int
    chunk_type: str
    metadata: Dict[str, Any]
    
    @property
    def is_table(self) -> bool:
        return self.chunk_type == "table"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
            "doc_hash": self.doc_hash,
            "filename": self.filename,
            "chunk_id": self.chunk_id,
            "type": self.chunk_type,
            "metadata": self.metadata
        }


class VectorSearcher:
    """
    Semantic search interface using FAISS vector store.
    
    Features:
    - Single document or cross-document search
    - Context window expansion
    - Score normalization
    - Type filtering
    """
    
    def __init__(self, store: Optional[VectorStore] = None):
        """
        Initialize vector searcher.
        
        Args:
            store: VectorStore instance (uses global if None)
        """
        self._store = store
    
    @property
    def store(self) -> VectorStore:
        """Get the underlying vector store."""
        if self._store is None:
            self._store = get_vector_store()
        return self._store
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        doc_hash: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            doc_hash: Optional document filter
            min_score: Minimum similarity score threshold
            
        Returns:
            List of SearchResult objects sorted by score
        """
        raw_results = self.store.search(query_embedding, k=k, doc_hash=doc_hash)
        
        results = []
        for chunk, score in raw_results:
            if score < min_score:
                continue
            
            results.append(SearchResult(
                text=chunk.get("text", ""),
                score=self._normalize_score(score),
                doc_hash=chunk.get("doc_hash", ""),
                filename=chunk.get("filename", "unknown"),
                chunk_id=chunk.get("chunk_id", 0),
                chunk_type=chunk.get("type", "text"),
                metadata=chunk.get("metadata", {})
            ))
        
        return results
    
    def search_with_context(
        self,
        query_embedding: List[float],
        k: int = 10,
        context_window: int = 2,
        doc_hash: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search with adjacent chunk expansion.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            context_window: Number of adjacent chunks to include
            doc_hash: Optional document filter
            
        Returns:
            List of SearchResult with expanded context
        """
        raw_results = self.store.search_with_context(
            query_embedding,
            k=k,
            context_window=context_window,
            doc_hash=doc_hash
        )
        
        results = []
        for item in raw_results:
            results.append(SearchResult(
                text=item.get("text", ""),
                score=self._normalize_score(item.get("score", 0)),
                doc_hash=item.get("doc_hash", ""),
                filename=item.get("filename", "unknown"),
                chunk_id=item.get("chunk_ids", [0])[0] if item.get("chunk_ids") else 0,
                chunk_type=item.get("type", "text"),
                metadata={"num_chunks": item.get("num_chunks", 1)}
            ))
        
        return results
    
    def get_document_chunks(self, doc_hash: str) -> List[SearchResult]:
        """
        Get all chunks for a specific document.
        
        Args:
            doc_hash: Document hash
            
        Returns:
            List of all chunks sorted by chunk_id
        """
        chunks = self.store.get_document_chunks(doc_hash)
        
        return [
            SearchResult(
                text=chunk.get("text", ""),
                score=1.0,  # Not a search result
                doc_hash=doc_hash,
                filename=chunk.get("filename", "unknown"),
                chunk_id=chunk.get("chunk_id", 0),
                chunk_type=chunk.get("type", "text"),
                metadata=chunk.get("metadata", {})
            )
            for chunk in chunks
        ]
    
    def get_full_document_text(self, doc_hash: str) -> str:
        """Get concatenated text for a document."""
        return self.store.get_full_document_text(doc_hash)
    
    def get_document_tables(self, doc_hash: str) -> List[SearchResult]:
        """Get only table chunks for a document."""
        tables = self.store.get_document_tables(doc_hash)
        
        return [
            SearchResult(
                text=table.get("text", ""),
                score=1.0,
                doc_hash=doc_hash,
                filename=table.get("filename", "unknown"),
                chunk_id=table.get("chunk_id", 0),
                chunk_type="table",
                metadata=table.get("metadata", {})
            )
            for table in tables
        ]
    
    def has_document(self, doc_hash: str) -> bool:
        """Check if a document is indexed."""
        return self.store.has_document(doc_hash)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents."""
        return self.store.get_all_documents()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return self.store.get_stats()
    
    @staticmethod
    def _normalize_score(raw_score: float) -> float:
        """
        Normalize cosine similarity score to 0-1 range.
        
        Raw cosine similarity typically ranges:
        - 0.3-0.4: Weak match
        - 0.4-0.5: Moderate match
        - 0.5-0.6: Good match
        - 0.6+: Strong match
        """
        if raw_score > 0.7:
            return 0.95 + (raw_score - 0.7) * 0.15
        elif raw_score > 0.6:
            return 0.90 + (raw_score - 0.6) * 0.5
        elif raw_score > 0.5:
            return 0.80 + (raw_score - 0.5) * 1.0
        elif raw_score > 0.4:
            return 0.65 + (raw_score - 0.4) * 1.5
        elif raw_score > 0.3:
            return 0.45 + (raw_score - 0.3) * 2.0
        else:
            return raw_score * 1.5


# Singleton
_vector_searcher: Optional[VectorSearcher] = None


def get_vector_searcher() -> VectorSearcher:
    """Get global vector searcher instance."""
    global _vector_searcher
    if _vector_searcher is None:
        _vector_searcher = VectorSearcher()
    return _vector_searcher


def reset_vector_searcher() -> None:
    """Reset global instance (for testing)."""
    global _vector_searcher
    _vector_searcher = None
