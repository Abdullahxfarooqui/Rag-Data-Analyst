"""
Pure NumPy vector store with intelligent retrieval.
No FAISS dependency - works on any platform including Streamlit Cloud.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

# Default embedding dimension (OpenAI text-embedding-3-small = 1536)
DEFAULT_EMBEDDING_DIMS = 1536

# Storage paths - use absolute path relative to this file's location
_MODULE_DIR = Path(__file__).parent  # core/
_APP_DIR = _MODULE_DIR.parent  # rag-app/
DATA_DIR = _APP_DIR / "data"
VECTORS_PATH = DATA_DIR / "vectors.npy"
METADATA_PATH = DATA_DIR / "chunks_metadata.json"


def ensure_data_dir():
    """Ensure data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_embedding_dims() -> int:
    """Get current embedding dimensions from embedder module."""
    try:
        from core.embedder import get_embedding_dimensions
        return get_embedding_dimensions()
    except ImportError:
        return DEFAULT_EMBEDDING_DIMS


class VectorStore:
    """
    Pure NumPy vector store for RAG.
    Uses cosine similarity for search - no FAISS required.
    """
    
    def __init__(self, dimension: int = None):
        """Initialize vector store."""
        self._requested_dimension = dimension
        self.vectors: Optional[np.ndarray] = None  # Shape: (n, dimension)
        self.chunks: List[Dict[str, Any]] = []
        self._chunk_hashes: set = set()
        self._doc_chunks_cache: Dict[str, List[int]] = {}
        
        ensure_data_dir()
        self._load()
        
        # Set dimension
        if self.vectors is not None and len(self.vectors) > 0:
            self.dimension = self.vectors.shape[1]
        elif self._requested_dimension is not None:
            self.dimension = self._requested_dimension
        else:
            self.dimension = get_embedding_dims()
    
    def _load(self) -> bool:
        """Load existing index from disk."""
        if VECTORS_PATH.exists() and METADATA_PATH.exists():
            try:
                self.vectors = np.load(str(VECTORS_PATH))
                with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
                
                self._chunk_hashes = set(c.get("hash", "") for c in self.chunks if c.get("hash"))
                self._rebuild_doc_cache()
                return True
            except Exception as e:
                print(f"Error loading index: {e}")
                self.vectors = None
                self.chunks = []
        return False
    
    def _save(self) -> None:
        """Save index to disk."""
        if self.vectors is None:
            return
        
        ensure_data_dir()
        np.save(str(VECTORS_PATH), self.vectors)
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f)
    
    def _rebuild_doc_cache(self) -> None:
        """Rebuild document-to-chunks cache."""
        self._doc_chunks_cache = {}
        for i, chunk in enumerate(self.chunks):
            doc_hash = chunk.get("doc_hash", "")
            if doc_hash:
                if doc_hash not in self._doc_chunks_cache:
                    self._doc_chunks_cache[doc_hash] = []
                self._doc_chunks_cache[doc_hash].append(i)
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
    
    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        doc_hash: str
    ) -> int:
        """Add chunks with their embeddings."""
        if not chunks or not embeddings or len(chunks) != len(embeddings):
            return 0
        
        # Filter duplicates
        new_chunks = []
        new_embeddings = []
        
        for chunk, emb in zip(chunks, embeddings):
            chunk_hash = chunk.get("hash", "")
            if chunk_hash and chunk_hash in self._chunk_hashes:
                continue
            
            chunk["doc_hash"] = doc_hash
            new_chunks.append(chunk)
            new_embeddings.append(emb)
            
            if chunk_hash:
                self._chunk_hashes.add(chunk_hash)
        
        if not new_chunks:
            return 0
        
        # Convert to numpy and normalize
        new_vectors = np.array(new_embeddings, dtype=np.float32)
        new_vectors = self._normalize(new_vectors)
        
        # Initialize or extend vectors
        start_idx = len(self.chunks)
        
        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])
        
        self.chunks.extend(new_chunks)
        
        # Update doc cache
        for i, chunk in enumerate(new_chunks):
            idx = start_idx + i
            if doc_hash not in self._doc_chunks_cache:
                self._doc_chunks_cache[doc_hash] = []
            self._doc_chunks_cache[doc_hash].append(idx)
        
        self._save()
        return len(new_chunks)
    
    @property
    def index(self):
        """Compatibility property - returns self if vectors exist."""
        return self if self.vectors is not None and len(self.vectors) > 0 else None
    
    @property
    def ntotal(self) -> int:
        """Total number of vectors (FAISS compatibility)."""
        return len(self.vectors) if self.vectors is not None else 0
    
    @property
    def d(self) -> int:
        """Dimension of vectors (FAISS compatibility)."""
        return self.dimension
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        doc_hash: Optional[str] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar chunks using cosine similarity."""
        if self.vectors is None or len(self.vectors) == 0:
            return []
        
        # Normalize query
        query = np.array([query_embedding], dtype=np.float32)
        query = self._normalize(query)
        
        # Handle dimension mismatch
        query_dim = query.shape[1]
        index_dim = self.vectors.shape[1]
        
        if query_dim != index_dim:
            print(f"⚠️ Dimension mismatch: query={query_dim}, index={index_dim}")
            if query_dim > index_dim:
                query = query[:, :index_dim]
            else:
                padding = np.zeros((1, index_dim - query_dim), dtype=np.float32)
                query = np.concatenate([query, padding], axis=1)
                query = self._normalize(query)
        
        # Compute cosine similarity (dot product of normalized vectors)
        scores = np.dot(self.vectors, query.T).flatten()
        
        # Get top-k indices
        if doc_hash:
            # Filter by document
            valid_indices = self._doc_chunks_cache.get(doc_hash, [])
            if not valid_indices:
                return []
            
            doc_scores = [(idx, scores[idx]) for idx in valid_indices]
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in doc_scores[:k]]
            top_scores = [score for _, score in doc_scores[:k]]
        else:
            top_indices = np.argsort(scores)[::-1][:k]
            top_scores = scores[top_indices]
        
        results = []
        for idx, score in zip(top_indices, top_scores):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def search_with_context(
        self,
        query_embedding: List[float],
        k: int = 10,
        context_window: int = 2,
        doc_hash: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search and include adjacent chunks for context."""
        base_results = self.search(query_embedding, k=k, doc_hash=doc_hash)
        
        if not base_results:
            return []
        
        # Group results by document
        doc_results: Dict[str, List[Tuple[int, float]]] = {}
        
        for chunk, score in base_results:
            d_hash = chunk.get("doc_hash", "")
            chunk_id = chunk.get("chunk_id", 0)
            
            if d_hash not in doc_results:
                doc_results[d_hash] = []
            doc_results[d_hash].append((chunk_id, score))
        
        # Build context-enriched results
        enriched_results = []
        
        for d_hash, matches in doc_results.items():
            doc_indices = self._doc_chunks_cache.get(d_hash, [])
            doc_chunks = [(self.chunks[i], i) for i in doc_indices]
            doc_chunks.sort(key=lambda x: x[0].get("chunk_id", 0))
            
            id_to_pos = {c[0].get("chunk_id"): pos for pos, c in enumerate(doc_chunks)}
            
            for chunk_id, score in matches:
                if chunk_id not in id_to_pos:
                    continue
                
                pos = id_to_pos[chunk_id]
                start_pos = max(0, pos - context_window)
                end_pos = min(len(doc_chunks), pos + context_window + 1)
                
                context_chunks = [doc_chunks[i][0] for i in range(start_pos, end_pos)]
                merged_text = "\n\n---\n\n".join(c.get("text", "") for c in context_chunks)
                
                result = {
                    "text": merged_text,
                    "score": score,
                    "doc_hash": d_hash,
                    "filename": context_chunks[0].get("filename", ""),
                    "is_dataset": context_chunks[0].get("is_dataset", False),
                    "chunk_ids": [c.get("chunk_id") for c in context_chunks],
                    "num_chunks": len(context_chunks),
                    "type": context_chunks[pos - start_pos].get("type", "text") if context_chunks else "text"
                }
                
                enriched_results.append(result)
        
        enriched_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return enriched_results[:k]
    
    def get_document_chunks(self, doc_hash: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document."""
        indices = self._doc_chunks_cache.get(doc_hash, [])
        chunks = [self.chunks[i] for i in indices if i < len(self.chunks)]
        chunks.sort(key=lambda x: x.get("chunk_id", 0))
        return chunks
    
    def get_full_document_text(self, doc_hash: str) -> str:
        """Get complete text for a document."""
        chunks = self.get_document_chunks(doc_hash)
        return "\n\n".join(c.get("text", "") for c in chunks)
    
    def get_document_tables(self, doc_hash: str) -> List[Dict[str, Any]]:
        """Get all table chunks for a document."""
        chunks = self.get_document_chunks(doc_hash)
        return [c for c in chunks if c.get("type") == "table"]
    
    def has_document(self, doc_hash: str) -> bool:
        """Check if document is indexed."""
        return doc_hash in self._doc_chunks_cache
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get summary of all indexed documents."""
        docs = []
        for doc_hash in self._doc_chunks_cache.keys():
            chunks = self.get_document_chunks(doc_hash)
            if chunks:
                docs.append({
                    "doc_hash": doc_hash,
                    "filename": chunks[0].get("filename", "unknown"),
                    "num_chunks": len(chunks),
                    "is_dataset": chunks[0].get("is_dataset", False)
                })
        return docs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(self._doc_chunks_cache),
            "total_vectors": len(self.vectors) if self.vectors is not None else 0,
            "dimension": self.dimension
        }
    
    def clear(self) -> None:
        """Clear entire index."""
        self.vectors = None
        self.chunks = []
        self._chunk_hashes = set()
        self._doc_chunks_cache = {}
        
        if VECTORS_PATH.exists():
            VECTORS_PATH.unlink()
        if METADATA_PATH.exists():
            METADATA_PATH.unlink()
    
    def delete_document(self, doc_hash: str) -> int:
        """Delete all chunks for a document."""
        if doc_hash not in self._doc_chunks_cache:
            return 0
        
        count = len(self._doc_chunks_cache[doc_hash])
        
        # Get indices to keep
        indices_to_delete = set(self._doc_chunks_cache[doc_hash])
        indices_to_keep = [i for i in range(len(self.chunks)) if i not in indices_to_delete]
        
        if indices_to_keep and self.vectors is not None:
            self.vectors = self.vectors[indices_to_keep]
            self.chunks = [self.chunks[i] for i in indices_to_keep]
            self._rebuild_doc_cache()
            self._chunk_hashes = set(c.get("hash", "") for c in self.chunks if c.get("hash"))
            self._save()
        else:
            self.clear()
        
        return count


# Singleton instance
_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get global vector store instance."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def get_index_dimensions() -> Optional[int]:
    """Get dimensions of the current index, if loaded."""
    store = get_vector_store()
    if store.vectors is not None and len(store.vectors) > 0:
        return store.vectors.shape[1]
    return None


def reset_vector_store() -> None:
    """Reset global vector store instance."""
    global _store
    _store = None
