"""
Production-grade embeddings using local sentence-transformers.
No API calls needed - runs entirely locally for embeddings.

Uses all-MiniLM-L6-v2 (384 dimensions) - lightweight model suitable for 
Streamlit Cloud's memory constraints (~1GB RAM limit).

SIMPLIFIED VERSION: No auto-detection, always uses 384d model.
"""
from typing import List, Dict, Any, Optional, Callable
import hashlib

# Configuration - fixed model for stability
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMS = 384

# Module-level state
_embedding_model = None
_model_loaded = False


def _load_embedding_model():
    """Lazy load the embedding model (only once)."""
    global _embedding_model, _model_loaded
    
    if _model_loaded:
        return
    
    _model_loaded = True  # Prevent re-entry
    
    try:
        from sentence_transformers import SentenceTransformer
        print(f"🔄 Loading embedding model: {MODEL_NAME}...")
        _embedding_model = SentenceTransformer(MODEL_NAME)
        print(f"✅ Embedding model loaded: {MODEL_NAME} ({EMBEDDING_DIMS}d)")
    except ImportError:
        print("⚠️ sentence-transformers not installed. Install with: pip install sentence-transformers")
        _embedding_model = None
    except Exception as e:
        print(f"⚠️ Failed to load embedding model: {e}")
        _embedding_model = None


def get_model_info() -> Dict[str, Any]:
    """Get information about the current embedding model."""
    _load_embedding_model()
    return {
        "model_name": MODEL_NAME,
        "dimensions": EMBEDDING_DIMS,
        "is_loaded": _embedding_model is not None,
    }


def truncate_text(text: str, max_chars: int = 8000) -> str:
    """Truncate text to avoid memory issues."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def embed_batch_local(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts using local model.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    _load_embedding_model()
    
    if _embedding_model is None:
        print("⚠️ Local embeddings not available, returning zero vectors")
        return [[0.0] * EMBEDDING_DIMS for _ in texts]
    
    # Truncate long texts
    prepared_texts = [truncate_text(t) for t in texts]
    
    try:
        embeddings = _embedding_model.encode(
            prepared_texts, 
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        return [emb.tolist() for emb in embeddings]
    except Exception as e:
        print(f"Local embedding error: {e}")
        return [[0.0] * EMBEDDING_DIMS for _ in texts]


def embed_texts_sequential(
    texts: List[str],
    batch_size: int = 32,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    is_query: bool = False
) -> List[List[float]]:
    """
    Embed texts sequentially with batching using local model.
    
    Args:
        texts: List of texts to embed
        batch_size: Number of texts per batch
        progress_callback: Optional callback(completed, total)
        is_query: Ignored (kept for compatibility)
        
    Returns:
        List of embedding vectors in order
    """
    if not texts:
        return []
    
    _load_embedding_model()
    all_embeddings = []
    total = len(texts)
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            embeddings = embed_batch_local(batch)
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"Batch {i} failed: {e}")
            all_embeddings.extend([[0.0] * EMBEDDING_DIMS for _ in batch])
        
        if progress_callback:
            progress_callback(min(i + batch_size, total), total)
    
    return all_embeddings


def embed_query(query: str) -> List[float]:
    """
    Generate embedding for a single query using local model.
    
    Args:
        query: Query text to embed
        
    Returns:
        Embedding vector
    """
    _load_embedding_model()
    
    if _embedding_model is None:
        print("⚠️ Local embeddings not available for query")
        return [0.0] * EMBEDDING_DIMS
    
    prepared_query = truncate_text(query)
    
    try:
        embedding = _embedding_model.encode(
            [prepared_query], 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding[0].tolist()
    except Exception as e:
        print(f"Query embedding error: {e}")
        return [0.0] * EMBEDDING_DIMS


def embed_chunks(
    chunks: List[Dict[str, Any]],
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[List[float]]:
    """
    Generate embeddings for document chunks using local model.
    
    Args:
        chunks: List of chunk dictionaries with 'text' key
        progress_callback: Optional callback(completed, total)
        
    Returns:
        List of embedding vectors
    """
    texts = [chunk.get("text", "") for chunk in chunks]
    return embed_texts_sequential(texts, batch_size=32, progress_callback=progress_callback)


def compute_doc_hash(file_content: bytes) -> str:
    """Compute a unique hash for a document."""
    return hashlib.sha256(file_content).hexdigest()[:16]


def get_embedding_dimensions() -> int:
    """Get the current embedding dimensions."""
    return EMBEDDING_DIMS
