"""
Production-grade embeddings using local sentence-transformers.
No API calls needed - runs entirely locally for embeddings.

Phase 6: Upgraded to BAAI/bge-base-en-v1.5 (768 dimensions) for better retrieval quality.
         Auto-detects existing FAISS index dimensions for backward compatibility.
"""
from typing import List, Dict, Any, Optional, Callable
import time
import hashlib
from pathlib import Path
import json
import os

# Configuration - can be overridden via environment variable
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"  # Phase 6: Upgraded from all-MiniLM-L6-v2
FALLBACK_MODEL = "all-MiniLM-L6-v2"  # Fallback if BGE fails to load
LEGACY_MODEL = "all-MiniLM-L6-v2"  # For backward compatibility with existing indices
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)

# Use local sentence-transformers for embeddings (no API needed)
_embedding_model = None
EMBEDDING_DIMS = 384  # Start with legacy default, will be updated on load
HAS_LOCAL_EMBEDDINGS = False
ACTIVE_MODEL_NAME = None
_INDEX_DIMS_DETECTED = None  # Store detected index dimensions


def _detect_existing_index_dimensions() -> Optional[int]:
    """
    Detect dimensions of existing FAISS index for backward compatibility.
    Returns None if no index exists.
    """
    global _INDEX_DIMS_DETECTED
    
    if _INDEX_DIMS_DETECTED is not None:
        return _INDEX_DIMS_DETECTED
    
    # Check multiple possible index locations
    possible_paths = [
        Path("data/faiss_index.bin"),  # workspace root/data
        Path(__file__).parent / "data" / "faiss_index.bin",  # core/data
        Path(__file__).parent.parent / "data" / "faiss_index.bin",  # rag-app/data
        Path(__file__).parent.parent.parent / "data" / "faiss_index.bin",  # Demo AI/data
    ]
    
    for index_path in possible_paths:
        if index_path.exists():
            try:
                import faiss
                index = faiss.read_index(str(index_path))
                _INDEX_DIMS_DETECTED = index.d
                print(f"📐 Detected existing FAISS index ({index_path}) with {_INDEX_DIMS_DETECTED} dimensions")
                return _INDEX_DIMS_DETECTED
            except Exception as e:
                print(f"⚠️ Could not read index at {index_path}: {e}")
    
    return None


def _select_model_for_dimensions(target_dims: int) -> str:
    """Select the appropriate model for target dimensions."""
    if target_dims == 384:
        return LEGACY_MODEL  # all-MiniLM-L6-v2
    elif target_dims == 768:
        return DEFAULT_MODEL  # BAAI/bge-base-en-v1.5
    elif target_dims == 1024:
        return "BAAI/bge-large-en-v1.5"
    else:
        # Unknown dimension, try default
        return DEFAULT_MODEL


def _load_embedding_model(force_dims: Optional[int] = None):
    """
    Lazy load the embedding model.
    
    If an existing FAISS index is detected, loads a compatible model.
    Otherwise, loads the configured default model.
    
    Args:
        force_dims: Force loading a model with specific dimensions
    """
    global _embedding_model, HAS_LOCAL_EMBEDDINGS, EMBEDDING_DIMS, ACTIVE_MODEL_NAME
    
    if _embedding_model is not None:
        return
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Check for existing index dimensions (backward compatibility)
        existing_dims = force_dims or _detect_existing_index_dimensions()
        
        if existing_dims is not None:
            # Load model compatible with existing index
            compatible_model = _select_model_for_dimensions(existing_dims)
            print(f"🔄 Loading model compatible with existing {existing_dims}d index: {compatible_model}")
            model_to_load = compatible_model
        else:
            # No existing index, use configured default
            model_to_load = EMBEDDING_MODEL_NAME
        
        # Try loading the selected model
        try:
            _embedding_model = SentenceTransformer(model_to_load)
            ACTIVE_MODEL_NAME = model_to_load
            
            # Get actual dimensions from loaded model
            EMBEDDING_DIMS = _embedding_model.get_sentence_embedding_dimension()
            
            HAS_LOCAL_EMBEDDINGS = True
            print(f"✅ Embedding model loaded: {model_to_load} ({EMBEDDING_DIMS}d)")
            
        except Exception as e:
            print(f"⚠️ Failed to load {model_to_load}: {e}")
            print(f"⚠️ Falling back to {FALLBACK_MODEL}")
            
            _embedding_model = SentenceTransformer(FALLBACK_MODEL)
            ACTIVE_MODEL_NAME = FALLBACK_MODEL
            EMBEDDING_DIMS = 384
            HAS_LOCAL_EMBEDDINGS = True
            print(f"✅ Fallback model loaded: {FALLBACK_MODEL} ({EMBEDDING_DIMS}d)")
            
    except ImportError:
        HAS_LOCAL_EMBEDDINGS = False
        EMBEDDING_DIMS = 384  # Default to legacy dimensions
        _embedding_model = None
        print("⚠️ sentence-transformers not installed. Install with: pip install sentence-transformers")


def get_model_info() -> Dict[str, Any]:
    """Get information about the current embedding model."""
    _load_embedding_model()
    return {
        "model_name": ACTIVE_MODEL_NAME,
        "dimensions": EMBEDDING_DIMS,
        "is_loaded": HAS_LOCAL_EMBEDDINGS,
        "configured_model": EMBEDDING_MODEL_NAME,
        "fallback_model": FALLBACK_MODEL,
        "detected_index_dims": _INDEX_DIMS_DETECTED,
        "is_legacy_mode": ACTIVE_MODEL_NAME == LEGACY_MODEL
    }


def reset_to_new_model():
    """
    Reset embedder to use the new model (BGE).
    Call this when rebuilding the index from scratch.
    
    Returns:
        Dict with new model info
    """
    global _embedding_model, _INDEX_DIMS_DETECTED
    
    # Clear cached state
    _embedding_model = None
    _INDEX_DIMS_DETECTED = None  # Ignore existing index
    
    # Force load with new model dimensions (768 for BGE)
    _load_embedding_model(force_dims=768)
    
    return get_model_info()


def truncate_text(text: str, max_chars: int = 8000) -> str:
    """Truncate text to avoid memory issues."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _prepare_text_for_bge(text: str, is_query: bool = False) -> str:
    """
    Prepare text for BGE model embedding.
    
    BGE models work better with instruction prefixes for queries.
    """
    text = truncate_text(text)
    
    # BGE query prefix for better retrieval
    if is_query and ACTIVE_MODEL_NAME and "bge" in ACTIVE_MODEL_NAME.lower():
        return f"Represent this sentence for searching relevant passages: {text}"
    
    return text


def embed_batch_local(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """
    Embed a batch of texts using local model.
    
    Args:
        texts: List of texts to embed
        is_query: Whether these are query texts (adds BGE prefix)
        
    Returns:
        List of embedding vectors
    """
    _load_embedding_model()
    
    if not HAS_LOCAL_EMBEDDINGS or _embedding_model is None:
        print("⚠️ Local embeddings not available, returning zero vectors")
        return [[0.0] * EMBEDDING_DIMS for _ in texts]
    
    # Prepare texts (with BGE prefix for queries)
    prepared_texts = [_prepare_text_for_bge(t, is_query) for t in texts]
    
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
        is_query: Whether these are query texts
        
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
            embeddings = embed_batch_local(batch, is_query=is_query)
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"Batch {i} failed: {e}")
            # Return zero vectors for failed batch
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
    
    if not HAS_LOCAL_EMBEDDINGS or _embedding_model is None:
        print("⚠️ Local embeddings not available for query")
        return [0.0] * EMBEDDING_DIMS
    
    # Use BGE query prefix
    prepared_query = _prepare_text_for_bge(query, is_query=True)
    
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
    return embed_texts_sequential(texts, batch_size=32, progress_callback=progress_callback, is_query=False)


def compute_doc_hash(file_content: bytes) -> str:
    """Compute a unique hash for a document."""
    return hashlib.sha256(file_content).hexdigest()[:16]


def get_embedding_dimensions() -> int:
    """Get the current embedding dimensions."""
    _load_embedding_model()
    return EMBEDDING_DIMS
