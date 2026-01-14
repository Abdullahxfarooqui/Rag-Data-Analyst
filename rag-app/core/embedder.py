"""
Production-grade embeddings using OpenAI API.
Lightweight - no PyTorch/CUDA dependencies.

Uses text-embedding-3-small (1536 dimensions) - fast and affordable.
"""
import os
import hashlib
import requests
from typing import List, Dict, Any, Optional, Callable

# Configuration
MODEL_NAME = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
OPENAI_API_URL = "https://api.openai.com/v1/embeddings"


def _get_api_key() -> str:
    """Get OpenAI API key from Streamlit secrets or environment."""
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY", "")


def get_model_info() -> Dict[str, Any]:
    """Get information about the current embedding model."""
    return {
        "model_name": MODEL_NAME,
        "dimensions": EMBEDDING_DIMS,
        "is_loaded": bool(_get_api_key()),
    }


def truncate_text(text: str, max_chars: int = 8000) -> str:
    """Truncate text to avoid token limits."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def embed_batch_openai(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts using OpenAI API.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    api_key = _get_api_key()
    
    if not api_key:
        print("⚠️ OpenAI API key not configured, returning zero vectors")
        return [[0.0] * EMBEDDING_DIMS for _ in texts]
    
    # Truncate and clean texts
    prepared_texts = [truncate_text(t.replace("\n", " ")) for t in texts]
    
    try:
        response = requests.post(
            OPENAI_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL_NAME,
                "input": prepared_texts
            },
            timeout=60
        )
        response.raise_for_status()
        
        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings
        
    except Exception as e:
        print(f"OpenAI embedding error: {e}")
        return [[0.0] * EMBEDDING_DIMS for _ in texts]


def embed_texts_sequential(
    texts: List[str],
    batch_size: int = 100,  # OpenAI can handle larger batches
    progress_callback: Optional[Callable[[int, int], None]] = None,
    is_query: bool = False
) -> List[List[float]]:
    """
    Embed texts sequentially with batching using OpenAI API.
    
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
    
    all_embeddings = []
    total = len(texts)
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            embeddings = embed_batch_openai(batch)
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"Batch {i} failed: {e}")
            all_embeddings.extend([[0.0] * EMBEDDING_DIMS for _ in batch])
        
        if progress_callback:
            progress_callback(min(i + batch_size, total), total)
    
    return all_embeddings


def embed_query(query: str) -> List[float]:
    """
    Generate embedding for a single query using OpenAI API.
    
    Args:
        query: Query text to embed
        
    Returns:
        Embedding vector
    """
    prepared_query = truncate_text(query.replace("\n", " "))
    
    try:
        embeddings = embed_batch_openai([prepared_query])
        return embeddings[0]
    except Exception as e:
        print(f"Query embedding error: {e}")
        return [0.0] * EMBEDDING_DIMS


def embed_chunks(
    chunks: List[Dict[str, Any]],
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[List[float]]:
    """
    Generate embeddings for document chunks using OpenAI API.
    
    Args:
        chunks: List of chunk dictionaries with 'text' key
        progress_callback: Optional callback(completed, total)
        
    Returns:
        List of embedding vectors
    """
    texts = [chunk.get("text", "") for chunk in chunks]
    return embed_texts_sequential(texts, batch_size=100, progress_callback=progress_callback)


def compute_doc_hash(file_content: bytes) -> str:
    """Compute a unique hash for a document."""
    return hashlib.sha256(file_content).hexdigest()[:16]


def get_embedding_dimensions() -> int:
    """Get the current embedding dimensions."""
    return EMBEDDING_DIMS
