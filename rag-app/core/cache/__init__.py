"""
Cache package.

This package provides both the original TTL cache system (for backward compatibility)
and new production-grade caching capabilities:

Original (backward compatible):
- TTLCache with automatic expiration
- Global cache instances (llm_cache, search_cache, etc.)
- hash_key function

New production features:
- LRU in-memory caching
- Disk-based persistent caching
- Multi-tier caching (memory + disk)
- Specialized caches (FAISS, LLM, Embeddings)
"""

# ============================================================================
# BACKWARD COMPATIBLE IMPORTS (from original cache.py, now ttl_cache.py)
# ============================================================================

from core.ttl_cache import (
    TTLCache,
    CacheEntry as TTLCacheEntry,
    hash_key,
    cached as ttl_cached,
    llm_cache,
    classification_cache,
    search_cache,
    embedding_cache,
    dataframe_cache,
    get_all_cache_stats,
    clear_all_caches,
)

# Alias for backward compatibility
cached = ttl_cached


# ============================================================================
# NEW PRODUCTION CACHE IMPORTS
# ============================================================================

from .production_cache import (
    # Key generation
    generate_cache_key,
    semantic_hash,
    
    # Caches
    LRUCache,
    DiskCache,
    TieredCache,
    CacheEntry,
    
    # Specialized
    FAISSRetrievalCache,
    LLMResponseCache,
    EmbeddingCache,
    
    # Decorators
    memoize,
    
    # Manager
    CacheManager,
    cache_manager,
)

__all__ = [
    # Backward compatible
    "TTLCache",
    "TTLCacheEntry",
    "hash_key",
    "cached",
    "ttl_cached",
    "llm_cache",
    "classification_cache",
    "search_cache",
    "embedding_cache", 
    "dataframe_cache",
    "get_all_cache_stats",
    "clear_all_caches",
    
    # New production features
    "generate_cache_key",
    "semantic_hash",
    "LRUCache",
    "DiskCache",
    "TieredCache",
    "CacheEntry",
    "FAISSRetrievalCache",
    "LLMResponseCache",
    "EmbeddingCache",
    "memoize",
    "CacheManager",
    "cache_manager",
]
