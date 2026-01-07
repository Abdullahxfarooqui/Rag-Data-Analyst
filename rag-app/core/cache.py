"""
In-memory TTL cache for RAG system.

Provides caching for:
- LLM responses (keyed by prompt hash)
- Query classifications
- Search results
- DataFrame references

Phase 2 implementation as per refactoring plan.
"""
from __future__ import annotations
import hashlib
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TypeVar, Generic, Callable
from functools import wraps

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Single cache entry with TTL tracking."""
    value: T
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 300.0  # Default 5 minutes
    hits: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds
    
    def touch(self) -> None:
        """Record a cache hit."""
        self.hits += 1


class TTLCache(Generic[T]):
    """
    Thread-safe in-memory cache with TTL expiration.
    
    Features:
    - Automatic expiration based on TTL
    - Thread-safe operations
    - Hit/miss statistics
    - Configurable max size with LRU eviction
    
    Usage:
        cache = TTLCache[str](default_ttl=300, max_size=1000)
        cache.set("key", "value")
        result = cache.get("key")  # Returns "value" or None if expired
    """
    
    def __init__(
        self,
        default_ttl: float = 300.0,
        max_size: int = 1000,
        cleanup_interval: float = 60.0
    ):
        """
        Initialize cache.
        
        Args:
            default_ttl: Default time-to-live in seconds
            max_size: Maximum number of entries (0 = unlimited)
            cleanup_interval: How often to run cleanup (seconds)
        """
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        
        # Statistics
        self._hits = 0
        self._misses = 0
    
    def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed."""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = now
    
    def _cleanup_expired(self) -> None:
        """Remove all expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries if over max size."""
        if self._max_size <= 0:
            return
            
        with self._lock:
            while len(self._cache) >= self._max_size:
                # Find entry with oldest created_at (simple LRU)
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].created_at
                )
                del self._cache[oldest_key]
    
    def get(self, key: str) -> Optional[T]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        self._maybe_cleanup()
        
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None
            
            entry.touch()
            self._hits += 1
            return entry.value
    
    def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None
    ) -> None:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Time-to-live in seconds (uses default if None)
        """
        self._maybe_cleanup()
        self._evict_lru()
        
        with self._lock:
            self._cache[key] = CacheEntry(
                value=value,
                ttl_seconds=ttl if ttl is not None else self._default_ttl
            )
    
    def delete(self, key: str) -> bool:
        """
        Remove entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry existed, False otherwise
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Remove all entries from cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def has(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None
    
    @property
    def size(self) -> int:
        """Current number of entries (including expired)."""
        return len(self._cache)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        
        return {
            "size": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
            "max_size": self._max_size,
            "default_ttl": self._default_ttl
        }


def hash_key(*args, **kwargs) -> str:
    """
    Generate a cache key from arguments.
    
    Creates MD5 hash of string representation of args/kwargs.
    """
    key_str = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_str.encode()).hexdigest()


def cached(
    cache: TTLCache,
    ttl: Optional[float] = None,
    key_fn: Optional[Callable[..., str]] = None
):
    """
    Decorator to cache function results.
    
    Args:
        cache: TTLCache instance to use
        ttl: Time-to-live for cached results
        key_fn: Custom function to generate cache key from args
        
    Usage:
        cache = TTLCache[str](default_ttl=300)
        
        @cached(cache, ttl=60)
        def expensive_function(x, y):
            return x + y
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = f"{func.__name__}:{hash_key(*args, **kwargs)}"
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(key, result, ttl)
            
            return result
        
        # Attach cache reference for manual operations
        wrapper.cache = cache
        wrapper.cache_key_fn = key_fn or (lambda *a, **kw: f"{func.__name__}:{hash_key(*a, **kw)}")
        
        return wrapper
    
    return decorator


# ============================================================================
# Global Cache Instances
# ============================================================================

# LLM response cache - longer TTL for expensive calls
llm_cache: TTLCache[str] = TTLCache(
    default_ttl=600.0,  # 10 minutes
    max_size=500
)

# Query classification cache - medium TTL
classification_cache: TTLCache[Dict[str, Any]] = TTLCache(
    default_ttl=300.0,  # 5 minutes
    max_size=1000
)

# Search results cache - shorter TTL
search_cache: TTLCache[Any] = TTLCache(
    default_ttl=120.0,  # 2 minutes
    max_size=500
)

# Embedding cache - longer TTL since embeddings don't change
embedding_cache: TTLCache[Any] = TTLCache(
    default_ttl=3600.0,  # 1 hour
    max_size=2000
)

# DataFrame reference cache - very long TTL
dataframe_cache: TTLCache[Any] = TTLCache(
    default_ttl=7200.0,  # 2 hours
    max_size=50  # DataFrames are large, limit count
)


def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all global caches."""
    return {
        "llm_cache": llm_cache.stats,
        "classification_cache": classification_cache.stats,
        "search_cache": search_cache.stats,
        "embedding_cache": embedding_cache.stats,
        "dataframe_cache": dataframe_cache.stats
    }


def clear_all_caches() -> None:
    """Clear all global caches."""
    llm_cache.clear()
    classification_cache.clear()
    search_cache.clear()
    embedding_cache.clear()
    dataframe_cache.clear()
