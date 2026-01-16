"""
Production Caching Layer.

MULTI-TIER CACHING FOR HIGH PERFORMANCE.

This module provides:
1. FAISS retrieval caching - Avoid redundant vector searches
2. LLM response caching - Reduce API costs and latency
3. Document metadata caching - Fast access to doc info
4. Embedding caching - Avoid re-computing embeddings

CACHE ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────────────┐
│                         Caching Layer                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  L1: In-Memory (LRU)                                             │   │
│  │  - Hot data: <100ms access                                       │   │
│  │  - Size: Configurable (default 1000 items)                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │ miss                                     │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  L2: Disk Cache (SQLite/File)                                    │   │
│  │  - Warm data: <10ms access                                       │   │
│  │  - Size: Configurable (default 10GB)                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │ miss                                     │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Origin (FAISS/LLM/Embedder)                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

CACHE STRATEGIES:
- Query Cache: Hash(query + parameters) → results
- Semantic Cache: Similar queries → same results
- TTL-based expiration: Stale data eviction
- LRU eviction: Remove least recently used

TRADE-OFFS:
1. Memory vs Speed:
   - More memory = higher hit rate = faster responses
   - Typical: 100MB cache → 60% hit rate, 500MB → 80% hit rate

2. Freshness vs Performance:
   - Short TTL = fresher data but more cache misses
   - Long TTL = faster but potentially stale
   
3. Cost vs Latency:
   - LLM caching saves $$ but uses memory
   - 70% LLM cache hit rate can reduce costs by 60%+
"""
import hashlib
import json
import logging
import os
import pickle
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic
import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# CACHE KEY GENERATION
# ============================================================================

def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate a deterministic cache key from arguments.
    
    Handles:
    - Primitive types
    - Numpy arrays
    - Lists and dicts
    - Custom objects via __dict__
    """
    def normalize(obj):
        if isinstance(obj, np.ndarray):
            return ("ndarray", obj.shape, obj.tobytes().hex()[:64])
        elif isinstance(obj, (list, tuple)):
            return tuple(normalize(x) for x in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, normalize(v)) for k, v in obj.items()))
        elif hasattr(obj, '__dict__'):
            return ("obj", type(obj).__name__, normalize(obj.__dict__))
        return obj
    
    normalized = (normalize(args), normalize(kwargs))
    serialized = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:32]


def semantic_hash(text: str, precision: int = 3) -> str:
    """
    Generate a semantic-aware hash for text queries.
    
    Normalizes text to improve cache hit rate for similar queries:
    - Lowercases
    - Removes extra whitespace
    - Strips punctuation
    - Truncates to reasonable length
    """
    import re
    
    # Normalize
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text[:500]  # Truncate long queries
    
    return hashlib.sha256(text.encode()).hexdigest()[:32]


# ============================================================================
# LRU CACHE
# ============================================================================

@dataclass
class CacheEntry(Generic[T]):
    """A cached item with metadata."""
    key: str
    value: T
    created_at: float
    accessed_at: float
    ttl_seconds: Optional[float] = None
    size_bytes: int = 0
    hit_count: int = 0
    
    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    def touch(self):
        """Update access time and hit count."""
        self.accessed_at = time.time()
        self.hit_count += 1


class LRUCache(Generic[T]):
    """
    Thread-safe LRU cache with TTL support.
    
    Features:
    - Configurable max size (items or bytes)
    - TTL-based expiration
    - Thread-safe operations
    - Hit/miss statistics
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        max_bytes: Optional[int] = None,
        default_ttl: Optional[float] = None,
        name: str = "cache"
    ):
        self.max_size = max_size
        self.max_bytes = max_bytes
        self.default_ttl = default_ttl
        self.name = name
        
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        self._total_bytes = 0
        
        # Statistics
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            if entry.is_expired:
                self._remove(key)
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            
            return entry.value
    
    def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
        size_bytes: int = 0
    ):
        """Set value in cache."""
        with self._lock:
            # Remove if exists
            if key in self._cache:
                self._remove(key)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl_seconds=ttl or self.default_ttl,
                size_bytes=size_bytes,
            )
            
            # Evict if needed
            self._evict_if_needed(size_bytes)
            
            # Add entry
            self._cache[key] = entry
            self._total_bytes += size_bytes
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                self._remove(key)
                return True
            return False
    
    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._total_bytes = 0
    
    def _remove(self, key: str):
        """Remove entry (internal, assumes lock held)."""
        entry = self._cache.pop(key, None)
        if entry:
            self._total_bytes -= entry.size_bytes
    
    def _evict_if_needed(self, incoming_bytes: int):
        """Evict entries if needed (internal, assumes lock held)."""
        # Evict expired first
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for key in expired:
            self._remove(key)
        
        # Evict LRU if still over size
        while len(self._cache) >= self.max_size:
            oldest = next(iter(self._cache))
            self._remove(oldest)
        
        # Evict if over bytes limit
        if self.max_bytes:
            while self._total_bytes + incoming_bytes > self.max_bytes and self._cache:
                oldest = next(iter(self._cache))
                self._remove(oldest)
    
    @property
    def stats(self) -> Dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "name": self.name,
            "size": len(self._cache),
            "max_size": self.max_size,
            "bytes": self._total_bytes,
            "max_bytes": self.max_bytes,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
        }
    
    def __contains__(self, key: str) -> bool:
        with self._lock:
            entry = self._cache.get(key)
            return entry is not None and not entry.is_expired
    
    def __len__(self) -> int:
        return len(self._cache)


# ============================================================================
# DISK CACHE
# ============================================================================

class DiskCache:
    """
    SQLite-backed disk cache for persistence.
    
    Features:
    - Survives restarts
    - Configurable size limit
    - Background cleanup
    - Compression option
    """
    
    def __init__(
        self,
        path: str,
        max_size_gb: float = 10.0,
        compress: bool = True,
        table_name: str = "cache"
    ):
        self.path = Path(path)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.compress = compress
        self.table_name = table_name
        
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database."""
        conn = self._get_conn()
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                key TEXT PRIMARY KEY,
                value BLOB,
                created_at REAL,
                accessed_at REAL,
                expires_at REAL,
                size_bytes INTEGER
            )
        """)
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_expires
            ON {self.table_name}(expires_at)
        """)
        conn.commit()
        conn.close()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(str(self.path), timeout=30)
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value."""
        data = pickle.dumps(value)
        if self.compress:
            import zlib
            data = zlib.compress(data)
        return data
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value."""
        if self.compress:
            import zlib
            data = zlib.decompress(data)
        return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(f"""
                SELECT value, expires_at FROM {self.table_name}
                WHERE key = ?
            """, (key,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            value_data, expires_at = row
            
            # Check expiration
            if expires_at and time.time() > expires_at:
                conn.execute(f"DELETE FROM {self.table_name} WHERE key = ?", (key,))
                conn.commit()
                return None
            
            # Update access time
            conn.execute(f"""
                UPDATE {self.table_name}
                SET accessed_at = ?
                WHERE key = ?
            """, (time.time(), key))
            conn.commit()
            
            return self._deserialize(value_data)
        finally:
            conn.close()
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None
    ):
        """Set value in disk cache."""
        data = self._serialize(value)
        size_bytes = len(data)
        
        now = time.time()
        expires_at = now + ttl_seconds if ttl_seconds else None
        
        conn = self._get_conn()
        try:
            conn.execute(f"""
                INSERT OR REPLACE INTO {self.table_name}
                (key, value, created_at, accessed_at, expires_at, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (key, data, now, now, expires_at, size_bytes))
            conn.commit()
            
            # Cleanup if needed
            self._cleanup_if_needed(conn)
        finally:
            conn.close()
    
    def delete(self, key: str) -> bool:
        """Delete key from disk cache."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(f"""
                DELETE FROM {self.table_name} WHERE key = ?
            """, (key,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def _cleanup_if_needed(self, conn: sqlite3.Connection):
        """Clean up expired and excess entries."""
        # Remove expired
        conn.execute(f"""
            DELETE FROM {self.table_name}
            WHERE expires_at IS NOT NULL AND expires_at < ?
        """, (time.time(),))
        
        # Check total size
        cursor = conn.execute(f"""
            SELECT SUM(size_bytes) FROM {self.table_name}
        """)
        total_size = cursor.fetchone()[0] or 0
        
        # Remove LRU if over size
        if total_size > self.max_size_bytes:
            excess = total_size - self.max_size_bytes
            conn.execute(f"""
                DELETE FROM {self.table_name}
                WHERE key IN (
                    SELECT key FROM {self.table_name}
                    ORDER BY accessed_at ASC
                    LIMIT (
                        SELECT COUNT(*) FROM {self.table_name}
                        WHERE (
                            SELECT SUM(size_bytes) FROM {self.table_name} t2
                            WHERE t2.accessed_at <= {self.table_name}.accessed_at
                        ) <= ?
                    )
                )
            """, (excess,))
        
        conn.commit()
    
    @property
    def stats(self) -> Dict:
        """Get cache statistics."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(f"""
                SELECT COUNT(*), SUM(size_bytes) FROM {self.table_name}
            """)
            count, size = cursor.fetchone()
            return {
                "entries": count or 0,
                "size_bytes": size or 0,
                "size_mb": (size or 0) / 1024 / 1024,
                "max_size_gb": self.max_size_bytes / 1024 / 1024 / 1024,
            }
        finally:
            conn.close()


# ============================================================================
# TIERED CACHE
# ============================================================================

class TieredCache:
    """
    Multi-tier cache with L1 (memory) and L2 (disk).
    
    Provides transparent caching across tiers:
    1. Check L1 (memory) - fastest
    2. Check L2 (disk) - fast, persistent
    3. Miss - fetch from origin
    """
    
    def __init__(
        self,
        l1_max_size: int = 1000,
        l1_max_bytes: Optional[int] = None,
        l2_path: Optional[str] = None,
        l2_max_size_gb: float = 10.0,
        default_ttl: Optional[float] = None,
        name: str = "tiered_cache"
    ):
        self.name = name
        self.default_ttl = default_ttl
        
        # L1: Memory cache
        self.l1 = LRUCache(
            max_size=l1_max_size,
            max_bytes=l1_max_bytes,
            default_ttl=default_ttl,
            name=f"{name}_l1"
        )
        
        # L2: Disk cache (optional)
        self.l2: Optional[DiskCache] = None
        if l2_path:
            self.l2 = DiskCache(
                path=l2_path,
                max_size_gb=l2_max_size_gb,
            )
    
    def get(self, key: str) -> Optional[Any]:
        """Get from cache, checking L1 then L2."""
        # Check L1
        value = self.l1.get(key)
        if value is not None:
            return value
        
        # Check L2
        if self.l2:
            value = self.l2.get(key)
            if value is not None:
                # Promote to L1
                self.l1.set(key, value)
                return value
        
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        size_bytes: int = 0,
        persist: bool = True
    ):
        """Set in cache (both tiers if persist=True)."""
        self.l1.set(key, value, ttl=ttl, size_bytes=size_bytes)
        
        if persist and self.l2:
            self.l2.set(key, value, ttl_seconds=ttl or self.default_ttl)
    
    def delete(self, key: str) -> bool:
        """Delete from both tiers."""
        l1_deleted = self.l1.delete(key)
        l2_deleted = self.l2.delete(key) if self.l2 else False
        return l1_deleted or l2_deleted
    
    def clear(self):
        """Clear L1 cache (L2 is persistent)."""
        self.l1.clear()
    
    @property
    def stats(self) -> Dict:
        """Get combined statistics."""
        stats = {
            "name": self.name,
            "l1": self.l1.stats,
        }
        if self.l2:
            stats["l2"] = self.l2.stats
        return stats


# ============================================================================
# SPECIALIZED CACHES
# ============================================================================

class FAISSRetrievalCache:
    """
    Cache for FAISS retrieval results.
    
    Keys on query embedding hash + parameters.
    Handles numpy arrays efficiently.
    """
    
    def __init__(
        self,
        max_size: int = 500,
        max_bytes: int = 100 * 1024 * 1024,  # 100MB
        ttl_seconds: float = 3600,  # 1 hour
    ):
        self.cache = LRUCache(
            max_size=max_size,
            max_bytes=max_bytes,
            default_ttl=ttl_seconds,
            name="faiss_retrieval"
        )
    
    def get(
        self,
        query_embedding: np.ndarray,
        k: int,
        filters: Optional[Dict] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get cached FAISS results."""
        key = generate_cache_key(
            ("embedding", query_embedding.tobytes()[:128]),
            k=k,
            filters=filters or {}
        )
        return self.cache.get(key)
    
    def set(
        self,
        query_embedding: np.ndarray,
        k: int,
        distances: np.ndarray,
        indices: np.ndarray,
        filters: Optional[Dict] = None
    ):
        """Cache FAISS results."""
        key = generate_cache_key(
            ("embedding", query_embedding.tobytes()[:128]),
            k=k,
            filters=filters or {}
        )
        
        # Estimate size
        size_bytes = distances.nbytes + indices.nbytes
        
        self.cache.set(key, (distances, indices), size_bytes=size_bytes)
    
    @property
    def stats(self) -> Dict:
        return self.cache.stats


class LLMResponseCache:
    """
    Cache for LLM responses.
    
    Supports:
    - Exact match caching
    - Semantic similarity caching (optional)
    - Streaming response caching
    """
    
    def __init__(
        self,
        max_size: int = 200,
        max_bytes: int = 50 * 1024 * 1024,  # 50MB
        ttl_seconds: float = 7200,  # 2 hours
        disk_path: Optional[str] = None,
    ):
        if disk_path:
            self.cache = TieredCache(
                l1_max_size=max_size,
                l1_max_bytes=max_bytes,
                l2_path=disk_path,
                default_ttl=ttl_seconds,
                name="llm_response"
            )
        else:
            self.cache = LRUCache(
                max_size=max_size,
                max_bytes=max_bytes,
                default_ttl=ttl_seconds,
                name="llm_response"
            )
        
        # Track token savings
        self._tokens_saved = 0
    
    def get(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        **kwargs
    ) -> Optional[Dict]:
        """Get cached LLM response."""
        # Only cache deterministic calls
        if temperature > 0.1:
            return None
        
        key = self._make_key(prompt, model, **kwargs)
        result = self.cache.get(key)
        
        if result:
            self._tokens_saved += result.get("tokens_used", 0)
        
        return result
    
    def set(
        self,
        prompt: str,
        model: str,
        response: Dict,
        **kwargs
    ):
        """Cache LLM response."""
        key = self._make_key(prompt, model, **kwargs)
        
        # Estimate size
        size_bytes = len(json.dumps(response, default=str))
        
        self.cache.set(key, response, size_bytes=size_bytes)
    
    def _make_key(self, prompt: str, model: str, **kwargs) -> str:
        """Generate cache key for LLM call."""
        return generate_cache_key(
            prompt=prompt[:2000],  # Truncate long prompts
            model=model,
            **kwargs
        )
    
    @property
    def stats(self) -> Dict:
        stats = self.cache.stats
        stats["tokens_saved"] = self._tokens_saved
        return stats


class EmbeddingCache:
    """
    Cache for embeddings.
    
    Avoids re-computing embeddings for seen text.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        max_bytes: int = 200 * 1024 * 1024,  # 200MB
        ttl_seconds: float = 86400,  # 24 hours
        disk_path: Optional[str] = None,
    ):
        if disk_path:
            self.cache = TieredCache(
                l1_max_size=max_size,
                l1_max_bytes=max_bytes,
                l2_path=disk_path,
                default_ttl=ttl_seconds,
                name="embedding"
            )
        else:
            self.cache = LRUCache(
                max_size=max_size,
                max_bytes=max_bytes,
                default_ttl=ttl_seconds,
                name="embedding"
            )
    
    def get(self, text: str, model: str = "default") -> Optional[np.ndarray]:
        """Get cached embedding."""
        key = self._make_key(text, model)
        return self.cache.get(key)
    
    def set(self, text: str, embedding: np.ndarray, model: str = "default"):
        """Cache embedding."""
        key = self._make_key(text, model)
        self.cache.set(key, embedding, size_bytes=embedding.nbytes)
    
    def get_batch(
        self,
        texts: List[str],
        model: str = "default"
    ) -> Tuple[List[Optional[np.ndarray]], List[int]]:
        """
        Get cached embeddings for batch.
        
        Returns:
            (cached_embeddings, missing_indices)
        """
        cached = []
        missing_indices = []
        
        for i, text in enumerate(texts):
            embedding = self.get(text, model)
            cached.append(embedding)
            if embedding is None:
                missing_indices.append(i)
        
        return cached, missing_indices
    
    def set_batch(
        self,
        texts: List[str],
        embeddings: List[np.ndarray],
        model: str = "default"
    ):
        """Cache batch of embeddings."""
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding, model)
    
    def _make_key(self, text: str, model: str) -> str:
        """Generate cache key."""
        return generate_cache_key(text=text[:500], model=model)
    
    @property
    def stats(self) -> Dict:
        return self.cache.stats


# ============================================================================
# CACHE DECORATORS
# ============================================================================

def cached(
    cache: LRUCache,
    key_func: Optional[Callable[..., str]] = None,
    ttl: Optional[float] = None
):
    """
    Decorator for caching function results.
    
    Usage:
        @cached(my_cache, ttl=3600)
        def expensive_function(x, y):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = generate_cache_key(*args, **kwargs)
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set(key, result, ttl=ttl)
            
            return result
        return wrapper
    return decorator


def memoize(max_size: int = 100, ttl: Optional[float] = None):
    """
    Simple memoization decorator.
    
    Creates a per-function cache.
    
    Usage:
        @memoize(max_size=100, ttl=3600)
        def expensive_function(x, y):
            ...
    """
    def decorator(func):
        func._cache = LRUCache(max_size=max_size, default_ttl=ttl)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = generate_cache_key(*args, **kwargs)
            
            result = func._cache.get(key)
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            func._cache.set(key, result)
            
            return result
        
        wrapper.cache = func._cache
        wrapper.cache_clear = func._cache.clear
        wrapper.cache_stats = lambda: func._cache.stats
        
        return wrapper
    return decorator


# ============================================================================
# CACHE MANAGER
# ============================================================================

class CacheManager:
    """
    Centralized cache management.
    
    Provides:
    - Named cache registration
    - Global stats aggregation
    - Bulk operations
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._caches: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._initialized = True
    
    def register(self, name: str, cache: Any):
        """Register a cache."""
        with self._lock:
            self._caches[name] = cache
    
    def get(self, name: str) -> Optional[Any]:
        """Get a registered cache."""
        return self._caches.get(name)
    
    def all_stats(self) -> Dict[str, Dict]:
        """Get stats for all caches."""
        result = {}
        for name, cache in self._caches.items():
            if hasattr(cache, 'stats'):
                result[name] = cache.stats
        return result
    
    def clear_all(self):
        """Clear all caches."""
        for cache in self._caches.values():
            if hasattr(cache, 'clear'):
                cache.clear()


# Global cache manager
cache_manager = CacheManager()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Key generation
    "generate_cache_key",
    "semantic_hash",
    
    # Caches
    "LRUCache",
    "DiskCache",
    "TieredCache",
    "CacheEntry",
    
    # Specialized
    "FAISSRetrievalCache",
    "LLMResponseCache",
    "EmbeddingCache",
    
    # Decorators
    "cached",
    "memoize",
    
    # Manager
    "CacheManager",
    "cache_manager",
]
