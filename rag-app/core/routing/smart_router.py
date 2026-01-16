"""
Smart Query Routing with FAISS Bypass.

CORE PRINCIPLE:
"Not every question needs vectors."

BYPASS CONDITIONS (skip FAISS entirely):
1. Pure aggregations: "total sales", "count of orders", "average price"
2. Direct column lookups: "show me the price column"
3. Simple filters: "products over $100", "orders from last week"
4. Sorting/ranking: "top 10 products", "bottom 5 performers"

FAISS REQUIRED:
1. Semantic similarity: "products similar to X"
2. Document search: "find documents about Y"
3. Context retrieval: "what did we discuss about Z"
4. Concept matching: "marketing-related items"

ARCHITECTURE:
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Query Router   │────▶│  Pattern Match  │
└────────┬────────┘     └─────────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐  ┌───────┐
│ FAISS │  │BYPASS │
│ Path  │  │ Path  │
└───────┘  └───────┘
"""
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# QUERY CLASSIFICATION
# ============================================================================

class QueryIntent(Enum):
    """Primary intent of a query."""
    AGGREGATE = auto()      # sum, count, average, total
    FILTER = auto()         # where, filter, only, exclude
    SORT = auto()           # top, bottom, highest, lowest
    LOOKUP = auto()         # show column, get value
    COMPARE = auto()        # vs, compare, difference between
    TREND = auto()          # over time, growth, change
    SEMANTIC = auto()       # similar to, like, related to
    DOCUMENT = auto()       # find doc, search for, what about
    COMPOUND = auto()       # multiple intents combined
    UNKNOWN = auto()        # fallback to FAISS


class ProcessingPath(Enum):
    """Which processing path to use."""
    PANDAS_ONLY = auto()         # Pure Python/Pandas, no LLM
    PANDAS_WITH_LLM = auto()     # Pandas for data, LLM for insight
    FAISS_WITH_LLM = auto()      # Full RAG pipeline
    LLM_ONLY = auto()            # Direct LLM query


@dataclass
class QueryClassification:
    """Result of query classification."""
    query: str
    intent: QueryIntent
    path: ProcessingPath
    confidence: float
    entities: Dict[str, Any] = field(default_factory=dict)
    bypass_faiss: bool = False
    cache_key: str = ""
    reasoning: str = ""


# ============================================================================
# PATTERN DEFINITIONS
# ============================================================================

# Patterns that indicate FAISS should be BYPASSED
BYPASS_PATTERNS = {
    # Aggregation patterns
    "aggregate": [
        r"\b(total|sum|count|average|avg|mean|median|min|max|std|variance)\b",
        r"\bhow many\b",
        r"\bwhat is the (total|sum|average|count)\b",
    ],
    
    # Filtering patterns
    "filter": [
        r"\bwhere\b.*\b(is|are|equals?|greater|less|more|fewer)\b",
        r"\bfilter\b",
        r"\bonly\b.*\b(show|include|display)\b",
        r"\bexclude\b",
        r"\bwith\b.*\b(value|price|count)\b.*\b(over|under|above|below)\b",
    ],
    
    # Sorting/ranking patterns
    "sort": [
        r"\b(top|bottom|highest|lowest|best|worst)\s+\d+\b",
        r"\brank(ed|ing)?\b",
        r"\bsort(ed)?\s+by\b",
        r"\border\s+by\b",
        r"\b(ascending|descending)\b",
    ],
    
    # Lookup patterns
    "lookup": [
        r"\bshow\s+(me\s+)?(the\s+)?(\w+)\s+(column|field|values?)\b",
        r"\bget\s+(the\s+)?(\w+)\s+(for|from)\b",
        r"\bwhat\s+(is|are)\s+the\s+(\w+)\s+(of|for)\b",
        r"\blist\s+(all\s+)?(\w+)\b",
    ],
    
    # Time-based patterns (can be computed without FAISS)
    "temporal": [
        r"\b(last|previous|this|next)\s+(week|month|year|quarter|day)\b",
        r"\b(daily|weekly|monthly|yearly|quarterly)\b",
        r"\bfrom\s+\d{4}.*to\s+\d{4}\b",
        r"\bsince\s+(january|february|march|april|may|june|july|august|september|october|november|december|\d{4})\b",
    ],
    
    # Statistical patterns
    "statistical": [
        r"\bdistribution\s+of\b",
        r"\bcorrelation\s+between\b",
        r"\bpercentile\b",
        r"\boutliers?\b",
        r"\bstandard\s+deviation\b",
    ]
}

# Patterns that REQUIRE FAISS
FAISS_REQUIRED_PATTERNS = [
    r"\bsimilar\s+to\b",
    r"\blike\s+\w+\b",
    r"\brelated\s+to\b",
    r"\babout\s+\w+\b",
    r"\bfind\s+(documents?|files?|content)\b",
    r"\bsearch\s+for\b",
    r"\bwhat\s+(did|does|do)\s+\w+\s+(say|mention|discuss)\b",
    r"\bcontext\b",
    r"\bsemantic\b",
    r"\bmeaning\b",
]


# ============================================================================
# QUERY CLASSIFIER
# ============================================================================

class SmartQueryClassifier:
    """
    Classifies queries to determine optimal processing path.
    
    This is the first step in query processing - determines
    whether we need FAISS at all, or can compute directly.
    """
    
    def __init__(self, column_names: List[str] = None):
        self._column_names = set(c.lower() for c in (column_names or []))
        self._entity_extractors = self._build_entity_extractors()
    
    def _build_entity_extractors(self) -> Dict[str, Callable]:
        """Build entity extraction functions."""
        return {
            "columns": self._extract_columns,
            "numbers": self._extract_numbers,
            "dates": self._extract_dates,
            "comparisons": self._extract_comparisons,
        }
    
    def update_columns(self, column_names: List[str]):
        """Update known column names."""
        self._column_names = set(c.lower() for c in column_names)
    
    def _extract_columns(self, query: str) -> List[str]:
        """Extract column names mentioned in query."""
        query_lower = query.lower()
        return [col for col in self._column_names if col in query_lower]
    
    def _extract_numbers(self, query: str) -> List[float]:
        """Extract numeric values from query."""
        pattern = r'\b(\d+(?:\.\d+)?)\b'
        matches = re.findall(pattern, query)
        return [float(m) for m in matches]
    
    def _extract_dates(self, query: str) -> List[str]:
        """Extract date references from query."""
        patterns = [
            r'\b(\d{4}-\d{2}-\d{2})\b',
            r'\b(\d{1,2}/\d{1,2}/\d{4})\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b',
            r'\b(last|this|next)\s+(week|month|year|quarter)\b',
        ]
        dates = []
        for pattern in patterns:
            matches = re.findall(pattern, query.lower())
            if matches:
                if isinstance(matches[0], tuple):
                    dates.extend([' '.join(m) for m in matches])
                else:
                    dates.extend(matches)
        return dates
    
    def _extract_comparisons(self, query: str) -> List[Tuple[str, str]]:
        """Extract comparison operators and values."""
        patterns = [
            (r'(greater|more|over|above)\s+(?:than\s+)?(\d+)', '>'),
            (r'(less|fewer|under|below)\s+(?:than\s+)?(\d+)', '<'),
            (r'(equals?|is)\s+(\d+)', '='),
            (r'between\s+(\d+)\s+and\s+(\d+)', 'between'),
        ]
        comparisons = []
        for pattern, op in patterns:
            matches = re.findall(pattern, query.lower())
            for match in matches:
                if op == 'between':
                    comparisons.append((op, f"{match[0]},{match[1]}"))
                else:
                    comparisons.append((op, match[-1]))
        return comparisons
    
    def _calculate_bypass_score(self, query: str) -> Tuple[float, str]:
        """
        Calculate confidence score for bypassing FAISS.
        
        Returns:
            Tuple of (score, category) where score > 0.5 means bypass
        """
        query_lower = query.lower()
        
        # Check FAISS-required patterns first
        for pattern in FAISS_REQUIRED_PATTERNS:
            if re.search(pattern, query_lower):
                return (0.1, "semantic")
        
        # Check bypass patterns
        max_score = 0.0
        best_category = "unknown"
        
        for category, patterns in BYPASS_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    # Score based on pattern specificity
                    pattern_specificity = len(pattern) / 100  # Longer patterns = more specific
                    score = 0.6 + pattern_specificity
                    if score > max_score:
                        max_score = score
                        best_category = category
        
        # Boost score if query mentions known columns
        mentioned_columns = self._extract_columns(query)
        if mentioned_columns:
            max_score = min(max_score + 0.1 * len(mentioned_columns), 0.95)
        
        return (max_score, best_category)
    
    def _determine_intent(self, query: str, bypass_category: str) -> QueryIntent:
        """Determine primary intent from query."""
        intent_mapping = {
            "aggregate": QueryIntent.AGGREGATE,
            "filter": QueryIntent.FILTER,
            "sort": QueryIntent.SORT,
            "lookup": QueryIntent.LOOKUP,
            "temporal": QueryIntent.TREND,
            "statistical": QueryIntent.AGGREGATE,
            "semantic": QueryIntent.SEMANTIC,
            "unknown": QueryIntent.UNKNOWN,
        }
        return intent_mapping.get(bypass_category, QueryIntent.UNKNOWN)
    
    def _determine_path(
        self,
        intent: QueryIntent,
        bypass_score: float,
        has_entities: bool
    ) -> ProcessingPath:
        """Determine optimal processing path."""
        # High confidence bypass with known entities = pure Pandas
        if bypass_score > 0.7 and has_entities:
            if intent in (QueryIntent.AGGREGATE, QueryIntent.FILTER, 
                         QueryIntent.SORT, QueryIntent.LOOKUP):
                return ProcessingPath.PANDAS_ONLY
        
        # Medium confidence = Pandas with LLM for insight
        if bypass_score > 0.5:
            return ProcessingPath.PANDAS_WITH_LLM
        
        # Semantic queries = full FAISS
        if intent in (QueryIntent.SEMANTIC, QueryIntent.DOCUMENT):
            return ProcessingPath.FAISS_WITH_LLM
        
        # Default = FAISS with LLM
        return ProcessingPath.FAISS_WITH_LLM
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        # Normalize query for caching
        normalized = re.sub(r'\s+', ' ', query.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def classify(self, query: str) -> QueryClassification:
        """
        Classify a query and determine processing path.
        
        Args:
            query: User's query string
            
        Returns:
            QueryClassification with intent, path, and metadata
        """
        # Calculate bypass score
        bypass_score, bypass_category = self._calculate_bypass_score(query)
        
        # Extract entities
        entities = {}
        for name, extractor in self._entity_extractors.items():
            extracted = extractor(query)
            if extracted:
                entities[name] = extracted
        
        has_entities = bool(entities.get("columns")) or bool(entities.get("comparisons"))
        
        # Determine intent and path
        intent = self._determine_intent(query, bypass_category)
        path = self._determine_path(intent, bypass_score, has_entities)
        
        # Build classification
        classification = QueryClassification(
            query=query,
            intent=intent,
            path=path,
            confidence=bypass_score,
            entities=entities,
            bypass_faiss=(path in (ProcessingPath.PANDAS_ONLY, ProcessingPath.PANDAS_WITH_LLM)),
            cache_key=self._generate_cache_key(query),
            reasoning=f"Category: {bypass_category}, Score: {bypass_score:.2f}"
        )
        
        logger.info(f"Query classified: {intent.name} -> {path.name} (bypass={classification.bypass_faiss})")
        
        return classification


# ============================================================================
# QUERY CACHE
# ============================================================================

@dataclass
class CachedResult:
    """Cached query result with metadata."""
    query: str
    result: Any
    timestamp: float
    ttl_seconds: float
    hit_count: int = 0
    
    @property
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp


class QueryCache:
    """
    Intelligent query cache with TTL and LRU eviction.
    
    CACHING STRATEGY:
    1. Cache deterministic results (aggregations, filters) longer
    2. Cache LLM results shorter (may vary)
    3. Don't cache semantic search results (context-dependent)
    4. Invalidate on data changes
    """
    
    # TTL by path type
    DEFAULT_TTL = {
        ProcessingPath.PANDAS_ONLY: 3600,       # 1 hour - deterministic
        ProcessingPath.PANDAS_WITH_LLM: 1800,   # 30 min - semi-deterministic
        ProcessingPath.FAISS_WITH_LLM: 300,     # 5 min - context-dependent
        ProcessingPath.LLM_ONLY: 600,           # 10 min - may vary
    }
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, CachedResult] = {}
        self._max_size = max_size
        self._data_version: int = 0  # Increment on data changes
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }
    
    def _evict_if_needed(self):
        """Evict oldest entries if cache is full."""
        if len(self._cache) < self._max_size:
            return
        
        # Remove expired entries first
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for key in expired:
            del self._cache[key]
            self._stats["evictions"] += 1
        
        # If still full, remove least recently used (lowest hit count)
        if len(self._cache) >= self._max_size:
            sorted_keys = sorted(
                self._cache.keys(),
                key=lambda k: (self._cache[k].hit_count, -self._cache[k].age_seconds)
            )
            # Remove bottom 10%
            remove_count = max(1, len(sorted_keys) // 10)
            for key in sorted_keys[:remove_count]:
                del self._cache[key]
                self._stats["evictions"] += 1
    
    def get(
        self,
        cache_key: str,
        data_version: int = None
    ) -> Optional[Any]:
        """
        Get cached result if available and valid.
        
        Args:
            cache_key: Cache key (from QueryClassification)
            data_version: Current data version (for invalidation)
            
        Returns:
            Cached result or None
        """
        cached = self._cache.get(cache_key)
        
        if cached is None:
            self._stats["misses"] += 1
            return None
        
        # Check if expired
        if cached.is_expired:
            del self._cache[cache_key]
            self._stats["misses"] += 1
            return None
        
        # Check data version (invalidate if data changed)
        if data_version is not None and data_version > self._data_version:
            # Data has changed since caching, invalidate all
            self._cache.clear()
            self._data_version = data_version
            self._stats["misses"] += 1
            return None
        
        # Valid cache hit
        cached.hit_count += 1
        self._stats["hits"] += 1
        
        logger.debug(f"Cache hit for {cache_key[:8]}... (hits: {cached.hit_count})")
        return cached.result
    
    def set(
        self,
        cache_key: str,
        result: Any,
        path: ProcessingPath,
        ttl_seconds: float = None
    ):
        """
        Cache a query result.
        
        Args:
            cache_key: Cache key
            result: Result to cache
            path: Processing path (determines default TTL)
            ttl_seconds: Custom TTL (overrides default)
        """
        self._evict_if_needed()
        
        ttl = ttl_seconds or self.DEFAULT_TTL.get(path, 300)
        
        self._cache[cache_key] = CachedResult(
            query=cache_key,
            result=result,
            timestamp=time.time(),
            ttl_seconds=ttl
        )
        
        logger.debug(f"Cached result for {cache_key[:8]}... (TTL: {ttl}s)")
    
    def invalidate(self, pattern: str = None):
        """
        Invalidate cache entries.
        
        Args:
            pattern: Regex pattern to match keys (None = clear all)
        """
        if pattern is None:
            self._cache.clear()
            logger.info("Cache cleared")
        else:
            regex = re.compile(pattern)
            keys_to_remove = [k for k in self._cache if regex.search(k)]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries")
    
    def on_data_change(self):
        """Call when underlying data changes."""
        self._data_version += 1
        self._cache.clear()
        logger.info(f"Data changed, cache invalidated (version: {self._data_version})")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "hit_rate": hit_rate,
            "data_version": self._data_version,
        }


# ============================================================================
# SMART ROUTER
# ============================================================================

class SmartRouter:
    """
    Routes queries to optimal processing path.
    
    Combines classification and caching for efficient query handling.
    
    Usage:
        router = SmartRouter(column_names=df.columns.tolist())
        
        # Route a query
        routing = router.route(query)
        
        if routing.use_cache:
            return routing.cached_result
        
        if routing.bypass_faiss:
            result = pandas_compute(query)
        else:
            result = faiss_search(query)
        
        router.cache_result(routing.classification, result)
    """
    
    def __init__(
        self,
        column_names: List[str] = None,
        cache_enabled: bool = True,
        max_cache_size: int = 1000
    ):
        self._classifier = SmartQueryClassifier(column_names)
        self._cache = QueryCache(max_cache_size) if cache_enabled else None
        self._routing_history: List[Dict] = []
    
    def update_schema(self, column_names: List[str]):
        """Update known column names."""
        self._classifier.update_columns(column_names)
    
    def on_data_change(self):
        """Call when underlying data changes."""
        if self._cache:
            self._cache.on_data_change()
    
    def route(
        self,
        query: str,
        check_cache: bool = True
    ) -> "RoutingResult":
        """
        Route a query to optimal processing path.
        
        Args:
            query: User's query
            check_cache: Whether to check cache
            
        Returns:
            RoutingResult with classification and caching info
        """
        # Classify query
        classification = self._classifier.classify(query)
        
        # Check cache
        cached_result = None
        if check_cache and self._cache:
            cached_result = self._cache.get(classification.cache_key)
        
        routing = RoutingResult(
            classification=classification,
            use_cache=cached_result is not None,
            cached_result=cached_result
        )
        
        # Track routing history
        self._routing_history.append({
            "query": query,
            "intent": classification.intent.name,
            "path": classification.path.name,
            "bypass_faiss": classification.bypass_faiss,
            "cache_hit": routing.use_cache,
            "timestamp": time.time()
        })
        
        return routing
    
    def cache_result(
        self,
        classification: QueryClassification,
        result: Any,
        ttl_seconds: float = None
    ):
        """Cache a query result."""
        if self._cache:
            self._cache.set(
                classification.cache_key,
                result,
                classification.path,
                ttl_seconds
            )
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        if not self._routing_history:
            return {"total_queries": 0}
        
        history = self._routing_history[-1000:]  # Last 1000
        
        bypass_count = sum(1 for r in history if r["bypass_faiss"])
        cache_hits = sum(1 for r in history if r["cache_hit"])
        
        intent_counts = {}
        path_counts = {}
        
        for r in history:
            intent_counts[r["intent"]] = intent_counts.get(r["intent"], 0) + 1
            path_counts[r["path"]] = path_counts.get(r["path"], 0) + 1
        
        return {
            "total_queries": len(history),
            "bypass_rate": bypass_count / len(history) if history else 0,
            "cache_hit_rate": cache_hits / len(history) if history else 0,
            "intent_distribution": intent_counts,
            "path_distribution": path_counts,
            "cache_stats": self._cache.get_stats() if self._cache else None,
        }


@dataclass
class RoutingResult:
    """Result of query routing."""
    classification: QueryClassification
    use_cache: bool
    cached_result: Any = None
    
    @property
    def bypass_faiss(self) -> bool:
        return self.classification.bypass_faiss
    
    @property
    def path(self) -> ProcessingPath:
        return self.classification.path
    
    @property
    def intent(self) -> QueryIntent:
        return self.classification.intent


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "SmartQueryClassifier",
    "SmartRouter",
    "QueryCache",
    "QueryClassification",
    "RoutingResult",
    "QueryIntent",
    "ProcessingPath",
]
