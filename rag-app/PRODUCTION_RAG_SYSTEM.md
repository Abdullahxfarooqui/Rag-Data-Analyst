# Production RAG Data Analyst System

## Overview

This document describes the production-grade RAG (Retrieval-Augmented Generation) Data Analyst system that has been implemented. The system is designed to:

- **Ingest any document type** dynamically (Excel, CSV, PDF, DOCX, JSON, Parquet, etc.)
- **Generate structured outputs** that adapt to document content
- **Provide fast, progressive insights** with async processing
- **Scale to thousands of documents** and concurrent users

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ProductionRAGSystem                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Ingestor    │───▶│ Vector Store │───▶│  Task Queue  │───▶│   Schema     │  │
│  │  (Dynamic)   │    │   (FAISS)    │    │   (Async)    │    │  Generator   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │                   │           │
│         ▼                   ▼                   ▼                   ▼           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Cache Layer                                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │   │
│  │  │  Embedding  │  │    FAISS    │  │     LLM     │                      │   │
│  │  │   Cache     │  │    Cache    │  │    Cache    │                      │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     Observability Layer                                  │   │
│  │  ┌───────┐  ┌─────────┐  ┌────────┐  ┌─────────┐                        │   │
│  │  │Logger │  │ Metrics │  │ Tracer │  │ Alerter │                        │   │
│  │  └───────┘  └─────────┘  └────────┘  └─────────┘                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Modules Implemented

### 1. Dynamic Ingestion Engine (`core/ingestion/dynamic_ingestor.py`)

**Purpose**: Auto-detect and extract content from any document type.

**Features**:
- Document type detection via magic bytes + extension
- Column type detection (NUMERIC, CATEGORICAL, DATETIME, TEXT, BOOLEAN, ID)
- Unit detection patterns (currency, percentage, weight, etc.)
- Chunk generation for FAISS indexing

**Usage**:
```python
from core.ingestion import DynamicIngestor

ingestor = DynamicIngestor()
schema = ingestor.ingest("data.xlsx")

# Access extracted data
df = schema.get_primary_dataframe()
metrics = schema.get_all_metrics()
chunks = ingestor.generate_chunks(schema)
```

### 2. Production FAISS Vector Store (`core/retrieval/production_vector_store.py`)

**Purpose**: Scalable vector indexing with memory optimization.

**Index Type Selection**:
| Scale | Index Type | Recall | Speed | Memory |
|-------|------------|--------|-------|--------|
| <10k vectors | Flat | 100% | O(n) | 1x |
| 10k-100k | IVF | ~95% | O(√n) | 1.1x |
| 100k-1M | IVF-PQ | ~90% | O(√n) | 0.25x |
| >1M | HNSW | ~95% | O(log n) | 1.3x |

**Usage**:
```python
from core.retrieval.production_vector_store import ProductionVectorStore, IndexConfig

config = IndexConfig(dimension=384, index_type="auto")
store = ProductionVectorStore(storage_path="./faiss_data", config=config)

# Add chunks
store.add_chunks(chunks, embeddings)

# Search
results = store.search(query_embedding, k=10)
```

### 3. Async Task Queue (`core/queue/task_queue.py`)

**Purpose**: Handle concurrent users with priority scheduling.

**Features**:
- Priority queue (CRITICAL/HIGH/NORMAL/LOW)
- Circuit breaker for failing services
- Exponential backoff retry
- Progress callbacks for UI updates
- Graceful shutdown

**Scaling Guidelines**:
| Users | Workers | Batch Size | Queue Size |
|-------|---------|------------|------------|
| 1-2 | 1-2 | 1 | 100 |
| 10 | 4 | 4 | 500 |
| 100 | 10 | 10 | 2000 |

**Usage**:
```python
from core.queue import AsyncTaskQueue, TaskPriority

queue = AsyncTaskQueue(worker_count=4)
queue.register_handler("analyze", analyze_function)

task_id = queue.submit(
    task_type="analyze",
    payload={"query": "..."},
    priority=TaskPriority.HIGH
)
```

### 4. Dynamic Schema Generator (`core/schema/dynamic_schema.py`)

**Purpose**: Generate structured JSON output adapting to any document.

**Output Format**:
```json
{
  "metrics": [
    {"name": "total_sales", "value": 1000000, "unit": "USD", ...}
  ],
  "comparisons": [
    {"item_a": "Product A", "item_b": "Product B", "difference": 15000, ...}
  ],
  "rankings": [
    {"rank": 1, "item": "Product A", "value": 50000, ...}
  ],
  "trends": [
    {"period": "Q1-Q2", "direction": "up", "change_percent": 12.5, ...}
  ],
  "anomalies": [
    {"item": "Outlier Product", "z_score": 3.2, ...}
  ],
  "confidence_level": "high"
}
```

**Usage**:
```python
from core.schema import DynamicSchemaGenerator

generator = DynamicSchemaGenerator()
output = generator.generate(dataframe, "What are top sellers?")
```

### 5. Observability & Monitoring (`core/observability/monitoring.py`)

**Purpose**: Track system health and performance.

**Metrics Tracked**:
- LLM calls: count, latency, tokens, errors
- FAISS queries: count, latency, hit rate
- Document ingestion: count, size, time
- Task queue: depth, wait time, completion rate
- Memory: heap usage, FAISS index size
- Cache: hit rate, size, evictions

**Usage**:
```python
from core.observability import metrics, timer, track_llm_call

# Track metrics
metrics.counter("documents_processed").inc()

# Time operations
with timer("llm_latency_seconds"):
    response = llm.call(...)

# Export metrics
stats = metrics.export()
```

### 6. Production Caching Layer (`core/cache/production_cache.py`)

**Purpose**: Multi-tier caching for performance.

**Cache Tiers**:
1. **L1 (Memory)**: Hot data, <100ms access
2. **L2 (Disk)**: Warm data, <10ms access

**Specialized Caches**:
- `EmbeddingCache`: Avoid re-computing embeddings
- `FAISSRetrievalCache`: Cache vector search results
- `LLMResponseCache`: Reduce API costs

**Usage**:
```python
from core.cache import LRUCache, EmbeddingCache, memoize

# LRU Cache
cache = LRUCache(max_size=1000, default_ttl=3600)
cache.set("key", value)
result = cache.get("key")

# Memoization
@memoize(max_size=100, ttl=3600)
def expensive_function(x):
    return compute(x)
```

### 7. Integration Layer (`core/integration/production_system.py`)

**Purpose**: Unified entry point for the production system.

**Usage**:
```python
from core.integration import ProductionRAGSystem, SystemConfig

config = SystemConfig(
    data_dir="data",
    num_workers=4,
    enable_monitoring=True
)

system = ProductionRAGSystem(config)
await system.initialize()

# Set embedder and LLM
system.set_embedder(my_embed_function)
system.set_llm(my_llm_function)

# Ingest documents
schema = await system.ingest_document("data.xlsx")

# Query
result = await system.query("What are top selling products?")
print(result.structured_output)
```

## Trade-offs Explained

### Latency vs UX

| Approach | Latency | User Experience |
|----------|---------|-----------------|
| Synchronous | 500ms-5s | User waits |
| Async + Progress | Same total | Progressive updates |
| Cached | <100ms | Instant (if cached) |

**Recommendation**: Use async with progress callbacks for long operations.

### Memory vs Scaling

| Index Type | Memory per 1M vectors | Recall |
|------------|----------------------|--------|
| Flat | ~1.5GB | 100% |
| IVF | ~1.6GB | ~95% |
| IVF-PQ | ~400MB | ~90% |
| HNSW | ~2GB | ~95% |

**Recommendation**: Use IVF-PQ for >100k vectors unless recall is critical.

### Cost vs Caching

| Cache Hit Rate | API Cost Reduction | Memory Overhead |
|----------------|-------------------|-----------------|
| 50% | ~40% | 100MB |
| 70% | ~60% | 200MB |
| 90% | ~80% | 500MB |

**Recommendation**: 70% hit rate is good balance for most applications.

## Testing

Run the test suite:
```bash
python tests/test_production_system.py
```

All 7 tests should pass:
- ✓ Dynamic Ingestor
- ✓ Production Vector Store
- ✓ Task Queue
- ✓ Dynamic Schema Generator
- ✓ Observability
- ✓ Caching
- ✓ Production System Integration

## File Structure

```
core/
├── cache/
│   ├── __init__.py
│   └── production_cache.py      # Multi-tier caching
├── ingestion/
│   ├── __init__.py
│   └── dynamic_ingestor.py      # Document ingestion
├── integration/
│   ├── __init__.py
│   └── production_system.py     # Unified system
├── observability/
│   ├── __init__.py
│   └── monitoring.py            # Metrics & logging
├── queue/
│   ├── __init__.py
│   └── task_queue.py            # Async processing
├── retrieval/
│   └── production_vector_store.py  # FAISS indexing
├── schema/
│   ├── __init__.py
│   └── dynamic_schema.py        # Structured output
└── ttl_cache.py                 # Original TTL cache
```

## Dependencies

Required packages:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `faiss-cpu` - Vector indexing
- `psutil` - System monitoring
- `openpyxl` - Excel support (optional)
- `pypdf` - PDF support (optional)
