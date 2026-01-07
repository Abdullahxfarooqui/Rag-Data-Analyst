# Phase 1: Architecture Refactor - Migration Notes

## Overview

Phase 1 introduces a modular architecture that decomposes the monolithic `rag_engine.py` (~3000 LOC) into focused, testable modules while maintaining full backward compatibility.

## New Module Structure

```
core/
├── engine.py           # NEW: Thin RAGEngine orchestrator
├── cache.py            # NEW: TTL caching layer
├── llm/
│   ├── __init__.py
│   ├── client.py       # LLMClient with NVIDIA Nemotron
│   └── prompts.py      # All prompt templates
├── retrieval/
│   ├── __init__.py
│   ├── vector_search.py    # VectorSearcher wrapping FAISS
│   ├── keyword_search.py   # BM25Searcher for hybrid search
│   └── reranker.py         # Cross-encoder re-ranking
├── routing/
│   ├── __init__.py
│   ├── classifier.py       # QueryClassifier with LLM
│   └── handlers.py         # Mode-specific handlers
├── analytics/
│   ├── __init__.py
│   ├── statistics.py       # Python-based statistics
│   └── visualizations.py   # Visualization configuration
└── rag_engine.py       # LEGACY: Kept for backward compatibility
```

## Key Changes

### 1. RAGEngine Orchestrator (`engine.py`)

The new `RAGEngine` class provides:
- Dependency injection for all components
- Clear separation of concerns
- Standardized `RAGResponse` dataclass
- Factory method `RAGEngine.create_default()` for easy instantiation

```python
from core.engine import RAGEngine, query

# New way (recommended)
engine = RAGEngine.create_default()
response = engine.query("What is oil production?", dataframe=df)

# Old way (still works)
result = query("What is oil production?", dataframe=df)
```

### 2. LLM Module (`llm/`)

Extracted LLM-related code into:
- `LLMClient`: OpenRouter client with streaming support
- `LLMConfig`: Configuration dataclass
- `LLMResponse`: Structured response wrapper
- Centralized prompts in `prompts.py`

### 3. Retrieval Module (`retrieval/`)

Provides multiple search strategies:
- `VectorSearcher`: Wraps FAISS with cleaner interface
- `BM25Searcher`: Keyword search for hybrid retrieval
- `reciprocal_rank_fusion()`: Combines vector + keyword results
- `Reranker` protocol with `CrossEncoderReranker` and `NoOpReranker`

### 4. Routing Module (`routing/`)

Semantic query classification:
- `QueryClassifier`: LLM-based classification
- Four modes: `DATA_QUERY`, `DOC_OVERVIEW`, `FREEFORM_QUERY`, `SYSTEM_TASK`
- Mode handlers with `ModeHandler` protocol
- Confidence threshold (0.6) with automatic fallback

### 5. Analytics Module (`analytics/`)

Statistics and visualization:
- `compute_data_statistics()`: Python-based metrics
- `detect_specific_metrics()`: Query metric detection
- `get_target_columns()`: Column matching
- `VisualizationConfig`: Chart configuration

### 6. Cache Module (`cache.py`)

TTL-based caching:
- `TTLCache`: Generic cache with expiration
- `@cached` decorator for function results
- Global caches for LLM, classifications, search, embeddings

## Backward Compatibility

The following imports continue to work unchanged:

```python
from core.rag_engine import (
    query,
    summarize_document,
    list_documents,
    get_document_info,
    set_dataframe_cache,
    get_target_columns,
    detect_specific_metrics,
    METRIC_COLUMNS
)
```

The `app.py` file requires no changes - it will continue to work with the existing imports.

## Migration Path

### For New Code

Use the new modular imports:

```python
from core.engine import RAGEngine
from core.llm import LLMClient, LLMConfig
from core.routing import QueryClassifier, QueryMode
from core.retrieval import VectorSearcher, BM25Searcher
from core.analytics import compute_data_statistics
from core.cache import TTLCache, cached
```

### For Existing Code

No changes required. Legacy imports remain functional.

## Testing

Run the existing tests to verify backward compatibility:

```bash
cd rag-app
python -m pytest tests/ -v
```

New module tests can be added under `tests/`:
- `test_llm_client.py`
- `test_retrieval.py`
- `test_routing.py`
- `test_analytics.py`
- `test_cache.py`

## Phase 2 Preview

The cache module (`cache.py`) is already in place. Phase 2 will:
1. Integrate caching into the RAGEngine
2. Add cache statistics to the UI
3. Implement cache invalidation strategies

## Configuration

The system uses these defaults (configurable via `RAGConfig`):

| Setting | Default | Description |
|---------|---------|-------------|
| `default_k` | 10 | Chunks to retrieve |
| `context_window` | 2 | Adjacent chunks |
| `max_context_chars` | 8000 | Max context length |
| `confidence_threshold` | 0.6 | Classification threshold |
| `enable_cache` | True | Enable caching |
| `cache_ttl` | 300.0 | Cache TTL (seconds) |

## Constraints Maintained

Per the refactoring plan:
- ✅ Single LLM: NVIDIA Nemotron only
- ✅ No cloud APIs for embeddings
- ✅ FAISS for vector store
- ✅ Python/Pandas for statistics
- ✅ Streamlit for UI
- ✅ Backward compatibility preserved
