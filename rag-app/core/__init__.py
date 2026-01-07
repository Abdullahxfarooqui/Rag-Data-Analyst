"""
Core module for RAG application.
Contains extraction, chunking, embedding, vector store, RAG engine, and data analysis components.

Key modules:
- engine: New modular RAG engine orchestrator
- llm: LLM client for NVIDIA Nemotron
- retrieval: Vector and keyword search
- routing: Query classification and mode handlers
- analytics: Statistics computation
- cache: TTL caching layer
- data_engine: Python-based table extraction, statistics, trend/anomaly detection
- extractor: Unified document extraction
- chunker: Smart chunking with data preservation
- embedder: Embedding generation
- vector_store: FAISS-based vector storage
- rag_engine: Legacy query processing (backward compatibility)
"""
from core.extractor import extract_document, get_extraction_summary
from core.chunker import chunk_document, get_chunk_summary
from core.embedder import embed_chunks, embed_query, compute_doc_hash
from core.vector_store import get_vector_store, VectorStore

# New modular engine
from core.engine import (
    RAGEngine,
    RAGConfig,
    RAGResponse,
    get_engine,
    query,
    summarize_document,
    list_documents,
    get_document_info,
    set_dataframe_cache
)

# Legacy rag_engine for backward compatibility
from core.rag_engine import (
    compare_documents,
    answer_question,
    analyze_query,
    get_target_columns,
    detect_specific_metrics,
    is_general_query,
    METRIC_COLUMNS
)

# Import data engine functions
try:
    from core.data_engine import (
        extract_tables_from_pdf,
        extract_tables_from_excel,
        extract_tables_from_csv,
        merge_tables,
        compute_statistics,
        prepare_llm_chunks,
        format_full_table,
        format_sample_rows,
        format_statistics_summary
    )
    HAS_DATA_ENGINE = True
except ImportError:
    HAS_DATA_ENGINE = False

__all__ = [
    # New Engine
    "RAGEngine",
    "RAGConfig",
    "RAGResponse",
    "get_engine",
    # Extraction
    "extract_document",
    "get_extraction_summary",
    # Chunking
    "chunk_document",
    "get_chunk_summary",
    # Embedding
    "embed_chunks",
    "embed_query",
    "compute_doc_hash",
    # Vector Store
    "get_vector_store",
    "VectorStore",
    # RAG Engine
    "query",
    "summarize_document",
    "list_documents",
    "get_document_info",
    "set_dataframe_cache",
    "compare_documents",
    "answer_question",
    "analyze_query",
    "get_target_columns",
    "detect_specific_metrics",
    "is_general_query",
    "METRIC_COLUMNS",
    # Data Engine (if available)
    "extract_tables_from_pdf",
    "extract_tables_from_excel", 
    "extract_tables_from_csv",
    "merge_tables",
    "compute_statistics",
    "prepare_llm_chunks",
    "format_full_table",
    "format_sample_rows",
    "format_statistics_summary"
]
