"""
RAGEngine - Thin Orchestrator for the RAG Pipeline.

This is the main entry point for the RAG system. It coordinates:
1. Query classification (routing)
2. Context retrieval (vector + keyword search with hybrid fusion)
3. Response generation (LLM with streaming support)

Design principles:
- No business logic - delegates to specialized modules
- Dependency injection for testability
- Single responsibility: orchestration only

Phases implemented:
- Phase 1: Architecture refactor
- Phase 2: TTL caching
- Phase 3: Hybrid retrieval (BM25 + FAISS + RRF)
- Phase 4: Semantic routing
- Phase 5: Streaming responses
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Generator, Callable
import pandas as pd

from core.cache import (
    TTLCache, 
    classification_cache, 
    search_cache, 
    llm_cache,
    hash_key
)
from core.routing import QueryClassifier, QueryMode, ClassificationResult
from core.routing.handlers import (
    ModeHandler,
    DataQueryHandler,
    DocumentOverviewHandler,
    FreeformHandler,
    SystemTaskHandler
)
from core.llm import LLMClient, LLMConfig
from core.retrieval import VectorSearcher, BM25Searcher, Reranker, NoOpReranker
from core.retrieval.keyword_search import reciprocal_rank_fusion
from core.analytics import compute_data_statistics, get_target_columns


logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG engine."""
    # Retrieval settings
    default_k: int = 10
    context_window: int = 2
    max_context_chars: int = 8000
    
    # Hybrid search settings (Phase 3)
    enable_hybrid_search: bool = True
    vector_weight: float = 0.6
    keyword_weight: float = 0.4
    
    # Re-ranking settings
    enable_reranking: bool = False  # Disabled by default (requires cross-encoder model)
    
    # Classification settings (Phase 4)
    confidence_threshold: float = 0.6
    
    # LLM settings
    max_tokens_concise: int = 1500
    max_tokens_detailed: int = 3000
    
    # Streaming settings (Phase 5)
    enable_streaming: bool = True
    
    # Caching (Phase 2)
    enable_cache: bool = True
    cache_ttl: float = 300.0


@dataclass
class RAGResponse:
    """Standard response from RAG engine."""
    answer: str
    query_mode: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Visualization hints
    show_visualizations: bool = False
    specific_metrics: List[str] = field(default_factory=list)
    target_columns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "answer": self.answer,
            "query_mode": self.query_mode,
            "sources": self.sources,
            "show_visualizations": self.show_visualizations,
            "specific_metrics": self.specific_metrics,
            "target_columns": self.target_columns,
            **self.metadata
        }


class RAGEngine:
    """
    Main RAG Engine - Orchestrates the entire pipeline.
    
    This is a thin orchestrator that delegates work to:
    - QueryClassifier: Determines query intent (Phase 4: LLM-based)
    - ModeHandlers: Handle different query types
    - VectorSearcher: FAISS-based retrieval
    - BM25Searcher: Keyword-based retrieval (Phase 3: hybrid)
    - Reranker: Re-ranks results (optional)
    - LLMClient: Generates responses (Phase 5: streaming)
    
    Usage:
        engine = RAGEngine.create_default()
        response = engine.query("What is oil production?", dataframe=df)
        
        # Streaming
        for token in engine.query_stream("What is oil production?", dataframe=df):
            print(token, end="")
    """
    
    def __init__(
        self,
        config: RAGConfig,
        classifier: QueryClassifier,
        llm_client: LLMClient,
        vector_searcher: VectorSearcher,
        keyword_searcher: Optional[BM25Searcher] = None,
        reranker: Optional[Reranker] = None,
        handlers: Optional[Dict[QueryMode, ModeHandler]] = None
    ):
        """
        Initialize RAG engine with dependencies.
        
        Args:
            config: Engine configuration
            classifier: Query classifier for routing
            llm_client: LLM client for generation
            vector_searcher: Vector store searcher
            keyword_searcher: Optional BM25 searcher for hybrid
            reranker: Optional re-ranker for result improvement
            handlers: Custom mode handlers (uses defaults if None)
        """
        self.config = config
        self.classifier = classifier
        self.llm = llm_client
        self.vector_searcher = vector_searcher
        self.keyword_searcher = keyword_searcher
        self.reranker = reranker or NoOpReranker()
        
        # Initialize handlers
        self.handlers = handlers or self._create_default_handlers()
        
        # DataFrame cache for session
        self._dataframe_cache: Dict[str, pd.DataFrame] = {}
        
        # BM25 index state
        self._bm25_indexed = False
    
    def _create_default_handlers(self) -> Dict[QueryMode, ModeHandler]:
        """Create default mode handlers."""
        return {
            QueryMode.DATA_QUERY: DataQueryHandler(self.llm),
            QueryMode.DOC_OVERVIEW: DocumentOverviewHandler(self.llm),
            QueryMode.FREEFORM_QUERY: FreeformHandler(),
            QueryMode.SYSTEM_TASK: SystemTaskHandler()
        }
    
    @classmethod
    def create_default(cls, enable_hybrid: bool = True) -> "RAGEngine":
        """
        Factory method to create engine with default configuration.
        
        Args:
            enable_hybrid: Whether to enable hybrid search (Phase 3)
        
        This is the recommended way to create an engine instance.
        """
        from core.vector_store import get_vector_store
        
        config = RAGConfig(enable_hybrid_search=enable_hybrid)
        
        # Create LLM client
        llm_config = LLMConfig(
            model="nvidia/nemotron-nano-12b-v2-vl:free",
            temperature=0.1,
            max_tokens=config.max_tokens_concise
        )
        llm_client = LLMClient(llm_config)
        
        # Create classifier (Phase 4: LLM-based semantic routing)
        classifier = QueryClassifier(
            llm_client=llm_client,
            confidence_threshold=config.confidence_threshold
        )
        
        # Create vector searcher
        store = get_vector_store()
        vector_searcher = VectorSearcher(store)
        
        # Create keyword searcher for hybrid (Phase 3)
        keyword_searcher = BM25Searcher() if enable_hybrid else None
        
        # Create engine
        return cls(
            config=config,
            classifier=classifier,
            llm_client=llm_client,
            vector_searcher=vector_searcher,
            keyword_searcher=keyword_searcher
        )
    
    def _ensure_bm25_indexed(self) -> None:
        """Ensure BM25 index is built from vector store chunks."""
        if not self.keyword_searcher or self._bm25_indexed:
            return
        
        # Get all chunks from vector store
        try:
            docs = self.vector_searcher.store.get_all_documents()
            all_chunks = []
            
            for doc in docs:
                doc_hash = doc.get("doc_hash")
                if doc_hash:
                    chunks = self.vector_searcher.store.get_document_chunks(doc_hash)
                    for chunk in chunks:
                        chunk["doc_hash"] = doc_hash
                        chunk["filename"] = doc.get("filename", "unknown")
                    all_chunks.extend(chunks)
            
            if all_chunks:
                self.keyword_searcher.index(all_chunks)
                self._bm25_indexed = True
                logger.info(f"BM25 index built with {len(all_chunks)} chunks")
        except Exception as e:
            logger.warning(f"Failed to build BM25 index: {e}")
    
    def set_dataframe_cache(self, cache: Dict[str, pd.DataFrame]) -> None:
        """Update the DataFrame cache from session state."""
        self._dataframe_cache = cache
    
    def get_dataframe(self, filename: str) -> Optional[pd.DataFrame]:
        """Get DataFrame from cache by filename."""
        return self._dataframe_cache.get(filename)
    
    def query(
        self,
        user_query: str,
        doc_hash: Optional[str] = None,
        k: Optional[int] = None,
        dataframe: Optional[pd.DataFrame] = None
    ) -> RAGResponse:
        """
        Main query method - routes and processes user queries.
        
        Args:
            user_query: The user's question
            doc_hash: Optional specific document to query
            k: Number of chunks to retrieve
            dataframe: Optional DataFrame for direct analysis
            
        Returns:
            RAGResponse with answer, sources, and metadata
        """
        k = k or self.config.default_k
        
        # ====================================================================
        # STEP 1: Query Classification (Phase 4: LLM-based semantic routing)
        # ====================================================================
        cache_key = f"classify:{hash_key(user_query)}"
        classification = None
        
        if self.config.enable_cache:
            cached = classification_cache.get(cache_key)
            if cached:
                classification = ClassificationResult(**cached)
        
        if classification is None:
            df_columns = list(dataframe.columns) if dataframe is not None else []
            classification = self.classifier.classify(user_query, df_columns)
            
            if self.config.enable_cache:
                classification_cache.set(cache_key, {
                    "mode": classification.mode,
                    "confidence": classification.confidence,
                    "reason": classification.reason
                })
        
        logger.info(f"Query classified as {classification.mode.value} "
                   f"(confidence: {classification.confidence:.2f})")
        
        # ====================================================================
        # STEP 2: Route to Handler
        # ====================================================================
        mode = classification.mode
        handler = self.handlers.get(mode)
        
        if handler is None:
            logger.warning(f"No handler for mode {mode}, falling back to freeform")
            handler = self.handlers[QueryMode.FREEFORM_QUERY]
        
        # ====================================================================
        # STEP 3: Handle Non-Data Queries (Fast Path)
        # ====================================================================
        if mode in (QueryMode.FREEFORM_QUERY, QueryMode.SYSTEM_TASK):
            result = handler.handle(
                query=user_query,
                context="",
                dataframe=None
            )
            return RAGResponse(
                answer=result.get("answer", ""),
                query_mode=mode.value,
                metadata=result.get("metadata", {})
            )
        
        # ====================================================================
        # STEP 4: Retrieve Context (Phase 3: Hybrid search)
        # ====================================================================
        from core.embedder import embed_query
        
        query_embedding = embed_query(user_query)
        
        # Vector search
        vector_results = self.vector_searcher.search_with_context(
            query_embedding,
            k=k,
            context_window=self.config.context_window,
            doc_hash=doc_hash
        )
        
        # Phase 3: Hybrid search with BM25 + RRF fusion
        results = vector_results
        if self.config.enable_hybrid_search and self.keyword_searcher:
            self._ensure_bm25_indexed()
            
            if self.keyword_searcher.is_indexed:
                keyword_results = self.keyword_searcher.search(user_query, k=k, doc_hash=doc_hash)
                
                if keyword_results:
                    # Reciprocal Rank Fusion
                    results = reciprocal_rank_fusion(
                        vector_results,
                        keyword_results,
                        vector_weight=self.config.vector_weight,
                        keyword_weight=self.config.keyword_weight
                    )
                    logger.info(f"Hybrid search: {len(vector_results)} vector + {len(keyword_results)} keyword = {len(results)} fused")
        
        # Optional: Re-rank results
        if self.config.enable_reranking and self.reranker and results:
            results = self.reranker.rerank(user_query, results, top_k=k)
        
        # ====================================================================
        # STEP 5: Build Context
        # ====================================================================
        context = self._build_context(results)
        
        # Get filename from results
        filename = results[0].get("filename") if results else None
        
        # ====================================================================
        # STEP 6: Compute Statistics (if DataFrame available)
        # ====================================================================
        stats_block = ""
        specific_metrics = self._detect_metrics(user_query)
        target_columns = []
        
        if dataframe is not None and not dataframe.empty:
            if specific_metrics:
                target_columns = get_target_columns(
                    specific_metrics, 
                    dataframe.columns.tolist()
                )
            stats_block = compute_data_statistics(
                dataframe, 
                specific_metrics, 
                target_columns
            )
            if stats_block:
                context = f"{stats_block}\n\n---\n\n{context}"
        
        # ====================================================================
        # STEP 7: Generate Response via Handler
        # ====================================================================
        result = handler.handle(
            query=user_query,
            context=context,
            dataframe=dataframe,
            stats_block=stats_block,
            specific_metrics=specific_metrics
        )
        
        # ====================================================================
        # STEP 8: Format Sources
        # ====================================================================
        sources = self._format_sources(results)
        
        # ====================================================================
        # STEP 9: Build Response
        # ====================================================================
        return RAGResponse(
            answer=result.get("answer", ""),
            query_mode=mode.value,
            sources=sources,
            show_visualizations=(mode == QueryMode.DATA_QUERY),
            specific_metrics=specific_metrics,
            target_columns=target_columns,
            metadata={
                "detail_mode": result.get("detail_mode", "normal"),
                "num_chunks": len(results),
                "has_statistics": bool(stats_block),
                "confidence": classification.confidence,
                "hybrid_search": self.config.enable_hybrid_search
            }
        )
    
    def query_stream(
        self,
        user_query: str,
        doc_hash: Optional[str] = None,
        k: Optional[int] = None,
        dataframe: Optional[pd.DataFrame] = None,
        on_token: Optional[Callable[[str], None]] = None
    ) -> Generator[str, None, RAGResponse]:
        """
        Stream query response tokens (Phase 5).
        
        Args:
            user_query: The user's question
            doc_hash: Optional specific document to query
            k: Number of chunks to retrieve
            dataframe: Optional DataFrame for direct analysis
            on_token: Optional callback for each token
            
        Yields:
            Individual response tokens
            
        Returns:
            Final RAGResponse (access after iteration completes)
        """
        k = k or self.config.default_k
        
        # Classification (same as regular query)
        df_columns = list(dataframe.columns) if dataframe is not None else []
        classification = self.classifier.classify(user_query, df_columns)
        mode = classification.mode
        
        # Fast path for non-data queries
        if mode in (QueryMode.FREEFORM_QUERY, QueryMode.SYSTEM_TASK):
            handler = self.handlers.get(mode)
            result = handler.handle(query=user_query, context="", dataframe=None)
            answer = result.get("answer", "")
            yield answer
            return RAGResponse(answer=answer, query_mode=mode.value)
        
        # Retrieve context
        from core.embedder import embed_query
        from core.llm.prompts import SYSTEM_PROMPT_CONCISE, get_data_query_prompt
        
        query_embedding = embed_query(user_query)
        vector_results = self.vector_searcher.search_with_context(
            query_embedding, k=k, context_window=self.config.context_window, doc_hash=doc_hash
        )
        
        # Hybrid search
        results = vector_results
        if self.config.enable_hybrid_search and self.keyword_searcher:
            self._ensure_bm25_indexed()
            if self.keyword_searcher.is_indexed:
                keyword_results = self.keyword_searcher.search(user_query, k=k, doc_hash=doc_hash)
                if keyword_results:
                    results = reciprocal_rank_fusion(
                        vector_results, keyword_results,
                        vector_weight=self.config.vector_weight,
                        keyword_weight=self.config.keyword_weight
                    )
        
        # Build context
        context = self._build_context(results)
        
        # Compute stats
        stats_block = ""
        specific_metrics = self._detect_metrics(user_query)
        target_columns = []
        
        if dataframe is not None and not dataframe.empty:
            if specific_metrics:
                target_columns = get_target_columns(specific_metrics, dataframe.columns.tolist())
            stats_block = compute_data_statistics(dataframe, specific_metrics, target_columns)
            if stats_block:
                context = f"{stats_block}\n\n---\n\n{context}"
        
        # Build prompt
        user_prompt = get_data_query_prompt(
            user_query=user_query,
            stats_block=stats_block,
            context=context,
            is_detailed=False
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_CONCISE},
            {"role": "user", "content": user_prompt}
        ]
        
        # Stream response
        full_answer = []
        for token in self.llm.stream(messages, on_token=on_token):
            full_answer.append(token)
            yield token
        
        # Return final response
        sources = self._format_sources(results)
        return RAGResponse(
            answer="".join(full_answer),
            query_mode=mode.value,
            sources=sources,
            show_visualizations=(mode == QueryMode.DATA_QUERY),
            specific_metrics=specific_metrics,
            target_columns=target_columns,
            metadata={
                "num_chunks": len(results),
                "has_statistics": bool(stats_block),
                "confidence": classification.confidence,
                "streamed": True
            }
        )
    
    def _build_context(
        self, 
        results: List[Dict[str, Any]], 
        max_chars: Optional[int] = None
    ) -> str:
        """Build context string from search results."""
        max_chars = max_chars or self.config.max_context_chars
        
        context_parts = []
        total_chars = 0
        
        for r in results:
            text = r.get("text", "")
            if total_chars + len(text) > max_chars:
                # Truncate to fit
                remaining = max_chars - total_chars
                if remaining > 100:
                    context_parts.append(text[:remaining])
                break
            context_parts.append(text)
            total_chars += len(text)
        
        return "\n\n---\n\n".join(context_parts)
    
    def _format_sources(
        self, 
        results: List[Dict[str, Any]], 
        max_sources: int = 5
    ) -> List[Dict[str, Any]]:
        """Format search results as source citations."""
        sources = []
        
        for r in results[:max_sources]:
            raw_score = r.get("score", 0)
            # Normalize score to 0-1 range
            if isinstance(raw_score, float) and raw_score <= 1.0:
                display_score = raw_score
            else:
                display_score = min(1.0, max(0.0, (raw_score + 1) / 2))
            
            sources.append({
                "filename": r.get("filename", "unknown"),
                "doc_hash": r.get("doc_hash", ""),
                "score": display_score,
                "raw_score": raw_score,
                "preview": r.get("text", "")[:300] + "..." if len(r.get("text", "")) > 300 else r.get("text", ""),
                "type": r.get("type", "text"),
                "sources": r.get("sources", ["vector"])  # Track which search found it
            })
        
        return sources
    
    def _detect_metrics(self, query: str) -> List[str]:
        """Detect which metrics are mentioned in the query."""
        query_lower = query.lower()
        
        metric_keywords = {
            'oil': ['oil', 'crude', 'petroleum'],
            'gas': ['gas', 'natural gas', 'ng'],
            'water': ['water', 'wat', 'produced water'],
            'condensate': ['condensate', 'cond'],
            'lpg': ['lpg'],
            'ngl': ['ngl'],
            'energy': ['energy', 'btu', 'heat'],
            'injection': ['injection', 'inj', 'inject'],
            'production': ['production', 'prod', 'produce'],
            'sales': ['sales', 'sold', 'sell']
        }
        
        detected = []
        for metric, keywords in metric_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected.append(metric)
        
        return detected
    
    def summarize_document(
        self,
        doc_hash: str,
        dataframe: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generate document summary.
        
        Args:
            doc_hash: Document hash to summarize
            dataframe: Optional DataFrame for enhanced summary
            
        Returns:
            Summary result with status, summary text, and metadata
        """
        # Get document info
        docs = self.vector_searcher.store.get_all_documents()
        doc_info = next((d for d in docs if d.get("doc_hash") == doc_hash), {})
        filename = doc_info.get("filename", "unknown")
        
        # Use document overview handler
        handler = self.handlers.get(QueryMode.DOC_OVERVIEW)
        
        if dataframe is not None and not dataframe.empty:
            # Generate Python-based executive summary
            result = handler.handle(
                query="Generate executive summary",
                context="",
                dataframe=dataframe
            )
            return {
                "status": "success",
                "summary": result.get("answer", ""),
                "filename": filename,
                "num_chunks": doc_info.get("num_chunks", 0),
                "is_dataset": True
            }
        
        # Fallback to chunk-based summary
        chunks = self.vector_searcher.store.get_document_chunks(doc_hash)
        tables = self.vector_searcher.store.get_document_tables(doc_hash)
        
        if not chunks and not tables:
            return {
                "status": "error",
                "message": "Document not found or empty"
            }
        
        # Build context from chunks/tables
        if tables:
            context = "\n\n---\n\n".join(
                t.get("text", "")[:3000] for t in tables[:2]
            )
        else:
            context = self.vector_searcher.store.get_full_document_text(doc_hash)[:8000]
        
        result = handler.handle(
            query="Generate document summary",
            context=context,
            dataframe=None
        )
        
        return {
            "status": "success",
            "summary": result.get("answer", ""),
            "filename": filename,
            "num_chunks": len(chunks),
            "num_tables": len(tables),
            "is_dataset": doc_info.get("is_dataset", False)
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents."""
        return self.vector_searcher.store.get_all_documents()
    
    def get_document_info(self, doc_hash: str) -> Dict[str, Any]:
        """Get detailed document information."""
        chunks = self.vector_searcher.store.get_document_chunks(doc_hash)
        tables = self.vector_searcher.store.get_document_tables(doc_hash)
        
        if not chunks:
            return {"error": "Document not found"}
        
        docs = self.vector_searcher.store.get_all_documents()
        doc_info = next((d for d in docs if d.get("doc_hash") == doc_hash), {})
        
        # Extract column info from tables
        columns = []
        sample_data = []
        
        for table in tables[:1]:
            text = table.get("text", "")
            lines = text.strip().split("\n")
            if lines and "|" in lines[0]:
                headers = [h.strip() for h in lines[0].split("|") if h.strip()]
                columns = headers
                
                for line in lines[2:12]:
                    if "|" in line:
                        row = [c.strip() for c in line.split("|") if c.strip()]
                        sample_data.append(row)
        
        return {
            "doc_hash": doc_hash,
            "filename": doc_info.get("filename", "unknown"),
            "num_chunks": len(chunks),
            "num_tables": len(tables),
            "is_dataset": doc_info.get("is_dataset", False),
            "columns": columns,
            "sample_data": sample_data[:10],
            "total_tokens": sum(c.get("tokens", 0) for c in chunks)
        }
    
    def rebuild_bm25_index(self) -> int:
        """
        Force rebuild of BM25 index.
        
        Returns:
            Number of chunks indexed
        """
        if not self.keyword_searcher:
            return 0
        
        self._bm25_indexed = False
        self.keyword_searcher.clear()
        self._ensure_bm25_indexed()
        return self.keyword_searcher.num_documents
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        from core.cache import get_all_cache_stats
        from core.embedder import get_model_info
        
        return {
            "config": {
                "hybrid_search": self.config.enable_hybrid_search,
                "reranking": self.config.enable_reranking,
                "streaming": self.config.enable_streaming,
                "caching": self.config.enable_cache,
                "confidence_threshold": self.config.confidence_threshold
            },
            "vector_store": self.vector_searcher.get_stats(),
            "bm25": {
                "indexed": self._bm25_indexed,
                "num_documents": self.keyword_searcher.num_documents if self.keyword_searcher else 0
            },
            "embedding_model": get_model_info(),
            "cache_stats": get_all_cache_stats()
        }


# ============================================================================
# Backward Compatibility Layer
# ============================================================================

# Global engine instance (lazy-initialized)
_engine: Optional[RAGEngine] = None


def get_engine() -> RAGEngine:
    """Get or create global engine instance."""
    global _engine
    if _engine is None:
        _engine = RAGEngine.create_default()
    return _engine


def reset_engine() -> None:
    """Reset the global engine (for testing)."""
    global _engine
    _engine = None


def query(
    user_query: str,
    doc_hash: Optional[str] = None,
    k: int = 10,
    dataframe: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Legacy query function for backward compatibility.
    
    Delegates to RAGEngine.query() and returns dict.
    """
    engine = get_engine()
    response = engine.query(user_query, doc_hash, k, dataframe)
    return response.to_dict()


def summarize_document(
    doc_hash: str,
    dataframe: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Legacy summarize function."""
    engine = get_engine()
    return engine.summarize_document(doc_hash, dataframe)


def list_documents() -> List[Dict[str, Any]]:
    """Legacy list documents function."""
    engine = get_engine()
    return engine.list_documents()


def get_document_info(doc_hash: str) -> Dict[str, Any]:
    """Legacy get document info function."""
    engine = get_engine()
    return engine.get_document_info(doc_hash)


def set_dataframe_cache(cache: Dict[str, pd.DataFrame]) -> None:
    """Set DataFrame cache on engine."""
    engine = get_engine()
    engine.set_dataframe_cache(cache)
