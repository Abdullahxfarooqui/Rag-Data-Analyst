"""
Production RAG System Integration.

UNIFIED ENTRY POINT FOR THE PRODUCTION RAG SYSTEM.

This module integrates all production components:
1. Dynamic Ingestion Engine
2. Production FAISS Vector Store
3. Async Task Queue
4. Dynamic Schema Generator
5. Observability & Monitoring
6. Production Caching

SYSTEM ARCHITECTURE:
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

REQUEST FLOW:
1. Query arrives → check cache → if miss, continue
2. Generate embedding → check embedding cache
3. FAISS retrieval → check retrieval cache
4. LLM generation → check LLM cache
5. Dynamic schema output → return structured JSON
6. Cache results at each layer
7. Track metrics throughout

USAGE:
    from core.integration.production_system import ProductionRAGSystem
    
    # Initialize
    system = ProductionRAGSystem(config)
    await system.initialize()
    
    # Ingest documents
    await system.ingest_document("path/to/document.csv")
    
    # Query
    result = await system.query("What are the top selling products?")
    
    # Get structured output
    print(result.metrics)
    print(result.rankings)
    print(result.trends)
"""
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

# Core imports (with graceful fallback)
try:
    from ..ingestion.dynamic_ingestor import DynamicIngestor, DocumentSchema
except ImportError:
    DynamicIngestor = None
    DocumentSchema = None

try:
    from ..retrieval.production_vector_store import ProductionVectorStore, IndexConfig
except ImportError:
    ProductionVectorStore = None
    IndexConfig = None

try:
    from ..queue.task_queue import AsyncTaskQueue, Task, TaskPriority
except ImportError:
    AsyncTaskQueue = None
    Task = None
    TaskPriority = None

try:
    from ..schema.dynamic_schema import DynamicSchemaGenerator, DynamicOutput
except ImportError:
    DynamicSchemaGenerator = None
    DynamicOutput = None

try:
    from ..observability.monitoring import (
        metrics, tracer, StructuredLogger, SystemMonitor,
        Alerter, AlertRule, AlertLevel,
        track_llm_call, track_faiss_query, track_cache
    )
except ImportError:
    metrics = None
    tracer = None
    StructuredLogger = None

try:
    from ..cache.production_cache import (
        LRUCache, TieredCache, FAISSRetrievalCache,
        LLMResponseCache, EmbeddingCache, cache_manager
    )
except ImportError:
    LRUCache = None
    TieredCache = None
    cache_manager = None

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SystemConfig:
    """Configuration for the production RAG system."""
    
    # Data paths
    data_dir: str = "data"
    cache_dir: str = "data/cache"
    index_dir: str = "data/faiss_index"
    upload_dir: str = "data/uploads"
    
    # FAISS settings
    embedding_dim: int = 384
    faiss_index_type: str = "auto"  # auto, flat, ivf, ivf_pq, hnsw
    faiss_nlist: int = 100
    
    # Task queue settings
    num_workers: int = 4
    max_queue_size: int = 1000
    task_timeout: float = 300.0
    
    # Cache settings
    embedding_cache_size: int = 10000
    faiss_cache_size: int = 500
    llm_cache_size: int = 200
    cache_ttl_seconds: float = 3600.0
    use_disk_cache: bool = True
    
    # LLM settings
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096
    
    # Monitoring settings
    enable_monitoring: bool = True
    metrics_interval: float = 10.0
    enable_alerts: bool = True
    
    # Performance tuning
    batch_size: int = 32
    concurrent_requests: int = 10


@dataclass
class QueryResult:
    """Result from a query."""
    query: str
    answer: str
    structured_output: Optional[Dict] = None
    sources: List[Dict] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False
    trace_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "structured_output": self.structured_output,
            "sources": self.sources,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "trace_id": self.trace_id,
        }


# ============================================================================
# PRODUCTION RAG SYSTEM
# ============================================================================

class ProductionRAGSystem:
    """
    Production-grade RAG system with all components integrated.
    
    Features:
    - Dynamic document ingestion
    - Scalable FAISS indexing
    - Async task processing
    - Multi-tier caching
    - Comprehensive monitoring
    - Dynamic structured output
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self._initialized = False
        
        # Components (initialized lazily)
        self.ingestor: Optional[DynamicIngestor] = None
        self.vector_store: Optional[ProductionVectorStore] = None
        self.task_queue: Optional[AsyncTaskQueue] = None
        self.schema_generator: Optional[DynamicSchemaGenerator] = None
        
        # Caches
        self.embedding_cache: Optional[EmbeddingCache] = None
        self.faiss_cache: Optional[FAISSRetrievalCache] = None
        self.llm_cache: Optional[LLMResponseCache] = None
        
        # Monitoring
        self.system_monitor: Optional[SystemMonitor] = None
        self.alerter: Optional[Alerter] = None
        self.logger = StructuredLogger(__name__) if StructuredLogger else logger
        
        # Document registry
        self._documents: Dict[str, DocumentSchema] = {}
        self._embedder: Optional[Callable] = None
        self._llm: Optional[Callable] = None
    
    async def initialize(self):
        """Initialize all system components."""
        if self._initialized:
            return
        
        self.logger.info("Initializing Production RAG System...")
        
        # Create directories
        for dir_path in [self.config.data_dir, self.config.cache_dir, 
                         self.config.index_dir, self.config.upload_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        await self._init_ingestor()
        await self._init_vector_store()
        await self._init_task_queue()
        await self._init_schema_generator()
        await self._init_caches()
        await self._init_monitoring()
        
        self._initialized = True
        self.logger.info("Production RAG System initialized successfully")
    
    async def _init_ingestor(self):
        """Initialize document ingestor."""
        if DynamicIngestor:
            self.ingestor = DynamicIngestor()
            self.logger.info("Dynamic Ingestor initialized")
    
    async def _init_vector_store(self):
        """Initialize FAISS vector store."""
        if ProductionVectorStore and IndexConfig:
            config = IndexConfig(
                dimension=self.config.embedding_dim,
                index_type=self.config.faiss_index_type,
                nlist=self.config.faiss_nlist,
            )
            self.vector_store = ProductionVectorStore(
                config=config,
                persist_dir=self.config.index_dir,
            )
            self.logger.info("Production Vector Store initialized")
    
    async def _init_task_queue(self):
        """Initialize async task queue."""
        if AsyncTaskQueue:
            self.task_queue = AsyncTaskQueue(
                num_workers=self.config.num_workers,
                max_queue_size=self.config.max_queue_size,
            )
            self.task_queue.start()
            self.logger.info("Async Task Queue initialized")
    
    async def _init_schema_generator(self):
        """Initialize schema generator."""
        if DynamicSchemaGenerator:
            self.schema_generator = DynamicSchemaGenerator()
            self.logger.info("Dynamic Schema Generator initialized")
    
    async def _init_caches(self):
        """Initialize caching layer."""
        disk_path = os.path.join(self.config.cache_dir, "cache.db") if self.config.use_disk_cache else None
        
        if EmbeddingCache:
            self.embedding_cache = EmbeddingCache(
                max_size=self.config.embedding_cache_size,
                ttl_seconds=self.config.cache_ttl_seconds,
                disk_path=disk_path,
            )
            if cache_manager:
                cache_manager.register("embeddings", self.embedding_cache)
        
        if FAISSRetrievalCache:
            self.faiss_cache = FAISSRetrievalCache(
                max_size=self.config.faiss_cache_size,
                ttl_seconds=self.config.cache_ttl_seconds,
            )
            if cache_manager:
                cache_manager.register("faiss", self.faiss_cache)
        
        if LLMResponseCache:
            self.llm_cache = LLMResponseCache(
                max_size=self.config.llm_cache_size,
                ttl_seconds=self.config.cache_ttl_seconds,
                disk_path=disk_path.replace(".db", "_llm.db") if disk_path else None,
            )
            if cache_manager:
                cache_manager.register("llm", self.llm_cache)
        
        self.logger.info("Caching layer initialized")
    
    async def _init_monitoring(self):
        """Initialize monitoring components."""
        if not self.config.enable_monitoring:
            return
        
        if SystemMonitor:
            self.system_monitor = SystemMonitor(
                interval_seconds=self.config.metrics_interval
            )
            self.system_monitor.start()
        
        if self.config.enable_alerts and Alerter and AlertRule:
            self.alerter = Alerter()
            
            # Add default alert rules
            self.alerter.add_rule(AlertRule(
                name="high_latency",
                metric_name="llm_latency_seconds",
                threshold=5.0,
                comparison="gt",
                level=AlertLevel.WARNING,
            ))
            
            self.alerter.add_rule(AlertRule(
                name="high_error_rate",
                metric_name="llm_errors_total",
                threshold=10,
                comparison="gt",
                level=AlertLevel.ERROR,
            ))
            
            self.alerter.add_rule(AlertRule(
                name="high_memory",
                metric_name="memory_usage_bytes",
                threshold=4 * 1024 * 1024 * 1024,  # 4GB
                comparison="gt",
                level=AlertLevel.WARNING,
            ))
        
        self.logger.info("Monitoring initialized")
    
    def set_embedder(self, embedder: Callable[[List[str]], np.ndarray]):
        """Set the embedding function."""
        self._embedder = embedder
    
    def set_llm(self, llm: Callable[[str], str]):
        """Set the LLM function."""
        self._llm = llm
    
    # ========================================================================
    # INGESTION
    # ========================================================================
    
    async def ingest_document(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        callback: Optional[Callable[[float, str], None]] = None
    ) -> DocumentSchema:
        """
        Ingest a document into the system.
        
        Args:
            file_path: Path to document file
            document_id: Optional document ID
            metadata: Additional metadata
            callback: Progress callback (progress: float, message: str)
        
        Returns:
            DocumentSchema with extracted information
        """
        start_time = time.time()
        
        if not self.ingestor:
            raise RuntimeError("Ingestor not initialized")
        
        self.logger.info(f"Ingesting document: {file_path}")
        
        if callback:
            callback(0.1, "Detecting document type...")
        
        # Ingest document
        schema = self.ingestor.ingest(file_path)
        
        if document_id:
            schema.document_id = document_id
        
        if metadata:
            schema.metadata.update(metadata)
        
        if callback:
            callback(0.3, "Extracting content...")
        
        # Generate chunks
        chunks = self.ingestor.generate_chunks(schema)
        
        if callback:
            callback(0.5, f"Generated {len(chunks)} chunks")
        
        # Generate embeddings
        if self._embedder and self.vector_store:
            if callback:
                callback(0.6, "Generating embeddings...")
            
            texts = [c["content"] for c in chunks]
            embeddings = await self._get_embeddings(texts)
            
            if callback:
                callback(0.8, "Indexing vectors...")
            
            # Add to vector store
            self.vector_store.add(
                embeddings=embeddings,
                metadatas=chunks,
                ids=[c["chunk_id"] for c in chunks],
            )
        
        # Register document
        self._documents[schema.document_id] = schema
        
        if callback:
            callback(1.0, "Ingestion complete")
        
        # Track metrics
        if metrics:
            metrics.counter("documents_ingested_total").inc()
            metrics.counter("chunks_created_total").inc(len(chunks))
            metrics.histogram("ingestion_time_seconds").observe(time.time() - start_time)
        
        self.logger.info(
            f"Document ingested: {schema.document_id}",
            chunks=len(chunks),
            tables=len(schema.tables),
            latency_ms=(time.time() - start_time) * 1000
        )
        
        return schema
    
    async def ingest_batch(
        self,
        file_paths: List[str],
        callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[DocumentSchema]:
        """
        Ingest multiple documents.
        
        Args:
            file_paths: List of file paths
            callback: Progress callback (completed: int, total: int, message: str)
        
        Returns:
            List of DocumentSchema objects
        """
        results = []
        total = len(file_paths)
        
        for i, path in enumerate(file_paths):
            try:
                schema = await self.ingest_document(path)
                results.append(schema)
                
                if callback:
                    callback(i + 1, total, f"Ingested {path}")
            
            except Exception as e:
                self.logger.error(f"Failed to ingest {path}: {e}")
                if callback:
                    callback(i + 1, total, f"Failed: {path}")
        
        return results
    
    # ========================================================================
    # RETRIEVAL
    # ========================================================================
    
    async def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings with caching."""
        if not self._embedder:
            raise RuntimeError("Embedder not set")
        
        # Check cache
        if self.embedding_cache:
            cached, missing_indices = self.embedding_cache.get_batch(texts)
            
            if not missing_indices:
                # All cached
                if track_cache:
                    for _ in texts:
                        track_cache(hit=True)
                return np.array([c for c in cached if c is not None])
            
            # Compute missing
            missing_texts = [texts[i] for i in missing_indices]
            new_embeddings = self._embedder(missing_texts)
            
            # Cache new embeddings
            self.embedding_cache.set_batch(missing_texts, list(new_embeddings))
            
            # Merge results
            result = []
            new_idx = 0
            for i, emb in enumerate(cached):
                if emb is not None:
                    result.append(emb)
                    if track_cache:
                        track_cache(hit=True)
                else:
                    result.append(new_embeddings[new_idx])
                    new_idx += 1
                    if track_cache:
                        track_cache(hit=False)
            
            return np.array(result)
        
        # No cache, compute directly
        return self._embedder(texts)
    
    async def retrieve(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Query string
            k: Number of results
            filters: Metadata filters
        
        Returns:
            List of relevant chunks with scores
        """
        start_time = time.time()
        
        if not self.vector_store or not self._embedder:
            return []
        
        # Get query embedding
        query_embedding = (await self._get_embeddings([query]))[0]
        
        # Check FAISS cache
        if self.faiss_cache:
            cached = self.faiss_cache.get(query_embedding, k, filters)
            if cached:
                distances, indices = cached
                if track_faiss_query:
                    track_faiss_query(time.time() - start_time, len(indices))
                if track_cache:
                    track_cache(hit=True)
                
                # Retrieve metadata
                return self.vector_store.get_by_ids(indices.tolist())
        
        # Perform search
        distances, indices, metadatas = self.vector_store.search(
            query_embedding=query_embedding,
            k=k,
        )
        
        # Cache results
        if self.faiss_cache:
            self.faiss_cache.set(query_embedding, k, distances, indices, filters)
            if track_cache:
                track_cache(hit=False)
        
        # Track metrics
        if track_faiss_query:
            track_faiss_query(time.time() - start_time, len(metadatas))
        
        return metadatas
    
    # ========================================================================
    # QUERY PROCESSING
    # ========================================================================
    
    async def query(
        self,
        query: str,
        k: int = 10,
        generate_structured: bool = True,
        use_cache: bool = True
    ) -> QueryResult:
        """
        Process a query end-to-end.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            generate_structured: Whether to generate structured output
            use_cache: Whether to use caching
        
        Returns:
            QueryResult with answer and structured data
        """
        start_time = time.time()
        trace_id = None
        
        # Start trace
        if tracer:
            span = tracer.start_trace("query")
            trace_id = span.trace_id
            span.set_tag("query", query[:100])
        
        try:
            # Check LLM cache
            cached_response = None
            if use_cache and self.llm_cache:
                cached_response = self.llm_cache.get(
                    prompt=query,
                    model=self.config.llm_model,
                )
            
            if cached_response:
                # Return cached result
                result = QueryResult(
                    query=query,
                    answer=cached_response.get("answer", ""),
                    structured_output=cached_response.get("structured"),
                    sources=cached_response.get("sources", []),
                    confidence=cached_response.get("confidence", 0.8),
                    latency_ms=(time.time() - start_time) * 1000,
                    cached=True,
                    trace_id=trace_id,
                )
                return result
            
            # Retrieve context
            if tracer:
                with tracer.trace("retrieve"):
                    sources = await self.retrieve(query, k=k)
            else:
                sources = await self.retrieve(query, k=k)
            
            # Generate answer
            answer = ""
            if self._llm:
                if tracer:
                    with tracer.trace("llm_generate"):
                        answer = await self._generate_answer(query, sources)
                else:
                    answer = await self._generate_answer(query, sources)
            
            # Generate structured output
            structured = None
            if generate_structured and self.schema_generator:
                if tracer:
                    with tracer.trace("schema_generate"):
                        structured = self._generate_structured_output(query, sources)
                else:
                    structured = self._generate_structured_output(query, sources)
            
            # Build result
            result = QueryResult(
                query=query,
                answer=answer,
                structured_output=structured,
                sources=sources,
                confidence=0.8,  # TODO: Calculate actual confidence
                latency_ms=(time.time() - start_time) * 1000,
                cached=False,
                trace_id=trace_id,
            )
            
            # Cache result
            if use_cache and self.llm_cache:
                self.llm_cache.set(
                    prompt=query,
                    model=self.config.llm_model,
                    response={
                        "answer": answer,
                        "structured": structured,
                        "sources": sources,
                        "confidence": result.confidence,
                    }
                )
            
            return result
        
        finally:
            if tracer and span:
                span.finish()
    
    async def _generate_answer(
        self,
        query: str,
        sources: List[Dict]
    ) -> str:
        """Generate answer using LLM."""
        if not self._llm:
            return ""
        
        # Build context
        context_parts = []
        for i, source in enumerate(sources[:5]):  # Top 5 sources
            content = source.get("content", "")[:500]
            context_parts.append(f"[Source {i+1}]: {content}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        start_time = time.time()
        try:
            answer = self._llm(prompt)
            
            if track_llm_call:
                # Estimate tokens (rough)
                tokens = len(prompt.split()) + len(answer.split())
                track_llm_call(tokens, time.time() - start_time, error=False)
            
            return answer
        
        except Exception as e:
            if track_llm_call:
                track_llm_call(0, time.time() - start_time, error=True)
            raise
    
    def _generate_structured_output(
        self,
        query: str,
        sources: List[Dict]
    ) -> Optional[Dict]:
        """Generate structured output from sources."""
        if not self.schema_generator:
            return None
        
        # Extract data from sources
        data_rows = []
        for source in sources:
            if "data" in source:
                data_rows.extend(source["data"])
        
        if not data_rows:
            # Try to extract from content
            for source in sources:
                content = source.get("content", "")
                # Simple extraction - could be enhanced
                if ":" in content:
                    parts = content.split("\n")
                    row = {}
                    for part in parts:
                        if ":" in part:
                            key, val = part.split(":", 1)
                            row[key.strip()] = val.strip()
                    if row:
                        data_rows.append(row)
        
        if not data_rows:
            return None
        
        # Generate dynamic output
        try:
            output = self.schema_generator.generate(data_rows, query)
            return output.to_dict()
        except Exception as e:
            self.logger.error(f"Schema generation failed: {e}")
            return None
    
    # ========================================================================
    # ASYNC TASK PROCESSING
    # ========================================================================
    
    async def submit_task(
        self,
        task_type: str,
        payload: Dict,
        priority: str = "normal",
        callback: Optional[Callable] = None
    ) -> str:
        """
        Submit an async task.
        
        Args:
            task_type: Type of task (ingest, query, analyze)
            payload: Task parameters
            priority: Priority level (critical, high, normal, low)
            callback: Completion callback
        
        Returns:
            Task ID
        """
        if not self.task_queue or not Task or not TaskPriority:
            raise RuntimeError("Task queue not initialized")
        
        # Map priority
        priority_map = {
            "critical": TaskPriority.CRITICAL,
            "high": TaskPriority.HIGH,
            "normal": TaskPriority.NORMAL,
            "low": TaskPriority.LOW,
        }
        task_priority = priority_map.get(priority, TaskPriority.NORMAL)
        
        # Create handler
        async def handler(p):
            if task_type == "ingest":
                return await self.ingest_document(p["file_path"])
            elif task_type == "query":
                return await self.query(p["query"])
            elif task_type == "analyze":
                return await self._analyze_documents(p)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        
        # Submit task
        task = Task(
            task_id=f"{task_type}_{int(time.time() * 1000)}",
            handler=handler,
            payload=payload,
            priority=task_priority,
        )
        
        task_id = self.task_queue.submit(task)
        
        self.logger.info(f"Task submitted: {task_id}", task_type=task_type)
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Dict:
        """Get task status."""
        if not self.task_queue:
            return {"status": "unknown"}
        
        result = self.task_queue.get_result(task_id)
        if result:
            return {
                "status": "completed" if result.success else "failed",
                "result": result.result if result.success else None,
                "error": result.error,
                "duration_ms": result.execution_time * 1000 if result.execution_time else None,
            }
        
        return {"status": "pending"}
    
    async def _analyze_documents(self, payload: Dict) -> Dict:
        """Analyze documents based on payload."""
        document_ids = payload.get("document_ids", list(self._documents.keys()))
        
        results = {
            "total_documents": len(document_ids),
            "total_tables": 0,
            "total_columns": 0,
            "document_summaries": [],
        }
        
        for doc_id in document_ids:
            schema = self._documents.get(doc_id)
            if schema:
                results["total_tables"] += len(schema.tables)
                for table in schema.tables:
                    results["total_columns"] += len(table.columns)
                
                results["document_summaries"].append({
                    "id": doc_id,
                    "file_name": schema.file_name,
                    "tables": len(schema.tables),
                    "rows": sum(t.row_count for t in schema.tables),
                })
        
        return results
    
    # ========================================================================
    # MONITORING & STATS
    # ========================================================================
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        stats = {
            "initialized": self._initialized,
            "documents": len(self._documents),
            "components": {
                "ingestor": self.ingestor is not None,
                "vector_store": self.vector_store is not None,
                "task_queue": self.task_queue is not None,
                "schema_generator": self.schema_generator is not None,
            },
        }
        
        # Vector store stats
        if self.vector_store:
            stats["vector_store"] = {
                "total_vectors": self.vector_store.count,
            }
        
        # Cache stats
        if cache_manager:
            stats["caches"] = cache_manager.all_stats()
        
        # Task queue stats
        if self.task_queue:
            stats["task_queue"] = {
                "pending": self.task_queue.pending_count,
                "running": self.task_queue.active_count,
            }
        
        # System metrics
        if metrics:
            stats["metrics"] = metrics.export()
        
        return stats
    
    def get_health(self) -> Dict:
        """Get system health status."""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
        }
        
        # Check components
        checks = {
            "ingestor": self.ingestor is not None,
            "vector_store": self.vector_store is not None,
            "embedder": self._embedder is not None,
            "llm": self._llm is not None,
        }
        
        health["checks"] = checks
        
        if not all(checks.values()):
            health["status"] = "degraded"
        
        return health
    
    # ========================================================================
    # SHUTDOWN
    # ========================================================================
    
    async def shutdown(self):
        """Gracefully shutdown the system."""
        self.logger.info("Shutting down Production RAG System...")
        
        # Stop task queue
        if self.task_queue:
            self.task_queue.shutdown(wait=True)
        
        # Stop monitoring
        if self.system_monitor:
            self.system_monitor.stop()
        
        # Save vector store
        if self.vector_store:
            self.vector_store.save()
        
        self.logger.info("System shutdown complete")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_production_system(
    data_dir: str = "data",
    **kwargs
) -> ProductionRAGSystem:
    """
    Factory function to create a configured ProductionRAGSystem.
    
    Args:
        data_dir: Base data directory
        **kwargs: Additional configuration options
    
    Returns:
        Configured ProductionRAGSystem instance
    """
    config = SystemConfig(
        data_dir=data_dir,
        cache_dir=os.path.join(data_dir, "cache"),
        index_dir=os.path.join(data_dir, "faiss_index"),
        upload_dir=os.path.join(data_dir, "uploads"),
        **kwargs
    )
    
    return ProductionRAGSystem(config)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ProductionRAGSystem",
    "SystemConfig",
    "QueryResult",
    "create_production_system",
]
