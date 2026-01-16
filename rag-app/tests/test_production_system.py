"""
Test suite for Production RAG System.

Tests all production components and their integration.
"""
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# TEST UTILITIES
# ============================================================================

class TestResult:
    """Test result container."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.duration_ms = 0
    
    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status} {self.name} ({self.duration_ms:.1f}ms)"


def run_test(name: str, test_func):
    """Run a single test."""
    result = TestResult(name)
    start = time.time()
    
    try:
        test_func()
        result.passed = True
    except Exception as e:
        result.error = str(e)
        logger.error(f"Test {name} failed: {e}", exc_info=True)
    
    result.duration_ms = (time.time() - start) * 1000
    return result


async def run_async_test(name: str, test_func):
    """Run an async test."""
    result = TestResult(name)
    start = time.time()
    
    try:
        await test_func()
        result.passed = True
    except Exception as e:
        result.error = str(e)
        logger.error(f"Test {name} failed: {e}", exc_info=True)
    
    result.duration_ms = (time.time() - start) * 1000
    return result


# ============================================================================
# COMPONENT TESTS
# ============================================================================

def test_dynamic_ingestor():
    """Test the Dynamic Ingestor component."""
    from core.ingestion.dynamic_ingestor import (
        DynamicIngestor, DocumentType, detect_document_type
    )
    
    ingestor = DynamicIngestor()
    assert ingestor is not None
    
    # Test document type detection (module-level function)
    assert detect_document_type("test.csv") == DocumentType.CSV
    assert detect_document_type("test.xlsx") == DocumentType.EXCEL
    assert detect_document_type("test.pdf") == DocumentType.PDF
    assert detect_document_type("test.txt") == DocumentType.TEXT
    assert detect_document_type("test.json") == DocumentType.JSON
    
    logger.info("Dynamic Ingestor tests passed")


def test_production_vector_store():
    """Test the Production Vector Store component."""
    from core.retrieval.production_vector_store import (
        ProductionVectorStore, IndexConfig
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = IndexConfig(dimension=128, index_type="flat")
        store = ProductionVectorStore(storage_path=tmpdir, dimension=128, config=config)
        
        # Test adding vectors
        embeddings = np.random.randn(10, 128).astype(np.float32)
        chunks = [
            {"text": f"This is document {i}", "metadata": {"doc_id": f"doc_{i}"}} 
            for i in range(10)
        ]
        
        store.add_chunks(chunks, embeddings)
        assert store.num_vectors == 10
        
        # Test search
        query = np.random.randn(128).astype(np.float32)
        results = store.search(query, k=5)
        
        assert len(results) <= 5
        
        # Test persistence
        store.save()
        
        # Load into new store (auto-loads on init)
        store2 = ProductionVectorStore(storage_path=tmpdir, dimension=128, config=config)
        assert store2.num_vectors == 10
    
    logger.info("Production Vector Store tests passed")


def test_task_queue():
    """Test the Async Task Queue component."""
    from core.queue.task_queue import AsyncTaskQueue, TaskPriority
    
    # Queue auto-starts workers on init
    queue = AsyncTaskQueue(worker_count=2)
    
    # Register a handler
    def simple_handler(payload):
        time.sleep(0.1)
        return payload["value"] * 2
    
    queue.register_handler("multiply", simple_handler)
    
    # Submit tasks
    task_ids = []
    for i in range(5):
        task_id = queue.submit(
            task_type="multiply",
            payload={"value": i},
            priority=TaskPriority.NORMAL,
        )
        task_ids.append(task_id)
    
    # Wait for completion
    time.sleep(2)
    
    # Check results
    completed = []
    for task_id in task_ids:
        result = queue.get_result(task_id)
        if result and result.is_success:
            completed.append(result.result)
    
    queue.shutdown(wait=True)
    
    assert len(completed) == 5
    assert sorted(completed) == [0, 2, 4, 6, 8]
    
    logger.info("Task Queue tests passed")


def test_dynamic_schema_generator():
    """Test the Dynamic Schema Generator component."""
    from core.schema.dynamic_schema import DynamicSchemaGenerator
    import pandas as pd
    
    generator = DynamicSchemaGenerator()
    
    # Sample data as DataFrame
    data = pd.DataFrame([
        {"product": "Widget A", "sales": 1000, "region": "North"},
        {"product": "Widget B", "sales": 1500, "region": "South"},
        {"product": "Widget C", "sales": 800, "region": "East"},
        {"product": "Widget D", "sales": 2000, "region": "West"},
        {"product": "Widget E", "sales": 1200, "region": "North"},
    ])
    
    query = "What are the top selling products?"
    output = generator.generate(data, query)
    
    # Check that output is generated
    assert output is not None
    
    # Check that we have some structured output
    output_dict = output.to_dict()
    assert "confidence_level" in output_dict
    
    # Check rankings are generated (since we asked for "top")
    assert len(output.rankings) > 0
    
    logger.info("Dynamic Schema Generator tests passed")


def test_observability():
    """Test the Observability & Monitoring component."""
    from core.observability.monitoring import (
        metrics, Counter, Gauge, Histogram, timer, timed
    )
    
    # Test counter
    counter = metrics.counter("test_counter")
    counter.inc()
    counter.inc(5)
    assert counter.value == 6
    
    # Test gauge
    gauge = metrics.gauge("test_gauge")
    gauge.set(100)
    assert gauge.value == 100
    gauge.inc(10)
    assert gauge.value == 110
    
    # Test histogram
    histogram = metrics.histogram("test_histogram")
    for i in range(100):
        histogram.observe(i / 100)
    assert histogram.count == 100
    assert histogram.mean > 0
    
    # Test timer
    with timer("test_timer"):
        time.sleep(0.1)
    
    # Test export
    exported = metrics.export()
    assert "test_counter" in exported
    assert "test_gauge" in exported
    
    logger.info("Observability tests passed")


def test_caching():
    """Test the Production Caching component."""
    from core.cache.production_cache import (
        LRUCache, generate_cache_key, memoize
    )
    
    # Test LRU Cache
    cache = LRUCache(max_size=5)
    
    for i in range(10):
        cache.set(f"key_{i}", f"value_{i}")
    
    # Only last 5 should be present
    assert cache.get("key_9") == "value_9"
    assert cache.get("key_0") is None  # Evicted
    
    # Test cache key generation
    key1 = generate_cache_key("test", x=1, y=2)
    key2 = generate_cache_key("test", x=1, y=2)
    key3 = generate_cache_key("test", x=1, y=3)
    
    assert key1 == key2
    assert key1 != key3
    
    # Test memoize decorator
    call_count = [0]
    
    @memoize(max_size=10)
    def expensive_func(x):
        call_count[0] += 1
        return x * 2
    
    assert expensive_func(5) == 10
    assert expensive_func(5) == 10  # Cached
    assert call_count[0] == 1  # Only called once
    
    logger.info("Caching tests passed")


async def test_production_system():
    """Test the integrated Production RAG System."""
    from core.integration.production_system import (
        ProductionRAGSystem, SystemConfig, create_production_system
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SystemConfig(
            data_dir=tmpdir,
            cache_dir=os.path.join(tmpdir, "cache"),
            index_dir=os.path.join(tmpdir, "index"),
            upload_dir=os.path.join(tmpdir, "uploads"),
            num_workers=2,
            enable_monitoring=False,  # Disable for tests
        )
        
        system = ProductionRAGSystem(config)
        
        # Just test basic initialization (without vector store for now)
        # since initialization requires embedding model
        assert system is not None
        assert system.config.data_dir == tmpdir
        
        # Check health before initialization
        health = system.get_health()
        assert health["status"] in ["healthy", "degraded"]
    
    logger.info("Production System integration tests passed")


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("Production RAG System Test Suite")
    print("=" * 60 + "\n")
    
    results = []
    
    # Sync tests
    sync_tests = [
        ("Dynamic Ingestor", test_dynamic_ingestor),
        ("Production Vector Store", test_production_vector_store),
        ("Task Queue", test_task_queue),
        ("Dynamic Schema Generator", test_dynamic_schema_generator),
        ("Observability", test_observability),
        ("Caching", test_caching),
    ]
    
    for name, test_func in sync_tests:
        result = run_test(name, test_func)
        results.append(result)
        print(result)
    
    # Async tests
    async def run_async_tests():
        async_results = []
        
        async_tests = [
            ("Production System Integration", test_production_system),
        ]
        
        for name, test_func in async_tests:
            result = await run_async_test(name, test_func)
            async_results.append(result)
            print(result)
        
        return async_results
    
    async_results = asyncio.run(run_async_tests())
    results.extend(async_results)
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total_time = sum(r.duration_ms for r in results)
    
    print("\n" + "-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print(f"Total time: {total_time:.1f}ms")
    print("-" * 60)
    
    if failed > 0:
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.error}")
    
    return failed == 0


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
