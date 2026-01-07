"""
Comprehensive Tests for Modular RAG System.

Phase 7: Testing & Observability

Tests cover:
- LLM client
- Query classification (routing)
- Retrieval (vector + BM25 + hybrid)
- Analytics (statistics)
- Cache
- Full engine integration
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# CACHE TESTS
# ============================================================================

class TestTTLCache:
    """Tests for TTL cache module."""
    
    def test_cache_set_get(self):
        """Test basic cache set and get."""
        from core.cache import TTLCache
        
        cache = TTLCache[str](default_ttl=60)
        cache.set("key1", "value1")
        
        assert cache.get("key1") == "value1"
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        from core.cache import TTLCache
        
        cache = TTLCache[str](default_ttl=60)
        
        assert cache.get("nonexistent") is None
    
    def test_cache_expiration(self):
        """Test cache entry expiration."""
        from core.cache import TTLCache
        import time
        
        cache = TTLCache[str](default_ttl=0.1)  # 100ms TTL
        cache.set("key1", "value1")
        
        # Should exist immediately
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should be expired
        assert cache.get("key1") is None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        from core.cache import TTLCache
        
        cache = TTLCache[str](default_ttl=60)
        cache.set("key1", "value1")
        
        # Hit
        cache.get("key1")
        # Miss
        cache.get("key2")
        
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate_percent"] == 50.0
    
    def test_cache_delete(self):
        """Test cache deletion."""
        from core.cache import TTLCache
        
        cache = TTLCache[str](default_ttl=60)
        cache.set("key1", "value1")
        
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("key1") is False
    
    def test_cache_clear(self):
        """Test cache clearing."""
        from core.cache import TTLCache
        
        cache = TTLCache[str](default_ttl=60)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.clear()
        
        assert cache.size == 0
        assert cache.get("key1") is None
    
    def test_hash_key(self):
        """Test cache key hashing."""
        from core.cache import hash_key
        
        key1 = hash_key("query1", k=10)
        key2 = hash_key("query1", k=10)
        key3 = hash_key("query2", k=10)
        
        assert key1 == key2
        assert key1 != key3


# ============================================================================
# ANALYTICS TESTS
# ============================================================================

class TestAnalytics:
    """Tests for analytics module."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample production DataFrame."""
        return pd.DataFrame({
            "ITEM_NAME": ["Well-A", "Well-A", "Well-B", "Well-B"],
            "START_DATETIME": pd.date_range("2024-01-01", periods=4),
            "PROD_OIL_VOL": [100.0, 150.0, 200.0, 250.0],
            "PROD_GAS_VOL": [1000.0, 1500.0, 2000.0, 2500.0],
            "PROD_WAT_VOL": [50.0, 75.0, 100.0, 125.0],
            "PROD_OIL_VOL_UOM": ["bbl", "bbl", "bbl", "bbl"],
            "PROD_GAS_VOL_UOM": ["MCF", "MCF", "MCF", "MCF"],
        })
    
    def test_detect_specific_metrics(self):
        """Test metric detection from queries."""
        from core.analytics import detect_specific_metrics
        
        assert "oil" in detect_specific_metrics("What is oil production?")
        assert "gas" in detect_specific_metrics("Show gas trends")
        assert "water" in detect_specific_metrics("water injection volume")
        assert detect_specific_metrics("hello world") == []
    
    def test_get_target_columns(self, sample_df):
        """Test target column selection."""
        from core.analytics import get_target_columns
        
        columns = get_target_columns(["oil"], sample_df.columns.tolist())
        assert "PROD_OIL_VOL" in columns
        
        columns = get_target_columns(["gas"], sample_df.columns.tolist())
        assert "PROD_GAS_VOL" in columns
    
    def test_compute_data_statistics(self, sample_df):
        """Test statistics computation."""
        from core.analytics import compute_data_statistics
        
        stats = compute_data_statistics(
            sample_df,
            specific_metrics=["oil"],
            target_columns=["PROD_OIL_VOL"]
        )
        
        assert "700" in stats  # Total: 100+150+200+250 = 700
        assert "PROD_OIL_VOL" in stats
    
    def test_detect_detail_mode(self):
        """Test detail mode detection."""
        from core.analytics import detect_detail_mode
        
        assert detect_detail_mode("give me detailed analysis") == "detailed"
        assert detect_detail_mode("brief summary") == "brief"
        assert detect_detail_mode("what is oil?") == "normal"


# ============================================================================
# RETRIEVAL TESTS
# ============================================================================

class TestBM25Search:
    """Tests for BM25 keyword search."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for indexing."""
        return [
            {"text": "Oil production increased by 10% in Q1", "doc_hash": "doc1", "chunk_id": 0, "filename": "report.csv"},
            {"text": "Gas volumes remained stable throughout the year", "doc_hash": "doc1", "chunk_id": 1, "filename": "report.csv"},
            {"text": "Water injection rates were optimized", "doc_hash": "doc1", "chunk_id": 2, "filename": "report.csv"},
            {"text": "Total oil output reached 1 million barrels", "doc_hash": "doc2", "chunk_id": 0, "filename": "summary.csv"},
        ]
    
    def test_bm25_indexing(self, sample_chunks):
        """Test BM25 index building."""
        from core.retrieval import BM25Searcher
        
        searcher = BM25Searcher()
        searcher.index(sample_chunks)
        
        assert searcher.is_indexed
        assert searcher.num_documents == 4
    
    def test_bm25_search(self, sample_chunks):
        """Test BM25 search."""
        from core.retrieval import BM25Searcher
        
        searcher = BM25Searcher()
        searcher.index(sample_chunks)
        
        results = searcher.search("oil production", k=2)
        
        assert len(results) <= 2
        # Oil-related chunks should rank higher
        assert any("oil" in r.text.lower() for r in results)
    
    def test_bm25_doc_filter(self, sample_chunks):
        """Test BM25 search with document filter."""
        from core.retrieval import BM25Searcher
        
        searcher = BM25Searcher()
        searcher.index(sample_chunks)
        
        results = searcher.search("oil", k=10, doc_hash="doc2")
        
        # Should only return chunks from doc2
        assert all(r.doc_hash == "doc2" for r in results)
    
    def test_reciprocal_rank_fusion(self, sample_chunks):
        """Test RRF fusion of vector and keyword results."""
        from core.retrieval.keyword_search import reciprocal_rank_fusion, BM25Result
        from core.retrieval.vector_search import SearchResult
        
        # Mock vector results
        vector_results = [
            SearchResult(text="Oil production", score=0.9, doc_hash="doc1", filename="f.csv", chunk_id=0, chunk_type="text", metadata={}),
            SearchResult(text="Gas volumes", score=0.7, doc_hash="doc1", filename="f.csv", chunk_id=1, chunk_type="text", metadata={}),
        ]
        
        # Mock keyword results
        keyword_results = [
            BM25Result(text="Gas volumes", score=5.0, doc_hash="doc1", chunk_id=1, filename="f.csv"),
            BM25Result(text="Oil production", score=4.0, doc_hash="doc1", chunk_id=0, filename="f.csv"),
        ]
        
        fused = reciprocal_rank_fusion(vector_results, keyword_results)
        
        assert len(fused) == 2
        # Both sources should be tracked
        assert all("sources" in r for r in fused)


# ============================================================================
# ROUTING TESTS
# ============================================================================

class TestQueryClassifier:
    """Tests for query classification."""
    
    def test_query_mode_enum(self):
        """Test QueryMode enum parsing."""
        from core.routing import QueryMode
        
        assert QueryMode.from_string("DATA_QUERY") == QueryMode.DATA_QUERY
        assert QueryMode.from_string("doc_overview") == QueryMode.DOC_OVERVIEW
        assert QueryMode.from_string("invalid") == QueryMode.FREEFORM_QUERY
    
    def test_classification_result_properties(self):
        """Test ClassificationResult properties."""
        from core.routing import ClassificationResult, QueryMode
        
        result = ClassificationResult(
            mode=QueryMode.DATA_QUERY,
            confidence=0.85,
            reason="Data analysis query"
        )
        
        assert result.is_high_confidence
        assert result.should_use_rag
        assert result.show_visualizations
    
    def test_low_confidence_classification(self):
        """Test low confidence defaults to FREEFORM."""
        from core.routing import ClassificationResult, QueryMode
        
        result = ClassificationResult(
            mode=QueryMode.DATA_QUERY,
            confidence=0.4,  # Below threshold
            reason="Uncertain"
        )
        
        assert not result.is_high_confidence


# ============================================================================
# HANDLER TESTS
# ============================================================================

class TestModeHandlers:
    """Tests for mode handlers."""
    
    def test_freeform_handler(self):
        """Test freeform query handler."""
        from core.routing.handlers import FreeformHandler
        
        handler = FreeformHandler()
        result = handler.handle(query="What is the meaning of life?", context="")
        
        assert "answer" in result
        assert result["query_mode"] == "FREEFORM_QUERY"
        assert result["show_visualizations"] is False
    
    def test_system_task_handler(self):
        """Test system task handler."""
        from core.routing.handlers import SystemTaskHandler
        
        handler = SystemTaskHandler()
        result = handler.handle(query="How do I use this system?", context="")
        
        assert "answer" in result
        assert result["query_mode"] == "SYSTEM_TASK"
        assert "Upload" in result["answer"] or "upload" in result["answer"]


# ============================================================================
# LLM CLIENT TESTS
# ============================================================================

class TestLLMClient:
    """Tests for LLM client."""
    
    def test_llm_config_defaults(self):
        """Test LLMConfig default values."""
        from core.llm import LLMConfig
        
        config = LLMConfig()
        
        assert config.model == "nvidia/nemotron-nano-12b-v2-vl:free"
        assert config.temperature == 0.1
        assert config.max_tokens == 2000
    
    def test_llm_response_error_property(self):
        """Test LLMResponse error detection."""
        from core.llm.client import LLMResponse
        
        success_response = LLMResponse(content="Hello", model="test")
        error_response = LLMResponse(content="", model="test", error="API error")
        
        assert not success_response.is_error
        assert error_response.is_error
    
    @patch('core.llm.client.requests.post')
    def test_llm_client_call(self, mock_post):
        """Test LLM client API call."""
        from core.llm import LLMClient, LLMConfig
        
        # Mock response
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Test response"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 100}
        }
        mock_post.return_value.raise_for_status = Mock()
        
        config = LLMConfig()
        client = LLMClient(config, api_key="test-key")
        
        response = client.call([{"role": "user", "content": "Hello"}])
        
        assert response.content == "Test response"
        assert not response.is_error


# ============================================================================
# ENGINE INTEGRATION TESTS
# ============================================================================

class TestRAGEngine:
    """Integration tests for RAG engine."""
    
    def test_rag_config_defaults(self):
        """Test RAGConfig default values."""
        from core.engine import RAGConfig
        
        config = RAGConfig()
        
        assert config.default_k == 10
        assert config.enable_hybrid_search is True
        assert config.confidence_threshold == 0.6
        assert config.enable_cache is True
    
    def test_rag_response_to_dict(self):
        """Test RAGResponse serialization."""
        from core.engine import RAGResponse
        
        response = RAGResponse(
            answer="Test answer",
            query_mode="DATA_QUERY",
            sources=[{"filename": "test.csv", "score": 0.9}],
            show_visualizations=True,
            specific_metrics=["oil"],
            target_columns=["PROD_OIL_VOL"]
        )
        
        result = response.to_dict()
        
        assert result["answer"] == "Test answer"
        assert result["query_mode"] == "DATA_QUERY"
        assert result["show_visualizations"] is True
        assert "oil" in result["specific_metrics"]
    
    def test_metric_detection(self):
        """Test metric detection in engine."""
        from core.engine import RAGEngine, RAGConfig
        from unittest.mock import MagicMock
        
        # Create minimal engine with mocks
        config = RAGConfig()
        engine = RAGEngine(
            config=config,
            classifier=MagicMock(),
            llm_client=MagicMock(),
            vector_searcher=MagicMock()
        )
        
        metrics = engine._detect_metrics("What is oil production?")
        assert "oil" in metrics
        assert "production" in metrics
        
        metrics = engine._detect_metrics("gas sales volume")
        assert "gas" in metrics
        assert "sales" in metrics


# ============================================================================
# EMBEDDER TESTS
# ============================================================================

class TestEmbedder:
    """Tests for embedding module."""
    
    def test_get_model_info(self):
        """Test embedding model info retrieval."""
        from core.embedder import get_model_info
        
        info = get_model_info()
        
        assert "model_name" in info
        assert "dimensions" in info
        assert "is_loaded" in info
    
    def test_embed_query(self):
        """Test query embedding."""
        from core.embedder import embed_query, get_embedding_dimensions
        
        embedding = embed_query("test query")
        dims = get_embedding_dimensions()
        
        assert isinstance(embedding, list)
        assert len(embedding) == dims
    
    def test_embed_chunks(self):
        """Test chunk embedding."""
        from core.embedder import embed_chunks
        
        chunks = [
            {"text": "First chunk"},
            {"text": "Second chunk"}
        ]
        
        embeddings = embed_chunks(chunks)
        
        assert len(embeddings) == 2
        assert all(isinstance(e, list) for e in embeddings)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
