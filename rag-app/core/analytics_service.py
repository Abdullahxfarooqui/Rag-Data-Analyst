"""
Production Analytics Service - Unified Query Processing.

This is the main entry point for all analytics queries. It:
1. Classifies queries using SmartRouter
2. Routes to optimal processing path (Pandas vs FAISS)
3. Executes via AsyncPipeline for progressive loading
4. Returns structured results via LLM Orchestrator

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────────┐
│                        AnalyticsService                             │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────────┐  │
│  │ SmartRouter │─▶│ AsyncPipeline│─▶│ LLMOrchestrator            │  │
│  │ (classify)  │  │ (execute)    │  │ (generate structured out)  │  │
│  └─────────────┘  └──────────────┘  └────────────────────────────┘  │
│         │                │                       │                  │
│         ▼                ▼                       ▼                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────────┐  │
│  │ QueryCache  │  │ ProgressResp │  │ StructuredInsight          │  │
│  │ (memoize)   │  │ (partial UI) │  │ (typed output)             │  │
│  └─────────────┘  └──────────────┘  └────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

USAGE:
    service = AnalyticsService(df=my_dataframe)
    
    # Execute query with progressive loading
    response = service.query("What are the top 10 products by revenue?")
    
    # Response is immediately available for stats/charts
    # LLM insights stream in progressively
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd

# Local imports
from core.routing.smart_router import (
    SmartRouter,
    QueryClassification,
    ProcessingPath,
    QueryIntent,
)
from core.pipeline.async_pipeline import (
    AsyncPipeline,
    ProgressiveResponse,
    StageResult,
    StageStatus,
)
from core.llm.orchestrator import (
    LLMOrchestrator,
    StructuredInsight,
    ToolRouter,
)

logger = logging.getLogger(__name__)


# ============================================================================
# RESULT TYPES
# ============================================================================

@dataclass
class AnalyticsResult:
    """
    Complete result from analytics query.
    
    Contains all components needed for UI rendering:
    - Raw data (for tables)
    - Statistics (for metrics display)
    - Charts (Plotly configs)
    - Insights (LLM-generated narrative)
    - Metadata (timing, confidence, etc.)
    """
    query: str
    success: bool
    
    # Data components (immediately available)
    raw_data: Optional[pd.DataFrame] = None
    statistics: Optional[Dict[str, Any]] = None
    charts: Optional[List[Dict]] = None
    
    # Insight components (may be pending)
    insights: Optional[StructuredInsight] = None
    insights_pending: bool = False
    insights_error: Optional[str] = None
    
    # Anomalies and trends (Python-computed)
    anomalies: Optional[List[Dict]] = None
    trends: Optional[List[Dict]] = None
    
    # Metadata
    classification: Optional[QueryClassification] = None
    processing_path: Optional[ProcessingPath] = None
    cached: bool = False
    timing_ms: float = 0
    stage_timings: Dict[str, float] = field(default_factory=dict)
    
    @property
    def has_data(self) -> bool:
        return self.raw_data is not None or self.statistics is not None
    
    @property
    def has_insights(self) -> bool:
        return self.insights is not None and not self.insights_pending
    
    @property
    def is_complete(self) -> bool:
        return self.has_data and (self.has_insights or not self.insights_pending)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "query": self.query,
            "success": self.success,
            "has_data": self.has_data,
            "has_insights": self.has_insights,
            "is_complete": self.is_complete,
            "cached": self.cached,
            "timing_ms": self.timing_ms,
        }
        
        if self.statistics:
            result["statistics"] = self.statistics
        
        if self.insights:
            result["insights"] = {
                "summary": self.insights.summary,
                "key_metrics": [m.to_dict() for m in self.insights.key_metrics],
                "comparisons": [c.to_dict() for c in self.insights.comparisons],
                "trends": [t.to_dict() for t in self.insights.trends],
                "risks": [r.to_dict() for r in self.insights.risks],
                "recommendations": self.insights.recommendations,
                "confidence_level": self.insights.confidence_level,
            }
        
        if self.anomalies:
            result["anomalies"] = self.anomalies
        
        if self.trends:
            result["trends"] = self.trends
        
        if self.classification:
            result["classification"] = {
                "intent": self.classification.intent.name,
                "path": self.classification.path.name,
                "confidence": self.classification.confidence,
                "bypass_faiss": self.classification.bypass_faiss,
            }
        
        return result


# ============================================================================
# PANDAS COMPUTATION LAYER
# ============================================================================

class PandasCompute:
    """
    Pure Python/Pandas computation layer.
    
    Handles all deterministic computations without LLM:
    - Aggregations (sum, count, avg, etc.)
    - Filtering
    - Sorting/ranking
    - Statistical analysis
    - Anomaly detection
    - Trend detection
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        self._categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self._datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive statistics."""
        stats = {
            "row_count": len(self.df),
            "column_count": len(self.df.columns),
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "numeric_summary": {},
            "categorical_summary": {},
            "missing_values": {},
        }
        
        # Numeric summaries
        for col in self._numeric_columns:
            stats["numeric_summary"][col] = {
                "mean": float(self.df[col].mean()) if pd.notna(self.df[col].mean()) else None,
                "median": float(self.df[col].median()) if pd.notna(self.df[col].median()) else None,
                "std": float(self.df[col].std()) if pd.notna(self.df[col].std()) else None,
                "min": float(self.df[col].min()) if pd.notna(self.df[col].min()) else None,
                "max": float(self.df[col].max()) if pd.notna(self.df[col].max()) else None,
                "sum": float(self.df[col].sum()) if pd.notna(self.df[col].sum()) else None,
            }
        
        # Categorical summaries
        for col in self._categorical_columns[:10]:  # Limit to first 10
            value_counts = self.df[col].value_counts().head(10)
            stats["categorical_summary"][col] = {
                "unique_count": int(self.df[col].nunique()),
                "top_values": value_counts.to_dict(),
            }
        
        # Missing values
        for col in self.df.columns:
            missing = int(self.df[col].isna().sum())
            if missing > 0:
                stats["missing_values"][col] = {
                    "count": missing,
                    "percentage": round(missing / len(self.df) * 100, 2),
                }
        
        return stats
    
    def compute_top_n(
        self,
        column: str,
        n: int = 10,
        ascending: bool = False
    ) -> pd.DataFrame:
        """Get top/bottom N rows by column."""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        return self.df.nlargest(n, column) if not ascending else self.df.nsmallest(n, column)
    
    def compute_aggregation(
        self,
        group_by: str,
        agg_column: str,
        agg_func: str = "sum"
    ) -> pd.DataFrame:
        """Compute grouped aggregation."""
        if group_by not in self.df.columns:
            raise ValueError(f"Group column '{group_by}' not found")
        if agg_column not in self.df.columns:
            raise ValueError(f"Aggregation column '{agg_column}' not found")
        
        return self.df.groupby(group_by)[agg_column].agg(agg_func).reset_index()
    
    def detect_anomalies(self, threshold: float = 2.5) -> List[Dict]:
        """
        Detect statistical anomalies using Z-score method.
        
        Returns list of anomalies with column, value, and z-score.
        """
        anomalies = []
        
        for col in self._numeric_columns:
            if self.df[col].std() == 0:
                continue
            
            z_scores = (self.df[col] - self.df[col].mean()) / self.df[col].std()
            outlier_mask = z_scores.abs() > threshold
            
            for idx in self.df[outlier_mask].index:
                anomalies.append({
                    "column": col,
                    "row_index": int(idx),
                    "value": float(self.df.loc[idx, col]),
                    "z_score": float(z_scores[idx]),
                    "severity": "high" if abs(z_scores[idx]) > 3 else "medium",
                })
        
        return anomalies[:50]  # Limit to 50
    
    def detect_trends(self, date_column: str = None) -> List[Dict]:
        """
        Detect trends in time-series data.
        
        If no date column specified, tries to find one automatically.
        """
        trends = []
        
        # Auto-detect date column
        if date_column is None:
            if self._datetime_columns:
                date_column = self._datetime_columns[0]
            else:
                # Try to parse columns that look like dates
                for col in self._categorical_columns:
                    try:
                        pd.to_datetime(self.df[col].head(100))
                        date_column = col
                        break
                    except:
                        continue
        
        if date_column is None:
            return trends
        
        # Compute trends for numeric columns
        df_sorted = self.df.sort_values(date_column)
        
        for col in self._numeric_columns[:5]:  # Top 5 numeric columns
            values = df_sorted[col].dropna()
            if len(values) < 3:
                continue
            
            # Simple trend detection: compare first third to last third
            third = len(values) // 3
            if third == 0:
                continue
            
            first_avg = values.iloc[:third].mean()
            last_avg = values.iloc[-third:].mean()
            
            if first_avg == 0:
                continue
            
            change_pct = ((last_avg - first_avg) / first_avg) * 100
            
            trends.append({
                "column": col,
                "direction": "increasing" if change_pct > 5 else "decreasing" if change_pct < -5 else "stable",
                "change_percent": round(change_pct, 2),
                "first_period_avg": round(first_avg, 2),
                "last_period_avg": round(last_avg, 2),
            })
        
        return trends


# ============================================================================
# CHART BUILDER
# ============================================================================

class ChartBuilder:
    """Builds Plotly chart configurations."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        self._categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def build_summary_charts(self) -> List[Dict]:
        """Build automatic summary charts based on data."""
        charts = []
        
        # Top categories bar chart
        if self._categorical_columns and self._numeric_columns:
            cat_col = self._categorical_columns[0]
            num_col = self._numeric_columns[0]
            
            agg_data = self.df.groupby(cat_col)[num_col].sum().nlargest(10)
            
            charts.append({
                "type": "bar",
                "title": f"Top 10 {cat_col} by {num_col}",
                "data": {
                    "x": agg_data.index.tolist(),
                    "y": agg_data.values.tolist(),
                },
                "layout": {
                    "xaxis": {"title": cat_col},
                    "yaxis": {"title": num_col},
                }
            })
        
        # Numeric distribution histogram
        if self._numeric_columns:
            num_col = self._numeric_columns[0]
            charts.append({
                "type": "histogram",
                "title": f"Distribution of {num_col}",
                "data": {
                    "x": self.df[num_col].dropna().tolist(),
                },
                "layout": {
                    "xaxis": {"title": num_col},
                    "yaxis": {"title": "Count"},
                }
            })
        
        return charts
    
    def build_comparison_chart(
        self,
        x_column: str,
        y_columns: List[str]
    ) -> Dict:
        """Build comparison bar chart."""
        return {
            "type": "bar",
            "title": f"Comparison by {x_column}",
            "data": [
                {
                    "x": self.df[x_column].tolist(),
                    "y": self.df[col].tolist(),
                    "name": col,
                }
                for col in y_columns if col in self.df.columns
            ],
            "layout": {
                "barmode": "group",
                "xaxis": {"title": x_column},
            }
        }
    
    def build_trend_chart(
        self,
        date_column: str,
        value_column: str
    ) -> Dict:
        """Build time-series trend chart."""
        df_sorted = self.df.sort_values(date_column)
        
        return {
            "type": "scatter",
            "title": f"{value_column} Over Time",
            "data": {
                "x": df_sorted[date_column].astype(str).tolist(),
                "y": df_sorted[value_column].tolist(),
                "mode": "lines+markers",
            },
            "layout": {
                "xaxis": {"title": date_column},
                "yaxis": {"title": value_column},
            }
        }


# ============================================================================
# ANALYTICS SERVICE
# ============================================================================

class AnalyticsService:
    """
    Main analytics service - unified entry point for all queries.
    
    Features:
    - Smart query routing (FAISS bypass when possible)
    - Async pipeline for progressive loading
    - Structured LLM output via orchestrator
    - Query caching for repeated queries
    - Comprehensive statistics and charts
    """
    
    def __init__(
        self,
        df: pd.DataFrame = None,
        llm_client: Any = None,
        vector_store: Any = None,
        cache_enabled: bool = True,
    ):
        """
        Initialize analytics service.
        
        Args:
            df: DataFrame for analytics
            llm_client: LLM client for insights
            vector_store: Vector store for semantic search
            cache_enabled: Enable query caching
        """
        self._df = df
        self._llm_client = llm_client
        self._vector_store = vector_store
        
        # Initialize components
        self._router = SmartRouter(
            column_names=df.columns.tolist() if df is not None else [],
            cache_enabled=cache_enabled
        )
        
        self._orchestrator = LLMOrchestrator(llm_client) if llm_client else None
        
        if df is not None:
            self._pandas = PandasCompute(df)
            self._charts = ChartBuilder(df)
        else:
            self._pandas = None
            self._charts = None
        
        logger.info("AnalyticsService initialized")
    
    def set_dataframe(self, df: pd.DataFrame):
        """Set or update the DataFrame."""
        self._df = df
        self._pandas = PandasCompute(df)
        self._charts = ChartBuilder(df)
        self._router.update_schema(df.columns.tolist())
        self._router.on_data_change()
        logger.info(f"DataFrame updated: {len(df)} rows, {len(df.columns)} columns")
    
    def query(
        self,
        query: str,
        wait_for_insights: bool = False,
        insight_timeout: float = 30.0,
    ) -> AnalyticsResult:
        """
        Execute an analytics query.
        
        Args:
            query: User's query string
            wait_for_insights: If True, wait for LLM insights
            insight_timeout: Timeout for LLM insights
            
        Returns:
            AnalyticsResult with data, charts, and insights
        """
        start_time = time.time()
        
        # Route query
        routing = self._router.route(query)
        
        # Check cache
        if routing.use_cache:
            logger.info(f"Cache hit for query: {query[:50]}...")
            cached = routing.cached_result
            cached.cached = True
            cached.timing_ms = (time.time() - start_time) * 1000
            return cached
        
        # Initialize result
        result = AnalyticsResult(
            query=query,
            success=True,
            classification=routing.classification,
            processing_path=routing.path,
        )
        
        try:
            # Execute based on processing path
            if routing.path == ProcessingPath.PANDAS_ONLY:
                self._execute_pandas_only(result)
            elif routing.path == ProcessingPath.PANDAS_WITH_LLM:
                self._execute_pandas_with_llm(result, wait_for_insights, insight_timeout)
            elif routing.path == ProcessingPath.FAISS_WITH_LLM:
                self._execute_faiss_with_llm(result, wait_for_insights, insight_timeout)
            else:
                self._execute_llm_only(result)
            
        except Exception as e:
            logger.exception(f"Query execution failed: {e}")
            result.success = False
            result.insights_error = str(e)
        
        # Record timing
        result.timing_ms = (time.time() - start_time) * 1000
        
        # Cache result (if successful and not LLM-only)
        if result.success and routing.path != ProcessingPath.LLM_ONLY:
            self._router.cache_result(routing.classification, result)
        
        return result
    
    def _execute_pandas_only(self, result: AnalyticsResult):
        """Execute pure Pandas query (no LLM)."""
        if not self._pandas:
            raise ValueError("No DataFrame loaded")
        
        stage_start = time.time()
        
        # Compute statistics
        result.statistics = self._pandas.compute_statistics()
        result.stage_timings["compute_stats"] = (time.time() - stage_start) * 1000
        
        # Detect anomalies
        stage_start = time.time()
        result.anomalies = self._pandas.detect_anomalies()
        result.stage_timings["detect_anomalies"] = (time.time() - stage_start) * 1000
        
        # Detect trends
        stage_start = time.time()
        result.trends = self._pandas.detect_trends()
        result.stage_timings["detect_trends"] = (time.time() - stage_start) * 1000
        
        # Build charts
        stage_start = time.time()
        result.charts = self._charts.build_summary_charts()
        result.stage_timings["build_charts"] = (time.time() - stage_start) * 1000
        
        # No LLM insights for pure Pandas
        result.insights_pending = False
        
        logger.info(f"Pandas-only execution complete: {sum(result.stage_timings.values()):.0f}ms")
    
    def _execute_pandas_with_llm(
        self,
        result: AnalyticsResult,
        wait_for_insights: bool,
        insight_timeout: float
    ):
        """Execute Pandas computation + LLM insights."""
        # First, execute Pandas stages (synchronously)
        self._execute_pandas_only(result)
        
        # Then, generate LLM insights (optionally async)
        if self._orchestrator:
            result.insights_pending = True
            
            # Prepare context for LLM
            context = {
                "statistics": result.statistics,
                "anomalies": result.anomalies,
                "trends": result.trends,
                "query": result.query,
            }
            
            if wait_for_insights:
                stage_start = time.time()
                try:
                    insight = self._orchestrator.execute_tool(
                        "generate_insights",
                        query=result.query,
                        context=context
                    )
                    result.insights = insight
                    result.insights_pending = False
                except Exception as e:
                    logger.error(f"LLM insight generation failed: {e}")
                    result.insights_error = str(e)
                    result.insights_pending = False
                
                result.stage_timings["generate_insights"] = (time.time() - stage_start) * 1000
        else:
            result.insights_pending = False
    
    def _execute_faiss_with_llm(
        self,
        result: AnalyticsResult,
        wait_for_insights: bool,
        insight_timeout: float
    ):
        """Execute full RAG pipeline with FAISS."""
        # Execute Pandas stages first
        if self._pandas:
            self._execute_pandas_only(result)
        
        # Vector search (if available)
        if self._vector_store:
            stage_start = time.time()
            # TODO: Integrate actual vector search
            # search_results = self._vector_store.search(result.query)
            result.stage_timings["vector_search"] = (time.time() - stage_start) * 1000
        
        # Generate insights with RAG context
        if self._orchestrator:
            result.insights_pending = True
            
            if wait_for_insights:
                stage_start = time.time()
                try:
                    insight = self._orchestrator.execute_tool(
                        "answer_question",
                        query=result.query,
                        context={
                            "statistics": result.statistics,
                            "anomalies": result.anomalies,
                            # "rag_context": search_results  # TODO
                        }
                    )
                    result.insights = insight
                    result.insights_pending = False
                except Exception as e:
                    logger.error(f"LLM insight generation failed: {e}")
                    result.insights_error = str(e)
                    result.insights_pending = False
                
                result.stage_timings["generate_insights"] = (time.time() - stage_start) * 1000
        else:
            result.insights_pending = False
    
    def _execute_llm_only(self, result: AnalyticsResult):
        """Execute LLM-only query (no data processing)."""
        if not self._orchestrator:
            raise ValueError("No LLM client configured")
        
        stage_start = time.time()
        try:
            insight = self._orchestrator.execute_tool(
                "answer_question",
                query=result.query,
                context={}
            )
            result.insights = insight
            result.insights_pending = False
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            result.insights_error = str(e)
        
        result.stage_timings["llm_query"] = (time.time() - stage_start) * 1000
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current DataFrame statistics."""
        if not self._pandas:
            return {}
        return self._pandas.compute_statistics()
    
    def get_charts(self) -> List[Dict]:
        """Get auto-generated charts."""
        if not self._charts:
            return []
        return self._charts.build_summary_charts()
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return self._router.get_routing_stats()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "AnalyticsService",
    "AnalyticsResult",
    "PandasCompute",
    "ChartBuilder",
]
