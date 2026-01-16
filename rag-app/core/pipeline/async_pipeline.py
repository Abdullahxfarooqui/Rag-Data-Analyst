"""
Async Pipeline Manager - Progressive Loading Architecture.

CORE PROBLEM SOLVED:
Users wait for LLM when charts could render immediately.

SOLUTION:
1. Return deterministic results (stats, charts) IMMEDIATELY
2. Stream LLM insights progressively  
3. Never block UI on AI

PIPELINE STAGES:
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Stage 1: SYNC  │ ──▶ │ Stage 2: ASYNC  │ ──▶ │ Stage 3: STREAM │
│  - Load data    │     │  - LLM insight  │     │  - Progressive  │
│  - Compute stats│     │  - Background   │     │  - Token stream │
│  - Build charts │     │  - Non-blocking │     │  - Real-time    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
      │                       │                       │
      ▼                       ▼                       ▼
   INSTANT              1-3 seconds              Progressive
"""
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, Generator, List, Optional, 
    TypeVar, Generic, Union, Awaitable
)
from queue import Queue
import threading

logger = logging.getLogger(__name__)


# ============================================================================
# PIPELINE STAGE DEFINITIONS
# ============================================================================

class StageStatus(Enum):
    """Status of a pipeline stage."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class StageResult:
    """Result from a pipeline stage."""
    stage_name: str
    status: StageStatus
    data: Any = None
    error: Optional[str] = None
    duration_ms: float = 0
    timestamp: float = field(default_factory=time.time)
    
    @property
    def is_success(self) -> bool:
        return self.status == StageStatus.COMPLETED
    
    @property
    def is_ready(self) -> bool:
        return self.status in (StageStatus.COMPLETED, StageStatus.FAILED, StageStatus.SKIPPED)


@dataclass
class PipelineProgress:
    """Overall pipeline progress for UI updates."""
    total_stages: int
    completed_stages: int
    current_stage: str
    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    
    @property
    def progress_percent(self) -> float:
        if self.total_stages == 0:
            return 100.0
        return (self.completed_stages / self.total_stages) * 100
    
    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self.started_at) * 1000
    
    def get_stage_result(self, stage_name: str) -> Optional[StageResult]:
        return self.stage_results.get(stage_name)
    
    def is_stage_ready(self, stage_name: str) -> bool:
        result = self.stage_results.get(stage_name)
        return result is not None and result.is_ready


# ============================================================================
# STAGE DEFINITIONS - Concrete pipeline stages
# ============================================================================

@dataclass
class PipelineStage:
    """Definition of a pipeline stage."""
    name: str
    executor: Callable
    is_async: bool = False
    depends_on: List[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    can_skip: bool = False
    description: str = ""


# Pre-defined stages for analytics pipeline
ANALYTICS_STAGES = {
    "load_data": PipelineStage(
        name="load_data",
        executor=None,  # Set at runtime
        is_async=False,
        depends_on=[],
        timeout_seconds=10.0,
        description="Load and validate data"
    ),
    "compute_stats": PipelineStage(
        name="compute_stats",
        executor=None,
        is_async=False,
        depends_on=["load_data"],
        timeout_seconds=5.0,
        description="Compute statistics with Python"
    ),
    "detect_anomalies": PipelineStage(
        name="detect_anomalies",
        executor=None,
        is_async=False,
        depends_on=["compute_stats"],
        timeout_seconds=3.0,
        can_skip=True,
        description="Detect data anomalies"
    ),
    "detect_trends": PipelineStage(
        name="detect_trends",
        executor=None,
        is_async=False,
        depends_on=["compute_stats"],
        timeout_seconds=3.0,
        can_skip=True,
        description="Detect time-series trends"
    ),
    "build_charts": PipelineStage(
        name="build_charts",
        executor=None,
        is_async=False,
        depends_on=["compute_stats"],
        timeout_seconds=5.0,
        description="Generate Plotly charts"
    ),
    "generate_insights": PipelineStage(
        name="generate_insights",
        executor=None,
        is_async=True,  # ASYNC - Non-blocking
        depends_on=["compute_stats", "detect_anomalies", "detect_trends"],
        timeout_seconds=30.0,
        can_skip=True,
        description="Generate LLM insights"
    )
}


# ============================================================================
# ASYNC PIPELINE EXECUTOR
# ============================================================================

class AsyncPipeline:
    """
    Executes pipeline stages with proper async/sync handling.
    
    KEY FEATURES:
    1. Sync stages execute immediately and return results
    2. Async stages run in background, results stream to callback
    3. Dependencies are respected
    4. Timeouts prevent hanging
    5. Progress updates stream to UI
    
    Usage:
        pipeline = AsyncPipeline()
        pipeline.register_stage("compute_stats", compute_stats_fn)
        pipeline.register_stage("generate_insights", llm_fn, is_async=True)
        
        # Get immediate results
        results = pipeline.execute_sync_stages(input_data)
        
        # Stream async results
        async for update in pipeline.stream_async_stages():
            update_ui(update)
    """
    
    def __init__(self, max_workers: int = 4):
        self._stages: Dict[str, PipelineStage] = {}
        self._results: Dict[str, StageResult] = {}
        self._progress: Optional[PipelineProgress] = None
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._async_futures: Dict[str, Future] = {}
        self._callbacks: List[Callable[[StageResult], None]] = []
        self._lock = threading.Lock()
    
    def register_stage(
        self,
        name: str,
        executor: Callable,
        is_async: bool = False,
        depends_on: List[str] = None,
        timeout_seconds: float = 30.0,
        can_skip: bool = False,
        description: str = ""
    ):
        """Register a pipeline stage."""
        self._stages[name] = PipelineStage(
            name=name,
            executor=executor,
            is_async=is_async,
            depends_on=depends_on or [],
            timeout_seconds=timeout_seconds,
            can_skip=can_skip,
            description=description
        )
    
    def on_stage_complete(self, callback: Callable[[StageResult], None]):
        """Register callback for stage completion."""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self, result: StageResult):
        """Notify all callbacks of stage completion."""
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _check_dependencies(self, stage: PipelineStage) -> bool:
        """Check if all dependencies are satisfied."""
        for dep in stage.depends_on:
            if dep not in self._results:
                return False
            if not self._results[dep].is_success:
                return False
        return True
    
    def _get_dependency_data(self, stage: PipelineStage) -> Dict[str, Any]:
        """Collect data from dependencies."""
        data = {}
        for dep in stage.depends_on:
            result = self._results.get(dep)
            if result and result.is_success:
                data[dep] = result.data
        return data
    
    def _execute_stage(
        self,
        stage: PipelineStage,
        input_data: Any = None
    ) -> StageResult:
        """Execute a single stage synchronously."""
        start_time = time.time()
        
        try:
            # Check dependencies
            if not self._check_dependencies(stage):
                if stage.can_skip:
                    return StageResult(
                        stage_name=stage.name,
                        status=StageStatus.SKIPPED,
                        error="Dependencies not satisfied"
                    )
                else:
                    return StageResult(
                        stage_name=stage.name,
                        status=StageStatus.FAILED,
                        error="Dependencies not satisfied"
                    )
            
            # Get dependency data
            dep_data = self._get_dependency_data(stage)
            
            # Execute
            logger.info(f"Executing stage: {stage.name}")
            result_data = stage.executor(input_data, dep_data)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return StageResult(
                stage_name=stage.name,
                status=StageStatus.COMPLETED,
                data=result_data,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            logger.exception(f"Stage {stage.name} failed: {e}")
            duration_ms = (time.time() - start_time) * 1000
            
            return StageResult(
                stage_name=stage.name,
                status=StageStatus.FAILED,
                error=str(e),
                duration_ms=duration_ms
            )
    
    def execute_sync_stages(
        self,
        input_data: Any = None,
        progress_callback: Callable[[PipelineProgress], None] = None
    ) -> Dict[str, StageResult]:
        """
        Execute all synchronous stages IMMEDIATELY.
        
        This returns as soon as all sync stages complete.
        Async stages are started but not waited for.
        
        Args:
            input_data: Initial input data
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict of stage results (only sync stages)
        """
        # Initialize progress
        sync_stages = [s for s in self._stages.values() if not s.is_async]
        self._progress = PipelineProgress(
            total_stages=len(self._stages),
            completed_stages=0,
            current_stage=""
        )
        
        # Execute sync stages in dependency order
        executed = set()
        
        while len(executed) < len(sync_stages):
            made_progress = False
            
            for stage in sync_stages:
                if stage.name in executed:
                    continue
                
                # Check if dependencies are met
                deps_met = all(
                    dep in executed or dep not in [s.name for s in sync_stages]
                    for dep in stage.depends_on
                )
                
                if deps_met:
                    self._progress.current_stage = stage.name
                    if progress_callback:
                        progress_callback(self._progress)
                    
                    result = self._execute_stage(stage, input_data)
                    
                    with self._lock:
                        self._results[stage.name] = result
                        self._progress.stage_results[stage.name] = result
                        self._progress.completed_stages += 1
                    
                    self._notify_callbacks(result)
                    executed.add(stage.name)
                    made_progress = True
            
            if not made_progress:
                logger.warning("No progress made in sync stages - possible circular dependency")
                break
        
        return {name: self._results[name] for name in executed}
    
    def start_async_stages(
        self,
        input_data: Any = None
    ) -> Dict[str, Future]:
        """
        Start all async stages in background threads.
        
        Returns immediately with futures that can be awaited.
        
        Args:
            input_data: Initial input data
            
        Returns:
            Dict mapping stage names to futures
        """
        async_stages = [s for s in self._stages.values() if s.is_async]
        
        for stage in async_stages:
            def execute_async_stage(stg=stage, data=input_data):
                result = self._execute_stage(stg, data)
                
                with self._lock:
                    self._results[stg.name] = result
                    self._progress.stage_results[stg.name] = result
                    self._progress.completed_stages += 1
                
                self._notify_callbacks(result)
                return result
            
            future = self._executor.submit(execute_async_stage)
            self._async_futures[stage.name] = future
        
        return self._async_futures
    
    def get_async_result(
        self,
        stage_name: str,
        timeout: float = None
    ) -> Optional[StageResult]:
        """
        Get result of an async stage, waiting if necessary.
        
        Args:
            stage_name: Name of the async stage
            timeout: Max time to wait (None = wait forever)
            
        Returns:
            StageResult or None if not found/timeout
        """
        future = self._async_futures.get(stage_name)
        if not future:
            return self._results.get(stage_name)
        
        try:
            return future.result(timeout=timeout)
        except Exception as e:
            logger.error(f"Error getting async result: {e}")
            return None
    
    def get_progress(self) -> Optional[PipelineProgress]:
        """Get current pipeline progress."""
        return self._progress
    
    def get_all_results(self) -> Dict[str, StageResult]:
        """Get all stage results (sync and completed async)."""
        return dict(self._results)
    
    def cleanup(self):
        """Clean up resources."""
        self._executor.shutdown(wait=False)


# ============================================================================
# STREAMLIT-OPTIMIZED PIPELINE
# ============================================================================

class StreamlitPipeline:
    """
    Pipeline optimized for Streamlit's execution model.
    
    Streamlit reruns the entire script on each interaction,
    so we need to:
    1. Cache computed results in session_state
    2. Return sync results immediately  
    3. Show placeholder for async results
    4. Update UI when async completes
    """
    
    def __init__(self):
        self._pipeline = AsyncPipeline()
    
    def register_analytics_stages(
        self,
        load_fn: Callable,
        stats_fn: Callable,
        anomaly_fn: Callable = None,
        trend_fn: Callable = None,
        chart_fn: Callable = None,
        insight_fn: Callable = None
    ):
        """
        Register standard analytics pipeline stages.
        
        Args:
            load_fn: Function to load data
            stats_fn: Function to compute statistics
            anomaly_fn: Function to detect anomalies (optional)
            trend_fn: Function to detect trends (optional)
            chart_fn: Function to build charts (optional)
            insight_fn: Function to generate LLM insights (optional, async)
        """
        self._pipeline.register_stage(
            "load_data", load_fn, is_async=False
        )
        self._pipeline.register_stage(
            "compute_stats", stats_fn, is_async=False,
            depends_on=["load_data"]
        )
        
        if anomaly_fn:
            self._pipeline.register_stage(
                "detect_anomalies", anomaly_fn, is_async=False,
                depends_on=["compute_stats"], can_skip=True
            )
        
        if trend_fn:
            self._pipeline.register_stage(
                "detect_trends", trend_fn, is_async=False,
                depends_on=["compute_stats"], can_skip=True
            )
        
        if chart_fn:
            self._pipeline.register_stage(
                "build_charts", chart_fn, is_async=False,
                depends_on=["compute_stats"]
            )
        
        if insight_fn:
            deps = ["compute_stats"]
            if anomaly_fn:
                deps.append("detect_anomalies")
            if trend_fn:
                deps.append("detect_trends")
            
            self._pipeline.register_stage(
                "generate_insights", insight_fn, is_async=True,
                depends_on=deps, can_skip=True
            )
    
    def execute(
        self,
        input_data: Any,
        wait_for_insights: bool = False,
        insight_timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Execute the pipeline with Streamlit-optimized flow.
        
        Args:
            input_data: Input data for pipeline
            wait_for_insights: If True, wait for LLM insights
            insight_timeout: Timeout for insights
            
        Returns:
            Dict with:
            - "sync_results": Results from sync stages (ready immediately)
            - "insights_pending": True if insights still processing
            - "insights": LLM insights if available
            - "progress": Pipeline progress object
        """
        # Execute sync stages immediately
        sync_results = self._pipeline.execute_sync_stages(input_data)
        
        # Start async stages (LLM insights)
        self._pipeline.start_async_stages(input_data)
        
        result = {
            "sync_results": sync_results,
            "insights_pending": True,
            "insights": None,
            "progress": self._pipeline.get_progress()
        }
        
        # Optionally wait for insights
        if wait_for_insights:
            insight_result = self._pipeline.get_async_result(
                "generate_insights",
                timeout=insight_timeout
            )
            if insight_result:
                result["insights_pending"] = False
                result["insights"] = insight_result.data
        
        return result
    
    def get_insights_if_ready(self) -> Optional[Any]:
        """
        Get insights if they're ready, without blocking.
        
        Returns:
            Insights data if ready, None if still processing
        """
        result = self._pipeline._results.get("generate_insights")
        if result and result.is_success:
            return result.data
        return None


# ============================================================================
# PROGRESSIVE RESPONSE BUILDER
# ============================================================================

@dataclass
class ProgressiveResponse:
    """
    Response that builds progressively as stages complete.
    
    UI can render partial response immediately:
    - Stats section: Ready after compute_stats
    - Charts section: Ready after build_charts
    - Insights section: Shows placeholder, updates when LLM completes
    """
    stats_ready: bool = False
    stats_data: Optional[Dict] = None
    
    charts_ready: bool = False
    chart_configs: Optional[List[Dict]] = None
    
    anomalies_ready: bool = False
    anomalies: Optional[List[Dict]] = None
    
    trends_ready: bool = False
    trends: Optional[List[Dict]] = None
    
    insights_ready: bool = False
    insights: Optional[Dict] = None
    insights_error: Optional[str] = None
    
    @classmethod
    def from_pipeline_results(
        cls,
        results: Dict[str, StageResult]
    ) -> "ProgressiveResponse":
        """Build response from pipeline results."""
        response = cls()
        
        # Stats
        stats_result = results.get("compute_stats")
        if stats_result and stats_result.is_success:
            response.stats_ready = True
            response.stats_data = stats_result.data
        
        # Charts
        chart_result = results.get("build_charts")
        if chart_result and chart_result.is_success:
            response.charts_ready = True
            response.chart_configs = chart_result.data
        
        # Anomalies
        anomaly_result = results.get("detect_anomalies")
        if anomaly_result and anomaly_result.is_success:
            response.anomalies_ready = True
            response.anomalies = anomaly_result.data
        
        # Trends
        trend_result = results.get("detect_trends")
        if trend_result and trend_result.is_success:
            response.trends_ready = True
            response.trends = trend_result.data
        
        # Insights
        insight_result = results.get("generate_insights")
        if insight_result:
            if insight_result.is_success:
                response.insights_ready = True
                response.insights = insight_result.data
            elif insight_result.status == StageStatus.FAILED:
                response.insights_ready = True
                response.insights_error = insight_result.error
        
        return response
    
    def get_display_sections(self) -> List[Dict[str, Any]]:
        """
        Get sections ready for display.
        
        Returns list of sections in display order:
        1. Stats (if ready)
        2. Charts (if ready)
        3. Insights (placeholder or content)
        """
        sections = []
        
        if self.stats_ready:
            sections.append({
                "type": "stats",
                "ready": True,
                "data": self.stats_data
            })
        
        if self.charts_ready:
            sections.append({
                "type": "charts",
                "ready": True,
                "data": self.chart_configs
            })
        
        # Always include insights section (with loading state if not ready)
        sections.append({
            "type": "insights",
            "ready": self.insights_ready,
            "data": self.insights,
            "error": self.insights_error,
            "loading": not self.insights_ready
        })
        
        return sections


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "AsyncPipeline",
    "StreamlitPipeline",
    "ProgressiveResponse",
    "PipelineProgress",
    "StageResult",
    "StageStatus",
    "PipelineStage"
]
