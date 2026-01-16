"""
Production Observability & Monitoring.

COMPREHENSIVE SYSTEM OBSERVABILITY.

This module provides:
1. Structured logging with context
2. Metrics collection (latency, throughput, errors)
3. Token usage tracking for LLM calls
4. Memory monitoring
5. Request tracing
6. Alerting thresholds

METRICS TRACKED:
- LLM calls: count, latency, tokens, errors
- FAISS queries: count, latency, hit rate
- Document ingestion: count, size, time
- Task queue: depth, wait time, completion rate
- Memory: heap usage, FAISS index size
- Cache: hit rate, size, evictions

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────────────┐
│                         Observability Layer                              │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Logger      │  │ Metrics     │  │ Tracer      │  │ Alerter     │    │
│  │ (structured)│  │ (counters)  │  │ (spans)     │  │ (thresholds)│    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
│         │                │                │                │            │
│         └────────────────┼────────────────┼────────────────┘            │
│                          ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    MetricsStore                                  │   │
│  │  - In-memory counters/histograms                                 │   │
│  │  - Rolling windows for rates                                     │   │
│  │  - Export to Prometheus/JSON                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

TRADE-OFFS:
- Observability overhead: ~1-5% CPU, ~10-50MB RAM
- Log verbosity: More logs = better debugging but more storage
- Metric granularity: Fine-grained = more insights but more memory
"""
import json
import logging
import os
import psutil
import sys
import threading
import time
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import uuid

logger = logging.getLogger(__name__)


# ============================================================================
# STRUCTURED LOGGING
# ============================================================================

class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogContext:
    """Context for structured logging."""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    operation: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class StructuredLogger:
    """
    Structured logger with context propagation.
    
    Outputs JSON-formatted logs for easy parsing
    by log aggregation systems.
    """
    
    _context = threading.local()
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Add JSON formatter if no handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JsonFormatter())
            self.logger.addHandler(handler)
    
    @classmethod
    def set_context(cls, **kwargs):
        """Set thread-local context."""
        if not hasattr(cls._context, 'data'):
            cls._context.data = {}
        cls._context.data.update(kwargs)
    
    @classmethod
    def clear_context(cls):
        """Clear thread-local context."""
        cls._context.data = {}
    
    @classmethod
    def get_context(cls) -> Dict:
        """Get current context."""
        return getattr(cls._context, 'data', {})
    
    def _log(self, level: int, message: str, **kwargs):
        """Log with context."""
        context = self.get_context()
        extra = {
            'structured': True,
            'context': context,
            'extra': kwargs,
            'timestamp': datetime.utcnow().isoformat(),
        }
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        if exc_info:
            kwargs['traceback'] = traceback.format_exc()
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs):
        if exc_info:
            kwargs['traceback'] = traceback.format_exc()
        self._log(logging.CRITICAL, message, **kwargs)


class JsonFormatter(logging.Formatter):
    """JSON log formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        if hasattr(record, 'structured') and record.structured:
            log_obj['context'] = getattr(record, 'context', {})
            log_obj['extra'] = getattr(record, 'extra', {})
        
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_obj, default=str)


# ============================================================================
# METRICS COLLECTION
# ============================================================================

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"         # Monotonically increasing
    GAUGE = "gauge"             # Current value
    HISTOGRAM = "histogram"     # Distribution
    RATE = "rate"               # Events per second


@dataclass
class MetricValue:
    """Single metric value with metadata."""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class Counter:
    """Thread-safe counter metric."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0
        self._lock = threading.Lock()
    
    def inc(self, amount: float = 1):
        with self._lock:
            self._value += amount
    
    @property
    def value(self) -> float:
        return self._value


class Gauge:
    """Thread-safe gauge metric."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0
        self._lock = threading.Lock()
    
    def set(self, value: float):
        with self._lock:
            self._value = value
    
    def inc(self, amount: float = 1):
        with self._lock:
            self._value += amount
    
    def dec(self, amount: float = 1):
        with self._lock:
            self._value -= amount
    
    @property
    def value(self) -> float:
        return self._value


class Histogram:
    """Thread-safe histogram for latency tracking."""
    
    DEFAULT_BUCKETS = [
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ]
    
    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: List[float] = None
    ):
        self.name = name
        self.description = description
        self.buckets = buckets or self.DEFAULT_BUCKETS
        
        self._sum = 0.0
        self._count = 0
        self._bucket_counts = {b: 0 for b in self.buckets}
        self._bucket_counts[float('inf')] = 0
        self._lock = threading.Lock()
    
    def observe(self, value: float):
        with self._lock:
            self._sum += value
            self._count += 1
            
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[bucket] += 1
            self._bucket_counts[float('inf')] += 1
    
    @property
    def count(self) -> int:
        return self._count
    
    @property
    def sum(self) -> float:
        return self._sum
    
    @property
    def mean(self) -> float:
        if self._count == 0:
            return 0
        return self._sum / self._count
    
    def percentile(self, p: float) -> float:
        """Estimate percentile from histogram buckets."""
        if self._count == 0:
            return 0
        
        target = self._count * p
        cumulative = 0
        
        for bucket in sorted(self.buckets):
            cumulative += self._bucket_counts[bucket]
            if cumulative >= target:
                return bucket
        
        return self.buckets[-1]


class RollingRate:
    """
    Calculates rate over a rolling window.
    
    Useful for tracking events per second/minute.
    """
    
    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self._events: deque = deque()
        self._lock = threading.Lock()
    
    def add(self, count: int = 1):
        """Add events."""
        now = time.time()
        with self._lock:
            self._events.append((now, count))
            self._prune(now)
    
    def _prune(self, now: float):
        """Remove old events."""
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
    
    @property
    def rate(self) -> float:
        """Get events per second."""
        now = time.time()
        with self._lock:
            self._prune(now)
            if not self._events:
                return 0.0
            
            total = sum(count for _, count in self._events)
            return total / self.window_seconds


# ============================================================================
# METRICS REGISTRY
# ============================================================================

class MetricsRegistry:
    """
    Central registry for all metrics.
    
    Provides a single point for metric collection and export.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._initialized = True
        
        # Initialize standard metrics
        self._init_standard_metrics()
    
    def _init_standard_metrics(self):
        """Initialize standard system metrics."""
        # LLM metrics
        self.counter("llm_calls_total", "Total LLM API calls")
        self.counter("llm_tokens_total", "Total tokens used")
        self.counter("llm_errors_total", "Total LLM errors")
        self.histogram("llm_latency_seconds", "LLM call latency")
        
        # FAISS metrics
        self.counter("faiss_queries_total", "Total FAISS queries")
        self.histogram("faiss_latency_seconds", "FAISS query latency")
        self.gauge("faiss_index_size", "Number of vectors in FAISS")
        
        # Ingestion metrics
        self.counter("documents_ingested_total", "Total documents ingested")
        self.counter("chunks_created_total", "Total chunks created")
        self.histogram("ingestion_time_seconds", "Document ingestion time")
        
        # Cache metrics
        self.counter("cache_hits_total", "Cache hits")
        self.counter("cache_misses_total", "Cache misses")
        self.gauge("cache_size", "Current cache size")
        
        # Task queue metrics
        self.gauge("task_queue_depth", "Pending tasks in queue")
        self.counter("tasks_completed_total", "Completed tasks")
        self.counter("tasks_failed_total", "Failed tasks")
        self.histogram("task_wait_time_seconds", "Task wait time")
        self.histogram("task_execution_time_seconds", "Task execution time")
        
        # System metrics
        self.gauge("memory_usage_bytes", "Process memory usage")
        self.gauge("cpu_usage_percent", "Process CPU usage")
    
    def counter(self, name: str, description: str = "") -> Counter:
        """Get or create a counter."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Counter(name, description)
            return self._metrics[name]
    
    def gauge(self, name: str, description: str = "") -> Gauge:
        """Get or create a gauge."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Gauge(name, description)
            return self._metrics[name]
    
    def histogram(self, name: str, description: str = "", buckets: List[float] = None) -> Histogram:
        """Get or create a histogram."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Histogram(name, description, buckets)
            return self._metrics[name]
    
    def rate(self, name: str, window_seconds: float = 60.0) -> RollingRate:
        """Get or create a rate metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = RollingRate(window_seconds)
            return self._metrics[name]
    
    def export(self) -> Dict[str, Any]:
        """Export all metrics as dictionary."""
        result = {}
        
        with self._lock:
            for name, metric in self._metrics.items():
                if isinstance(metric, Counter):
                    result[name] = {
                        "type": "counter",
                        "value": metric.value
                    }
                elif isinstance(metric, Gauge):
                    result[name] = {
                        "type": "gauge",
                        "value": metric.value
                    }
                elif isinstance(metric, Histogram):
                    result[name] = {
                        "type": "histogram",
                        "count": metric.count,
                        "sum": metric.sum,
                        "mean": metric.mean,
                        "p50": metric.percentile(0.5),
                        "p95": metric.percentile(0.95),
                        "p99": metric.percentile(0.99),
                    }
                elif isinstance(metric, RollingRate):
                    result[name] = {
                        "type": "rate",
                        "value": metric.rate
                    }
        
        return result
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self._lock:
            for name, metric in self._metrics.items():
                safe_name = name.replace(".", "_")
                
                if isinstance(metric, Counter):
                    lines.append(f"# TYPE {safe_name} counter")
                    lines.append(f"{safe_name} {metric.value}")
                
                elif isinstance(metric, Gauge):
                    lines.append(f"# TYPE {safe_name} gauge")
                    lines.append(f"{safe_name} {metric.value}")
                
                elif isinstance(metric, Histogram):
                    lines.append(f"# TYPE {safe_name} histogram")
                    lines.append(f"{safe_name}_count {metric.count}")
                    lines.append(f"{safe_name}_sum {metric.sum}")
        
        return "\n".join(lines)


# Global registry instance
metrics = MetricsRegistry()


# ============================================================================
# TIMING UTILITIES
# ============================================================================

@contextmanager
def timer(histogram_name: str):
    """
    Context manager for timing code blocks.
    
    Usage:
        with timer("llm_latency_seconds"):
            response = llm.call(...)
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        metrics.histogram(histogram_name).observe(elapsed)


def timed(histogram_name: str):
    """
    Decorator for timing functions.
    
    Usage:
        @timed("llm_latency_seconds")
        def call_llm(...):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with timer(histogram_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# REQUEST TRACING
# ============================================================================

@dataclass
class Span:
    """A single span in a trace."""
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    operation: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "ok"
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0
    
    def log(self, message: str, **kwargs):
        self.logs.append({
            "timestamp": time.time(),
            "message": message,
            **kwargs
        })
    
    def set_tag(self, key: str, value: str):
        self.tags[key] = value
    
    def finish(self, status: str = "ok"):
        self.end_time = time.time()
        self.status = status


class Tracer:
    """
    Distributed tracing support.
    
    Creates traces for requests flowing through the system.
    """
    
    _local = threading.local()
    
    def __init__(self):
        self._spans: Dict[str, List[Span]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def start_trace(self, operation: str) -> Span:
        """Start a new trace."""
        trace_id = str(uuid.uuid4())[:16]
        span_id = str(uuid.uuid4())[:8]
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_id=None,
            operation=operation,
            start_time=time.time()
        )
        
        self._local.current_span = span
        
        with self._lock:
            self._spans[trace_id].append(span)
        
        return span
    
    def start_span(self, operation: str) -> Span:
        """Start a child span."""
        parent = getattr(self._local, 'current_span', None)
        
        if parent:
            trace_id = parent.trace_id
            parent_id = parent.span_id
        else:
            trace_id = str(uuid.uuid4())[:16]
            parent_id = None
        
        span_id = str(uuid.uuid4())[:8]
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            operation=operation,
            start_time=time.time()
        )
        
        self._local.current_span = span
        
        with self._lock:
            self._spans[trace_id].append(span)
        
        return span
    
    def current_span(self) -> Optional[Span]:
        """Get current span."""
        return getattr(self._local, 'current_span', None)
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        return self._spans.get(trace_id, [])
    
    @contextmanager
    def trace(self, operation: str):
        """Context manager for tracing."""
        span = self.start_span(operation)
        try:
            yield span
            span.finish("ok")
        except Exception as e:
            span.finish("error")
            span.set_tag("error", str(e))
            raise


# Global tracer instance
tracer = Tracer()


# ============================================================================
# SYSTEM MONITOR
# ============================================================================

class SystemMonitor:
    """
    Monitors system resources.
    
    Periodically updates memory, CPU, and other system metrics.
    """
    
    def __init__(self, interval_seconds: float = 10.0):
        self.interval = interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process()
    
    def start(self):
        """Start monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("System monitor started")
    
    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._update_metrics()
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            
            time.sleep(self.interval)
    
    def _update_metrics(self):
        """Update system metrics."""
        # Memory
        mem_info = self._process.memory_info()
        metrics.gauge("memory_usage_bytes").set(mem_info.rss)
        
        # CPU
        cpu_percent = self._process.cpu_percent()
        metrics.gauge("cpu_usage_percent").set(cpu_percent)
    
    def get_snapshot(self) -> Dict:
        """Get current system snapshot."""
        mem_info = self._process.memory_info()
        
        return {
            "memory_rss_mb": mem_info.rss / 1024 / 1024,
            "memory_vms_mb": mem_info.vms / 1024 / 1024,
            "cpu_percent": self._process.cpu_percent(),
            "threads": self._process.num_threads(),
            "open_files": len(self._process.open_files()),
        }


# ============================================================================
# ALERTING
# ============================================================================

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """An alert event."""
    name: str
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)


class AlertRule:
    """Rule for triggering alerts."""
    
    def __init__(
        self,
        name: str,
        metric_name: str,
        threshold: float,
        comparison: str = "gt",  # gt, lt, eq, gte, lte
        level: AlertLevel = AlertLevel.WARNING,
        cooldown_seconds: float = 300,  # 5 minutes
    ):
        self.name = name
        self.metric_name = metric_name
        self.threshold = threshold
        self.comparison = comparison
        self.level = level
        self.cooldown = cooldown_seconds
        
        self._last_alert_time: float = 0
    
    def check(self, current_value: float) -> Optional[Alert]:
        """Check if alert should fire."""
        # Check cooldown
        if time.time() - self._last_alert_time < self.cooldown:
            return None
        
        # Check condition
        triggered = False
        if self.comparison == "gt" and current_value > self.threshold:
            triggered = True
        elif self.comparison == "lt" and current_value < self.threshold:
            triggered = True
        elif self.comparison == "gte" and current_value >= self.threshold:
            triggered = True
        elif self.comparison == "lte" and current_value <= self.threshold:
            triggered = True
        elif self.comparison == "eq" and current_value == self.threshold:
            triggered = True
        
        if triggered:
            self._last_alert_time = time.time()
            return Alert(
                name=self.name,
                level=self.level,
                message=f"{self.metric_name} {self.comparison} {self.threshold} (current: {current_value})",
                metric_name=self.metric_name,
                current_value=current_value,
                threshold=self.threshold,
            )
        
        return None


class Alerter:
    """
    Alert manager.
    
    Evaluates rules and triggers alerts.
    """
    
    def __init__(self):
        self._rules: List[AlertRule] = []
        self._handlers: List[Callable[[Alert], None]] = []
        self._alerts: deque = deque(maxlen=1000)
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self._rules.append(rule)
    
    def add_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler."""
        self._handlers.append(handler)
    
    def check_all(self):
        """Check all rules against current metrics."""
        exported = metrics.export()
        
        for rule in self._rules:
            metric_data = exported.get(rule.metric_name)
            if not metric_data:
                continue
            
            value = metric_data.get("value") or metric_data.get("mean", 0)
            alert = rule.check(value)
            
            if alert:
                self._alerts.append(alert)
                self._fire_alert(alert)
    
    def _fire_alert(self, alert: Alert):
        """Fire alert to handlers."""
        logger.warning(f"ALERT [{alert.level.value}] {alert.name}: {alert.message}")
        
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def get_recent_alerts(self, count: int = 100) -> List[Alert]:
        """Get recent alerts."""
        return list(self._alerts)[-count:]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def track_llm_call(tokens: int, latency: float, error: bool = False):
    """Track an LLM call."""
    metrics.counter("llm_calls_total").inc()
    metrics.counter("llm_tokens_total").inc(tokens)
    metrics.histogram("llm_latency_seconds").observe(latency)
    
    if error:
        metrics.counter("llm_errors_total").inc()


def track_faiss_query(latency: float, results: int):
    """Track a FAISS query."""
    metrics.counter("faiss_queries_total").inc()
    metrics.histogram("faiss_latency_seconds").observe(latency)


def track_cache(hit: bool):
    """Track a cache access."""
    if hit:
        metrics.counter("cache_hits_total").inc()
    else:
        metrics.counter("cache_misses_total").inc()


def track_task(wait_time: float, execution_time: float, success: bool):
    """Track a task execution."""
    metrics.histogram("task_wait_time_seconds").observe(wait_time)
    metrics.histogram("task_execution_time_seconds").observe(execution_time)
    
    if success:
        metrics.counter("tasks_completed_total").inc()
    else:
        metrics.counter("tasks_failed_total").inc()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Logging
    "StructuredLogger",
    "LogContext",
    "JsonFormatter",
    
    # Metrics
    "MetricsRegistry",
    "Counter",
    "Gauge",
    "Histogram",
    "RollingRate",
    "metrics",
    
    # Timing
    "timer",
    "timed",
    
    # Tracing
    "Tracer",
    "Span",
    "tracer",
    
    # System
    "SystemMonitor",
    
    # Alerting
    "Alerter",
    "AlertRule",
    "Alert",
    "AlertLevel",
    
    # Convenience
    "track_llm_call",
    "track_faiss_query",
    "track_cache",
    "track_task",
]
