"""
Observability package.

Provides:
- Structured logging with JSON output
- Metrics collection (counters, gauges, histograms)
- Request tracing with spans
- System resource monitoring
- Alerting with thresholds
"""
from .monitoring import (
    # Logging
    StructuredLogger,
    LogContext,
    JsonFormatter,
    
    # Metrics
    MetricsRegistry,
    Counter,
    Gauge,
    Histogram,
    RollingRate,
    metrics,
    
    # Timing
    timer,
    timed,
    
    # Tracing
    Tracer,
    Span,
    tracer,
    
    # System
    SystemMonitor,
    
    # Alerting
    Alerter,
    AlertRule,
    Alert,
    AlertLevel,
    
    # Convenience
    track_llm_call,
    track_faiss_query,
    track_cache,
    track_task,
)

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
