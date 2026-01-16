"""
Routing module - Query classification and mode handlers.
"""
from core.routing.classifier import (
    QueryMode,
    ClassificationResult,
    QueryClassifier,
    get_query_classifier,
)
from core.routing.handlers import (
    ModeHandler,
    DataQueryHandler,
    DocumentOverviewHandler,
    FreeformHandler,
    SystemTaskHandler,
    get_handler_for_mode,
)
from core.routing.smart_router import (
    SmartQueryClassifier,
    SmartRouter,
    QueryCache,
    QueryClassification,
    RoutingResult,
    QueryIntent,
    ProcessingPath,
)

__all__ = [
    # Original exports
    "QueryMode",
    "ClassificationResult",
    "QueryClassifier",
    "get_query_classifier",
    "ModeHandler",
    "DataQueryHandler",
    "DocumentOverviewHandler",
    "FreeformHandler",
    "SystemTaskHandler",
    "get_handler_for_mode",
    # New smart routing
    "SmartQueryClassifier",
    "SmartRouter",
    "QueryCache",
    "QueryClassification",
    "RoutingResult",
    "QueryIntent",
    "ProcessingPath",
]
