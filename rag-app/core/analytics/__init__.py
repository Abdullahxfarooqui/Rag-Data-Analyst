"""
Analytics module - Statistics computation and visualization configuration.
"""
from core.analytics.statistics import (
    compute_data_statistics,
    detect_specific_metrics,
    get_target_columns,
    generate_dataset_overview,
    detect_detail_mode,
    METRIC_COLUMNS,
    METRIC_KEYWORDS,
)
from core.analytics.visualizations import (
    VisualizationConfig,
    get_visualization_config,
    should_show_visualizations,
)

__all__ = [
    "compute_data_statistics",
    "detect_specific_metrics",
    "get_target_columns",
    "generate_dataset_overview",
    "detect_detail_mode",
    "METRIC_COLUMNS",
    "METRIC_KEYWORDS",
    "VisualizationConfig",
    "get_visualization_config",
    "should_show_visualizations",
]
