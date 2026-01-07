"""
Visualization Configuration Module.

Provides configuration for visualizations based on query type and detected metrics.
Actual chart rendering is done in the Streamlit UI.
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd


class ChartType(str, Enum):
    """Supported chart types."""
    TIME_SERIES = "time_series"
    BAR = "bar"
    PIE = "pie"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    SCATTER = "scatter"


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    show: bool
    chart_types: List[ChartType]
    target_columns: List[str]
    category_column: Optional[str] = None
    date_column: Optional[str] = None
    title: str = ""
    metrics: List[str] = field(default_factory=list)
    max_categories: int = 10
    max_series: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "show": self.show,
            "chart_types": [ct.value for ct in self.chart_types],
            "target_columns": self.target_columns,
            "category_column": self.category_column,
            "date_column": self.date_column,
            "title": self.title,
            "metrics": self.metrics,
            "max_categories": self.max_categories,
            "max_series": self.max_series
        }


def get_visualization_config(
    query: str,
    df: Optional[pd.DataFrame],
    specific_metrics: List[str],
    target_columns: List[str],
    query_mode: str
) -> VisualizationConfig:
    """
    Determine visualization configuration based on query and data.
    
    Args:
        query: User query
        df: DataFrame (may be None)
        specific_metrics: Detected metrics from query
        target_columns: Target columns for metrics
        query_mode: The query mode (DATA_QUERY, DOC_OVERVIEW, etc.)
        
    Returns:
        VisualizationConfig with appropriate settings
    """
    # Only show visualizations for DATA_QUERY mode
    if query_mode != "DATA_QUERY" or df is None or df.empty:
        return VisualizationConfig(
            show=False,
            chart_types=[],
            target_columns=[]
        )
    
    # Detect date column
    date_column = None
    for col in df.columns:
        if 'DATE' in col.upper() or 'TIME' in col.upper():
            try:
                pd.to_datetime(df[col].dropna().head(10), errors='coerce')
                date_column = col
                break
            except:
                pass
    
    # Detect category column
    category_column = None
    cat_candidates = [c for c in df.columns if df[c].dtype == 'object' and df[c].nunique() <= 50]
    if 'ITEM_NAME' in df.columns:
        category_column = 'ITEM_NAME'
    elif cat_candidates:
        category_column = cat_candidates[0]
    
    # Determine chart types based on query
    q_lower = query.lower()
    chart_types = []
    
    # Check for explicit chart requests
    if any(kw in q_lower for kw in ['trend', 'over time', 'time series', 'timeline']):
        chart_types.append(ChartType.TIME_SERIES)
    if any(kw in q_lower for kw in ['bar', 'compare', 'comparison', 'by']):
        chart_types.append(ChartType.BAR)
    if any(kw in q_lower for kw in ['pie', 'distribution', 'breakdown', 'proportion']):
        chart_types.append(ChartType.PIE)
    if any(kw in q_lower for kw in ['histogram', 'distribution of', 'frequency']):
        chart_types.append(ChartType.HISTOGRAM)
    if any(kw in q_lower for kw in ['correlation', 'heatmap', 'relationship']):
        chart_types.append(ChartType.HEATMAP)
    
    # Default chart types if none specified
    if not chart_types:
        if date_column:
            chart_types.append(ChartType.TIME_SERIES)
        if category_column:
            chart_types.append(ChartType.BAR)
        if len(target_columns) >= 2:
            chart_types.append(ChartType.PIE)
    
    # Build title
    metric_title = ', '.join(specific_metrics).upper() if specific_metrics else "Production"
    title = f"{metric_title} Analysis"
    
    return VisualizationConfig(
        show=True,
        chart_types=chart_types[:3],  # Limit to 3 chart types
        target_columns=target_columns[:4],  # Limit to 4 columns
        category_column=category_column,
        date_column=date_column,
        title=title,
        metrics=specific_metrics
    )


def should_show_visualizations(query_mode: str, specific_metrics: List[str]) -> bool:
    """
    Quick check if visualizations should be shown.
    
    Args:
        query_mode: Query classification mode
        specific_metrics: Detected metrics
        
    Returns:
        True if visualizations should be shown
    """
    return query_mode == "DATA_QUERY" and bool(specific_metrics)


def get_chart_data_requirements(chart_type: ChartType) -> Dict[str, Any]:
    """
    Get data requirements for a chart type.
    
    Args:
        chart_type: The chart type
        
    Returns:
        Dict with required columns and parameters
    """
    requirements = {
        ChartType.TIME_SERIES: {
            "requires_date": True,
            "requires_numeric": True,
            "min_columns": 1,
            "max_columns": 4,
            "aggregation": "sum"
        },
        ChartType.BAR: {
            "requires_date": False,
            "requires_category": True,
            "requires_numeric": True,
            "min_columns": 1,
            "max_columns": 4,
            "aggregation": "sum"
        },
        ChartType.PIE: {
            "requires_date": False,
            "requires_numeric": True,
            "min_columns": 2,
            "max_columns": 8,
            "aggregation": "sum"
        },
        ChartType.HISTOGRAM: {
            "requires_date": False,
            "requires_numeric": True,
            "min_columns": 1,
            "max_columns": 1,
            "aggregation": None
        },
        ChartType.HEATMAP: {
            "requires_date": False,
            "requires_numeric": True,
            "min_columns": 2,
            "max_columns": 15,
            "aggregation": None
        },
        ChartType.SCATTER: {
            "requires_date": False,
            "requires_numeric": True,
            "min_columns": 2,
            "max_columns": 2,
            "aggregation": None
        }
    }
    
    return requirements.get(chart_type, {})
