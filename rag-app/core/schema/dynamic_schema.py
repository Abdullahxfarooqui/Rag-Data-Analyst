"""
Dynamic JSON Schema Generator.

ADAPTS OUTPUT STRUCTURE TO ANY DOCUMENT TYPE.

This module generates structured JSON outputs that automatically
adapt to the content of ingested documents. The schema is not
fixed - it discovers metrics, comparisons, and rankings from
the actual data.

TARGET OUTPUT FORMAT:
{
  "summary": "High-level document summary",
  "metrics": [
    {"name": "<column_name>", "value": 123, "unit": "<detected_unit>", "context": "..."}
  ],
  "comparisons": [
    {"description": "...", "entities": ["<col1>", "<col2>"], "difference": 10, "significance": "high"}
  ],
  "rankings": [
    {"entity": "Entity Name", "metric": "Metric Name", "rank": 1, "value": 123}
  ],
  "trends": [
    {"metric": "...", "direction": "up", "change_percent": 15.2, "period": "Q4 2024"}
  ],
  "anomalies": [
    {"metric": "...", "value": 999, "expected_range": [0, 100], "severity": "high"}
  ],
  "confidence_level": "high|medium|low",
  "notes": ["Fallback message if LLM fails", "Schema mismatch warnings"]
}

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────────────┐
│                    DynamicSchemaGenerator                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Metric      │  │ Comparison  │  │ Ranking     │  │ Trend       │    │
│  │ Extractor   │  │ Generator   │  │ Generator   │  │ Detector    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
│         │                │                │                │            │
│         └────────────────┼────────────────┼────────────────┘            │
│                          ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Schema Assembler                              │   │
│  │  - Validates all components                                      │   │
│  │  - Adds confidence scoring                                       │   │
│  │  - Handles missing/invalid data                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

TRADE-OFFS:
- Flexibility vs Validation: Dynamic schemas adapt but need careful validation
- Completeness vs Noise: Including all metrics vs only significant ones
- Accuracy vs Coverage: Deep analysis of few fields vs shallow of all
"""
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# SCHEMA COMPONENTS
# ============================================================================

class ConfidenceLevel(Enum):
    """Confidence level for generated insights."""
    HIGH = "high"       # Strong statistical basis
    MEDIUM = "medium"   # Some uncertainty
    LOW = "low"         # Weak or missing data


class SignificanceLevel(Enum):
    """Significance level for comparisons/anomalies."""
    HIGH = "high"       # >20% difference or >2 std
    MEDIUM = "medium"   # 10-20% or 1-2 std
    LOW = "low"         # <10% or <1 std


@dataclass
class Metric:
    """A single extracted metric."""
    name: str
    value: Union[int, float]
    unit: Optional[str] = None
    context: Optional[str] = None
    source_table: Optional[str] = None
    computation: Optional[str] = None  # e.g., "sum", "mean", "count"
    
    def to_dict(self) -> Dict:
        result = {
            "name": self.name,
            "value": self.value,
        }
        if self.unit:
            result["unit"] = self.unit
        if self.context:
            result["context"] = self.context
        if self.computation:
            result["computation"] = self.computation
        return result


@dataclass
class Comparison:
    """Comparison between entities or time periods."""
    description: str
    entities: List[str]
    values: List[float]
    difference: float
    difference_percent: Optional[float] = None
    significance: SignificanceLevel = SignificanceLevel.MEDIUM
    metric_name: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "description": self.description,
            "entities": self.entities,
            "values": self.values,
            "difference": self.difference,
            "difference_percent": self.difference_percent,
            "significance": self.significance.value,
            "metric_name": self.metric_name,
        }


@dataclass
class Ranking:
    """Ranked entity by metric."""
    entity: str
    metric_name: str
    rank: int
    value: float
    unit: Optional[str] = None
    total_count: Optional[int] = None
    
    def to_dict(self) -> Dict:
        result = {
            "entity": self.entity,
            "metric": self.metric_name,
            "rank": self.rank,
            "value": self.value,
        }
        if self.unit:
            result["unit"] = self.unit
        if self.total_count:
            result["of_total"] = self.total_count
        return result


@dataclass
class Trend:
    """Detected trend in data."""
    metric_name: str
    direction: str  # "up", "down", "stable"
    change_percent: float
    period: Optional[str] = None
    start_value: Optional[float] = None
    end_value: Optional[float] = None
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    
    def to_dict(self) -> Dict:
        return {
            "metric": self.metric_name,
            "direction": self.direction,
            "change_percent": self.change_percent,
            "period": self.period,
            "start_value": self.start_value,
            "end_value": self.end_value,
            "confidence": self.confidence.value,
        }


@dataclass
class Anomaly:
    """Detected anomaly in data."""
    metric_name: str
    value: float
    expected_range: Tuple[float, float]
    deviation: float
    severity: SignificanceLevel
    description: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "metric": self.metric_name,
            "value": self.value,
            "expected_range": list(self.expected_range),
            "deviation": self.deviation,
            "severity": self.severity.value,
            "description": self.description,
        }


@dataclass
class DynamicOutput:
    """
    Complete dynamic output structure.
    
    This is the final output format that adapts to any document.
    """
    summary: str
    metrics: List[Metric] = field(default_factory=list)
    comparisons: List[Comparison] = field(default_factory=list)
    rankings: List[Ranking] = field(default_factory=list)
    trends: List[Trend] = field(default_factory=list)
    anomalies: List[Anomaly] = field(default_factory=list)
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    notes: List[str] = field(default_factory=list)
    
    # Metadata
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_document: Optional[str] = None
    query: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "summary": self.summary,
            "metrics": [m.to_dict() for m in self.metrics],
            "comparisons": [c.to_dict() for c in self.comparisons],
            "rankings": [r.to_dict() for r in self.rankings],
            "trends": [t.to_dict() for t in self.trends],
            "anomalies": [a.to_dict() for a in self.anomalies],
            "confidence_level": self.confidence_level.value,
            "notes": self.notes,
            "generated_at": self.generated_at,
            "source_document": self.source_document,
            "query": self.query,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ============================================================================
# METRIC EXTRACTOR
# ============================================================================

class MetricExtractor:
    """
    Extracts metrics from DataFrames.
    
    Automatically discovers and computes relevant metrics
    based on column types and data distribution.
    """
    
    def __init__(
        self,
        max_metrics: int = 20,
        min_significance: float = 0.01,  # Min % of total to be significant
    ):
        self.max_metrics = max_metrics
        self.min_significance = min_significance
    
    def extract(
        self,
        df: pd.DataFrame,
        table_name: str = None,
        column_metadata: Dict[str, Dict] = None,
    ) -> List[Metric]:
        """
        Extract metrics from DataFrame.
        
        Args:
            df: Source DataFrame
            table_name: Optional table name for context
            column_metadata: Optional column metadata with units
            
        Returns:
            List of extracted metrics
        """
        metrics = []
        column_metadata = column_metadata or {}
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            col_meta = column_metadata.get(col, {})
            unit = col_meta.get("unit")
            
            # Skip ID columns
            if col.lower() in ['id', 'index'] or col.lower().endswith('_id'):
                continue
            
            # Total/Sum
            total = df[col].sum()
            if pd.notna(total) and total != 0:
                metrics.append(Metric(
                    name=f"Total {col}",
                    value=round(total, 2),
                    unit=unit,
                    source_table=table_name,
                    computation="sum"
                ))
            
            # Average
            mean = df[col].mean()
            if pd.notna(mean):
                metrics.append(Metric(
                    name=f"Average {col}",
                    value=round(mean, 2),
                    unit=unit,
                    source_table=table_name,
                    computation="mean"
                ))
            
            # Max
            max_val = df[col].max()
            if pd.notna(max_val):
                metrics.append(Metric(
                    name=f"Maximum {col}",
                    value=round(max_val, 2),
                    unit=unit,
                    source_table=table_name,
                    computation="max"
                ))
            
            # Min (if different from max and meaningful)
            min_val = df[col].min()
            if pd.notna(min_val) and min_val != max_val:
                metrics.append(Metric(
                    name=f"Minimum {col}",
                    value=round(min_val, 2),
                    unit=unit,
                    source_table=table_name,
                    computation="min"
                ))
        
        # Count metrics
        metrics.append(Metric(
            name="Total Records",
            value=len(df),
            unit="count",
            source_table=table_name,
            computation="count"
        ))
        
        # Categorical counts
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols[:3]:  # Limit to top 3
            unique = df[col].nunique()
            if unique > 0 and unique < len(df) * 0.9:  # Not all unique
                metrics.append(Metric(
                    name=f"Unique {col} Count",
                    value=unique,
                    unit="count",
                    source_table=table_name,
                    computation="unique_count"
                ))
        
        # Sort by importance and limit
        return metrics[:self.max_metrics]


# ============================================================================
# COMPARISON GENERATOR
# ============================================================================

class ComparisonGenerator:
    """
    Generates comparisons between entities.
    
    Automatically identifies comparable entities and
    computes meaningful differences.
    """
    
    def __init__(
        self,
        max_comparisons: int = 10,
        significance_threshold: float = 0.1,  # 10% difference
    ):
        self.max_comparisons = max_comparisons
        self.significance_threshold = significance_threshold
    
    def generate(
        self,
        df: pd.DataFrame,
        group_column: str = None,
        value_column: str = None,
    ) -> List[Comparison]:
        """
        Generate comparisons from DataFrame.
        
        Args:
            df: Source DataFrame
            group_column: Column to group by
            value_column: Column to compare values
            
        Returns:
            List of comparisons
        """
        comparisons = []
        
        # Auto-detect columns if not specified
        if not group_column:
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                # Pick column with moderate cardinality
                for col in cat_cols:
                    nunique = df[col].nunique()
                    if 2 <= nunique <= 20:
                        group_column = col
                        break
        
        if not value_column:
            num_cols = df.select_dtypes(include=['number']).columns
            if len(num_cols) > 0:
                value_column = num_cols[0]
        
        if not group_column or not value_column:
            return comparisons
        
        # Compute aggregations
        grouped = df.groupby(group_column)[value_column].sum().sort_values(ascending=False)
        
        if len(grouped) < 2:
            return comparisons
        
        # Top vs second
        if len(grouped) >= 2:
            top_entity = grouped.index[0]
            second_entity = grouped.index[1]
            top_val = grouped.iloc[0]
            second_val = grouped.iloc[1]
            
            diff = top_val - second_val
            diff_pct = (diff / second_val * 100) if second_val != 0 else 0
            
            significance = self._get_significance(diff_pct)
            
            comparisons.append(Comparison(
                description=f"{top_entity} has {abs(diff_pct):.1f}% more {value_column} than {second_entity}",
                entities=[str(top_entity), str(second_entity)],
                values=[float(top_val), float(second_val)],
                difference=float(diff),
                difference_percent=float(diff_pct),
                significance=significance,
                metric_name=value_column,
            ))
        
        # Top vs bottom
        if len(grouped) >= 3:
            top_entity = grouped.index[0]
            bottom_entity = grouped.index[-1]
            top_val = grouped.iloc[0]
            bottom_val = grouped.iloc[-1]
            
            diff = top_val - bottom_val
            diff_pct = (diff / bottom_val * 100) if bottom_val != 0 else 0
            
            significance = self._get_significance(diff_pct)
            
            comparisons.append(Comparison(
                description=f"{top_entity} has {abs(diff_pct):.1f}% more {value_column} than {bottom_entity}",
                entities=[str(top_entity), str(bottom_entity)],
                values=[float(top_val), float(bottom_val)],
                difference=float(diff),
                difference_percent=float(diff_pct),
                significance=significance,
                metric_name=value_column,
            ))
        
        return comparisons[:self.max_comparisons]
    
    def _get_significance(self, pct_diff: float) -> SignificanceLevel:
        """Determine significance level from percentage difference."""
        abs_diff = abs(pct_diff)
        if abs_diff > 20:
            return SignificanceLevel.HIGH
        elif abs_diff > 10:
            return SignificanceLevel.MEDIUM
        else:
            return SignificanceLevel.LOW


# ============================================================================
# RANKING GENERATOR
# ============================================================================

class RankingGenerator:
    """
    Generates rankings from data.
    
    Identifies top/bottom performers by various metrics.
    """
    
    def __init__(
        self,
        top_n: int = 10,
        include_bottom: bool = True,
    ):
        self.top_n = top_n
        self.include_bottom = include_bottom
    
    def generate(
        self,
        df: pd.DataFrame,
        entity_column: str = None,
        value_column: str = None,
        unit: str = None,
    ) -> List[Ranking]:
        """
        Generate rankings from DataFrame.
        
        Args:
            df: Source DataFrame
            entity_column: Column with entity names
            value_column: Column with values to rank
            unit: Unit for values
            
        Returns:
            List of rankings
        """
        rankings = []
        
        # Auto-detect columns
        if not entity_column:
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                entity_column = cat_cols[0]
        
        if not value_column:
            num_cols = df.select_dtypes(include=['number']).columns
            if len(num_cols) > 0:
                value_column = num_cols[0]
        
        if not entity_column or not value_column:
            return rankings
        
        # Aggregate if needed
        if df[entity_column].duplicated().any():
            aggregated = df.groupby(entity_column)[value_column].sum()
        else:
            aggregated = df.set_index(entity_column)[value_column]
        
        # Sort and rank
        sorted_data = aggregated.sort_values(ascending=False)
        total_count = len(sorted_data)
        
        # Top N
        for rank, (entity, value) in enumerate(sorted_data.head(self.top_n).items(), 1):
            rankings.append(Ranking(
                entity=str(entity),
                metric_name=value_column,
                rank=rank,
                value=float(value),
                unit=unit,
                total_count=total_count,
            ))
        
        # Bottom N (if enabled and enough data)
        if self.include_bottom and total_count > self.top_n * 2:
            bottom_start_rank = total_count - self.top_n + 1
            for i, (entity, value) in enumerate(sorted_data.tail(self.top_n).items()):
                rankings.append(Ranking(
                    entity=str(entity),
                    metric_name=value_column,
                    rank=bottom_start_rank + i,
                    value=float(value),
                    unit=unit,
                    total_count=total_count,
                ))
        
        return rankings


# ============================================================================
# TREND DETECTOR
# ============================================================================

class TrendDetector:
    """
    Detects trends in time-series data.
    """
    
    def __init__(
        self,
        min_data_points: int = 3,
        significance_threshold: float = 0.05,  # 5% change
    ):
        self.min_data_points = min_data_points
        self.significance_threshold = significance_threshold
    
    def detect(
        self,
        df: pd.DataFrame,
        date_column: str = None,
        value_columns: List[str] = None,
    ) -> List[Trend]:
        """
        Detect trends in DataFrame.
        
        Args:
            df: Source DataFrame
            date_column: Column with dates
            value_columns: Columns to analyze for trends
            
        Returns:
            List of detected trends
        """
        trends = []
        
        # Auto-detect date column
        if not date_column:
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]':
                    date_column = col
                    break
                # Try parsing
                try:
                    pd.to_datetime(df[col].head(10))
                    date_column = col
                    break
                except:
                    continue
        
        if not date_column:
            return trends
        
        # Auto-detect value columns
        if not value_columns:
            value_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        # Sort by date
        df = df.copy()
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column)
        except:
            return trends
        
        # Analyze each value column
        for col in value_columns[:5]:  # Limit to top 5
            values = df[col].dropna()
            
            if len(values) < self.min_data_points:
                continue
            
            # Compare first third to last third
            third = len(values) // 3
            if third == 0:
                continue
            
            first_avg = values.iloc[:third].mean()
            last_avg = values.iloc[-third:].mean()
            
            if first_avg == 0:
                continue
            
            change_pct = ((last_avg - first_avg) / first_avg) * 100
            
            if abs(change_pct) < self.significance_threshold * 100:
                direction = "stable"
            elif change_pct > 0:
                direction = "up"
            else:
                direction = "down"
            
            # Determine confidence based on data points
            if len(values) >= 30:
                confidence = ConfidenceLevel.HIGH
            elif len(values) >= 10:
                confidence = ConfidenceLevel.MEDIUM
            else:
                confidence = ConfidenceLevel.LOW
            
            trends.append(Trend(
                metric_name=col,
                direction=direction,
                change_percent=round(change_pct, 2),
                start_value=round(first_avg, 2),
                end_value=round(last_avg, 2),
                confidence=confidence,
            ))
        
        return trends


# ============================================================================
# ANOMALY DETECTOR
# ============================================================================

class AnomalyDetector:
    """
    Detects anomalies using statistical methods.
    """
    
    def __init__(
        self,
        z_threshold: float = 2.5,
        max_anomalies: int = 10,
    ):
        self.z_threshold = z_threshold
        self.max_anomalies = max_anomalies
    
    def detect(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
    ) -> List[Anomaly]:
        """
        Detect anomalies in DataFrame.
        
        Args:
            df: Source DataFrame
            columns: Columns to check (defaults to all numeric)
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if not columns:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        for col in columns:
            values = df[col].dropna()
            
            if len(values) < 10:  # Need enough data
                continue
            
            mean = values.mean()
            std = values.std()
            
            if std == 0:
                continue
            
            # Calculate Z-scores
            z_scores = (values - mean) / std
            
            # Find anomalies
            for idx, z in z_scores.items():
                if abs(z) > self.z_threshold:
                    value = values[idx]
                    expected_min = mean - 2 * std
                    expected_max = mean + 2 * std
                    
                    if abs(z) > 3:
                        severity = SignificanceLevel.HIGH
                    else:
                        severity = SignificanceLevel.MEDIUM
                    
                    anomalies.append(Anomaly(
                        metric_name=col,
                        value=float(value),
                        expected_range=(round(expected_min, 2), round(expected_max, 2)),
                        deviation=round(z, 2),
                        severity=severity,
                        description=f"Value {value:.2f} is {abs(z):.1f} standard deviations from mean"
                    ))
        
        # Sort by severity and deviation
        anomalies.sort(key=lambda a: (-1 if a.severity == SignificanceLevel.HIGH else 0, -abs(a.deviation)))
        
        return anomalies[:self.max_anomalies]


# ============================================================================
# DYNAMIC SCHEMA GENERATOR
# ============================================================================

class DynamicSchemaGenerator:
    """
    Generates complete dynamic output from data.
    
    Combines all extractors and generators to produce
    a comprehensive, structured analysis.
    
    Usage:
        generator = DynamicSchemaGenerator()
        output = generator.generate(
            df=my_dataframe,
            query="Analyze sales performance",
            llm_summary="AI-generated summary..."
        )
        
        # Get JSON
        json_output = output.to_json()
    """
    
    def __init__(
        self,
        max_metrics: int = 20,
        max_comparisons: int = 10,
        max_rankings: int = 20,
        include_anomalies: bool = True,
        include_trends: bool = True,
    ):
        self.metric_extractor = MetricExtractor(max_metrics=max_metrics)
        self.comparison_generator = ComparisonGenerator(max_comparisons=max_comparisons)
        self.ranking_generator = RankingGenerator(top_n=max_rankings // 2)
        self.anomaly_detector = AnomalyDetector() if include_anomalies else None
        self.trend_detector = TrendDetector() if include_trends else None
    
    def generate(
        self,
        df: pd.DataFrame,
        query: str = None,
        llm_summary: str = None,
        table_name: str = None,
        column_metadata: Dict = None,
    ) -> DynamicOutput:
        """
        Generate dynamic output from DataFrame.
        
        Args:
            df: Source DataFrame
            query: User's query (for context)
            llm_summary: LLM-generated summary
            table_name: Source table name
            column_metadata: Column metadata with units
            
        Returns:
            DynamicOutput with all components
        """
        notes = []
        
        # Extract metrics
        try:
            metrics = self.metric_extractor.extract(df, table_name, column_metadata)
        except Exception as e:
            logger.error(f"Metric extraction failed: {e}")
            metrics = []
            notes.append(f"Metric extraction failed: {str(e)}")
        
        # Generate comparisons
        try:
            comparisons = self.comparison_generator.generate(df)
        except Exception as e:
            logger.error(f"Comparison generation failed: {e}")
            comparisons = []
            notes.append(f"Comparison generation failed: {str(e)}")
        
        # Generate rankings
        try:
            rankings = self.ranking_generator.generate(df)
        except Exception as e:
            logger.error(f"Ranking generation failed: {e}")
            rankings = []
            notes.append(f"Ranking generation failed: {str(e)}")
        
        # Detect trends
        trends = []
        if self.trend_detector:
            try:
                trends = self.trend_detector.detect(df)
            except Exception as e:
                logger.error(f"Trend detection failed: {e}")
                notes.append(f"Trend detection failed: {str(e)}")
        
        # Detect anomalies
        anomalies = []
        if self.anomaly_detector:
            try:
                anomalies = self.anomaly_detector.detect(df)
            except Exception as e:
                logger.error(f"Anomaly detection failed: {e}")
                notes.append(f"Anomaly detection failed: {str(e)}")
        
        # Generate summary if not provided
        if not llm_summary:
            llm_summary = self._generate_fallback_summary(df, metrics, comparisons, rankings)
            notes.append("Summary generated from data (LLM not used)")
        
        # Determine confidence
        confidence = self._calculate_confidence(df, metrics, comparisons)
        
        return DynamicOutput(
            summary=llm_summary,
            metrics=metrics,
            comparisons=comparisons,
            rankings=rankings,
            trends=trends,
            anomalies=anomalies,
            confidence_level=confidence,
            notes=notes,
            source_document=table_name,
            query=query,
        )
    
    def _generate_fallback_summary(
        self,
        df: pd.DataFrame,
        metrics: List[Metric],
        comparisons: List[Comparison],
        rankings: List[Ranking],
    ) -> str:
        """Generate summary without LLM."""
        parts = [f"Analysis of {len(df)} records with {len(df.columns)} columns."]
        
        if metrics:
            top_metrics = metrics[:3]
            metric_strs = [f"{m.name}: {m.value}{m.unit or ''}" for m in top_metrics]
            parts.append(f"Key metrics: {', '.join(metric_strs)}.")
        
        if comparisons:
            comp = comparisons[0]
            parts.append(comp.description)
        
        if rankings:
            top = rankings[0]
            parts.append(f"Top performer: {top.entity} with {top.value} {top.metric_name}.")
        
        return " ".join(parts)
    
    def _calculate_confidence(
        self,
        df: pd.DataFrame,
        metrics: List[Metric],
        comparisons: List[Comparison],
    ) -> ConfidenceLevel:
        """Calculate overall confidence level."""
        score = 0
        
        # Data size
        if len(df) >= 1000:
            score += 2
        elif len(df) >= 100:
            score += 1
        
        # Metrics extracted
        if len(metrics) >= 10:
            score += 2
        elif len(metrics) >= 5:
            score += 1
        
        # Comparisons generated
        if len(comparisons) >= 3:
            score += 1
        
        # Column completeness
        null_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if null_pct < 0.1:
            score += 1
        
        if score >= 5:
            return ConfidenceLevel.HIGH
        elif score >= 3:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


# ============================================================================
# PROMPT GENERATOR FOR LLM
# ============================================================================

class PromptGenerator:
    """
    Generates dynamic prompts for LLM based on data.
    
    Creates context-aware prompts that adapt to the
    specific metrics and patterns in the data.
    """
    
    def generate_insight_prompt(
        self,
        df: pd.DataFrame,
        query: str,
        metrics: List[Metric] = None,
        comparisons: List[Comparison] = None,
        rankings: List[Ranking] = None,
    ) -> str:
        """
        Generate insight prompt for LLM.
        
        Args:
            df: Source DataFrame
            query: User's query
            metrics: Pre-extracted metrics
            comparisons: Pre-generated comparisons
            rankings: Pre-generated rankings
            
        Returns:
            Formatted prompt string
        """
        # Build context section
        context_parts = [
            f"Data contains {len(df)} records with columns: {', '.join(df.columns[:15])}",
        ]
        
        if metrics:
            metric_strs = [f"- {m.name}: {m.value}{m.unit or ''}" for m in metrics[:10]]
            context_parts.append("Key Metrics:\n" + "\n".join(metric_strs))
        
        if comparisons:
            comp_strs = [f"- {c.description}" for c in comparisons[:5]]
            context_parts.append("Comparisons:\n" + "\n".join(comp_strs))
        
        if rankings:
            rank_strs = [f"- #{r.rank}: {r.entity} ({r.value} {r.metric_name})" for r in rankings[:5]]
            context_parts.append("Top Rankings:\n" + "\n".join(rank_strs))
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Analyze the following data and provide insights.

USER QUERY: {query}

DATA CONTEXT:
{context}

Provide a concise analysis focusing on:
1. Key findings relevant to the query
2. Notable patterns or trends
3. Actionable recommendations

Keep your response focused and data-driven."""
        
        return prompt


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "DynamicSchemaGenerator",
    "DynamicOutput",
    "Metric",
    "Comparison",
    "Ranking",
    "Trend",
    "Anomaly",
    "ConfidenceLevel",
    "SignificanceLevel",
    "MetricExtractor",
    "ComparisonGenerator",
    "RankingGenerator",
    "TrendDetector",
    "AnomalyDetector",
    "PromptGenerator",
]
