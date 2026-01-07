"""
Statistics Computation Module.

All statistics are computed using Python/Pandas, NOT the LLM.
This ensures accuracy and prevents hallucination.
"""
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import re


# ============================================================================
# METRIC COLUMN MAPPING
# ============================================================================

METRIC_COLUMNS = {
    "oil": ["PROD_OIL_VOL", "SALES_OIL_VOL", "INJ_OIL_VOL", "OIL_RATE", "OIL_DENSITY"],
    "gas": ["PROD_GAS_VOL", "SALES_GAS_VOL", "FUEL_GAS_VOL", "INJ_GAS_VOL", "GAS_RATE", "GAS_DENSITY", "FLARE_GAS_VOL"],
    "water": ["PROD_WAT_VOL", "SALES_WAT_VOL", "INJ_WAT_VOL", "WATER_RATE", "WATER_DENSITY"],
    "condensate": ["PROD_COND_VOL", "SALES_COND_VOL", "COND_RATE", "COND_DENSITY"],
    "lpg": ["PROD_LPG_VOL", "SALES_LPG_VOL", "LPG_RATE"],
    "ngl": ["PROD_NGL_VOL", "SALES_NGL_VOL", "NGL_RATE"],
    "heat": ["VOL_HEAT_SALES", "HEAT_RATE", "ENERGY_PROD", "ENERGY_SOLD"],
    "energy": ["ENERGY_PROD", "ENERGY_SOLD", "ENERGY_RATE", "BTU"],
    "injection": ["INJ_GAS_VOL", "INJ_WAT_VOL", "INJ_OIL_VOL", "INJ_RATE"],
    "production": ["PROD_OIL_VOL", "PROD_GAS_VOL", "PROD_WAT_VOL", "PROD_COND_VOL", "PROD_LPG_VOL"],
    "sales": ["SALES_OIL_VOL", "SALES_GAS_VOL", "SALES_WAT_VOL", "SALES_COND_VOL", "SALES_LPG_VOL"]
}

METRIC_KEYWORDS = {
    "oil": ["oil", "crude", "petroleum"],
    "gas": ["gas", "natural gas", "mmcf", "mcf", "fuel gas", "flare"],
    "water": ["water", "wat", "h2o", "brine", "produced water"],
    "condensate": ["condensate", "cond"],
    "lpg": ["lpg", "liquefied petroleum"],
    "ngl": ["ngl", "natural gas liquid"],
    "heat": ["heat", "thermal"],
    "energy": ["energy", "btu", "mmbtu"],
    "injection": ["injection", "inject", "injected"],
    "production": ["production", "produced", "prod"],
    "sales": ["sales", "sold", "sale"]
}


# ============================================================================
# METRIC DETECTION
# ============================================================================

def detect_specific_metrics(query: str) -> List[str]:
    """
    Detect which specific metrics the user is asking about.
    
    Args:
        query: User query string
        
    Returns:
        List of detected metric types (e.g., ['oil'], ['gas', 'water'])
    """
    q = query.lower()
    metrics = []
    
    specific_resources = ['oil', 'gas', 'water', 'condensate', 'lpg', 'ngl', 'heat', 'energy']
    
    for metric_type, keywords in METRIC_KEYWORDS.items():
        if any(keyword in q for keyword in keywords):
            metrics.append(metric_type)
    
    # If specific resources detected, remove generic categories
    has_specific_resource = any(m in specific_resources for m in metrics)
    
    if has_specific_resource:
        metrics = [m for m in metrics if m in specific_resources]
    
    return metrics


def get_target_columns(metrics: List[str], df_columns: List[str]) -> List[str]:
    """
    Get the exact columns to use based on detected metrics.
    
    Args:
        metrics: List of detected metric types
        df_columns: List of columns in the DataFrame
        
    Returns:
        List of matching column names
    """
    if not metrics:
        return []
    
    specific_resources = ['oil', 'gas', 'water', 'condensate', 'lpg', 'ngl', 'heat', 'energy']
    has_specific = [m for m in metrics if m in specific_resources]
    has_production = 'production' in metrics
    has_sales = 'sales' in metrics
    has_injection = 'injection' in metrics
    
    matched_columns = []
    
    for col in df_columns:
        col_upper = col.upper()
        
        # Skip UOM columns
        if col_upper.endswith('_UOM'):
            continue
        
        # Skip non-metric columns
        if not any(kw in col_upper for kw in ['VOL', 'RATE', 'ENERGY', 'BTU', 'HEAT']):
            continue
        
        # Case 1: Specific resource requested
        if has_specific:
            resource_keywords = {
                'oil': ['OIL'],
                'gas': ['GAS'],
                'water': ['WAT'],
                'condensate': ['COND'],
                'lpg': ['LPG'],
                'ngl': ['NGL'],
                'heat': ['HEAT'],
                'energy': ['ENERGY', 'BTU']
            }
            
            col_matches_resource = False
            for resource in has_specific:
                keywords = resource_keywords.get(resource, [resource.upper()])
                if any(kw in col_upper for kw in keywords):
                    col_matches_resource = True
                    break
            
            if not col_matches_resource:
                continue
            
            # Filter by production/sales/injection if specified
            if has_production and 'PROD' not in col_upper:
                continue
            if has_sales and 'SALES' not in col_upper:
                continue
            if has_injection and 'INJ' not in col_upper:
                continue
            
            matched_columns.append(col)
        
        # Case 2: Only generic category
        elif has_production or has_sales or has_injection:
            if has_production and 'PROD' in col_upper:
                matched_columns.append(col)
            elif has_sales and 'SALES' in col_upper:
                matched_columns.append(col)
            elif has_injection and 'INJ' in col_upper:
                matched_columns.append(col)
    
    return matched_columns


def detect_detail_mode(query: str) -> str:
    """
    Detect the detail level requested by the user.
    
    Returns:
        'detailed', 'normal', or 'brief'
    """
    q = query.lower()
    
    detailed_triggers = [
        'in detail', 'detailed', 'full', 'complete', 'comprehensive',
        'everything', 'all', 'entire', 'whole', 'in-depth', 'thorough',
        'elaborate', 'deep dive', 'full breakdown', 'full report',
        'complete analysis', 'detailed analysis', 'extensively'
    ]
    
    if any(trigger in q for trigger in detailed_triggers):
        return 'detailed'
    
    brief_triggers = [
        'short', 'brief', 'quick', 'concise', 'simple',
        'just tell me', 'one line', 'summary only', 'tldr', 'briefly'
    ]
    
    if any(trigger in q for trigger in brief_triggers):
        return 'brief'
    
    return 'normal'


# ============================================================================
# STATISTICS COMPUTATION
# ============================================================================

def compute_data_statistics(
    df: pd.DataFrame,
    specific_metrics: List[str],
    target_columns: List[str]
) -> str:
    """
    Compute actual statistics from DataFrame.
    
    Args:
        df: The DataFrame to analyze
        specific_metrics: Detected metric types
        target_columns: Columns to compute stats for
        
    Returns:
        Formatted statistics string
    """
    if df is None or df.empty:
        return "No data available."
    
    stats_lines = []
    stats_lines.append("## 📊 Computed Statistics\n")
    stats_lines.append(f"**Dataset Size:** {len(df):,} rows × {len(df.columns)} columns\n")
    
    metric_totals = {}
    
    # Compute stats for target columns
    if target_columns:
        stats_lines.append("### Requested Metric Statistics:")
        stats_lines.append("| Column | Total | Average | Min | Max | Count |")
        stats_lines.append("|--------|-------|---------|-----|-----|-------|")
        
        for col in target_columns[:10]:
            try:
                data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(data) > 0:
                    total = data.sum()
                    avg = data.mean()
                    min_val = data.min()
                    max_val = data.max()
                    count = len(data)
                    
                    uom = ""
                    if col + "_UOM" in df.columns:
                        uom_vals = df[col + "_UOM"].dropna()
                        uom = f" {uom_vals.iloc[0]}" if len(uom_vals) > 0 else ""
                    
                    stats_lines.append(
                        f"| {col[:30]} | {total:,.2f}{uom} | {avg:,.2f}{uom} | "
                        f"{min_val:,.2f} | {max_val:,.2f} | {count:,} |"
                    )
                    
                    metric_totals[col] = {"total": total, "avg": avg, "unit": uom}
            except Exception:
                pass
        
        stats_lines.append("")
        
        # Add comparison if multiple metrics
        if len(specific_metrics) >= 2 and len(metric_totals) >= 2:
            stats_lines.append("### 📊 Metric Comparison:")
            stats_lines.append("| Metric | Total Volume | % of Combined |")
            stats_lines.append("|--------|--------------|---------------|")
            
            combined_total = sum(m["total"] for m in metric_totals.values())
            for col, data in metric_totals.items():
                pct = (data["total"] / combined_total * 100) if combined_total > 0 else 0
                stats_lines.append(f"| {col[:25]} | {data['total']:,.2f}{data['unit']} | {pct:.1f}% |")
            
            stats_lines.append(f"\n**Combined Total:** {combined_total:,.2f}")
            
            if len(metric_totals) >= 2:
                sorted_metrics = sorted(metric_totals.items(), key=lambda x: x[1]["total"], reverse=True)
                largest = sorted_metrics[0]
                smallest = sorted_metrics[-1]
                if largest[1]["total"] > 0 and smallest[1]["total"] > 0:
                    ratio = largest[1]["total"] / smallest[1]["total"]
                    stats_lines.append(f"\n**Comparison:** {largest[0]} is {ratio:.1f}x larger than {smallest[0]}")
            stats_lines.append("")
    else:
        # Fall back to key production columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        key_cols = [c for c in numeric_cols if any(x in c.upper() for x in ['PROD', 'SALES', 'VOL', 'ENERGY'])][:8]
        
        if key_cols:
            stats_lines.append("### Key Metric Statistics:")
            stats_lines.append("| Column | Total | Average | Min | Max | Count |")
            stats_lines.append("|--------|-------|---------|-----|-----|-------|")
            
            for col in key_cols:
                try:
                    data = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(data) > 0:
                        stats_lines.append(
                            f"| {col[:30]} | {data.sum():,.2f} | {data.mean():,.2f} | "
                            f"{data.min():,.2f} | {data.max():,.2f} | {len(data):,} |"
                        )
                except Exception:
                    pass
            stats_lines.append("")
    
    # Date range
    date_cols = [c for c in df.columns if 'DATE' in c.upper() or 'TIME' in c.upper()]
    for col in date_cols[:1]:
        try:
            dates = pd.to_datetime(df[col], errors='coerce').dropna()
            if len(dates) > 0:
                stats_lines.append(f"**Date Range:** {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
        except:
            pass
    
    return "\n".join(stats_lines)


def generate_dataset_overview(df: pd.DataFrame) -> str:
    """
    Generate comprehensive dataset overview.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Formatted overview string
    """
    if df is None or df.empty:
        return "No data available."
    
    overview_parts = []
    overview_parts.append("## 📘 Complete Dataset Overview\n")
    
    # Basic shape
    overview_parts.append("### 📐 Dataset Shape")
    overview_parts.append("| Metric | Value |")
    overview_parts.append("|--------|-------|")
    overview_parts.append(f"| **Total Rows** | {len(df):,} |")
    overview_parts.append(f"| **Total Columns** | {len(df.columns)} |")
    overview_parts.append(f"| **Memory Usage** | {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB |")
    overview_parts.append("")
    
    # Column categories
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    date_cols = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_cols.append(col)
    
    overview_parts.append("### 📊 Column Categories")
    overview_parts.append("| Category | Count |")
    overview_parts.append("|----------|-------|")
    overview_parts.append(f"| Numeric Columns | {len(numeric_cols)} |")
    overview_parts.append(f"| Text/Categorical Columns | {len(text_cols)} |")
    overview_parts.append(f"| Date/Time Columns | {len(date_cols)} |")
    overview_parts.append("")
    
    # Production columns
    prod_cols = [c for c in df.columns if 'PROD' in c.upper() and 'VOL' in c.upper()]
    sales_cols = [c for c in df.columns if 'SALES' in c.upper() and 'VOL' in c.upper()]
    
    overview_parts.append("### 🛢️ Metric Column Groups")
    
    if prod_cols:
        overview_parts.append(f"\n**Production Columns ({len(prod_cols)}):**")
        for col in prod_cols[:8]:
            try:
                total = pd.to_numeric(df[col], errors='coerce').dropna().sum()
                uom = ""
                if col + "_UOM" in df.columns:
                    uom_vals = df[col + "_UOM"].dropna()
                    uom = f" {uom_vals.iloc[0]}" if len(uom_vals) > 0 else ""
                overview_parts.append(f"- `{col}`: Total = {total:,.2f}{uom}")
            except:
                overview_parts.append(f"- `{col}`: (non-numeric)")
    
    if sales_cols:
        overview_parts.append(f"\n**Sales Columns ({len(sales_cols)}):**")
        for col in sales_cols[:8]:
            try:
                total = pd.to_numeric(df[col], errors='coerce').dropna().sum()
                overview_parts.append(f"- `{col}`: Total = {total:,.2f}")
            except:
                overview_parts.append(f"- `{col}`: (non-numeric)")
    
    overview_parts.append("")
    
    # Date ranges
    if date_cols:
        overview_parts.append("### 📅 Date Ranges")
        for col in date_cols[:3]:
            try:
                dates = pd.to_datetime(df[col], errors='coerce').dropna()
                if len(dates) > 0:
                    days = (dates.max() - dates.min()).days
                    overview_parts.append(f"- **{col}:** {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
                    overview_parts.append(f"  - Duration: {days} days ({days//30} months)")
            except:
                pass
        overview_parts.append("")
    
    # Missing values
    overview_parts.append("### 📋 Missing Values Analysis")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    total_cells = df.size
    completeness = ((total_cells - total_missing) / total_cells * 100) if total_cells > 0 else 0
    
    overview_parts.append(f"- **Overall Completeness:** {completeness:.1f}%")
    overview_parts.append(f"- **Total Missing Cells:** {total_missing:,} / {total_cells:,}")
    overview_parts.append("")
    
    overview_parts.append("---")
    overview_parts.append("*Use specific questions like 'Show me oil production' to explore the data in detail.*")
    
    return "\n".join(overview_parts)


def compute_quick_stats(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute quick statistics for specific columns.
    
    Args:
        df: DataFrame
        columns: Columns to compute stats for
        
    Returns:
        Dict mapping column names to their stats
    """
    stats = {}
    
    for col in columns:
        if col not in df.columns:
            continue
        
        try:
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(data) > 0:
                stats[col] = {
                    "total": float(data.sum()),
                    "mean": float(data.mean()),
                    "min": float(data.min()),
                    "max": float(data.max()),
                    "count": int(len(data)),
                    "std": float(data.std())
                }
        except:
            pass
    
    return stats
