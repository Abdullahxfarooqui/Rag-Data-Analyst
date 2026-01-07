"""
Mode Handlers for different query types.

Each handler implements the logic for a specific query mode.
"""
from typing import Dict, Any, Optional, List, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd

from core.routing.classifier import QueryMode


class ModeHandler(Protocol):
    """Protocol for mode handlers."""
    
    def handle(
        self,
        query: str,
        context: str,
        dataframe: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handle a query in this mode.
        
        Args:
            query: User query
            context: Retrieved context
            dataframe: Optional DataFrame for analysis
            **kwargs: Additional arguments
            
        Returns:
            Dict with 'answer', 'sources', and mode-specific fields
        """
        ...


@dataclass
class HandlerResult:
    """Standardized handler result."""
    answer: str
    sources: List[Dict[str, Any]]
    query_mode: str
    show_visualizations: bool
    specific_metrics: List[str]
    target_columns: List[str]
    detail_mode: str = "normal"
    stats_block: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query_mode": self.query_mode,
            "show_visualizations": self.show_visualizations,
            "specific_metrics": self.specific_metrics,
            "target_columns": self.target_columns,
            "detail_mode": self.detail_mode,
            "stats_block": self.stats_block
        }


class DataQueryHandler:
    """
    Handler for DATA_QUERY mode.
    
    Handles statistical queries, aggregations, and chart requests.
    Always returns visualizations flag as True.
    """
    
    def __init__(self, llm_client=None, stats_computer=None):
        """
        Initialize handler.
        
        Args:
            llm_client: LLM client for generating answers
            stats_computer: Statistics computer for DataFrame analysis
        """
        self._llm_client = llm_client
        self._stats_computer = stats_computer
    
    def handle(
        self,
        query: str,
        context: str,
        dataframe: Optional[pd.DataFrame] = None,
        specific_metrics: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        detail_mode: str = "normal",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handle a data query.
        
        Returns answer with statistics and visualization flag.
        """
        from core.llm.client import get_llm_client
        from core.llm.prompts import SYSTEM_PROMPT_CONCISE, SYSTEM_PROMPT_DETAILED, get_data_query_prompt
        from core.analytics.statistics import compute_data_statistics, detect_specific_metrics, get_target_columns
        
        llm_client = self._llm_client or get_llm_client()
        
        # Detect metrics if not provided
        if specific_metrics is None:
            specific_metrics = detect_specific_metrics(query)
        
        # Get target columns if not provided
        if target_columns is None and dataframe is not None:
            target_columns = get_target_columns(specific_metrics, dataframe.columns.tolist())
        
        target_columns = target_columns or []
        
        # Compute statistics
        stats_block = ""
        if dataframe is not None and not dataframe.empty:
            stats_block = compute_data_statistics(dataframe, specific_metrics, target_columns)
        
        # Select prompt based on detail mode
        is_detailed = (detail_mode == "detailed")
        system_prompt = SYSTEM_PROMPT_DETAILED if is_detailed else SYSTEM_PROMPT_CONCISE
        max_tokens = 3000 if is_detailed else 1500
        
        # Build user prompt
        user_prompt = get_data_query_prompt(
            user_query=query,
            stats_block=stats_block,
            context=context,
            is_detailed=is_detailed
        )
        
        # Call LLM
        response = llm_client.call_with_system(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens
        )
        
        answer = response.content if not response.is_error else f"Error: {response.error}"
        
        return {
            "answer": answer,
            "sources": [],
            "query_mode": QueryMode.DATA_QUERY.value,
            "show_visualizations": True,  # Always for data queries
            "specific_metrics": specific_metrics,
            "target_columns": target_columns,
            "detail_mode": detail_mode,
            "stats_block": stats_block
        }


class DocumentOverviewHandler:
    """
    Handler for DOC_OVERVIEW mode.
    
    Handles document summaries, explanations, and structure breakdowns.
    """
    
    def __init__(self, llm_client=None):
        self._llm_client = llm_client
    
    def handle(
        self,
        query: str,
        context: str,
        dataframe: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handle a document overview query.
        
        Returns comprehensive document breakdown.
        """
        from core.analytics.statistics import generate_dataset_overview
        
        # Check if executive summary is requested
        q_lower = query.lower()
        is_executive = any(phrase in q_lower for phrase in [
            "executive summary", "exec summary", "high level summary",
            "brief summary", "quick summary", "management summary"
        ])
        
        # Generate overview from DataFrame
        if dataframe is not None and not dataframe.empty:
            if is_executive:
                answer = self._generate_executive_summary(dataframe)
            else:
                answer = generate_dataset_overview(dataframe)
        else:
            answer = "📭 **No document loaded.** Please upload a document first to see its overview."
        
        # Detect available metrics for potential visualization
        detected_metrics = []
        if dataframe is not None:
            col_upper = [c.upper() for c in dataframe.columns]
            if any('OIL' in c for c in col_upper):
                detected_metrics.append('oil')
            if any('GAS' in c for c in col_upper):
                detected_metrics.append('gas')
            if any('WAT' in c for c in col_upper):
                detected_metrics.append('water')
            if any('COND' in c for c in col_upper):
                detected_metrics.append('condensate')
        
        return {
            "answer": answer,
            "sources": [],
            "query_mode": QueryMode.DOC_OVERVIEW.value,
            "show_visualizations": False,  # Overview doesn't show charts by default
            "specific_metrics": detected_metrics,
            "target_columns": list(dataframe.columns) if dataframe is not None else [],
            "detail_mode": "detailed"
        }
    
    def _generate_executive_summary(self, df: pd.DataFrame) -> str:
        """Generate executive summary for stakeholders."""
        summary_parts = []
        
        summary_parts.append("## 📋 Executive Summary\n")
        
        # Document overview
        summary_parts.append("### 📄 Document Overview")
        
        # Detect date range
        date_range_str = ""
        date_cols = [c for c in df.columns if 'DATE' in c.upper() or 'TIME' in c.upper()]
        for col in date_cols[:1]:
            try:
                dates = pd.to_datetime(df[col], errors='coerce').dropna()
                if len(dates) > 0:
                    start_date = dates.min().strftime('%B %d, %Y')
                    end_date = dates.max().strftime('%B %d, %Y')
                    date_range_str = f" covering the period from **{start_date}** to **{end_date}**"
                    break
            except:
                pass
        
        summary_parts.append(
            f"This dataset contains **{len(df):,} records** across **{len(df.columns)} data fields**{date_range_str}.\n"
        )
        
        # Key metrics
        summary_parts.append("### 📊 Key Performance Metrics\n")
        
        prod_cols = [c for c in df.columns if 'PROD' in c.upper() and 'VOL' in c.upper()]
        
        if prod_cols:
            summary_parts.append("| Metric | Total Volume | Average | Unit |")
            summary_parts.append("|--------|--------------|---------|------|")
            
            for col in prod_cols[:5]:
                try:
                    data = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(data) > 0 and data.sum() > 0:
                        total = data.sum()
                        avg = data.mean()
                        uom = "units"
                        if col + "_UOM" in df.columns:
                            uom_vals = df[col + "_UOM"].dropna()
                            uom = uom_vals.iloc[0] if len(uom_vals) > 0 else "units"
                        display_name = col.replace("PROD_", "").replace("_VOL", "").replace("_", " ").title()
                        summary_parts.append(f"| {display_name} Production | {total:,.2f} | {avg:,.2f} | {uom} |")
                except:
                    pass
        
        summary_parts.append("")
        
        # Data quality
        summary_parts.append("### 💡 Key Highlights\n")
        
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
        
        summary_parts.append(f"- **Data Completeness:** {completeness:.1f}% of all data fields are populated")
        
        key_id_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['item_name', 'well', 'facility'])]
        if key_id_cols:
            unique_count = df[key_id_cols[0]].nunique()
            col_name = key_id_cols[0].replace("_", " ").title()
            summary_parts.append(f"- **Unique {col_name}s:** {unique_count:,} distinct entries")
        
        summary_parts.append("")
        summary_parts.append("---")
        summary_parts.append("*Ask specific questions like 'Show oil production trends' for detailed analysis.*")
        
        return "\n".join(summary_parts)


class FreeformHandler:
    """
    Handler for FREEFORM_QUERY mode.
    
    Returns a polite refusal for non-data queries.
    """
    
    def handle(
        self,
        query: str,
        context: str,
        dataframe: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handle a freeform (non-data) query.
        
        Returns refusal message with guidance.
        """
        from core.llm.prompts import get_freeform_response
        
        return {
            "answer": get_freeform_response(),
            "sources": [],
            "query_mode": QueryMode.FREEFORM_QUERY.value,
            "show_visualizations": False,
            "specific_metrics": [],
            "target_columns": [],
            "detail_mode": "normal"
        }


class SystemTaskHandler:
    """
    Handler for SYSTEM_TASK mode.
    
    Provides guidance about system usage.
    """
    
    def handle(
        self,
        query: str,
        context: str,
        dataframe: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handle a system task query.
        
        Returns system guidance message.
        """
        from core.llm.prompts import get_system_task_response
        
        return {
            "answer": get_system_task_response(),
            "sources": [],
            "query_mode": QueryMode.SYSTEM_TASK.value,
            "show_visualizations": False,
            "specific_metrics": [],
            "target_columns": [],
            "detail_mode": "normal"
        }


def get_handler_for_mode(mode: QueryMode, **kwargs) -> ModeHandler:
    """
    Factory function to get the appropriate handler for a mode.
    
    Args:
        mode: The QueryMode to handle
        **kwargs: Arguments passed to handler constructor
        
    Returns:
        Appropriate ModeHandler instance
    """
    handlers = {
        QueryMode.DATA_QUERY: DataQueryHandler,
        QueryMode.DOC_OVERVIEW: DocumentOverviewHandler,
        QueryMode.FREEFORM_QUERY: FreeformHandler,
        QueryMode.SYSTEM_TASK: SystemTaskHandler
    }
    
    handler_class = handlers.get(mode, FreeformHandler)
    return handler_class(**kwargs)
