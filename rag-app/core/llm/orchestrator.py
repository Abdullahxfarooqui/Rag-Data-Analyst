"""
LLM Orchestrator - Production-Grade Tool-Based Architecture.

CORE PRINCIPLES:
1. SEPARATION: Deterministic analytics (Python) vs Narrative (LLM)
2. TOOLS: Each LLM capability is a typed, validated tool
3. ASYNC: Non-blocking operations for UX
4. STRUCTURED: All outputs are schema-validated JSON
5. FALLBACK: Graceful degradation when LLM fails

This module transforms the system from "LLM bolted onto analytics"
to "Analytics-first with AI-powered insights".
"""
import asyncio
import json
import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import (
    Any, Callable, Dict, List, Optional, Type, TypeVar, 
    Union, Generic, Awaitable
)
from enum import Enum, auto
from functools import wraps
import time

# ============================================================================
# LOGGING
# ============================================================================
logger = logging.getLogger(__name__)


# ============================================================================
# STRUCTURED OUTPUT SCHEMAS
# ============================================================================

@dataclass
class MetricSummary:
    """Schema for summarized metric data."""
    name: str
    total: float
    average: float
    min_value: float
    max_value: float
    unit: str = ""
    record_count: int = 0
    null_percentage: float = 0.0


@dataclass
class Comparison:
    """Schema for metric comparisons."""
    metric_a: str
    metric_b: str
    ratio: float
    difference: float
    larger: str
    insight: str


@dataclass
class Risk:
    """Schema for identified risks/anomalies."""
    severity: str  # "low", "medium", "high", "critical"
    category: str  # "data_quality", "operational", "trend"
    description: str
    affected_metric: str
    recommendation: str


@dataclass
class Trend:
    """Schema for trend analysis."""
    metric: str
    direction: str  # "increasing", "decreasing", "stable", "volatile"
    change_percent: float
    period: str
    confidence: float


@dataclass
class StructuredInsight:
    """
    MANDATORY OUTPUT SCHEMA for all LLM responses.
    
    This enforces consistent, parseable output that never breaks the UI.
    """
    summary: str
    key_metrics: List[MetricSummary] = field(default_factory=list)
    comparisons: List[Comparison] = field(default_factory=list)
    trends: List[Trend] = field(default_factory=list)
    risks: List[Risk] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence_level: str = "medium"  # "low", "medium", "high"
    data_quality_score: float = 1.0
    generated_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": self.summary,
            "key_metrics": [asdict(m) for m in self.key_metrics],
            "comparisons": [asdict(c) for c in self.comparisons],
            "trends": [asdict(t) for t in self.trends],
            "risks": [asdict(r) for r in self.risks],
            "recommendations": self.recommendations,
            "confidence_level": self.confidence_level,
            "data_quality_score": self.data_quality_score,
            "generated_at": self.generated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredInsight":
        """Create from dictionary with validation."""
        return cls(
            summary=data.get("summary", ""),
            key_metrics=[MetricSummary(**m) for m in data.get("key_metrics", [])],
            comparisons=[Comparison(**c) for c in data.get("comparisons", [])],
            trends=[Trend(**t) for t in data.get("trends", [])],
            risks=[Risk(**r) for r in data.get("risks", [])],
            recommendations=data.get("recommendations", []),
            confidence_level=data.get("confidence_level", "medium"),
            data_quality_score=data.get("data_quality_score", 1.0),
            generated_at=data.get("generated_at", "")
        )
    
    @classmethod
    def fallback(cls, error_msg: str = "") -> "StructuredInsight":
        """Create a fallback response when LLM fails."""
        return cls(
            summary=(
                "⚠️ Automated insights could not be generated. "
                "However, the data analysis and visualizations completed successfully. "
                "Please review the charts and statistics above for insights."
            ),
            confidence_level="low",
            data_quality_score=0.0,
            recommendations=[
                "Review the visualizations for patterns",
                "Check the computed statistics for exact values",
                "Try a more specific question"
            ]
        )


# ============================================================================
# TOOL DEFINITIONS - Each is a separate, typed capability
# ============================================================================

class ToolType(Enum):
    """Categories of LLM tools."""
    SUMMARIZE = auto()
    COMPARE = auto()
    ANALYZE_TRENDS = auto()
    DETECT_ANOMALIES = auto()
    GENERATE_INSIGHTS = auto()
    EXPLAIN_RANKINGS = auto()
    ANSWER_QUESTION = auto()


@dataclass
class ToolDefinition:
    """
    Schema for LLM tool definition.
    
    This enables function-calling style interactions where the LLM
    knows exactly what inputs it needs and what output format to use.
    """
    name: str
    description: str
    tool_type: ToolType
    input_schema: Dict[str, Any]
    output_schema: Type
    requires_llm: bool = True  # False for pure Python tools
    cacheable: bool = True
    max_tokens: int = 1500
    temperature: float = 0.1


# Define all available tools
TOOLS: Dict[str, ToolDefinition] = {
    "summarize_metrics": ToolDefinition(
        name="summarize_metrics",
        description="Generate a natural language summary of computed statistics",
        tool_type=ToolType.SUMMARIZE,
        input_schema={
            "type": "object",
            "properties": {
                "stats_json": {"type": "object", "description": "Pre-computed statistics"},
                "metric_names": {"type": "array", "items": {"type": "string"}},
                "context": {"type": "string", "description": "Additional context"}
            },
            "required": ["stats_json"]
        },
        output_schema=StructuredInsight,
        max_tokens=1000
    ),
    
    "compare_metrics": ToolDefinition(
        name="compare_metrics",
        description="Compare two or more metrics and explain the relationship",
        tool_type=ToolType.COMPARE,
        input_schema={
            "type": "object",
            "properties": {
                "metric_a": {"type": "object", "description": "First metric stats"},
                "metric_b": {"type": "object", "description": "Second metric stats"},
                "comparison_type": {"type": "string", "enum": ["ratio", "difference", "trend"]}
            },
            "required": ["metric_a", "metric_b"]
        },
        output_schema=StructuredInsight,
        max_tokens=800
    ),
    
    "analyze_trends": ToolDefinition(
        name="analyze_trends",
        description="Analyze time-series trends in the data",
        tool_type=ToolType.ANALYZE_TRENDS,
        input_schema={
            "type": "object",
            "properties": {
                "trend_data": {"type": "array", "description": "Time-series data points"},
                "metric_name": {"type": "string"},
                "period": {"type": "string", "description": "Time period analyzed"}
            },
            "required": ["trend_data", "metric_name"]
        },
        output_schema=StructuredInsight,
        max_tokens=1000
    ),
    
    "detect_anomalies": ToolDefinition(
        name="detect_anomalies",
        description="Explain detected anomalies and suggest causes",
        tool_type=ToolType.DETECT_ANOMALIES,
        input_schema={
            "type": "object",
            "properties": {
                "anomalies": {"type": "array", "description": "List of detected anomalies"},
                "context": {"type": "object", "description": "Dataset context"}
            },
            "required": ["anomalies"]
        },
        output_schema=StructuredInsight,
        max_tokens=1200
    ),
    
    "generate_insights": ToolDefinition(
        name="generate_insights",
        description="Generate business insights from analytics results",
        tool_type=ToolType.GENERATE_INSIGHTS,
        input_schema={
            "type": "object",
            "properties": {
                "stats": {"type": "object", "description": "Computed statistics"},
                "anomalies": {"type": "array"},
                "rankings": {"type": "array"},
                "trends": {"type": "array"},
                "query": {"type": "string", "description": "User's original question"}
            },
            "required": ["stats"]
        },
        output_schema=StructuredInsight,
        max_tokens=2000,
        temperature=0.2
    ),
    
    "explain_rankings": ToolDefinition(
        name="explain_rankings",
        description="Explain why certain items rank higher/lower",
        tool_type=ToolType.EXPLAIN_RANKINGS,
        input_schema={
            "type": "object",
            "properties": {
                "rankings": {"type": "array", "description": "Ranked items with scores"},
                "metric": {"type": "string", "description": "Metric used for ranking"},
                "top_n": {"type": "integer", "default": 5}
            },
            "required": ["rankings", "metric"]
        },
        output_schema=StructuredInsight,
        max_tokens=1000
    ),
    
    "answer_question": ToolDefinition(
        name="answer_question",
        description="Answer a specific user question using provided context",
        tool_type=ToolType.ANSWER_QUESTION,
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "context": {"type": "string", "description": "RAG context"},
                "stats": {"type": "object", "description": "Pre-computed stats"}
            },
            "required": ["question", "context"]
        },
        output_schema=StructuredInsight,
        max_tokens=1500
    )
}


# ============================================================================
# TOOL PROMPTS - Structured prompts for each tool
# ============================================================================

TOOL_PROMPTS = {
    "summarize_metrics": """You are a data analyst summarizing statistics.

INPUT STATISTICS (pre-computed by Python - these are EXACT):
{stats_json}

TASK: Generate a natural language summary of these statistics.

OUTPUT FORMAT (respond ONLY with this JSON):
{{
    "summary": "2-3 sentence summary of the key findings",
    "key_metrics": [
        {{"name": "metric_name", "total": 0, "average": 0, "unit": "bbl", "insight": "brief insight"}}
    ],
    "recommendations": ["actionable recommendation 1", "recommendation 2"],
    "confidence_level": "high"
}}

RULES:
- Use EXACT values from the statistics
- Do NOT estimate or guess numbers
- Keep summary under 100 words
- Focus on business relevance""",

    "compare_metrics": """You are comparing two metrics from production data.

METRIC A: {metric_a}
METRIC B: {metric_b}

COMPARISON TYPE: {comparison_type}

OUTPUT FORMAT (respond ONLY with this JSON):
{{
    "summary": "Brief comparison summary",
    "comparisons": [
        {{
            "metric_a": "name",
            "metric_b": "name", 
            "ratio": 1.5,
            "difference": 1000,
            "larger": "metric_a",
            "insight": "What this comparison means"
        }}
    ],
    "recommendations": ["Based on this comparison, consider..."],
    "confidence_level": "high"
}}

RULES:
- Calculate ratio and difference from provided values
- Explain what the comparison means operationally
- Suggest actions based on the comparison""",

    "generate_insights": """You are generating business insights from data analytics.

USER QUESTION: {query}

PRE-COMPUTED STATISTICS (EXACT - computed by Python):
{stats}

DETECTED ANOMALIES:
{anomalies}

DETECTED TRENDS:
{trends}

TOP RANKINGS:
{rankings}

OUTPUT FORMAT (respond ONLY with this JSON):
{{
    "summary": "Executive summary answering the user's question",
    "key_metrics": [
        {{"name": "metric", "total": 0, "average": 0, "unit": "unit", "record_count": 0}}
    ],
    "trends": [
        {{"metric": "name", "direction": "increasing", "change_percent": 5.2, "period": "monthly", "confidence": 0.85}}
    ],
    "risks": [
        {{"severity": "medium", "category": "operational", "description": "...", "affected_metric": "...", "recommendation": "..."}}
    ],
    "recommendations": ["actionable recommendation"],
    "confidence_level": "high"
}}

RULES:
- Answer the user's SPECIFIC question first
- Use ONLY the provided statistics - no estimation
- Be specific with numbers
- Provide actionable recommendations"""
}


# ============================================================================
# LLM ORCHESTRATOR - Main coordination class
# ============================================================================

class LLMOrchestrator:
    """
    Production-grade LLM orchestrator with tool-based architecture.
    
    Features:
    - Tool routing based on query intent
    - Structured output enforcement
    - Async execution support
    - Caching at query level
    - Graceful fallback on failures
    
    Usage:
        orchestrator = LLMOrchestrator()
        result = await orchestrator.execute_tool(
            "generate_insights",
            stats=computed_stats,
            query="What is oil production?"
        )
    """
    
    def __init__(self, llm_client=None, cache_enabled: bool = True):
        """
        Initialize the orchestrator.
        
        Args:
            llm_client: Optional LLM client (uses default if None)
            cache_enabled: Whether to cache LLM responses
        """
        self._llm_client = llm_client
        self._cache_enabled = cache_enabled
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 3600  # 1 hour
        self._cache_timestamps: Dict[str, float] = {}
    
    @property
    def llm_client(self):
        """Lazy-load LLM client."""
        if self._llm_client is None:
            from core.llm.client import get_llm_client
            self._llm_client = get_llm_client()
        return self._llm_client
    
    def _get_cache_key(self, tool_name: str, **kwargs) -> str:
        """Generate cache key from tool name and parameters."""
        param_str = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.md5(f"{tool_name}:{param_str}".encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[StructuredInsight]:
        """Check if result is cached and not expired."""
        if not self._cache_enabled:
            return None
        
        if cache_key not in self._cache:
            return None
        
        timestamp = self._cache_timestamps.get(cache_key, 0)
        if time.time() - timestamp > self._cache_ttl:
            # Expired
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None
        
        logger.info(f"Cache hit for {cache_key[:8]}...")
        return self._cache[cache_key]
    
    def _set_cache(self, cache_key: str, result: StructuredInsight):
        """Store result in cache."""
        if self._cache_enabled:
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()
    
    def get_tool(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name."""
        return TOOLS.get(tool_name)
    
    def select_tool(self, query: str, available_data: Dict[str, bool]) -> str:
        """
        Auto-select the best tool based on query and available data.
        
        Args:
            query: User's question
            available_data: Dict indicating what data is available
                           {"stats": True, "trends": True, "anomalies": False}
        
        Returns:
            Tool name to use
        """
        q = query.lower()
        
        # Comparison queries
        if any(w in q for w in ["compare", "vs", "versus", "difference", "ratio"]):
            return "compare_metrics"
        
        # Trend queries
        if any(w in q for w in ["trend", "over time", "change", "growth", "pattern"]):
            return "analyze_trends"
        
        # Anomaly queries
        if any(w in q for w in ["anomaly", "outlier", "unusual", "abnormal"]):
            return "detect_anomalies"
        
        # Ranking queries
        if any(w in q for w in ["top", "best", "worst", "highest", "lowest", "rank"]):
            return "explain_rankings"
        
        # Summary queries
        if any(w in q for w in ["summary", "summarize", "overview", "total"]):
            return "summarize_metrics"
        
        # Default to general insights
        return "generate_insights"
    
    def _build_prompt(self, tool_name: str, **kwargs) -> str:
        """Build the prompt for a tool with provided parameters."""
        template = TOOL_PROMPTS.get(tool_name)
        if not template:
            # Generic template
            template = """Analyze the following data and provide insights.

DATA: {data}

Respond with a JSON object containing:
- summary: Brief summary
- key_metrics: List of important metrics
- recommendations: List of actionable recommendations
- confidence_level: "low", "medium", or "high"
"""
        
        # Format template with kwargs
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            return template
    
    def _parse_llm_response(self, response_text: str) -> StructuredInsight:
        """
        Parse LLM response into structured format.
        
        Handles:
        - Clean JSON responses
        - JSON wrapped in markdown code blocks
        - Partial/malformed JSON
        - Complete failures
        """
        if not response_text:
            return StructuredInsight.fallback("Empty response")
        
        # Try to extract JSON from response
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines if they're code block markers
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        
        # Try to parse as JSON
        try:
            data = json.loads(text)
            return StructuredInsight.from_dict(data)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in text
        import re
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return StructuredInsight.from_dict(data)
            except json.JSONDecodeError:
                pass
        
        # Last resort: create insight from plain text
        logger.warning("Could not parse JSON, using plain text response")
        return StructuredInsight(
            summary=text[:500],
            confidence_level="low"
        )
    
    def execute_tool_sync(
        self,
        tool_name: str,
        **kwargs
    ) -> StructuredInsight:
        """
        Execute a tool synchronously.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Parameters for the tool
            
        Returns:
            StructuredInsight with results
        """
        tool = self.get_tool(tool_name)
        if not tool:
            logger.error(f"Unknown tool: {tool_name}")
            return StructuredInsight.fallback(f"Unknown tool: {tool_name}")
        
        # Check cache
        cache_key = self._get_cache_key(tool_name, **kwargs)
        cached = self._check_cache(cache_key)
        if cached:
            return cached
        
        # Build prompt
        prompt = self._build_prompt(tool_name, **kwargs)
        
        # Call LLM
        try:
            logger.info(f"Executing tool: {tool_name}")
            
            response = self.llm_client.call_with_system(
                system_prompt=(
                    "You are a data analysis AI. Respond ONLY with valid JSON. "
                    "No markdown, no explanation, just the JSON object."
                ),
                user_prompt=prompt,
                max_tokens=tool.max_tokens,
                temperature=tool.temperature
            )
            
            if response.is_error:
                logger.error(f"LLM error: {response.error}")
                return StructuredInsight.fallback(response.error)
            
            # Parse response
            result = self._parse_llm_response(response.content)
            
            # Cache if successful
            if result.confidence_level != "low" and tool.cacheable:
                self._set_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.exception(f"Tool execution failed: {e}")
            return StructuredInsight.fallback(str(e))
    
    async def execute_tool(
        self,
        tool_name: str,
        **kwargs
    ) -> StructuredInsight:
        """
        Execute a tool asynchronously.
        
        This allows the UI to remain responsive while the LLM processes.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Parameters for the tool
            
        Returns:
            StructuredInsight with results
        """
        # Run sync version in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.execute_tool_sync(tool_name, **kwargs)
        )
    
    def clear_cache(self):
        """Clear all cached results."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Cache cleared")


# ============================================================================
# TOOL ROUTER - Intelligent routing based on query analysis
# ============================================================================

class ToolRouter:
    """
    Routes queries to appropriate tools based on:
    - Query intent
    - Available data
    - Complexity requirements
    
    Also determines when to BYPASS LLM entirely.
    """
    
    # Queries that can be answered with pure Python (no LLM needed)
    DETERMINISTIC_PATTERNS = [
        r"^what is (the )?total",
        r"^how many",
        r"^count of",
        r"^sum of",
        r"^average of",
        r"^min(imum)? of",
        r"^max(imum)? of",
        r"^show (me )?(the )?data",
        r"^list (all )?(the )?columns",
    ]
    
    def __init__(self, orchestrator: LLMOrchestrator):
        self.orchestrator = orchestrator
        self._pattern_cache = {}
    
    def should_bypass_llm(self, query: str) -> bool:
        """
        Determine if this query can be answered without LLM.
        
        CRITICAL FOR LATENCY: Pure aggregations and simple lookups
        should NEVER hit the LLM.
        
        Returns:
            True if query can be answered deterministically
        """
        import re
        q = query.lower().strip()
        
        for pattern in self.DETERMINISTIC_PATTERNS:
            if re.match(pattern, q):
                logger.info(f"Query bypasses LLM (deterministic): {q[:50]}...")
                return True
        
        return False
    
    def route(
        self,
        query: str,
        stats: Optional[Dict] = None,
        trends: Optional[List] = None,
        anomalies: Optional[List] = None,
        rankings: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Route a query to the appropriate tool(s).
        
        Returns:
            Dict with:
            - "bypass_llm": bool
            - "tool": tool name if LLM needed
            - "params": parameters for the tool
            - "fallback_response": response if bypassing LLM
        """
        # Check for deterministic bypass
        if self.should_bypass_llm(query):
            return {
                "bypass_llm": True,
                "tool": None,
                "params": {},
                "fallback_response": self._generate_deterministic_response(
                    query, stats
                )
            }
        
        # Determine available data
        available = {
            "stats": stats is not None,
            "trends": bool(trends),
            "anomalies": bool(anomalies),
            "rankings": bool(rankings)
        }
        
        # Select tool
        tool_name = self.orchestrator.select_tool(query, available)
        
        # Build parameters
        params = {
            "query": query,
            "stats": json.dumps(stats, default=str) if stats else "{}",
            "trends": json.dumps(trends, default=str) if trends else "[]",
            "anomalies": json.dumps(anomalies, default=str) if anomalies else "[]",
            "rankings": json.dumps(rankings, default=str) if rankings else "[]"
        }
        
        return {
            "bypass_llm": False,
            "tool": tool_name,
            "params": params
        }
    
    def _generate_deterministic_response(
        self,
        query: str,
        stats: Optional[Dict]
    ) -> StructuredInsight:
        """
        Generate response for deterministic queries without LLM.
        
        This is INSTANT and uses Python-computed values.
        """
        if not stats:
            return StructuredInsight(
                summary="No statistics available. Please upload a document first.",
                confidence_level="high"
            )
        
        q = query.lower()
        
        # Total queries
        if "total" in q:
            totals = []
            for key, value in stats.items():
                if isinstance(value, dict) and "total" in value:
                    totals.append(MetricSummary(
                        name=key,
                        total=value["total"],
                        average=value.get("average", 0),
                        min_value=value.get("min", 0),
                        max_value=value.get("max", 0),
                        unit=value.get("unit", ""),
                        record_count=value.get("count", 0)
                    ))
            
            return StructuredInsight(
                summary=f"Found {len(totals)} metrics with computed totals.",
                key_metrics=totals,
                confidence_level="high",
                data_quality_score=1.0
            )
        
        # Count queries
        if "count" in q or "how many" in q:
            return StructuredInsight(
                summary=f"Dataset contains {stats.get('total_rows', 'unknown')} records.",
                confidence_level="high"
            )
        
        # Generic stats response
        return StructuredInsight(
            summary="Statistics computed successfully.",
            confidence_level="high"
        )


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_orchestrator: Optional[LLMOrchestrator] = None
_router: Optional[ToolRouter] = None


def get_orchestrator() -> LLMOrchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = LLMOrchestrator()
    return _orchestrator


def get_router() -> ToolRouter:
    """Get the global router instance."""
    global _router
    if _router is None:
        _router = ToolRouter(get_orchestrator())
    return _router


def reset_orchestrator():
    """Reset the global orchestrator (for testing)."""
    global _orchestrator, _router
    _orchestrator = None
    _router = None
