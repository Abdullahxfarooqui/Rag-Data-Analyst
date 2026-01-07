"""
Semantic Query Classifier using LLM.

Classifies queries into one of four modes using NVIDIA Nemotron.
Replaces keyword-based routing with semantic understanding.
"""
import json
import re
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from core.llm.client import LLMClient, get_llm_client
from core.llm.prompts import ROUTING_PROMPT_TEMPLATE


class QueryMode(str, Enum):
    """Query classification modes."""
    DATA_QUERY = "DATA_QUERY"
    DOC_OVERVIEW = "DOC_OVERVIEW"
    FREEFORM_QUERY = "FREEFORM_QUERY"
    SYSTEM_TASK = "SYSTEM_TASK"
    
    @classmethod
    def from_string(cls, s: str) -> "QueryMode":
        """Parse mode from string, with fallback."""
        s_upper = s.upper().strip()
        
        # Handle variations
        if s_upper in ("DATA_QUERY", "DATA", "QUERY"):
            return cls.DATA_QUERY
        elif s_upper in ("DOC_OVERVIEW", "DOCUMENT_OVERVIEW", "OVERVIEW"):
            return cls.DOC_OVERVIEW
        elif s_upper in ("FREEFORM_QUERY", "FREEFORM", "GENERAL"):
            return cls.FREEFORM_QUERY
        elif s_upper in ("SYSTEM_TASK", "SYSTEM"):
            return cls.SYSTEM_TASK
        else:
            return cls.FREEFORM_QUERY  # Safe fallback


@dataclass
class ClassificationResult:
    """Result of query classification."""
    mode: QueryMode
    confidence: float
    reason: str
    raw_response: Optional[str] = None
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if classification has high confidence (>= 0.6)."""
        return self.confidence >= 0.6
    
    @property
    def should_use_rag(self) -> bool:
        """Check if this mode requires RAG pipeline."""
        return self.mode in (QueryMode.DATA_QUERY, QueryMode.DOC_OVERVIEW)
    
    @property
    def show_visualizations(self) -> bool:
        """Check if this mode should show visualizations."""
        return self.mode == QueryMode.DATA_QUERY
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "should_use_rag": self.should_use_rag,
            "show_visualizations": self.show_visualizations
        }


class QueryClassifier:
    """
    Semantic query classifier using NVIDIA Nemotron.
    
    Uses LLM to classify queries into four modes with confidence scores.
    Low-confidence classifications are routed to FREEFORM_QUERY.
    """
    
    def __init__(
        self, 
        llm_client: Optional[LLMClient] = None,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize classifier.
        
        Args:
            llm_client: LLMClient instance (uses global if None)
            confidence_threshold: Threshold below which to route to FREEFORM_QUERY
        """
        self._llm_client = llm_client
        self.CONFIDENCE_THRESHOLD = confidence_threshold
    
    @property
    def llm_client(self) -> LLMClient:
        """Get the LLM client."""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client
    
    def classify(
        self, 
        query: str,
        df_columns: Optional[list] = None
    ) -> ClassificationResult:
        """
        Classify a user query into a mode.
        
        Args:
            query: The user's query string
            df_columns: Optional list of DataFrame column names for context
            
        Returns:
            ClassificationResult with mode, confidence, and reason
        """
        # Build the routing prompt
        prompt = ROUTING_PROMPT_TEMPLATE.format(query=query)
        
        # Call LLM for classification (low temperature for determinism)
        response = self.llm_client.classify(prompt, temperature=0.0)
        
        if response.is_error:
            # On LLM error, default to DATA_QUERY (safe for RAG system)
            return ClassificationResult(
                mode=QueryMode.DATA_QUERY,
                confidence=0.5,
                reason=f"LLM error, defaulting to DATA_QUERY: {response.error}",
                raw_response=None
            )
        
        # Parse the JSON response
        result = self._parse_response(response.content)
        result.raw_response = response.content
        
        # Enforce confidence threshold
        if result.confidence < self.CONFIDENCE_THRESHOLD:
            return ClassificationResult(
                mode=QueryMode.FREEFORM_QUERY,
                confidence=result.confidence,
                reason=f"Low confidence ({result.confidence:.2f}), routing to FREEFORM: {result.reason}",
                raw_response=response.content
            )
        
        return result
    
    def _parse_response(self, response_text: str) -> ClassificationResult:
        """
        Parse LLM response into ClassificationResult.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            ClassificationResult (may have default values on parse error)
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group())
                
                mode = QueryMode.from_string(data.get("mode", "FREEFORM_QUERY"))
                confidence = float(data.get("confidence", 0.5))
                reason = str(data.get("reason", "No reason provided"))
                
                # Clamp confidence to [0, 1]
                confidence = max(0.0, min(1.0, confidence))
                
                return ClassificationResult(
                    mode=mode,
                    confidence=confidence,
                    reason=reason
                )
            
            # No JSON found, try to extract mode from text
            response_upper = response_text.upper()
            
            if "DATA_QUERY" in response_upper:
                mode = QueryMode.DATA_QUERY
            elif "DOC_OVERVIEW" in response_upper or "DOCUMENT_OVERVIEW" in response_upper:
                mode = QueryMode.DOC_OVERVIEW
            elif "SYSTEM_TASK" in response_upper:
                mode = QueryMode.SYSTEM_TASK
            else:
                mode = QueryMode.FREEFORM_QUERY
            
            return ClassificationResult(
                mode=mode,
                confidence=0.5,
                reason="Extracted from non-JSON response"
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Parse error, return safe default
            return ClassificationResult(
                mode=QueryMode.DATA_QUERY,
                confidence=0.5,
                reason=f"Parse error: {str(e)}"
            )
    
    def classify_with_fallback(
        self,
        query: str,
        df_columns: Optional[list] = None
    ) -> ClassificationResult:
        """
        Classify with keyword fallback for when LLM is unavailable.
        
        This provides a safety net but should not be the primary path.
        
        Args:
            query: User query
            df_columns: Optional list of DataFrame column names
            
        Returns:
            ClassificationResult
        """
        # Try LLM classification first
        try:
            result = self.classify(query)
            if result.confidence >= self.CONFIDENCE_THRESHOLD:
                return result
        except Exception:
            pass
        
        # Fallback to simple heuristics
        return self._keyword_fallback(query, df_columns)
    
    def _keyword_fallback(
        self,
        query: str,
        df_columns: Optional[list] = None
    ) -> ClassificationResult:
        """
        Simple keyword-based classification fallback.
        
        Only used when LLM is unavailable.
        """
        q_lower = query.lower()
        
        # Check for overview patterns
        overview_patterns = [
            "what's in this", "what is in this", "describe", "overview",
            "summarize", "summary", "explain this", "tell me about"
        ]
        if any(p in q_lower for p in overview_patterns):
            return ClassificationResult(
                mode=QueryMode.DOC_OVERVIEW,
                confidence=0.7,
                reason="Keyword match: overview pattern"
            )
        
        # Check for system patterns
        system_patterns = ["fix", "bug", "error", "improve", "modify", "how to use"]
        if any(p in q_lower for p in system_patterns):
            return ClassificationResult(
                mode=QueryMode.SYSTEM_TASK,
                confidence=0.7,
                reason="Keyword match: system pattern"
            )
        
        # Check for data-related keywords
        data_keywords = [
            "oil", "gas", "water", "production", "sales", "total", "average",
            "chart", "plot", "trend", "compare", "statistics", "volume"
        ]
        if any(k in q_lower for k in data_keywords):
            return ClassificationResult(
                mode=QueryMode.DATA_QUERY,
                confidence=0.7,
                reason="Keyword match: data pattern"
            )
        
        # Check if column names appear in query
        if df_columns:
            for col in df_columns:
                if col.lower() in q_lower:
                    return ClassificationResult(
                        mode=QueryMode.DATA_QUERY,
                        confidence=0.8,
                        reason=f"Column name match: {col}"
                    )
        
        # Default to freeform
        return ClassificationResult(
            mode=QueryMode.FREEFORM_QUERY,
            confidence=0.5,
            reason="No pattern match, defaulting to FREEFORM"
        )


# Singleton
_query_classifier: Optional[QueryClassifier] = None


def get_query_classifier() -> QueryClassifier:
    """Get global query classifier instance."""
    global _query_classifier
    if _query_classifier is None:
        _query_classifier = QueryClassifier()
    return _query_classifier


def reset_query_classifier() -> None:
    """Reset global instance (for testing)."""
    global _query_classifier
    _query_classifier = None


# Backward compatibility functions
def classify_query(query: str) -> Tuple[str, float]:
    """Legacy function for backward compatibility."""
    result = get_query_classifier().classify(query)
    return (result.mode.value, result.confidence)


def should_use_rag(query: str) -> bool:
    """Legacy function - check if query should use RAG pipeline."""
    result = get_query_classifier().classify(query)
    return result.should_use_rag


def get_query_classification(query: str) -> Dict[str, Any]:
    """Legacy function - get detailed classification."""
    result = get_query_classifier().classify(query)
    return result.to_dict()
