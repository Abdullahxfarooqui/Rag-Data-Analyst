"""
LLM Client for OpenRouter API.

Features:
- Uses OpenRouter with free NVIDIA model
- Streaming support
- Configurable parameters
- Error handling with retries
- Safe response extraction
"""
import os
import json
import requests
import logging
from typing import Optional, List, Dict, Any, Generator, Callable, Union
from dataclasses import dataclass, field


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""
    # OpenRouter settings
    model: str = "nvidia/nemotron-3-nano-30b-a3b:free"
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    max_tokens: int = 2000
    temperature: float = 0.1
    timeout: int = 120
    referer: str = "https://rag-data-analyst.streamlit.app"
    app_title: str = "RAG Data Analyst"


@dataclass
class LLMResponse:
    """Structured response from LLM."""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    raw_response: Optional[Dict] = None
    
    @property
    def is_error(self) -> bool:
        return self.error is not None
    
    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)
    
    @property
    def success(self) -> bool:
        return self.error is None


def extract_llm_text(response: Union[Dict, Any], model: str = "unknown", fallback_message: str = None) -> LLMResponse:
    """
    Safely extract text content from any LLM API response format.
    
    Handles:
    - OpenAI ChatCompletion format: {"choices": [{"message": {"content": "..."}}]}
    - Error responses: {"error": {"message": "...", "type": "..."}}
    - Empty or malformed responses
    
    Args:
        response: Raw API response dict
        model: Model name for response object
        fallback_message: Message to return if extraction fails
        
    Returns:
        LLMResponse with extracted content or error details
    """
    default_fallback = (
        "Automated insights could not be generated, but data analysis "
        "and visualizations completed successfully."
    )
    fallback = fallback_message or default_fallback
    
    # Handle None response
    if response is None:
        logger.error("LLM response is None")
        return LLMResponse(
            content="",
            model=model,
            error="API returned null response",
            error_type="null_response"
        )
    
    # Ensure we have a dict
    if not isinstance(response, dict):
        logger.error(f"Unexpected response type: {type(response)}")
        return LLMResponse(
            content="",
            model=model,
            error=f"Expected dict, got {type(response).__name__}",
            error_type="invalid_type"
        )
    
    # Check for API error responses
    if "error" in response:
        error_obj = response["error"]
        if isinstance(error_obj, dict):
            error_type = error_obj.get("type", "unknown_error")
            error_message = error_obj.get("message", str(error_obj))
        else:
            error_type = "api_error"
            error_message = str(error_obj)
        
        logger.error(f"LLM API error: [{error_type}] {error_message}")
        return LLMResponse(
            content="",
            model=model,
            error=error_message,
            error_type=error_type,
            raw_response=response
        )
    
    # Extract from choices
    choices = response.get("choices")
    
    if choices is None:
        logger.error(f"Response missing 'choices' key. Keys: {list(response.keys())}")
        return LLMResponse(
            content="",
            model=response.get("model", model),
            error="Response missing 'choices' key",
            error_type="missing_choices",
            raw_response=response
        )
    
    if not isinstance(choices, list) or len(choices) == 0:
        logger.error(f"'choices' is empty or not a list")
        return LLMResponse(
            content="",
            model=response.get("model", model),
            error="'choices' is empty or invalid",
            error_type="empty_choices",
            raw_response=response
        )
    
    # Extract content
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return LLMResponse(
            content="",
            model=response.get("model", model),
            error="Invalid choice format",
            error_type="invalid_choice",
            raw_response=response
        )
    
    # Try standard message format
    message = first_choice.get("message", {})
    content = message.get("content", "")
    
    # Try delta format (streaming)
    if not content:
        delta = first_choice.get("delta", {})
        content = delta.get("content", "")
    
    # Try text format (older API)
    if not content:
        content = first_choice.get("text", "")
    
    return LLMResponse(
        content=content or "",
        model=response.get("model", model),
        usage=response.get("usage", {}),
        finish_reason=first_choice.get("finish_reason"),
        raw_response=response
    )


class LLMClient:
    """
    OpenRouter LLM client with free NVIDIA model.
    
    Features:
    - Synchronous and streaming calls
    - Configurable via LLMConfig
    - Thread-safe
    - Structured responses
    """
    
    def __init__(self, config: Optional[LLMConfig] = None, api_key: Optional[str] = None):
        """
        Initialize the LLM client.
        
        Args:
            config: LLMConfig instance (uses defaults if None)
            api_key: API key for OpenRouter (reads from env/secrets if None)
        """
        self.config = config or LLMConfig()
        self._api_key = api_key or self._get_api_key()
    
    @staticmethod
    def _get_api_key() -> str:
        """Get API key from Streamlit secrets or environment."""
        # Try Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and "OPENROUTER_API_KEY" in st.secrets:
                return st.secrets["OPENROUTER_API_KEY"]
        except Exception:
            pass
        
        # Fall back to environment variable
        return os.environ.get("OPENROUTER_API_KEY", "")
    
    @property
    def api_key(self) -> str:
        """Get the current API key (refreshes if needed)."""
        if not self._api_key:
            self._api_key = self._get_api_key()
        return self._api_key
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers for OpenRouter."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.config.referer,
            "X-Title": self.config.app_title
        }
    
    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Build request payload."""
        return {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "stream": stream
        }
    
    def call(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> LLMResponse:
        """
        Make a synchronous LLM call.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Override default max_tokens
            temperature: Override default temperature
            
        Returns:
            LLMResponse with content or error
        """
        if not self.api_key:
            return LLMResponse(
                content="",
                model=self.config.model,
                error="No API key configured. Please set OPENROUTER_API_KEY in Streamlit secrets.",
                error_type="no_api_key"
            )
        
        try:
            logger.info(f"LLM Request: model={self.config.model}")
            
            response = requests.post(
                self.config.base_url,
                headers=self._build_headers(),
                json=self._build_payload(messages, max_tokens, temperature),
                timeout=self.config.timeout
            )
            
            logger.info(f"LLM Response: status={response.status_code}")
            
            # Parse JSON first to get error details
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse response as JSON: {e}")
                return LLMResponse(
                    content="",
                    model=self.config.model,
                    error=f"Failed to parse API response: {str(e)}",
                    error_type="json_parse_error"
                )
            
            # Use safe extraction utility
            result = extract_llm_text(data, self.config.model)
            
            if result.success:
                logger.info(f"LLM Success: tokens={result.total_tokens}")
            else:
                logger.warning(f"LLM extraction issue: {result.error_type} - {result.error}")
            
            return result
            
        except requests.exceptions.Timeout:
            logger.error(f"LLM request timed out after {self.config.timeout}s")
            return LLMResponse(
                content="",
                model=self.config.model,
                error="Request timed out. Try a more specific query.",
                error_type="timeout"
            )
        except requests.exceptions.ConnectionError as e:
            logger.error(f"LLM connection error: {e}")
            return LLMResponse(
                content="",
                model=self.config.model,
                error="Connection error. Please check your internet connection.",
                error_type="connection_error"
            )
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            # Try to get more details from response
            try:
                if hasattr(e, 'response') and e.response is not None:
                    error_detail = e.response.json()
                    error_msg = f"{error_msg} - {error_detail}"
            except:
                pass
            logger.error(f"LLM request error: {error_msg}")
            return LLMResponse(
                content="",
                model=self.config.model,
                error=f"Error communicating with LLM: {error_msg}",
                error_type="request_error"
            )
        except Exception as e:
            logger.exception(f"Unexpected error in LLM call: {e}")
            return LLMResponse(
                content="",
                model=self.config.model,
                error=f"Unexpected error: {str(e)}",
                error_type="unexpected_error"
            )
    
    def stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        on_token: Optional[Callable[[str], None]] = None
    ) -> Generator[str, None, LLMResponse]:
        """
        Stream LLM response tokens.
        
        Args:
            messages: List of message dicts
            max_tokens: Override default max_tokens
            temperature: Override default temperature
            on_token: Optional callback for each token
            
        Yields:
            Individual tokens as strings
            
        Returns:
            Final LLMResponse (access via .send(None) after iteration)
        """
        if not self.api_key:
            yield ""
            return LLMResponse(
                content="",
                model=self.config.model,
                error="No API key configured."
            )
        
        full_content = []
        
        try:
            response = requests.post(
                self.config.base_url,
                headers=self._build_headers(),
                json=self._build_payload(messages, max_tokens, temperature, stream=True),
                timeout=self.config.timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                line_str = line.decode('utf-8')
                if not line_str.startswith("data: "):
                    continue
                
                data_str = line_str[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break
                
                try:
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    
                    if content:
                        full_content.append(content)
                        if on_token:
                            on_token(content)
                        yield content
                        
                except json.JSONDecodeError:
                    continue
            
            return LLMResponse(
                content="".join(full_content),
                model=self.config.model,
                finish_reason="stop"
            )
            
        except Exception as e:
            return LLMResponse(
                content="".join(full_content),
                model=self.config.model,
                error=str(e)
            )
    
    def call_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> LLMResponse:
        """
        Convenience method for system + user prompt pattern.
        
        Args:
            system_prompt: System message content
            user_prompt: User message content
            max_tokens: Override default max_tokens
            temperature: Override default temperature
            
        Returns:
            LLMResponse
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.call(messages, max_tokens, temperature)
    
    def classify(
        self,
        prompt: str,
        temperature: float = 0.0
    ) -> LLMResponse:
        """
        Make a classification call (deterministic, low temperature).
        
        Args:
            prompt: The classification prompt
            temperature: Temperature (default 0 for deterministic)
            
        Returns:
            LLMResponse with classification result
        """
        messages = [{"role": "user", "content": prompt}]
        return self.call(messages, max_tokens=200, temperature=temperature)


# Singleton instance
_llm_client: Optional[LLMClient] = None


def get_llm_client(config: Optional[LLMConfig] = None) -> LLMClient:
    """
    Get the global LLM client instance.
    
    Args:
        config: Optional config (only used on first call)
        
    Returns:
        LLMClient singleton
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient(config)
    return _llm_client


def reset_llm_client() -> None:
    """Reset the global LLM client (for testing)."""
    global _llm_client
    _llm_client = None
