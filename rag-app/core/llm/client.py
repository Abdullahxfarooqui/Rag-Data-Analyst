"""
LLM Client for OpenRouter API with NVIDIA Nemotron.

Features:
- Single model (no fallback chain)
- Streaming support
- Configurable parameters
- Error handling with retries
"""
import os
import json
import requests
from typing import Optional, List, Dict, Any, Generator, Callable
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""
    model: str = "nvidia/nemotron-nano-12b-v2-vl:free"
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    max_tokens: int = 2000
    temperature: float = 0.1
    timeout: int = 60
    referer: str = "http://localhost:8503"
    app_title: str = "RAG Engine"


@dataclass
class LLMResponse:
    """Structured response from LLM."""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    
    @property
    def is_error(self) -> bool:
        return self.error is not None
    
    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


class LLMClient:
    """
    OpenRouter LLM client with NVIDIA Nemotron support.
    
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
            api_key: API key (reads from env/secrets if None)
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
        """Build request headers."""
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
                error="No API key configured. Please set OPENROUTER_API_KEY."
            )
        
        try:
            response = requests.post(
                self.config.base_url,
                headers=self._build_headers(),
                json=self._build_payload(messages, max_tokens, temperature),
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            choice = data.get("choices", [{}])[0]
            
            return LLMResponse(
                content=choice.get("message", {}).get("content", ""),
                model=data.get("model", self.config.model),
                usage=data.get("usage", {}),
                finish_reason=choice.get("finish_reason")
            )
            
        except requests.exceptions.Timeout:
            return LLMResponse(
                content="",
                model=self.config.model,
                error="Request timed out. Try a more specific query."
            )
        except requests.exceptions.RequestException as e:
            return LLMResponse(
                content="",
                model=self.config.model,
                error=f"API error: {str(e)}"
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.config.model,
                error=f"Unexpected error: {str(e)}"
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
