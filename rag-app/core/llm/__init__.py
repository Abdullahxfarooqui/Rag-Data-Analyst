"""
LLM module - OpenRouter client and prompts.
"""
from core.llm.client import (
    LLMClient, 
    LLMConfig, 
    LLMResponse,
    get_llm_client,
    reset_llm_client
)
from core.llm.prompts import (
    SYSTEM_PROMPT_CONCISE,
    SYSTEM_PROMPT_DETAILED,
    SYSTEM_PROMPT_ANALYSIS,
    ROUTING_PROMPT_TEMPLATE,
    get_data_query_prompt,
    get_overview_prompt,
    get_freeform_response,
    get_system_task_response,
)

__all__ = [
    "LLMClient",
    "LLMConfig",
    "LLMResponse",
    "get_llm_client",
    "reset_llm_client",
    "SYSTEM_PROMPT_CONCISE",
    "SYSTEM_PROMPT_DETAILED",
    "SYSTEM_PROMPT_ANALYSIS",
    "ROUTING_PROMPT_TEMPLATE",
    "get_data_query_prompt",
    "get_overview_prompt",
    "get_freeform_response",
    "get_system_task_response",
]
