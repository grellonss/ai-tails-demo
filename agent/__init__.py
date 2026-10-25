"""AI-Tails explanatory agent (Azure OpenAI)."""

from .azure_client import AgentError, chat, generate_multimodal_explanation

__all__ = ["AgentError", "chat", "generate_multimodal_explanation"]
