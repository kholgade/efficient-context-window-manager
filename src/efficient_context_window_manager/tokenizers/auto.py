"""
Automatic tokenizer factory for model-agnostic selection.

Detects model type and returns appropriate tokenizer implementation.

Main function:
- AutoTokenizer: Factory function for tokenizer selection

Used by: context_manager.py, user code
Related: base.py, openai_tokenizer.py, huggingface_tokenizer.py, anthropic_tokenizer.py
"""

from typing import Union
from .base import BaseTokenizer
from .openai_tokenizer import OpenAITokenizer
from .huggingface_tokenizer import HuggingFaceTokenizer
from .anthropic_tokenizer import AnthropicTokenizer
from .simple import SimpleTokenizer


def AutoTokenizer(model_name: str, cache_enabled: bool = True) -> BaseTokenizer:
    """
    Automatically select appropriate tokenizer based on model name.

    Detects model provider and returns corresponding tokenizer implementation.

    Args:
        model_name: Model identifier (e.g., 'gpt-3.5-turbo', 'claude-3-opus', 'meta-llama/Llama-2-7b')
        cache_enabled: Enable token count caching

    Returns:
        Appropriate BaseTokenizer subclass instance

    Examples:
        tokenizer = AutoTokenizer("gpt-3.5-turbo")
        tokenizer = AutoTokenizer("claude-3-opus-20250219")
        tokenizer = AutoTokenizer("meta-llama/Llama-2-7b-hf")
    """
    model_lower = model_name.lower()

    # OpenAI models
    if any(provider in model_lower for provider in ["gpt-", "text-davinci", "text-curie"]):
        return OpenAITokenizer(model_name, cache_enabled)

    # Anthropic Claude models
    if "claude" in model_lower:
        return AnthropicTokenizer(model_name, cache_enabled)

    # Local/Ollama models - use simple estimation
    # These are typically available locally and don't require transformers
    if any(local_model in model_lower for local_model in ["llama", "mistral", "phi", "falcon", "ollama"]):
        # Try HuggingFace first if available, fallback to SimpleTokenizer
        try:
            return HuggingFaceTokenizer(model_name, cache_enabled)
        except (ImportError, OSError):
            # transformers not available or model not found, use simple estimation
            return SimpleTokenizer(model_name, cache_enabled)

    # HuggingFace models with '/' in name (org/model format)
    if "/" in model_name:
        try:
            return HuggingFaceTokenizer(model_name, cache_enabled)
        except (ImportError, OSError):
            # transformers not available or model not found, use simple estimation
            return SimpleTokenizer(model_name, cache_enabled)

    # Default to SimpleTokenizer for unknown models
    return SimpleTokenizer(model_name, cache_enabled)


__all__ = ["AutoTokenizer"]
