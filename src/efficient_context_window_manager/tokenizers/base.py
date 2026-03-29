"""
Base tokenizer interface for token counting across different models.

Key abstraction that enables model-agnostic token management.

Main class:
- BaseTokenizer: Abstract interface for token counting

Subclasses: openai_tokenizer.py, huggingface_tokenizer.py, anthropic_tokenizer.py
Used by: context_manager.py, chunking/*.py
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ..types import TokenizationResult


class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers.

    Different LLM providers use different tokenization schemes.
    This abstraction allows swapping tokenizers without changing downstream code.

    Attributes:
        model_name: Name of the model/tokenizer
        cache: Token count cache for performance
    """

    def __init__(self, model_name: str, cache_enabled: bool = True):
        """
        Initialize tokenizer.

        Args:
            model_name: Model identifier (e.g., 'gpt-3.5-turbo', 'claude-3-opus')
            cache_enabled: Enable caching of tokenization results
        """
        self.model_name = model_name
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, int] = {} if cache_enabled else None

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text to tokenize

        Returns:
            Number of tokens

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into token list.

        Args:
            text: Input text

        Returns:
            List of tokens

        Must be implemented by subclasses.
        """
        pass

    def tokenize_with_metadata(self, text: str) -> TokenizationResult:
        """
        Tokenize with full metadata.

        Args:
            text: Input text

        Returns:
            TokenizationResult with token count and tokens

        Default implementation using subclass methods.
        """
        token_count = self.count_tokens(text)
        tokens = self.tokenize(text)
        return TokenizationResult(
            token_count=token_count,
            tokens=tokens,
            metadata={"model": self.model_name}
        )

    def estimate_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Estimate tokens for message list (e.g., ChatGPT format).

        Args:
            messages: List of {"role": ..., "content": ...} dicts

        Returns:
            Estimated token count

        Default: count tokens in all content + overhead per message
        """
        token_count = 0
        for msg in messages:
            if "content" in msg:
                token_count += self.count_tokens(msg["content"])
            token_count += 4  # Message overhead
        token_count += 2  # General overhead
        return token_count

    def clear_cache(self):
        """Clear token count cache."""
        if self.cache_enabled:
            self._cache.clear()

    def _get_cached(self, text: str) -> Optional[int]:
        """Get cached token count if available."""
        if self.cache_enabled and text in self._cache:
            return self._cache[text]
        return None

    def _cache_result(self, text: str, count: int) -> int:
        """Cache tokenization result."""
        if self.cache_enabled:
            self._cache[text] = count
        return count
