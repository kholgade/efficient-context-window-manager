"""
Anthropic Claude tokenizer.

Provides token counting for Anthropic Claude models.

Main class:
- AnthropicTokenizer: Token counting for Claude models

Used by: context_manager.py, chunking/*.py
Related: base.py (parent), auto.py (factory)
"""

from typing import List
from .base import BaseTokenizer


class AnthropicTokenizer(BaseTokenizer):
    """
    Tokenizer for Anthropic Claude models.

    Uses approximate token counting based on model specifications.
    Note: For exact counts, use Anthropic's API.

    Attributes:
        model_name: Anthropic model identifier
        words_per_token: Approximate conversion for estimation
    """

    # Anthropic's approximate token ratios
    WORDS_PER_TOKEN = 0.75
    CHARS_PER_TOKEN = 3.5

    def __init__(self, model_name: str = "claude-3-opus-20250219", cache_enabled: bool = True):
        """
        Initialize Anthropic tokenizer.

        Args:
            model_name: Claude model identifier
            cache_enabled: Enable token count caching
        """
        super().__init__(model_name, cache_enabled)

    def count_tokens(self, text: str) -> int:
        """
        Estimate tokens for Claude using character-based approximation.

        Note: This is an estimate. For production, use Anthropic API's token counting.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        cached = self._get_cached(text)
        if cached is not None:
            return cached

        # Anthropic uses approximate 3.5 characters per token
        count = max(1, len(text) // int(1 / self.CHARS_PER_TOKEN * 10) // 10 or 1)
        # More accurate estimation: space-based word count
        word_count = len(text.split())
        estimated_tokens = int(word_count / self.WORDS_PER_TOKEN)

        return self._cache_result(text, max(1, estimated_tokens))

    def tokenize(self, text: str) -> List[str]:
        """
        Approximate tokenization (splits by whitespace and punctuation).

        Args:
            text: Input text

        Returns:
            Approximate token list
        """
        import re
        # Simple tokenization: split on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens

    def estimate_message_tokens(self, messages: List) -> int:
        """
        Estimate tokens for Claude message format.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Estimated token count
        """
        token_count = 0
        for msg in messages:
            token_count += 2  # Role/message markup
            if "content" in msg:
                token_count += self.count_tokens(msg["content"])
        return token_count
