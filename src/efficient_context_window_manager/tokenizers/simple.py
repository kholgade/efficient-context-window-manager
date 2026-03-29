"""
Simple character-based tokenizer for benchmarking and fallback purposes.

Uses character count estimation rather than actual tokenization.
Useful for local/Ollama models where actual tokenization isn't available.

Main class:
- SimpleTokenizer: Basic token estimation
"""

from .base import BaseTokenizer


class SimpleTokenizer(BaseTokenizer):
    """
    Simple token counter based on character estimation.

    Estimates tokens as roughly 1 token per 4 characters.
    This is a common approximation used for token counting.

    Useful for:
    - Benchmarking when actual tokenizer isn't available
    - Local models (Ollama, etc.)
    - Quick estimation
    """

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count based on character length.

        Args:
            text: Input text to estimate tokens for

        Returns:
            Estimated token count (roughly 1 token per 4 characters)
        """
        if self.cache_enabled and text in self._cache:
            return self._cache[text]

        # Common heuristic: ~1 token per 4 characters
        # This works for English text
        estimated_tokens = max(1, len(text) // 4)

        if self.cache_enabled:
            self._cache[text] = estimated_tokens

        return estimated_tokens

    def tokenize(self, text: str) -> list:
        """
        Tokenize text into simple word-based tokens.

        Args:
            text: Input text

        Returns:
            List of word tokens
        """
        return text.split()
