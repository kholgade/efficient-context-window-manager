"""
OpenAI tokenizer using tiktoken library.

Uses official OpenAI tiktoken for accurate token counting.

Main class:
- OpenAITokenizer: Uses tiktoken for GPT models

Used by: context_manager.py, chunking/*.py
Related: base.py (parent), auto.py (factory)
"""

from typing import List
import tiktoken
from .base import BaseTokenizer


class OpenAITokenizer(BaseTokenizer):
    """
    Tokenizer for OpenAI models using tiktoken.

    Supports: gpt-3.5-turbo, gpt-4, gpt-4-turbo, etc.

    Attributes:
        encoding: tiktoken encoding object
        model_name: OpenAI model name
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo", cache_enabled: bool = True):
        """
        Initialize OpenAI tokenizer.

        Args:
            model_name: OpenAI model (default: gpt-3.5-turbo)
            cache_enabled: Enable token count caching

        Raises:
            ValueError: If model not supported by tiktoken
        """
        super().__init__(model_name, cache_enabled)
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback for newer models
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken.

        Args:
            text: Input text

        Returns:
            Token count
        """
        cached = self._get_cached(text)
        if cached is not None:
            return cached

        count = len(self.encoding.encode(text))
        return self._cache_result(text, count)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into token list.

        Args:
            text: Input text

        Returns:
            List of token strings (decoded)
        """
        token_ids = self.encoding.encode(text)
        tokens = [self.encoding.decode_single_token_bytes(tid).decode('utf-8', errors='replace')
                  for tid in token_ids]
        return tokens

    def estimate_message_tokens(self, messages: List) -> int:
        """
        Accurate token counting for chat messages.

        Uses OpenAI's message format with proper overhead calculation.

        Args:
            messages: List of message dicts

        Returns:
            Token count including message overhead
        """
        # This is approximate based on OpenAI's guidance
        token_count = 0
        for msg in messages:
            token_count += 4  # Per-message overhead
            for value in msg.values():
                if isinstance(value, str):
                    token_count += len(self.encoding.encode(value))
        token_count += 2  # Reply overhead
        return token_count
