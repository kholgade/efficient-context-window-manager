"""
HuggingFace transformers tokenizer implementation.

Supports local models, open-source LLMs via HuggingFace.

Main class:
- HuggingFaceTokenizer: Uses transformers library

Supported models: Llama2, Mistral, Phi, etc.

Used by: context_manager.py, chunking/*.py
Related: base.py (parent), auto.py (factory)
"""

from typing import List
from .base import BaseTokenizer


class HuggingFaceTokenizer(BaseTokenizer):
    """
    Tokenizer for HuggingFace models using transformers.

    Supports: Llama2, Mistral, Phi, and other HF models.

    Attributes:
        tokenizer: transformers tokenizer object
        model_name: HuggingFace model identifier
    """

    def __init__(self, model_name: str, cache_enabled: bool = True):
        """
        Initialize HuggingFace tokenizer.

        Args:
            model_name: HuggingFace model identifier
            cache_enabled: Enable token count caching

        Raises:
            ImportError: If transformers not installed
            OSError: If model not found
        """
        super().__init__(model_name, cache_enabled)
        try:
            from transformers import AutoTokenizer as HFAutoTokenizer
            self.tokenizer = HFAutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except ImportError:
            raise ImportError("transformers library required for HuggingFaceTokenizer")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using HuggingFace tokenizer.

        Args:
            text: Input text

        Returns:
            Token count
        """
        cached = self._get_cached(text)
        if cached is not None:
            return cached

        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        count = len(tokens)
        return self._cache_result(text, count)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into token list.

        Args:
            text: Input text

        Returns:
            List of token strings
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        return tokens

    def estimate_message_tokens(self, messages: List) -> int:
        """
        Estimate tokens for chat messages.

        Uses HuggingFace's message format (simplified).

        Args:
            messages: List of message dicts

        Returns:
            Token count estimate
        """
        token_count = 0
        for msg in messages:
            token_count += 3  # Per-message overhead (lower than OpenAI)
            for value in msg.values():
                if isinstance(value, str):
                    tokens = self.tokenizer.encode(value, add_special_tokens=False)
                    token_count += len(tokens)
        return token_count
