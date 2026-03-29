"""
Semantic summarization compression strategy.

Uses LLM to create summaries of chunks, reducing token count.

Main class:
- SemanticSummarizationCompressor: LLM-based summarization

Note: Requires external LLM (slower but higher quality than masking)

Used by: compression/orchestrator.py, context_manager.py
Related: base.py (parent)
"""

from typing import Callable, Optional
from .base import BaseCompressor
from ..types import Chunk


class SemanticSummarizationCompressor(BaseCompressor):
    """
    Compress via semantic summarization.

    Uses an LLM to create summaries of chunks.
    More expensive but preserves meaning better than masking.

    Attributes:
        summarize_fn: LLM function for summarization
    """

    def __init__(self, config, summarize_fn: Optional[Callable] = None, **kwargs):
        """
        Initialize summarization compressor.

        Args:
            config: CompressionConfig
            summarize_fn: Function to summarize text (required)
            **kwargs: Additional config
        """
        super().__init__(config, **kwargs)
        self.summarize_fn = summarize_fn

    def compress(self, chunk: Chunk) -> Chunk:
        """
        Compress via summarization.

        Args:
            chunk: Input chunk

        Returns:
            Compressed chunk with summary
        """
        if self.summarize_fn is None:
            raise ValueError("summarize_fn required for SemanticSummarizationCompressor")

        cached = self._get_cached(chunk.content)
        if cached is not None:
            return cached

        # Create summary
        original_tokens = chunk.token_count
        target_tokens = int(original_tokens * (1 - self.config.compression_ratio))

        try:
            summary = self.summarize_fn(chunk.content, target_tokens)
        except Exception as e:
            # Fallback: just truncate
            words = chunk.content.split()
            target_words = int(len(words) * (1 - self.config.compression_ratio))
            summary = ' '.join(words[:target_words])

        compressed_chunk = Chunk(
            content=f"[Summary] {summary}",
            token_count=max(1, target_tokens),
            start_idx=chunk.start_idx,
            end_idx=chunk.end_idx,
            metadata={
                **chunk.metadata,
                "compression_method": "semantic_summarization",
                "original_tokens": original_tokens,
                "compressed_tokens": max(1, target_tokens),
                "is_summary": True,
            }
        )

        return self._cache_result(chunk.content, compressed_chunk)
