"""
Base compressor interface for reducing token usage.

Key abstraction for compression strategies based on research:
- Observation masking (simple, effective)
- Semantic summarization (LLM-based)
- Relevance filtering (content-based)
- LLM compression (LLMLingua-style)

Main class:
- BaseCompressor: Abstract interface

Subclasses: masking.py, summarization.py, relevance.py, llm_compression.py
Used by: context_manager.py, compression/orchestrator.py
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..types import Chunk, CompressionConfig


class BaseCompressor(ABC):
    """
    Abstract base class for compression strategies.

    Compression reduces token count while preserving semantic meaning
    and critical information.

    Attributes:
        config: Compression configuration
        cache: Results caching
    """

    def __init__(self, config: CompressionConfig, cache_enabled: bool = True):
        """
        Initialize compressor.

        Args:
            config: CompressionConfig with strategy settings
            cache_enabled: Enable result caching
        """
        self.config = config
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Chunk] = {} if cache_enabled else None

    @abstractmethod
    def compress(self, chunk: Chunk) -> Chunk:
        """
        Compress a single chunk.

        Args:
            chunk: Input chunk to compress

        Returns:
            Compressed chunk with reduced token count

        Must be implemented by subclasses.
        """
        pass

    def compress_multiple(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Compress multiple chunks.

        Args:
            chunks: List of chunks to compress

        Returns:
            List of compressed chunks
        """
        return [self.compress(chunk) for chunk in chunks]

    def _get_cached(self, chunk_content: str) -> Chunk:
        """Get cached compression result if available."""
        if self.cache_enabled and chunk_content in self._cache:
            return self._cache[chunk_content]
        return None

    def _cache_result(self, original_content: str, compressed_chunk: Chunk) -> Chunk:
        """Cache compression result."""
        if self.cache_enabled:
            self._cache[original_content] = compressed_chunk
        return compressed_chunk

    def clear_cache(self):
        """Clear compression cache."""
        if self.cache_enabled:
            self._cache.clear()

    def get_compression_ratio(self, original_tokens: int, compressed_tokens: int) -> float:
        """
        Calculate compression ratio.

        Args:
            original_tokens: Original token count
            compressed_tokens: Compressed token count

        Returns:
            Reduction ratio (0.3 = 30% reduction)
        """
        if original_tokens == 0:
            return 0.0
        return max(0.0, (original_tokens - compressed_tokens) / original_tokens)
