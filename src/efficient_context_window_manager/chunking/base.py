"""
Base chunker interface for breaking text into manageable pieces.

Key abstraction for different chunking strategies.

Main class:
- BaseChunker: Abstract interface for all chunking strategies

Subclasses: fixed_size.py, recursive.py, semantic.py, sliding_window.py
Used by: context_manager.py, chunking/factory.py
"""

from abc import ABC, abstractmethod
from typing import List
from ..types import Chunk
from ..tokenizers.base import BaseTokenizer


class BaseChunker(ABC):
    """
    Abstract base class for text chunking strategies.

    Different chunking approaches optimize for different use cases:
    - FixedSizeChunker: Simple, predictable
    - RecursiveChunker: Semantic coherence
    - SlidingWindowChunker: Overlapping context (SWAT approach)
    - SemanticChunker: Embedding-based boundaries

    Attributes:
        tokenizer: Token counter for accurate chunk sizing
        chunk_size: Target tokens per chunk
        overlap: Overlap between chunks (tokens)
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        chunk_size: int = 1000,
        overlap: int = 100,
    ):
        """
        Initialize chunker.

        Args:
            tokenizer: Token counter implementation
            chunk_size: Target tokens per chunk
            overlap: Overlapping tokens between chunks (for context)
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")

        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """
        Break text into chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of Chunk objects

        Must be implemented by subclasses.
        """
        pass

    def _create_chunk(
        self,
        content: str,
        start_idx: int,
        end_idx: int,
        metadata: dict = None,
    ) -> Chunk:
        """
        Create a Chunk with proper token counting.

        Args:
            content: Chunk text
            start_idx: Character position in original
            end_idx: Character position in original
            metadata: Optional additional info

        Returns:
            Chunk object with token count
        """
        token_count = self.tokenizer.count_tokens(content)
        return Chunk(
            content=content,
            token_count=token_count,
            start_idx=start_idx,
            end_idx=end_idx,
            metadata=metadata or {},
        )

    def validate_chunks(self, chunks: List[Chunk]) -> bool:
        """
        Validate that chunks are properly formed.

        Args:
            chunks: List of chunks to validate

        Returns:
            True if valid, raises exception otherwise
        """
        for i, chunk in enumerate(chunks):
            if len(chunk.content) == 0:
                raise ValueError(f"Chunk {i} is empty")
            if chunk.token_count == 0:
                raise ValueError(f"Chunk {i} has zero tokens")

        # Check no gaps between chunks
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                if chunks[i].end_idx > chunks[i + 1].start_idx:
                    # Overlap is OK, but not crossover
                    if chunks[i].end_idx > chunks[i + 1].end_idx:
                        raise ValueError(f"Chunk {i} and {i+1} have invalid overlap")

        return True
