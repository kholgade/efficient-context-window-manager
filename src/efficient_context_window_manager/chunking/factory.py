"""
Chunker factory for automatic selection and instantiation.

Selects appropriate chunking strategy based on configuration.

Main function:
- get_chunker: Factory function for chunker creation

Used by: context_manager.py, user code
Related: base.py, fixed_size.py, recursive.py, semantic.py, sliding_window.py
"""

from typing import Optional, Callable
from .base import BaseChunker
from .fixed_size import FixedSizeChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker
from .sliding_window import SlidingWindowChunker
from ..types import ChunkingStrategy
from ..tokenizers.base import BaseTokenizer


def get_chunker(
    tokenizer: BaseTokenizer,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    chunk_size: int = 1000,
    overlap: int = 100,
    embedding_fn: Optional[Callable] = None,
    **kwargs
) -> BaseChunker:
    """
    Factory function to create appropriate chunker.

    Args:
        tokenizer: Token counter implementation
        strategy: Chunking strategy to use
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks in tokens
        embedding_fn: Embedding function for semantic chunking
        **kwargs: Additional arguments for specific chunkers

    Returns:
        Configured chunker instance

    Examples:
        # Fixed size
        chunker = get_chunker(tokenizer, ChunkingStrategy.FIXED_SIZE, chunk_size=1000)

        # Recursive (recommended)
        chunker = get_chunker(tokenizer, ChunkingStrategy.RECURSIVE, chunk_size=1000)

        # Sliding window
        chunker = get_chunker(
            tokenizer,
            ChunkingStrategy.SLIDING_WINDOW,
            chunk_size=1000,
            stride=800
        )

        # Semantic (requires embedding function)
        chunker = get_chunker(
            tokenizer,
            ChunkingStrategy.SEMANTIC,
            chunk_size=1000,
            embedding_fn=my_embedding_fn
        )
    """
    if strategy == ChunkingStrategy.FIXED_SIZE:
        return FixedSizeChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    elif strategy == ChunkingStrategy.RECURSIVE:
        return RecursiveChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    elif strategy == ChunkingStrategy.SLIDING_WINDOW:
        stride = kwargs.get("stride", chunk_size - overlap)
        return SlidingWindowChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            overlap=overlap,
            stride=stride,
        )

    elif strategy == ChunkingStrategy.SEMANTIC:
        if embedding_fn is None:
            raise ValueError("embedding_fn required for semantic chunking")
        return SemanticChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            overlap=overlap,
            embedding_fn=embedding_fn,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


__all__ = ["get_chunker"]
