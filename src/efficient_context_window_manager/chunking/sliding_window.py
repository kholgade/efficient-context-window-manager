"""
Sliding window chunking strategy based on SWAT research.

Reference: "Sliding Window Attention Training for Efficient Large Language Models" (2502.18845)

Creates overlapping windows with configurable stride and heterogeneous window lengths.
Efficient for sequential processing and long-context scenarios.

Main class:
- SlidingWindowChunker: Overlapping windows with controlled stride

Used by: chunking/factory.py, context_manager.py
Related: base.py (parent)
"""

from typing import List
from .base import BaseChunker
from ..types import Chunk


class SlidingWindowChunker(BaseChunker):
    """
    Sliding window chunking following SWAT approach.

    Key properties:
    - Regular overlapping windows
    - Configurable stride (movement per window)
    - Efficient for sequential processing
    - Preserves continuity through overlap

    Attributes:
        stride: Tokens to advance per window (default: chunk_size - overlap)
    """

    def __init__(
        self,
        tokenizer,
        chunk_size: int = 1000,
        overlap: int = 200,
        stride: int = None,
    ):
        """
        Initialize sliding window chunker.

        Args:
            tokenizer: Token counter
            chunk_size: Window size in tokens
            overlap: Overlapping tokens between windows
            stride: How many tokens to advance (default: chunk_size - overlap)
        """
        super().__init__(tokenizer, chunk_size, overlap)
        self.stride = stride or (chunk_size - overlap)

        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if self.stride > self.chunk_size:
            raise ValueError(f"stride ({self.stride}) cannot exceed chunk_size ({self.chunk_size})")

    def chunk(self, text: str) -> List[Chunk]:
        """
        Create sliding windows over text.

        Args:
            text: Input text

        Returns:
            List of overlapping Chunk objects
        """
        if len(text) == 0:
            return []

        total_tokens = self.tokenizer.count_tokens(text)

        if total_tokens <= self.chunk_size:
            # Text fits in single window
            return [
                self._create_chunk(text, 0, len(text), {
                    "chunk_index": 0,
                    "window_index": 0,
                    "strategy": "sliding_window_single"
                })
            ]

        chunks = []
        # Estimate chars per token for position mapping
        chars_per_token = len(text) / total_tokens if total_tokens > 0 else 1

        window_index = 0
        pos_tokens = 0  # Position in tokens

        while pos_tokens < total_tokens:
            # Calculate character boundaries
            start_chars = int(pos_tokens * chars_per_token)
            end_tokens = min(pos_tokens + self.chunk_size, total_tokens)
            end_chars = int(end_tokens * chars_per_token)

            # Ensure we don't exceed text
            end_chars = min(end_chars, len(text))

            # Refine boundaries to word/sentence breaks
            if end_chars < len(text):
                # Backtrack to find space
                while end_chars > start_chars and text[end_chars - 1] not in ' \n\t':
                    end_chars -= 1
                if end_chars == start_chars:
                    end_chars = min(start_chars + 1, len(text))

            chunk_text = text[start_chars:end_chars].strip()
            if len(chunk_text) == 0:
                break

            chunk = self._create_chunk(
                chunk_text,
                start_chars,
                end_chars,
                {
                    "chunk_index": window_index,
                    "window_index": window_index,
                    "start_token": pos_tokens,
                    "end_token": end_tokens,
                    "strategy": "sliding_window"
                }
            )
            chunks.append(chunk)

            # Advance position by stride
            pos_tokens += self.stride
            window_index += 1

            # Safety limit
            if window_index > 10000:
                break

        self.validate_chunks(chunks)
        return chunks
