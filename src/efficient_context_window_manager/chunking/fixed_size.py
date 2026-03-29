"""
Fixed-size chunking strategy.

Simple, deterministic chunking by token count with optional overlap.

Main class:
- FixedSizeChunker: Break text into fixed-size token chunks

Used by: chunking/factory.py, context_manager.py
Related: base.py (parent)
"""

from typing import List
from .base import BaseChunker
from ..types import Chunk


class FixedSizeChunker(BaseChunker):
    """
    Break text into fixed-size token chunks.

    Simple and predictable. Good for homogeneous documents.
    Doesn't respect semantic boundaries.

    Example:
        chunk_size=1000, overlap=100
        Chunk 1: tokens 0-1000
        Chunk 2: tokens 900-1900 (900-1000 overlap)
        Chunk 3: tokens 1800-2800
    """

    def chunk(self, text: str) -> List[Chunk]:
        """
        Break text into fixed-size chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of Chunk objects with overlap
        """
        chunks = []
        text_len = len(text)

        if text_len == 0:
            return chunks

        # First pass: find approximate character boundaries for tokens
        # We estimate tokens per character, then refine
        token_count = self.tokenizer.count_tokens(text)

        if token_count <= self.chunk_size:
            # Text fits in single chunk
            return [self._create_chunk(text, 0, text_len, {"chunk_index": 0})]

        # Calculate approximate chars per token
        chars_per_token = text_len / token_count

        current_pos = 0
        chunk_index = 0

        while current_pos < text_len:
            # Calculate target end position
            if chunk_index == 0:
                # First chunk: full size
                target_tokens = self.chunk_size
            else:
                # Subsequent chunks: size minus overlap
                target_tokens = self.chunk_size - self.overlap

            # Estimate character position
            target_chars = int(target_tokens * chars_per_token)
            end_pos = min(current_pos + target_chars, text_len)

            # Refine: back up to reasonable boundary (space/punctuation)
            if end_pos < text_len:
                # Search backward for space or punctuation
                while end_pos > current_pos and text[end_pos] not in ' \n\t.!?,;:':
                    end_pos -= 1
                if end_pos == current_pos:
                    # Couldn't find boundary, use estimated position
                    end_pos = min(current_pos + target_chars, text_len)
                else:
                    end_pos += 1  # Include boundary char

            chunk_text = text[current_pos:end_pos].strip()
            if len(chunk_text) == 0:
                break

            chunk = self._create_chunk(
                chunk_text,
                current_pos,
                end_pos,
                {"chunk_index": chunk_index}
            )
            chunks.append(chunk)

            # Move position for next chunk
            if chunk_index == 0:
                # After first chunk, back up by overlap for next start
                current_pos = max(current_pos, end_pos - int(self.overlap * chars_per_token))
            else:
                current_pos = end_pos - int(self.overlap * chars_per_token)

            # Safety: if not progressing, jump forward
            if current_pos >= end_pos:
                current_pos = end_pos

            chunk_index += 1

            if chunk_index > 10000:  # Safety limit
                break

        self.validate_chunks(chunks)
        return chunks
