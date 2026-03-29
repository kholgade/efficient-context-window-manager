"""
Recursive chunking strategy inspired by LangChain's RecursiveCharacterTextSplitter.

Hierarchically splits by semantic delimiters (paragraphs, sentences, words).
Preserves context and coherence better than fixed-size.

Main class:
- RecursiveChunker: Split by delimiters in order of preference

Used by: chunking/factory.py, context_manager.py
Related: base.py (parent)
"""

from typing import List
from .base import BaseChunker
from ..types import Chunk


class RecursiveChunker(BaseChunker):
    """
    Recursively split text by semantic boundaries.

    Splitting order (tries each until pieces fit):
    1. Double newline (paragraph break)
    2. Single newline (line break)
    3. Space (word boundary)
    4. Character (last resort)

    Preserves document structure better than fixed-size chunking.
    """

    # Delimiters in order of preference
    SEPARATORS = ["\n\n", "\n", " ", ""]

    def chunk(self, text: str) -> List[Chunk]:
        """
        Recursively chunk text by semantic boundaries.

        Args:
            text: Input text to chunk

        Returns:
            List of Chunk objects
        """
        if len(text) == 0:
            return []

        good_splits = []
        separator = self.SEPARATORS[-1]  # Start with most granular

        for _s in self.SEPARATORS:
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                break

        # Now split by this separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Go through splits and combine to reach target chunk size
        good_splits = [s for s in splits if s]  # Remove empty

        if len(good_splits) == 0:
            return []

        # Merge splits into chunks respecting size
        chunks = self._merge_splits(good_splits, separator)

        self.validate_chunks(chunks)
        return chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[Chunk]:
        """
        Merge small splits into properly-sized chunks.

        Args:
            splits: List of text pieces
            separator: Separator used (for reconstruction)

        Returns:
            List of merged Chunk objects
        """
        separator_tokens = self.tokenizer.count_tokens(separator)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for split in splits:
            split_tokens = self.tokenizer.count_tokens(split)

            # If single split exceeds chunk_size, recurse
            if split_tokens > self.chunk_size:
                # Too big, recurse on next separator
                idx = len(self.SEPARATORS) - 1
                current_sep_idx = self.SEPARATORS.index(separator) if separator in self.SEPARATORS else 0
                if current_sep_idx < len(self.SEPARATORS) - 1:
                    next_sep = self.SEPARATORS[current_sep_idx + 1]
                else:
                    next_sep = ""

                # Recursive split
                sub_splits = split.split(next_sep) if next_sep else list(split)
                sub_chunks = self._merge_splits(
                    [s for s in sub_splits if s],
                    next_sep
                )
                chunks.extend(sub_chunks)
                current_chunk = []
                current_tokens = 0
            else:
                # Check if adding this split would exceed limit
                potential_tokens = current_tokens + split_tokens
                if len(current_chunk) > 0:
                    potential_tokens += separator_tokens

                if potential_tokens > self.chunk_size and len(current_chunk) > 0:
                    # Flush current chunk
                    merged_text = separator.join(current_chunk)
                    if merged_text.strip():
                        chunk = self._create_chunk_with_tracking(
                            merged_text,
                            len(chunks)
                        )
                        chunks.append(chunk)

                    current_chunk = [split]
                    current_tokens = split_tokens
                else:
                    # Add to current chunk
                    current_chunk.append(split)
                    if len(current_chunk) > 1:
                        current_tokens += separator_tokens
                    current_tokens += split_tokens

        # Flush remaining chunk
        if len(current_chunk) > 0:
            merged_text = separator.join(current_chunk)
            if merged_text.strip():
                chunk = self._create_chunk_with_tracking(
                    merged_text,
                    len(chunks)
                )
                chunks.append(chunk)

        # Apply overlap by merging adjacent chunks
        return self._apply_overlap(chunks)

    def _create_chunk_with_tracking(self, text: str, chunk_index: int) -> Chunk:
        """
        Create chunk with source tracking.

        Args:
            text: Chunk content
            chunk_index: Index in output

        Returns:
            Chunk object
        """
        token_count = self.tokenizer.count_tokens(text)
        return Chunk(
            content=text,
            token_count=token_count,
            start_idx=0,  # Simplified for recursive chunker
            end_idx=len(text),
            metadata={"chunk_index": chunk_index, "strategy": "recursive"}
        )

    def _apply_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Create overlapping chunks if overlap > 0.

        Args:
            chunks: List of non-overlapping chunks

        Returns:
            List of chunks with overlap
        """
        if self.overlap == 0 or len(chunks) <= 1:
            return chunks

        overlapped = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                # Create overlapping chunk by combining with previous
                prev_chunk = overlapped[-1]  # Use already overlapped chunk
                overlap_text = prev_chunk.content[-int(len(prev_chunk.content) * 0.2):]
                combined = overlap_text + " " + chunk.content

                overlapped.append(
                    Chunk(
                        content=combined,
                        token_count=self.tokenizer.count_tokens(combined),
                        start_idx=i,  # Simplified: use chunk index
                        end_idx=i + 1,  # Simplified: indicate this is chunk i
                        metadata={"chunk_index": i, "strategy": "recursive_overlapped"}
                    )
                )

        return overlapped
