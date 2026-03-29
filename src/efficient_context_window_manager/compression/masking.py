"""
Observation masking compression strategy.

Simple yet effective: marks and removes non-essential tokens.
Research shows it's as effective as complex summarization but much cheaper.

Reference: 2025 research on observation masking vs summarization

Main class:
- ObservationMaskingCompressor: Remove non-essential tokens

Used by: compression/orchestrator.py, context_manager.py
Related: base.py (parent)
"""

import re
from typing import List, Set
from .base import BaseCompressor
from ..types import Chunk


class ObservationMaskingCompressor(BaseCompressor):
    """
    Remove non-essential tokens via observation masking.

    Simple strategy: identify and remove low-value tokens:
    - Articles (a, an, the)
    - Redundant connectors
    - Filler words
    - Repeated phrases

    Maintains readability and semantic coherence while reducing tokens.

    Attributes:
        stop_words: Tokens to mark for potential removal
    """

    # Words commonly non-essential for understanding
    STOP_WORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'yet', 'so',
        'is', 'are', 'was', 'were', 'be', 'being',
        'that', 'this', 'these', 'those',
        'very', 'really', 'quite', 'just',
        'i', 'me', 'my', 'you', 'your',
        'he', 'she', 'it', 'we', 'they',
        'their', 'his', 'her', 'its',
    }

    def compress(self, chunk: Chunk) -> Chunk:
        """
        Compress chunk via observation masking.

        Removes non-essential tokens while preserving meaning.

        Args:
            chunk: Input chunk

        Returns:
            Compressed chunk
        """
        cached = self._get_cached(chunk.content)
        if cached is not None:
            return cached

        # Strategy: selectively remove non-essential words
        original_tokens = chunk.token_count
        target_tokens = int(original_tokens * (1 - self.config.compression_ratio))

        compressed_text = self._mask_non_essential(chunk.content, target_tokens)

        # If we didn't achieve target, be more aggressive
        if len(compressed_text.split()) < len(chunk.content.split()) * (1 - self.config.compression_ratio):
            compressed_text = self._aggressive_compress(chunk.content, target_tokens)

        # Create compressed chunk
        compressed_chunk = Chunk(
            content=compressed_text,
            token_count=max(1, int(target_tokens)),  # Estimate
            start_idx=chunk.start_idx,
            end_idx=chunk.end_idx,
            metadata={
                **chunk.metadata,
                "compression_method": "observation_masking",
                "original_tokens": original_tokens,
                "compressed_tokens": max(1, int(target_tokens)),
            }
        )

        return self._cache_result(chunk.content, compressed_chunk)

    def _mask_non_essential(self, text: str, target_tokens: int) -> str:
        """
        Remove non-essential words iteratively.

        Args:
            text: Input text
            target_tokens: Target token count

        Returns:
            Compressed text
        """
        words = text.split()

        # Mark non-essential words
        marked = []
        for word in words:
            word_lower = word.lower().strip('.,!?;:\'"')
            if word_lower in self.STOP_WORDS:
                marked.append((word, True))  # Essential
            else:
                marked.append((word, False))

        # Remove marked words until we reach target
        current_tokens = len(words)
        removal_idx = 0

        while current_tokens > target_tokens and removal_idx < len(marked):
            word, is_essential = marked[removal_idx]
            if not is_essential and len(word) > 1:
                marked[removal_idx] = (None, False)
                current_tokens -= 1
            removal_idx += 1

        # Reconstruct
        result = ' '.join([w for w, _ in marked if w is not None])
        return result.strip()

    def _aggressive_compress(self, text: str, target_tokens: int) -> str:
        """
        Aggressive compression: remove more tokens.

        Args:
            text: Input text
            target_tokens: Target token count

        Returns:
            More heavily compressed text
        """
        words = text.split()
        if len(words) <= target_tokens:
            return text

        # Keep only nouns, verbs, adjectives (simple POS heuristic)
        kept = []
        for word in words:
            # Simple heuristic: keep longer words and proper nouns
            if len(word) > 3 or word[0].isupper():
                kept.append(word)
            elif len(kept) < target_tokens:
                kept.append(word)

            if len(kept) >= target_tokens:
                break

        result = ' '.join(kept[:target_tokens])
        return result.strip()
