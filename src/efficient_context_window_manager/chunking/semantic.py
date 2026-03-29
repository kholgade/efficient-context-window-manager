"""
Semantic chunking using embedding-based boundary detection.

Advanced strategy that uses embeddings to identify natural semantic boundaries.
Higher quality but computationally more expensive than other strategies.

Main class:
- SemanticChunker: Embedding-based semantic boundary detection

Used by: chunking/factory.py, context_manager.py
Related: base.py (parent)
"""

from typing import List, Optional, Callable
import math
from .base import BaseChunker
from ..types import Chunk


class SemanticChunker(BaseChunker):
    """
    Chunk text based on semantic similarity boundaries.

    Uses embeddings to identify natural break points where content changes.
    Sentences/paragraphs with high dissimilarity from neighbors are chunk boundaries.

    Requires embedding function (e.g., from sentence-transformers).

    Attributes:
        embedding_fn: Function to compute embeddings
        similarity_threshold: Score below which indicates boundary
        split_strategy: 'sentences' or 'paragraphs'
    """

    def __init__(
        self,
        tokenizer,
        chunk_size: int = 1000,
        overlap: int = 100,
        embedding_fn: Optional[Callable] = None,
        similarity_threshold: float = 0.5,
        split_strategy: str = "sentences",
    ):
        """
        Initialize semantic chunker.

        Args:
            tokenizer: Token counter
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks
            embedding_fn: Function to compute embeddings (required)
            similarity_threshold: Cosine similarity threshold for boundaries
            split_strategy: 'sentences' or 'paragraphs' for initial split
        """
        super().__init__(tokenizer, chunk_size, overlap)
        self.embedding_fn = embedding_fn
        self.similarity_threshold = similarity_threshold
        self.split_strategy = split_strategy

        if embedding_fn is None:
            raise ValueError("embedding_fn is required for SemanticChunker")

    def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk using semantic boundary detection.

        Args:
            text: Input text

        Returns:
            List of semantically coherent chunks
        """
        if len(text) == 0:
            return []

        # Step 1: Split into sentences/paragraphs
        if self.split_strategy == "paragraphs":
            segments = text.split("\n\n")
        else:
            # Split by sentences (simple approach)
            import re
            segments = re.split(r'(?<=[.!?])\s+', text)

        if not segments:
            return [self._create_chunk(text, 0, len(text))]

        # Step 2: Get embeddings for each segment
        try:
            embeddings = [self.embedding_fn(seg) for seg in segments if seg.strip()]
        except Exception as e:
            # Fallback to recursive chunking if embeddings fail
            from .recursive import RecursiveChunker
            chunker = RecursiveChunker(self.tokenizer, self.chunk_size, self.overlap)
            return chunker.chunk(text)

        if not embeddings:
            return [self._create_chunk(text, 0, len(text))]

        # Step 3: Detect boundaries based on similarity
        boundaries = [0]  # Start is always a boundary
        for i in range(len(embeddings) - 1):
            similarity = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            if similarity < self.similarity_threshold:
                boundaries.append(i + 1)

        boundaries.append(len(segments))  # End is always a boundary

        # Step 4: Merge segments between boundaries into chunks
        chunks = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            chunk_segments = segments[start_idx:end_idx]

            merged = " ".join(seg.strip() for seg in chunk_segments if seg.strip())
            if not merged:
                continue

            chunk = self._create_chunk(
                merged,
                0,
                len(merged),
                {
                    "chunk_index": len(chunks),
                    "segments": end_idx - start_idx,
                    "strategy": "semantic"
                }
            )
            chunks.append(chunk)

        # Step 5: Merge small chunks if needed
        return self._merge_small_chunks(chunks)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Compute cosine similarity between embedding vectors.

        Args:
            a: First embedding vector
            b: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot_product / (mag_a * mag_b)

    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Merge chunks that are below target size.

        Args:
            chunks: Input chunks

        Returns:
            Merged chunks meeting size targets
        """
        if not chunks:
            return chunks

        merged = []
        current = None

        for chunk in chunks:
            if current is None:
                current = chunk
            elif current.token_count + chunk.token_count <= self.chunk_size:
                # Merge with current
                combined_text = current.content + " " + chunk.content
                current = self._create_chunk(
                    combined_text,
                    current.start_idx,
                    chunk.end_idx,
                    {
                        "chunk_index": len(merged),
                        "merged": True,
                        "strategy": "semantic_merged"
                    }
                )
            else:
                # Flush current and start new
                merged.append(current)
                current = chunk

        if current is not None:
            merged.append(current)

        return merged
