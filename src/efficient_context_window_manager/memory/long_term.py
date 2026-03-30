"""
Long-term memory with compression and semantic search.

Stores compressed messages and optional vector embeddings for semantic retrieval.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from ..types import Message, CompressionConfig
from ..compression.base import BaseCompressor
from ..compression.masking import ObservationMaskingCompressor


@dataclass
class LongTermMemory:
    """Compressed history with optional vector storage."""

    max_compressed_tokens: int = 2000
    compressed_messages: List[Message] = field(default_factory=list)
    compressor: Optional[BaseCompressor] = None
    vector_store: Optional[Any] = None

    def __post_init__(self):
        """Initialize compressor if not provided."""
        if self.compressor is None:
            config = CompressionConfig()
            self.compressor = ObservationMaskingCompressor(config)

    def store(self, messages: List[Message]) -> None:
        """
        Store and compress messages.

        Args:
            messages: Messages to store
        """
        for msg in messages:
            # Compress before storing
            from ..types import Chunk, CompressionConfig

            chunk = Chunk(
                content=msg.content,
                token_count=msg.token_count,
                start_idx=0,
                end_idx=len(msg.content),
                metadata={"role": msg.role, **msg.metadata},
            )

            compressed_chunk = self.compressor.compress(chunk)
            compressed_msg = Message(
                role=msg.role,
                content=compressed_chunk.content,
                metadata={
                    **msg.metadata,
                    "compressed": True,
                    "original_tokens": msg.token_count,
                    "compressed_tokens": compressed_chunk.token_count,
                },
            )
            compressed_msg.token_count = compressed_chunk.token_count

            self.compressed_messages.append(compressed_msg)

            # Add to vector store if available
            if self.vector_store:
                try:
                    self.vector_store.add({
                        "id": f"{msg.role}_{hash(msg.content)}",
                        "content": msg.content,
                        "metadata": msg.metadata,
                    })
                except Exception:
                    pass  # Vector store integration optional

    async def retrieve(
        self,
        query: str,
        max_tokens: int,
        strategy: str = "keyword",
    ) -> List[Message]:
        """
        Retrieve relevant messages.

        Args:
            query: Search query
            max_tokens: Max tokens to return
            strategy: "semantic" | "keyword" | "hybrid"

        Returns:
            List of relevant messages
        """
        if strategy == "semantic" and self.vector_store:
            try:
                query_embedding = self.vector_store.embed(query)
                results = await self.vector_store.search(
                    query_embedding, top_k=5
                )
                return results[:max_tokens]
            except Exception:
                pass  # Fall back to keyword

        if strategy in ["keyword", "hybrid"]:
            keywords = query.lower().split()
            relevant = [
                m
                for m in self.compressed_messages
                if any(kw in m.content.lower() for kw in keywords if len(kw) > 2)
            ]
            tokens = 0
            result = []
            for msg in relevant:
                if tokens + msg.token_count <= max_tokens:
                    result.append(msg)
                    tokens += msg.token_count
            return result

        return []

    def clear(self) -> None:
        """Clear all compressed messages."""
        self.compressed_messages.clear()
