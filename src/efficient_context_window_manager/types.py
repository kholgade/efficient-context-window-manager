"""
Core type definitions for context window management.

Key concepts:
- Chunk: Unit of text with token count and metadata
- ContextWindow: Current processing state (tokens used, chunks, budget)
- CompressionConfig: Settings for compression strategies
- ChunkingStrategy: Available chunking approaches

Relationships:
- ContextWindow contains multiple Chunks
- CompressionConfig used by compression modules (compression/)
- ChunkingStrategy used by chunking module factory (chunking/factory.py)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"


class ContextStrategyEnum(Enum):
    """Available context management strategies."""
    GREEDY = "greedy"                    # Fill with relevant chunks (RAG-style)
    SUMMARIZE = "summarize"              # Summarize older context when exceeding
    COMPRESS = "compress"                # Apply prompt compression
    SLIDING_WINDOW = "sliding_window"    # Use overlapping windows
    HYBRID = "hybrid"                    # Combine multiple strategies


class CompressionMethod(Enum):
    """Available compression methods."""
    OBSERVATION_MASKING = "observation_masking"
    SEMANTIC_SUMMARIZATION = "semantic_summarization"
    RELEVANCE_FILTERING = "relevance_filtering"
    LLM_COMPRESSION = "llm_compression"


@dataclass
class Chunk:
    """
    Represents a unit of text from chunking.

    Attributes:
        content: The text content of the chunk
        token_count: Estimated token count for this chunk
        start_idx: Character position in original document
        end_idx: Character position in original document
        metadata: Additional info (e.g., source, section, importance)
        embedding: Optional embedding vector for semantic operations

    Used by: chunking module, context_manager module
    """
    content: str
    token_count: int
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        """Validate chunk integrity."""
        if self.token_count < 0:
            raise ValueError(f"token_count must be non-negative, got {self.token_count}")
        if len(self.content) == 0:
            raise ValueError("Chunk content cannot be empty")


@dataclass
class ContextWindow:
    """
    Represents current context processing state.

    Attributes:
        current_tokens: Tokens used so far
        max_tokens: Model's native context window limit
        chunks: List of chunks in current window
        compressed_chunks: Chunks that have been compressed
        metadata: Tracking info (e.g., compression_ratio, strategy_used)

    Used by: context_manager module, compression module
    """
    current_tokens: int
    max_tokens: int
    chunks: List[Chunk] = field(default_factory=list)
    compressed_chunks: List[Chunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def available_tokens(self) -> int:
        """Calculate remaining token budget."""
        return max(0, self.max_tokens - self.current_tokens)

    @property
    def utilization(self) -> float:
        """Get utilization percentage (0.0 to 1.0)."""
        if self.max_tokens == 0:
            return 0.0
        return min(1.0, self.current_tokens / self.max_tokens)

    @property
    def all_chunks(self) -> List[Chunk]:
        """Get all chunks (original + compressed)."""
        return self.chunks + self.compressed_chunks

    def can_fit(self, chunk: Chunk, utilization_threshold: float = 0.9) -> bool:
        """
        Check if chunk fits within token budget considering utilization threshold.

        Args:
            chunk: Chunk to check
            utilization_threshold: Max allowed utilization (e.g., 0.9 = 90%)

        Returns:
            True if chunk fits within budget and threshold
        """
        max_allowed_tokens = int(self.max_tokens * utilization_threshold)
        return (self.current_tokens + chunk.token_count) <= max_allowed_tokens


@dataclass
class CompressionConfig:
    """
    Configuration for compression strategies.

    Attributes:
        method: Which compression method to use
        compression_ratio: Target % of tokens to remove (0.3 = 30%)
        preserve_recent_tokens: Min tokens to keep from recent context
        preserve_important: Keep chunks marked as important
        use_cache: Cache compression results

    Used by: compression module (compression/orchestrator.py)
    """
    method: CompressionMethod = CompressionMethod.OBSERVATION_MASKING
    compression_ratio: float = 0.3
    preserve_recent_tokens: int = 1000
    preserve_important: bool = True
    use_cache: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.compression_ratio <= 1.0:
            raise ValueError(f"compression_ratio must be 0-1, got {self.compression_ratio}")
        if self.preserve_recent_tokens < 0:
            raise ValueError(f"preserve_recent_tokens must be non-negative")


@dataclass
class TokenizationResult:
    """
    Result from tokenizing text.

    Attributes:
        token_count: Number of tokens
        tokens: List of actual tokens (if available)
        metadata: Additional info

    Used by: tokenizers module (tokenizers/base.py)
    """
    token_count: int
    tokens: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingRequest:
    """
    Request to process context through the pipeline.

    Attributes:
        content: Text to process
        model_name: Target model (for tokenizer selection)
        strategy: Context management strategy
        compression_config: Optional compression settings

    Used by: context_manager module
    """
    content: str
    model_name: str
    strategy: ContextStrategyEnum = ContextStrategyEnum.GREEDY
    compression_config: Optional[CompressionConfig] = None
