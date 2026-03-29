"""
Main context window manager orchestrator.

Coordinates chunking, compression, and context management strategies.
Key entry point for end users.

Main class:
- ContextWindowManager: Orchestrates entire pipeline

Used by: User code, integrations/
Related files:
- types.py: Data structures
- tokenizers/: Token counting
- chunking/: Chunking strategies
- compression/: Compression strategies
"""

from typing import List, Optional, Callable, Dict, Any
from .types import (
    Chunk, ContextWindow, CompressionConfig, ChunkingStrategy,
    ContextStrategyEnum, ProcessingRequest, CompressionMethod
)
from .tokenizers.base import BaseTokenizer
from .tokenizers.auto import AutoTokenizer
from .chunking.factory import get_chunker
from .chunking.base import BaseChunker
from .compression.base import BaseCompressor
from .compression.masking import ObservationMaskingCompressor
from .compression.summarization import SemanticSummarizationCompressor


class ContextWindowManager:
    """
    Main orchestrator for context window management.

    Coordinates:
    1. Document chunking (respecting semantic boundaries)
    2. Token budgeting (respecting model limits)
    3. Compression strategies (when approaching limits)
    4. Context window creation (for API calls)

    Attributes:
        tokenizer: Token counter
        chunker: Chunking strategy
        max_window_tokens: Model's context window size
        compression_config: Compression settings
        context_strategy: How to manage context
    """

    def __init__(
        self,
        tokenizer: Optional[BaseTokenizer] = None,
        model_name: str = "gpt-3.5-turbo",
        max_window_tokens: int = 4096,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 1000,
        overlap: int = 100,
        context_strategy: ContextStrategyEnum = ContextStrategyEnum.GREEDY,
        compression_config: Optional[CompressionConfig] = None,
        embedding_fn: Optional[Callable] = None,
    ):
        """
        Initialize context window manager.

        Args:
            tokenizer: Token counter (auto-detect if None)
            model_name: Target model name (for auto tokenizer)
            max_window_tokens: Model's context limit
            chunking_strategy: How to chunk documents
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks
            context_strategy: How to fill context windows
            compression_config: Optional compression settings
            embedding_fn: For semantic chunking
        """
        # Tokenizer
        self.tokenizer = tokenizer or AutoTokenizer(model_name)
        self.model_name = model_name
        self.max_window_tokens = max_window_tokens

        # Chunking
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunker = get_chunker(
            self.tokenizer,
            chunking_strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            embedding_fn=embedding_fn,
        )

        # Context strategy
        self.context_strategy = context_strategy

        # Compression
        self.compression_config = compression_config or CompressionConfig()
        self._init_compressor()

        # Tracking
        self.last_window: Optional[ContextWindow] = None
        self.compression_stats = {
            "total_compressions": 0,
            "total_tokens_saved": 0,
            "avg_compression_ratio": 0.0,
        }

    def _init_compressor(self) -> None:
        """Initialize compressor based on config."""
        if self.compression_config.method == CompressionMethod.OBSERVATION_MASKING:
            self.compressor = ObservationMaskingCompressor(self.compression_config)
        elif self.compression_config.method == CompressionMethod.SEMANTIC_SUMMARIZATION:
            self.compressor = SemanticSummarizationCompressor(self.compression_config)
        else:
            # Default
            self.compressor = ObservationMaskingCompressor(self.compression_config)

    def process_document(
        self,
        document: str,
        strategy: Optional[ContextStrategyEnum] = None,
    ) -> List[Chunk]:
        """
        Process a document through chunking pipeline.

        Args:
            document: Input document text
            strategy: Optional override of default strategy

        Returns:
            List of chunks ready for context windows
        """
        if not document or len(document) == 0:
            return []

        # Chunk document
        chunks = self.chunker.chunk(document)

        return chunks

    def create_context_window(
        self,
        chunks: List[Chunk],
        query: Optional[str] = None,
        preserve_recent: bool = True,
    ) -> ContextWindow:
        """
        Create a context window from chunks.

        Selects chunks to fit within model's token limit while maximizing relevance.

        Args:
            chunks: Available chunks
            query: Optional query for relevance (RAG-style)
            preserve_recent: Keep most recent chunks

        Returns:
            ContextWindow with selected chunks
        """
        if not chunks:
            return ContextWindow(
                current_tokens=0,
                max_tokens=self.max_window_tokens,
                chunks=[],
            )

        window = ContextWindow(
            current_tokens=0,
            max_tokens=self.max_window_tokens,
            chunks=[],
        )

        # Strategy-specific selection
        if self.context_strategy == ContextStrategyEnum.GREEDY:
            selected = self._select_greedy(chunks, window)
        elif self.context_strategy == ContextStrategyEnum.SUMMARIZE:
            selected = self._select_with_summarization(chunks, window)
        elif self.context_strategy == ContextStrategyEnum.COMPRESS:
            selected = self._select_with_compression(chunks, window)
        elif self.context_strategy == ContextStrategyEnum.SLIDING_WINDOW:
            selected = self._select_sliding_window(chunks, window)
        else:
            selected = self._select_greedy(chunks, window)

        window.chunks = selected
        self.last_window = window
        return window

    def _select_greedy(self, chunks: List[Chunk], window: ContextWindow) -> List[Chunk]:
        """
        Greedy selection: add chunks in order until full.

        Args:
            chunks: Available chunks
            window: Current context window

        Returns:
            Selected chunks
        """
        selected = []
        for chunk in chunks:
            if window.can_fit(chunk):
                selected.append(chunk)
                window.current_tokens += chunk.token_count
            else:
                # If window is mostly full, stop
                if window.utilization > 0.85:
                    break
                # Otherwise try next chunk

        return selected

    def _select_with_summarization(
        self,
        chunks: List[Chunk],
        window: ContextWindow,
    ) -> List[Chunk]:
        """
        Selection with on-demand summarization of old chunks.

        Args:
            chunks: Available chunks
            window: Current context window

        Returns:
            Selected and possibly compressed chunks
        """
        selected = []

        for i, chunk in enumerate(chunks):
            if window.can_fit(chunk):
                selected.append(chunk)
                window.current_tokens += chunk.token_count
            elif window.utilization < 1.0:
                # Try compressing older chunks to make room
                if len(selected) > 1:
                    # Compress older chunks
                    for j in range(len(selected) - 1):
                        old_chunk = selected[j]
                        if not old_chunk.metadata.get("is_summary"):
                            compressed = self.compressor.compress(old_chunk)
                            reduction = old_chunk.token_count - compressed.token_count
                            window.current_tokens -= reduction
                            selected[j] = compressed
                            self.compression_stats["total_compressions"] += 1
                            self.compression_stats["total_tokens_saved"] += reduction

                        if window.can_fit(chunk):
                            selected.append(chunk)
                            window.current_tokens += chunk.token_count
                            break
                else:
                    # Can't fit, stop
                    break
            else:
                break

        return selected

    def _select_with_compression(
        self,
        chunks: List[Chunk],
        window: ContextWindow,
    ) -> List[Chunk]:
        """
        Selection with upfront compression.

        Args:
            chunks: Available chunks
            window: Current context window

        Returns:
            Compressed chunks fitting in window
        """
        # Compress all chunks
        compressed_chunks = [self.compressor.compress(c) for c in chunks]

        selected = []
        for chunk in compressed_chunks:
            if window.can_fit(chunk):
                selected.append(chunk)
                window.current_tokens += chunk.token_count
            elif window.utilization > 0.9:
                break

        return selected

    def _select_sliding_window(
        self,
        chunks: List[Chunk],
        window: ContextWindow,
    ) -> List[Chunk]:
        """
        Sliding window selection: select contiguous chunks.

        Args:
            chunks: Available chunks
            window: Current context window

        Returns:
            Contiguous window of chunks
        """
        # Find best contiguous window
        best_window = []
        best_tokens = 0

        for start in range(len(chunks)):
            current = []
            current_tokens = 0

            for end in range(start, len(chunks)):
                chunk = chunks[end]
                if current_tokens + chunk.token_count <= int(self.max_window_tokens * 0.9):
                    current.append(chunk)
                    current_tokens += chunk.token_count
                else:
                    break

            if current_tokens > best_tokens:
                best_window = current
                best_tokens = current_tokens

        window.current_tokens = best_tokens
        return best_window

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens for text."""
        return self.tokenizer.count_tokens(text)

    def get_metrics(self) -> Dict[str, Any]:
        """Get compression and management metrics."""
        return {
            **self.compression_stats,
            "model": self.model_name,
            "max_window_tokens": self.max_window_tokens,
            "context_strategy": self.context_strategy.value,
            "last_window_utilization": (
                self.last_window.utilization if self.last_window else 0.0
            ),
        }
