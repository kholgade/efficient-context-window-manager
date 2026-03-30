"""
Efficient Context Window Manager

A production-ready Python package that breaks large context windows into smaller,
manageable chunks while honoring model native limits and preserving semantic coherence.

Key exports for end users:
- ContextWindowManager: Main orchestrator
- Chunkers: FixedSizeChunker, RecursiveChunker, SlidingWindowChunker
- Tokenizers: AutoTokenizer, OpenAITokenizer, HuggingFaceTokenizer
- Integrations: AnthropicAdapter, OpenAIAdapter, OllamaAdapter
- Types: Chunk, ContextWindow, CompressionConfig

Example:
    from efficient_context_window_manager import ContextWindowManager
    from efficient_context_window_manager.tokenizers import AutoTokenizer
    from efficient_context_window_manager.chunking import RecursiveChunker

    manager = ContextWindowManager(
        tokenizer=AutoTokenizer("gpt-3.5-turbo"),
        chunker=RecursiveChunker(chunk_size=2000),
        max_window_tokens=4000
    )
    chunks = manager.process_document(large_document)
"""

__version__ = "0.1.0"

from .types import (
    Chunk, ContextWindow, CompressionConfig, ChunkingStrategy, ProcessingMode,
    Message, ContextStrategyEnum
)
from .context_manager import ContextWindowManager
from .orchestrator import EfficientLLMCall, efficient_llm_call, LLMProvider
from .core import ContextManager
from .memory import ShortTermMemory, LongTermMemory, ExternalRetrieval
from .assembly import ContextAssembler
from .budgeting import TokenBudgeter
from .adapters import AutoGenAdapter, CrewAIAdapter, LangGraphAdapter

__all__ = [
    "ContextWindowManager",
    "ContextManager",
    "EfficientLLMCall",
    "efficient_llm_call",
    "LLMProvider",
    "Chunk",
    "ContextWindow",
    "CompressionConfig",
    "ChunkingStrategy",
    "ProcessingMode",
    "Message",
    "ContextStrategyEnum",
    "ShortTermMemory",
    "LongTermMemory",
    "ExternalRetrieval",
    "ContextAssembler",
    "TokenBudgeter",
    "AutoGenAdapter",
    "CrewAIAdapter",
    "LangGraphAdapter",
]
