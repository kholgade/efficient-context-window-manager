"""
RLM Orchestrator for high-level REPL-based context processing.

Provides interface similar to ContextWindowManager but using RLM strategies.
"""

from typing import Optional, Dict, Any, Callable
from .processor import RLMProcessor


class RLMOrchestrator:
    """
    High-level orchestrator for RLM-based context processing.

    Provides similar interface to ContextWindowManager but uses
    recursive decomposition instead of pre-processing.
    """

    def __init__(
        self,
        llm_call: Callable,
        model_name: str = "gpt-3.5-turbo",
        max_recursion_depth: int = 5,
        chunk_size: int = 2000,
    ):
        """
        Initialize RLM orchestrator.

        Args:
            llm_call: Function that takes text and returns response
            model_name: Name of the model (for logging)
            max_recursion_depth: Maximum recursion levels
            chunk_size: Chunk size for partition strategy
        """
        self.llm_call = llm_call
        self.model_name = model_name
        self.processor = RLMProcessor(
            llm_call=llm_call,
            max_recursion_depth=max_recursion_depth,
            chunk_size=chunk_size,
        )
        self.call_stats = {
            "total_calls": 0,
            "total_tokens_processed": 0,
            "strategies_used": {},
        }

    def process_document(
        self,
        document: str,
        query: str,
        strategy: str = "auto",
    ) -> Dict[str, Any]:
        """
        Process document with RLM strategies.

        Args:
            document: Full document text
            query: User query
            strategy: Strategy to use ('auto', 'grep', 'partition', 'summarize', 'peek')

        Returns:
            Result with response, metrics, and strategy used
        """
        result = self.processor.process(document, query, strategy)

        # Update stats
        self.call_stats["total_calls"] += result["llm_calls"]
        self.call_stats["total_tokens_processed"] += len(document)

        strategy_used = result["strategy"]
        if strategy_used not in self.call_stats["strategies_used"]:
            self.call_stats["strategies_used"][strategy_used] = 0
        self.call_stats["strategies_used"][strategy_used] += 1

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics."""
        return self.call_stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.call_stats = {
            "total_calls": 0,
            "total_tokens_processed": 0,
            "strategies_used": {},
        }
