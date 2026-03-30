"""
Context assembly from tiered memory sources.

Intelligently merges short-term, long-term, and external memory with
priority-based token allocation.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from ..types import Message, ContextStrategyEnum
from ..memory.short_term import ShortTermMemory
from ..memory.long_term import LongTermMemory
from ..memory.external import ExternalRetrieval


@dataclass
class AssemblyConfig:
    """Configuration for context assembly."""

    strategy: ContextStrategyEnum = ContextStrategyEnum.HYBRID
    # Allocation percentages (should sum to ~100%)
    tool_reserve_tokens: int = 500          # Always reserve for tools
    recent_tokens_pct: float = 0.4          # 40% for recent messages
    compressed_tokens_pct: float = 0.4      # 40% for compressed history
    external_tokens_pct: float = 0.2        # 20% for RAG/external
    preserve_recent_count: int = 5          # Always preserve N most recent


@dataclass
class AssembledContext:
    """Result of assembly operation."""

    messages: List[Message] = field(default_factory=list)
    total_tokens: int = 0
    tool_tokens: int = 0
    recent_tokens: int = 0
    compressed_tokens: int = 0
    external_tokens: int = 0
    metadata: dict = field(default_factory=dict)


class ContextAssembler:
    """
    Intelligently assembles context from tiered memory sources.

    Priority order:
    1. Tool results (highest - always preserved)
    2. Recent messages (short-term)
    3. Compressed history (long-term)
    4. External/RAG (lowest - included if space permits)

    Token allocation is configurable and respects model's context window.
    """

    def __init__(self, config: Optional[AssemblyConfig] = None):
        """
        Initialize assembler.

        Args:
            config: Assembly configuration (uses defaults if not provided)
        """
        self.config = config or AssemblyConfig()

    async def assemble(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        external: ExternalRetrieval,
        query: str,
        max_tokens: int,
        utilization_threshold: float = 0.9,
    ) -> AssembledContext:
        """
        Assemble context from all memory sources.

        Args:
            short_term: Recent messages
            long_term: Compressed history
            external: Tool results, RAG, real-time data
            query: Current query (for retrieving relevant history)
            max_tokens: Model's context window size
            utilization_threshold: Max utilization % (default 0.9 = 90%)

        Returns:
            AssembledContext with merged messages and token counts
        """
        budget = int(max_tokens * utilization_threshold)
        result = AssembledContext()
        tokens_used = 0

        # 1. Tools (highest priority - always preserve if possible)
        tool_messages = external.tool_results
        if tool_messages:
            tool_budget = self.config.tool_reserve_tokens
            tool_list = self._get_tool_messages(external, tool_budget)
            for msg in tool_list:
                if tokens_used + msg.token_count <= budget:
                    result.messages.append(msg)
                    tokens_used += msg.token_count
                    result.tool_tokens += msg.token_count

        remaining_budget = budget - tokens_used

        # 2. Recent messages (short-term)
        recent_budget = int(remaining_budget * self.config.recent_tokens_pct)
        recent_messages = short_term.get_recent(self.config.preserve_recent_count)
        for msg in recent_messages:
            if tokens_used + msg.token_count <= budget:
                result.messages.append(msg)
                tokens_used += msg.token_count
                result.recent_tokens += msg.token_count

        remaining_budget = budget - tokens_used

        # 3. Compressed history (long-term)
        compressed_budget = int(remaining_budget * self.config.compressed_tokens_pct)
        if compressed_budget > 0:
            compressed_messages = await long_term.retrieve(
                query,
                max_tokens=compressed_budget,
                strategy="hybrid"
            )
            for msg in compressed_messages:
                if tokens_used + msg.token_count <= budget:
                    result.messages.append(msg)
                    tokens_used += msg.token_count
                    result.compressed_tokens += msg.token_count

        remaining_budget = budget - tokens_used

        # 4. External/RAG (lowest priority)
        if remaining_budget > 100:  # Only if meaningful space remains
            external_budget = int(remaining_budget * self.config.external_tokens_pct)
            external_messages = await external.retrieve(
                query,
                max_tokens=external_budget,
                sources=["rag", "web"]
            )
            for msg in external_messages:
                if tokens_used + msg.token_count <= budget:
                    result.messages.append(msg)
                    tokens_used += msg.token_count
                    result.external_tokens += msg.token_count

        result.total_tokens = tokens_used
        result.metadata = {
            "strategy": self.config.strategy.value,
            "utilization": tokens_used / budget if budget > 0 else 0.0,
            "budget": budget,
        }

        return result

    def _get_tool_messages(
        self,
        external: ExternalRetrieval,
        budget: int
    ) -> List[Message]:
        """Extract tool result messages with token limit."""
        messages = []
        tokens_used = 0

        for tool_name, result in external.tool_results.items():
            msg = Message(
                role="tool",
                content=result,
                token_count=len(result) // 4,  # Estimate
                metadata={
                    "source": "tool",
                    "tool_name": tool_name,
                }
            )
            if tokens_used + msg.token_count <= budget:
                messages.append(msg)
                tokens_used += msg.token_count

        return messages

    def update_config(self, **kwargs) -> None:
        """Update assembly configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
