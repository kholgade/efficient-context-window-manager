"""
Context window management orchestrator.

Brings together memory tiers, assembly, budgeting, and compression
into a unified system that works with any framework.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from ..types import Message, ContextStrategyEnum, CompressionConfig
from ..memory.short_term import ShortTermMemory
from ..memory.long_term import LongTermMemory
from ..memory.external import ExternalRetrieval
from ..assembly.assembler import ContextAssembler, AssemblyConfig, AssembledContext
from ..budgeting.token_budgeter import TokenBudgeter, BudgetState
from ..compression.base import BaseCompressor
from ..compression.masking import ObservationMaskingCompressor


@dataclass
class ContextManagerConfig:
    """Configuration for context manager."""

    model_name: str
    max_tokens: int
    # Memory limits
    short_term_tokens: int = 4000
    long_term_tokens: int = 10000
    # Strategy
    strategy: ContextStrategyEnum = ContextStrategyEnum.HYBRID
    # Compression
    auto_compress: bool = True
    compression_threshold: float = 0.85  # Compress when > 85% utilization
    compressor: Optional[BaseCompressor] = None
    # Assembly
    assembly_config: Optional[AssemblyConfig] = None
    # Utilization threshold (default 90% of model's max)
    utilization_threshold: float = 0.9


class ContextManager:
    """
    Unified context window manager.

    Manages three memory tiers (short-term, long-term, external) and
    intelligently assembles context that fits within model's window.

    Works with any framework - users add messages, get assembled context
    that fits within token limit.

    Features:
    - Automatic memory tiering with FIFO eviction
    - Semantic compression of older messages
    - RAG/tool integration with priority handling
    - Token budget tracking and enforcement
    - Configurable assembly strategies
    """

    def __init__(self, config: ContextManagerConfig):
        """
        Initialize context manager.

        Args:
            config: Manager configuration
        """
        self.config = config
        self.budgeter = TokenBudgeter(
            max_tokens=int(config.max_tokens * config.utilization_threshold)
        )

        # Initialize memory tiers
        self.short_term = ShortTermMemory(max_tokens=config.short_term_tokens)
        self.long_term = LongTermMemory(max_compressed_tokens=config.long_term_tokens)
        self.external = ExternalRetrieval()

        # Initialize compressor
        if config.compressor is None:
            compression_cfg = CompressionConfig()
            self.compressor = ObservationMaskingCompressor(compression_cfg)
        else:
            self.compressor = config.compressor

        # Initialize assembler
        assembly_cfg = config.assembly_config or AssemblyConfig(strategy=config.strategy)
        self.assembler = ContextAssembler(assembly_cfg)

        # Metadata tracking
        self.metadata: Dict[str, Any] = {
            "total_messages_added": 0,
            "compressions_triggered": 0,
            "assembly_count": 0,
        }

    async def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Message]:
        """
        Add message to short-term memory.

        Messages are automatically promoted to long-term (compressed) if
        they exceed short-term capacity.

        Args:
            role: Message role (user, assistant, tool, etc)
            content: Message content
            metadata: Optional metadata

        Returns:
            Evicted message (if any) that was promoted to long-term
        """
        msg = Message(
            role=role,
            content=content,
            token_count=len(content) // 4,  # Estimate
            metadata=metadata or {}
        )

        self.metadata["total_messages_added"] += 1

        # Add to short-term, which may evict if over capacity
        evicted = self.short_term.add(msg)

        # Promote evicted messages to long-term
        if evicted:
            self.long_term.store([evicted])

        # Track budget
        self.budgeter.allocate(msg.token_count, source="short_term")

        # Trigger compression if needed
        if self.config.auto_compress and self.budgeter.state == BudgetState.CRITICAL:
            await self._trigger_compression()

        return evicted

    async def add_tool_result(self, tool_name: str, result: str) -> None:
        """
        Register tool result in external retrieval.

        Tool results have highest priority in assembly.

        Args:
            tool_name: Name of the tool
            result: Tool output
        """
        self.external.add_tool_result(tool_name, result)
        tokens = len(result) // 4
        self.budgeter.allocate(tokens, source="tool")

    def set_rag_engine(self, engine: Any) -> None:
        """Register RAG engine for semantic retrieval."""
        self.external.add_rag_engine(engine)

    def add_real_time_data(self, key: str, data: Any) -> None:
        """Add real-time data (web search, APIs, etc)."""
        self.external.add_real_time_data(key, data)

    async def assemble_context(
        self,
        query: str,
        max_tokens: Optional[int] = None,
    ) -> AssembledContext:
        """
        Assemble context from all memory sources.

        Intelligently merges recent messages, compressed history, and
        external sources to maximize relevant context within token limit.

        Args:
            query: Current query (for retrieving relevant history)
            max_tokens: Override default max tokens (use config if not specified)

        Returns:
            AssembledContext with merged messages
        """
        max_tokens = max_tokens or int(self.config.max_tokens * self.config.utilization_threshold)

        context = await self.assembler.assemble(
            short_term=self.short_term,
            long_term=self.long_term,
            external=self.external,
            query=query,
            max_tokens=max_tokens,
            utilization_threshold=1.0,  # Assembler respects its own limit
        )

        self.metadata["assembly_count"] += 1
        return context

    async def _trigger_compression(self) -> None:
        """Trigger compression of oldest messages in short-term."""
        self.metadata["compressions_triggered"] += 1

        # Get oldest messages (up to compression target)
        messages = self.short_term.get_all()
        target_reduction = self.budgeter.compression_target()

        if target_reduction > 0 and messages:
            # Move oldest messages to long-term (which compresses them)
            messages_to_promote = []
            tokens_to_reduce = 0

            for msg in messages:
                if tokens_to_reduce >= target_reduction:
                    break
                messages_to_promote.append(msg)
                tokens_to_reduce += msg.token_count

            # Remove from short-term
            for msg in messages_to_promote:
                self.short_term.messages.remove(msg)

            # Compress and store in long-term
            self.long_term.store(messages_to_promote)

            # Update budget
            for msg in messages_to_promote:
                self.budgeter.deallocate(msg.token_count, source="short_term")

    def clear(self) -> None:
        """Clear all memory and reset budget."""
        self.short_term.clear()
        self.long_term.clear()
        self.external.clear_tool_results()
        self.budgeter.reset()

    def get_status(self) -> Dict[str, Any]:
        """Get current status of memory and budgets."""
        return {
            "budget": self.budgeter.get_usage_report(),
            "short_term_messages": len(self.short_term.messages),
            "long_term_compressed_messages": len(self.long_term.compressed_messages),
            "tool_results": len(self.external.tool_results),
            "metadata": dict(self.metadata),
        }

    def update_strategy(self, strategy: ContextStrategyEnum) -> None:
        """Update context assembly strategy."""
        self.config.strategy = strategy
        self.assembler.update_config(strategy=strategy)
