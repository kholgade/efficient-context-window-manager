"""
LangGraph framework integration adapter.

Enables LangGraph workflows to use ContextManager for intelligent context
window management within graph nodes.

Example:
    from langgraph.graph import StateGraph
    from efficient_context_window_manager.adapters import LangGraphAdapter

    adapter = LangGraphAdapter(model_name="gpt-4", max_tokens=8000)

    def agent_node(state):
        context = adapter.get_context_for_state(state)
        # Use context in API call
        return {"messages": [...]}

    graph = StateGraph(...)
    # Use adapter in graph nodes
"""

from typing import Optional, Dict, Any, List
from ..core import ContextManager
from ..core.context_manager import ContextManagerConfig
from ..types import Message, ContextStrategyEnum


class LangGraphAdapter:
    """
    Integrates ContextManager with LangGraph workflows.

    Manages context within LangGraph state and provides methods to
    inject optimized context into graph nodes.
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 4096,
        strategy: ContextStrategyEnum = ContextStrategyEnum.HYBRID,
        short_term_tokens: int = 4000,
        long_term_tokens: int = 10000,
    ):
        """
        Initialize LangGraph adapter.

        Args:
            model_name: Target model (e.g., "gpt-4")
            max_tokens: Model's context window
            strategy: Context assembly strategy
            short_term_tokens: Tokens for recent messages
            long_term_tokens: Tokens for compressed history
        """
        config = ContextManagerConfig(
            model_name=model_name,
            max_tokens=max_tokens,
            short_term_tokens=short_term_tokens,
            long_term_tokens=long_term_tokens,
            strategy=strategy,
        )
        self.manager = ContextManager(config)
        self.model_name = model_name
        self.max_tokens = max_tokens

    async def add_message_from_state(
        self,
        message: Dict[str, str],
    ) -> None:
        """
        Add message from LangGraph state.

        Args:
            message: Message dict with 'role' and 'content'
        """
        await self.manager.add_message(
            role=message.get("role", "user"),
            content=message.get("content", ""),
            metadata=message.get("metadata", {})
        )

    async def add_messages_from_state(
        self,
        messages: List[Dict[str, str]],
    ) -> None:
        """
        Add multiple messages from LangGraph state.

        Args:
            messages: List of message dicts
        """
        for msg in messages:
            await self.add_message_from_state(msg)

    async def get_context_for_state(
        self,
        state: Dict[str, Any],
        query: str = "",
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Get optimized context from LangGraph state.

        Intelligently assembles context that respects token limits,
        formatted for use in graph nodes.

        Args:
            state: Current LangGraph state
            query: Query for relevance scoring
            max_tokens: Override default limit

        Returns:
            List of message dicts ready for LLM API
        """
        # Add any messages in state if not already added
        if "messages" in state:
            messages = state["messages"]
            if isinstance(messages, list):
                for msg in messages:
                    # Only add if not already in manager
                    # (Simple deduplication by content hash)
                    if isinstance(msg, dict):
                        await self.add_message_from_state(msg)

        context = await self.manager.assemble_context(query, max_tokens)
        return [
            {
                "role": msg.role,
                "content": msg.content,
                **(msg.metadata or {})
            }
            for msg in context.messages
        ]

    async def add_node_output(
        self,
        node_name: str,
        output: str,
        role: str = "assistant",
    ) -> None:
        """
        Add output from a graph node.

        Args:
            node_name: Name of the node
            output: Node's output
            role: Message role (default assistant)
        """
        await self.manager.add_message(
            role=role,
            content=output,
            metadata={"node": node_name}
        )

    async def add_tool_result_from_node(
        self,
        tool_name: str,
        result: str,
        node_name: str,
    ) -> None:
        """
        Register tool result from graph node.

        Args:
            tool_name: Name of the tool
            result: Tool output
            node_name: Node that called the tool
        """
        await self.manager.add_tool_result(tool_name, result)

    def get_state_update_for_context(
        self,
        current_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Get state update with assembled context.

        Can be used to inject context back into LangGraph state.

        Args:
            current_state: Current state dict

        Returns:
            State update with optimized context
        """
        status = self.manager.get_status()
        return {
            "context_metadata": {
                "tokens_used": status["budget"]["current_tokens"],
                "tokens_available": status["budget"]["available_tokens"],
                "utilization": status["budget"]["utilization_pct"],
                "state": status["budget"]["state"],
            }
        }

    def get_adapter_status(self) -> Dict[str, Any]:
        """Get current status of adapter and manager."""
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "manager_status": self.manager.get_status(),
        }
