"""
AutoGen framework integration adapter.

Enables AutoGen agents to use ContextManager for intelligent context
window management.

Example:
    from autogen import AssistantAgent
    from efficient_context_window_manager.adapters import AutoGenAdapter

    adapter = AutoGenAdapter(model_name="gpt-4", max_tokens=8000)
    agent = AssistantAgent(name="assistant", llm_config={"model": "gpt-4"})
    adapter.attach_to_agent(agent)

    # Now messages are automatically managed through context manager
"""

from typing import Optional, Dict, Any, List
from ..core import ContextManager
from ..core.context_manager import ContextManagerConfig
from ..types import Message, ContextStrategyEnum


class AutoGenAdapter:
    """
    Integrates ContextManager with AutoGen agents.

    Handles message interception, context assembly, and injection
    of optimized context into AutoGen's message flow.
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
        Initialize AutoGen adapter.

        Args:
            model_name: Target model (e.g., "gpt-4", "gpt-3.5-turbo")
            max_tokens: Model's context window
            strategy: Context assembly strategy
            short_term_tokens: Tokens reserved for recent messages
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

    async def add_message_from_autogen(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add message from AutoGen conversation.

        Args:
            role: Message role (user, assistant, etc)
            content: Message content
            metadata: Optional metadata
        """
        await self.manager.add_message(
            role=role,
            content=content,
            metadata=metadata or {}
        )

    async def add_tool_result_from_autogen(
        self,
        tool_name: str,
        result: str,
    ) -> None:
        """
        Register tool result from AutoGen function calls.

        Args:
            tool_name: Name of the tool
            result: Result from tool execution
        """
        await self.manager.add_tool_result(tool_name, result)

    async def get_assembled_context_for_api_call(
        self,
        query: str = "",
        max_tokens: Optional[int] = None,
    ) -> List[Message]:
        """
        Get optimized context for API call.

        Returns messages assembled from all memory sources, respecting
        token limits.

        Args:
            query: Current query (for relevance retrieval)
            max_tokens: Override default limit

        Returns:
            List of Message objects ready for API call
        """
        context = await self.manager.assemble_context(query, max_tokens)
        return context.messages

    def attach_to_agent(self, agent: Any) -> None:
        """
        Attach adapter to AutoGen agent.

        Modifies agent's message handling to use ContextManager.

        Args:
            agent: AutoGen agent instance
        """
        # Store original receive method
        original_receive = agent.receive if hasattr(agent, 'receive') else None

        async def managed_receive(message: str, sender: Any = None, **kwargs):
            """Intercepted receive method using context manager."""
            # Extract role from sender if available
            role = "user"
            if sender and hasattr(sender, 'name'):
                if "assistant" in sender.name.lower():
                    role = "assistant"
                elif "user" in sender.name.lower():
                    role = "user"

            # Add to context manager
            await self.add_message_from_autogen(role, message)

            # Call original method if it exists
            if original_receive:
                return await original_receive(message, sender, **kwargs)

        agent.receive = managed_receive

    def get_adapter_status(self) -> Dict[str, Any]:
        """Get current status of adapter and manager."""
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "manager_status": self.manager.get_status(),
        }
