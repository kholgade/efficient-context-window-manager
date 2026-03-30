"""
CrewAI framework integration adapter.

Enables CrewAI agents to use ContextManager for intelligent context
window management across task execution.

Example:
    from crewai import Agent, Task, Crew
    from efficient_context_window_manager.adapters import CrewAIAdapter

    adapter = CrewAIAdapter(model_name="gpt-4", max_tokens=8000)
    # Use adapter.get_context() before each task execution
"""

from typing import Optional, Dict, Any, List
from ..core import ContextManager
from ..core.context_manager import ContextManagerConfig
from ..types import Message, ContextStrategyEnum


class CrewAIAdapter:
    """
    Integrates ContextManager with CrewAI framework.

    Manages context across CrewAI agents and tasks, intelligently
    assembling context that respects token limits.
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
        Initialize CrewAI adapter.

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
        self.current_task = None
        self.task_history: Dict[str, List[Message]] = {}

    async def start_task(self, task_name: str) -> None:
        """
        Mark start of a task.

        Args:
            task_name: Name of the task starting
        """
        self.current_task = task_name
        if task_name not in self.task_history:
            self.task_history[task_name] = []

    async def add_agent_response(
        self,
        agent_name: str,
        response: str,
        task_name: Optional[str] = None,
    ) -> None:
        """
        Add agent's response to context.

        Args:
            agent_name: Name of the agent
            response: Agent's output
            task_name: Optional override for current task
        """
        task = task_name or self.current_task or "default"
        await self.manager.add_message(
            role="agent",
            content=response,
            metadata={
                "agent": agent_name,
                "task": task,
            }
        )

    async def add_task_input(
        self,
        task_name: str,
        task_input: str,
    ) -> None:
        """
        Add task input to context.

        Args:
            task_name: Name of the task
            task_input: Task description or input
        """
        await self.manager.add_message(
            role="task",
            content=task_input,
            metadata={"task": task_name}
        )

    async def get_context_for_agent(
        self,
        agent_name: str,
        task_name: Optional[str] = None,
        query: str = "",
    ) -> List[Dict[str, str]]:
        """
        Get optimized context for agent's next step.

        Returns context assembled from all sources, formatted for
        CrewAI's expected format.

        Args:
            agent_name: Agent name requesting context
            task_name: Current task name
            query: Query for relevance scoring

        Returns:
            List of message dicts with role and content
        """
        context = await self.manager.assemble_context(query)
        return [
            {"role": msg.role, "content": msg.content}
            for msg in context.messages
        ]

    async def add_tool_output(
        self,
        tool_name: str,
        output: str,
        agent_name: str,
    ) -> None:
        """
        Register tool output.

        Args:
            tool_name: Name of the tool executed
            output: Tool's output
            agent_name: Agent that executed the tool
        """
        await self.manager.add_tool_result(tool_name, output)

    async def end_task(
        self,
        task_name: str,
        result: Optional[str] = None,
    ) -> None:
        """
        Mark end of a task.

        Args:
            task_name: Name of the task
            result: Final result (optional)
        """
        if result:
            await self.manager.add_message(
                role="task_result",
                content=result,
                metadata={"task": task_name}
            )
        self.current_task = None

    async def get_task_summary(self, task_name: str) -> Dict[str, Any]:
        """
        Get summary of a task's context.

        Args:
            task_name: Name of the task

        Returns:
            Task summary with token stats and key messages
        """
        status = self.manager.get_status()
        return {
            "task": task_name,
            "total_tokens": status["budget"]["current_tokens"],
            "messages": len(status["short_term_messages"]),
            "status": status["budget"]["state"],
        }

    def get_adapter_status(self) -> Dict[str, Any]:
        """Get current status of adapter and manager."""
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "current_task": self.current_task,
            "manager_status": self.manager.get_status(),
        }
