"""
Framework-agnostic adapters for context window management.

Provides thin wrappers that integrate ContextManager with various
agent frameworks (AutoGen, CrewAI, LangGraph, etc).

Main classes:
- AutoGenAdapter: Integration with AutoGen agents
- CrewAIAdapter: Integration with CrewAI framework
- LangGraphAdapter: Integration with LangGraph workflows

Each adapter handles framework-specific message formatting and context
injection while using ContextManager internally for intelligent memory
management.
"""

from .autogen_adapter import AutoGenAdapter
from .crewai_adapter import CrewAIAdapter
from .langgraph_adapter import LangGraphAdapter

__all__ = [
    "AutoGenAdapter",
    "CrewAIAdapter",
    "LangGraphAdapter",
]
