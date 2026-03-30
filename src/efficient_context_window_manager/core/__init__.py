"""
Core context window management orchestration.

Brings together memory, assembly, budgeting, and compression into a
unified context manager that works with any framework.

Main classes:
- ContextManager: Primary API for intelligent context management

Used by: Framework adapters and applications
"""

from .context_manager import ContextManager

__all__ = ["ContextManager"]
