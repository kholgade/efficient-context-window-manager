"""
Context assembly from tiered memory sources.

Combines short-term, long-term, and external memory with intelligent
prioritization and token budgeting.

Main classes:
- ContextAssembler: Intelligently merges memory sources based on strategy

Used by: core/context_manager.py
Related: memory/short_term.py, memory/long_term.py, memory/external.py
"""

from .assembler import ContextAssembler

__all__ = ["ContextAssembler"]
