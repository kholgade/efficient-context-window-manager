"""
Token budget management for context windows.

Tracks token usage, enforces budget constraints, and determines when
compression is needed.

Main classes:
- TokenBudgeter: Manages token budgets and allocation

Used by: core/context_manager.py
"""

from .token_budgeter import TokenBudgeter, BudgetState

__all__ = ["TokenBudgeter", "BudgetState"]
