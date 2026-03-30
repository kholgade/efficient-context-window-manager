"""
Token budget management and allocation.

Tracks token usage across context window, enforces limits, and determines
when compression or pruning is needed.
"""

from typing import Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class BudgetState(Enum):
    """States of token budget."""

    HEALTHY = "healthy"              # < 70% utilization
    CAUTION = "caution"              # 70-85% utilization
    CRITICAL = "critical"            # > 85% utilization
    EXHAUSTED = "exhausted"          # >= 100% utilization


@dataclass
class TokenBudgeter:
    """
    Manages token budgets and allocation across context.

    Tracks usage against model's context window, enforces thresholds,
    and recommends compression when needed.
    """

    max_tokens: int
    current_tokens: int = 0
    # Thresholds for state transitions
    caution_threshold: float = 0.70      # 70% triggers caution
    critical_threshold: float = 0.85     # 85% triggers critical
    # Usage breakdown
    usage_breakdown: Dict[str, int] = field(default_factory=dict)

    @property
    def available_tokens(self) -> int:
        """Get remaining token budget."""
        return max(0, self.max_tokens - self.current_tokens)

    @property
    def utilization(self) -> float:
        """Get utilization percentage (0.0 to 1.0)."""
        if self.max_tokens == 0:
            return 0.0
        return min(1.0, self.current_tokens / self.max_tokens)

    @property
    def utilization_pct(self) -> float:
        """Get utilization as percentage (0-100)."""
        return self.utilization * 100

    @property
    def state(self) -> BudgetState:
        """Get current budget state."""
        util = self.utilization
        if util >= 1.0:
            return BudgetState.EXHAUSTED
        elif util >= self.critical_threshold:
            return BudgetState.CRITICAL
        elif util >= self.caution_threshold:
            return BudgetState.CAUTION
        else:
            return BudgetState.HEALTHY

    def can_fit(self, tokens: int, safety_margin: float = 0.05) -> bool:
        """
        Check if tokens fit within budget with safety margin.

        Args:
            tokens: Number of tokens to check
            safety_margin: Percentage to reserve (default 5%)

        Returns:
            True if tokens fit within budget
        """
        max_allowed = int(self.max_tokens * (1.0 - safety_margin))
        return (self.current_tokens + tokens) <= max_allowed

    def allocate(self, tokens: int, source: str = "unknown") -> bool:
        """
        Allocate tokens from budget.

        Args:
            tokens: Number of tokens to allocate
            source: Source identifier (for tracking)

        Returns:
            True if allocation successful, False if would exceed budget
        """
        if self.current_tokens + tokens > self.max_tokens:
            return False

        self.current_tokens += tokens
        self.usage_breakdown[source] = self.usage_breakdown.get(source, 0) + tokens
        return True

    def deallocate(self, tokens: int, source: str = "unknown") -> bool:
        """
        Return tokens to budget.

        Args:
            tokens: Number of tokens to return
            source: Source identifier

        Returns:
            True if deallocation successful
        """
        if source not in self.usage_breakdown or self.usage_breakdown[source] < tokens:
            return False

        self.current_tokens = max(0, self.current_tokens - tokens)
        self.usage_breakdown[source] -= tokens
        return True

    def reset(self) -> None:
        """Reset budget to zero usage."""
        self.current_tokens = 0
        self.usage_breakdown.clear()

    def should_compress(self) -> bool:
        """Determine if compression should be triggered."""
        return self.state in [BudgetState.CRITICAL, BudgetState.EXHAUSTED]

    def should_prune(self) -> bool:
        """Determine if pruning should be triggered."""
        return self.state == BudgetState.CRITICAL

    def compression_target(self) -> int:
        """
        Get target compression amount.

        Returns target tokens to reduce to reach caution threshold.
        """
        caution_tokens = int(self.max_tokens * self.caution_threshold)
        return max(0, self.current_tokens - caution_tokens)

    def get_usage_report(self) -> Dict[str, any]:
        """Get detailed usage report."""
        return {
            "current_tokens": self.current_tokens,
            "max_tokens": self.max_tokens,
            "available_tokens": self.available_tokens,
            "utilization_pct": self.utilization_pct,
            "state": self.state.value,
            "breakdown": dict(self.usage_breakdown),
        }
