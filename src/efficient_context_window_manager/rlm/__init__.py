"""
Recursive Language Models (RLM) for processing unbounded context.

Implements REPL-based context processing with adaptive strategies:
- Peeking: Observe context structure without full read
- Grepping: Pattern matching to narrow search space
- Partition+Map: Recursive calls on chunks with parallel processing
- Summarization: Hierarchical distillation of information

Main classes:
- RLMOrchestrator: REPL-based context orchestrator
- RLMProcessor: Process documents via recursive LLM calls
"""

from .orchestrator import RLMOrchestrator
from .processor import RLMProcessor

__all__ = ["RLMOrchestrator", "RLMProcessor"]
