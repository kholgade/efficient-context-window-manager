"""
Tiered memory management system.

Provides short-term, long-term, and external memory abstractions for
building context-aware systems.

Main classes:
- ShortTermMemory: Recent, uncompressed messages
- LongTermMemory: Compressed, searchable history
- ExternalRetrieval: RAG, tool outputs, real-time data
"""

from .short_term import ShortTermMemory
from .long_term import LongTermMemory
from .external import ExternalRetrieval

__all__ = [
    "ShortTermMemory",
    "LongTermMemory",
    "ExternalRetrieval",
]
