"""
Tiered memory management for context window optimization.

Manages short-term (recent), long-term (compressed), and external (RAG/tools)
memory sources with automatic tiering and intelligent retrieval.

Main classes:
- ShortTermMemory: Recent, uncompressed messages (0-4h)
- LongTermMemory: Compressed, searchable history (4h+)
- ExternalRetrieval: RAG, tool outputs, real-time data

Used by: assembly/assembler.py
Related: long_term.py, external.py
"""

import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from ..types import Message


@dataclass
class ShortTermMemory:
    """Recent, uncompressed messages with FIFO eviction."""

    max_tokens: int = 4000
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add(self, message: Message) -> Optional[Message]:
        """
        Add message, auto-evict oldest if over limit.

        Args:
            message: Message to add

        Returns:
            Evicted message (if any) for promotion to long-term
        """
        self.messages.append(message)
        return self._evict_if_needed()

    def _evict_if_needed(self) -> Optional[Message]:
        """FIFO eviction when over capacity."""
        while sum(m.token_count for m in self.messages) > self.max_tokens:
            if self.messages:
                return self.messages.pop(0)
        return None

    def get_all(self) -> List[Message]:
        """Return all messages in order."""
        return self.messages

    def get_recent(self, max_messages: int) -> List[Message]:
        """Get N most recent messages."""
        return self.messages[-max_messages:] if max_messages > 0 else []

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()
