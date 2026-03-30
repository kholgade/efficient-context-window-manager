"""
External retrieval for RAG, tool outputs, and real-time data.

Integrates with RAG systems, tool call results, and external APIs.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import time
from ..types import Message


@dataclass
class ExternalRetrieval:
    """Retrieve context from external sources."""

    tool_results: Dict[str, str] = field(default_factory=dict)
    rag_engine: Optional[Any] = None
    real_time_data: Dict[str, Any] = field(default_factory=dict)

    async def retrieve(
        self,
        query: str,
        max_tokens: int,
        sources: Optional[List[str]] = None,
    ) -> List[Message]:
        """
        Retrieve external context.

        Args:
            query: Search query
            max_tokens: Max tokens to return
            sources: Which sources to use (["rag", "tools", "web"])

        Returns:
            List of context messages
        """
        if sources is None:
            sources = ["tools", "rag"]

        results = []
        tokens_remaining = max_tokens

        # 1. Tool results (highest priority - always preserve)
        if "tools" in sources:
            for tool_name, result in self.tool_results.items():
                msg = Message(
                    role="tool",
                    content=result,
                    metadata={
                        "source": "tool",
                        "tool_name": tool_name,
                        "timestamp": time.time(),
                    },
                )
                msg.token_count = len(result) // 4  # Estimate

                if msg.token_count <= tokens_remaining:
                    results.append(msg)
                    tokens_remaining -= msg.token_count

        # 2. RAG results
        if "rag" in sources and self.rag_engine:
            try:
                rag_results = await self.rag_engine.retrieve(query, top_k=3)
                for result in rag_results:
                    content = (
                        result.get("content", str(result))
                        if isinstance(result, dict)
                        else str(result)
                    )
                    msg = Message(
                        role="context",
                        content=content,
                        metadata={
                            "source": "rag",
                            "score": result.get("score", 0.0)
                            if isinstance(result, dict)
                            else 0.0,
                        },
                    )
                    msg.token_count = len(content) // 4

                    if msg.token_count <= tokens_remaining:
                        results.append(msg)
                        tokens_remaining -= msg.token_count
            except Exception:
                pass  # RAG engine optional

        # 3. Real-time data
        if "web" in sources and self.real_time_data:
            for key, data in self.real_time_data.items():
                msg = Message(
                    role="context",
                    content=str(data),
                    metadata={"source": "real_time", "key": key},
                )
                msg.token_count = len(str(data)) // 4

                if msg.token_count <= tokens_remaining:
                    results.append(msg)
                    tokens_remaining -= msg.token_count

        return results

    def add_tool_result(self, tool_name: str, result: str) -> None:
        """Register tool call result."""
        self.tool_results[tool_name] = result

    def clear_tool_results(self) -> None:
        """Clear all tool results."""
        self.tool_results.clear()

    def add_rag_engine(self, engine: Any) -> None:
        """Register RAG engine."""
        self.rag_engine = engine

    def add_real_time_data(self, key: str, data: Any) -> None:
        """Add real-time data (web search, APIs, etc)."""
        self.real_time_data[key] = data
