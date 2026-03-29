"""
RLM Processor for recursive language model operations.

Implements REPL environment with adaptive context handling strategies.
"""

import re
from typing import List, Optional, Dict, Any, Callable
from ..types import Chunk


class RLMProcessor:
    """
    Process documents using Recursive Language Model approach.

    Treats input context as variables in a virtual environment,
    allowing the LLM to programmatically grep, partition, and recurse.
    """

    def __init__(
        self,
        llm_call: Callable,
        max_recursion_depth: int = 5,
        chunk_size: int = 2000,
    ):
        """
        Initialize RLM processor.

        Args:
            llm_call: Function that takes text and returns LLM response
            max_recursion_depth: Maximum recursion levels
            chunk_size: Size of chunks for partition+map
        """
        self.llm_call = llm_call
        self.max_recursion_depth = max_recursion_depth
        self.chunk_size = chunk_size
        self.call_count = 0
        self.max_calls = 100  # Safety limit

    def process(
        self,
        context: str,
        query: str,
        strategy: str = "auto",
    ) -> Dict[str, Any]:
        """
        Process document via adaptive RLM strategies.

        Args:
            context: Full input context
            query: User query
            strategy: 'auto', 'grep', 'partition', 'summarize', 'peek'

        Returns:
            Dict with response, strategy used, and metrics
        """
        self.call_count = 0

        # Automatically select strategy based on context and query
        if strategy == "auto":
            strategy = self._select_strategy(context, query)

        if strategy == "peek":
            result = self._peek(context, query)
        elif strategy == "grep":
            result = self._grep(context, query)
        elif strategy == "partition":
            result = self._partition_map(context, query)
        elif strategy == "summarize":
            result = self._summarize(context, query)
        else:
            # Default to partition for large contexts
            result = self._partition_map(context, query)

        result["strategy"] = strategy
        result["llm_calls"] = self.call_count
        return result

    def _select_strategy(self, context: str, query: str) -> str:
        """Automatically select best strategy based on context/query."""
        context_size = len(context)
        query_lower = query.lower()

        # Grep for keyword-based queries
        if any(kw in query_lower for kw in ["find", "search", "locate", "where"]):
            return "grep"

        # Summarize for aggregation queries
        if any(kw in query_lower for kw in ["summarize", "overview", "main", "key"]):
            return "summarize"

        # Peek for structure-learning queries
        if any(kw in query_lower for kw in ["structure", "format", "layout", "organization"]):
            return "peek"

        # Default: partition for large contexts
        if context_size > 50000:
            return "partition"

        return "partition"

    def _peek(self, context: str, query: str) -> Dict[str, Any]:
        """
        Peek strategy: Observe structure without full read.

        Shows sample lines to understand format.
        """
        lines = context.split("\n")
        sample_lines = min(10, len(lines))
        sample = "\n".join(lines[:sample_lines])

        prompt = f"""Given this sample of the document structure:
{sample}

And this query: {query}

What structure/format do you observe? List the main sections or patterns."""

        response = self.llm_call(prompt)
        self.call_count += 1

        return {
            "response": response,
            "context_size": len(context),
            "sample_size": len(sample),
        }

    def _grep(self, context: str, query: str) -> Dict[str, Any]:
        """
        Grep strategy: Pattern matching to narrow search.

        Extract query-relevant lines using regex/keywords.
        """
        # Extract keywords from query
        keywords = re.findall(r"\b\w+\b", query.lower())
        keywords = [kw for kw in keywords if len(kw) > 3]

        # Find matching lines
        lines = context.split("\n")
        matching_lines = []

        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in keywords):
                # Include context (line before and after)
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                matching_lines.extend(lines[start:end])

        filtered_context = "\n".join(matching_lines[-100:])  # Last 100 matching lines

        if not filtered_context:
            filtered_context = context[:10000]  # Fallback to start of document

        prompt = f"""Based on this filtered content:
{filtered_context}

Answer: {query}"""

        response = self.llm_call(prompt)
        self.call_count += 1

        return {
            "response": response,
            "original_size": len(context),
            "filtered_size": len(filtered_context),
            "reduction": 1 - len(filtered_context) / len(context),
        }

    def _partition_map(self, context: str, query: str) -> Dict[str, Any]:
        """
        Partition+Map strategy: Split and recursively process chunks.

        Divides context into chunks, processes each, then synthesizes.
        """
        # Split into chunks
        chunks = self._partition_context(context)

        # Process each chunk (if recursion depth allows)
        if self.call_count >= self.max_calls:
            return self._summarize(context, query)

        chunk_responses = []
        for chunk in chunks:
            if self.call_count >= self.max_calls:
                break

            prompt = f"""From this section:
{chunk}

Extract information relevant to: {query}
Be concise."""

            response = self.llm_call(prompt)
            self.call_count += 1
            chunk_responses.append(response)

        # Synthesize responses
        if chunk_responses:
            synthesis_prompt = f"""Synthesize these findings to answer: {query}

Findings:
{chr(10).join(chunk_responses)}"""
        else:
            synthesis_prompt = f"Answer: {query}"

        final_response = self.llm_call(synthesis_prompt)
        self.call_count += 1

        return {
            "response": final_response,
            "chunks_processed": len(chunks),
            "total_llm_calls": self.call_count,
        }

    def _summarize(self, context: str, query: str) -> Dict[str, Any]:
        """
        Summarization strategy: Hierarchical distillation.

        Summarize context sections and synthesize.
        """
        chunks = self._partition_context(context)

        # Summarize each chunk
        summaries = []
        for i, chunk in enumerate(chunks):
            if self.call_count >= self.max_calls:
                break

            prompt = f"""Summarize this section in 2-3 sentences:
{chunk}"""

            summary = self.llm_call(prompt)
            self.call_count += 1
            summaries.append(summary)

        # Answer query based on summaries
        combined_summary = "\n".join(summaries)
        final_prompt = f"""Based on these summaries:
{combined_summary}

Answer: {query}"""

        response = self.llm_call(final_prompt)
        self.call_count += 1

        return {
            "response": response,
            "summaries_created": len(summaries),
            "total_llm_calls": self.call_count,
        }

    def _partition_context(self, context: str, chunk_size: Optional[int] = None) -> List[str]:
        """Split context into chunks by paragraph, then by size if needed."""
        if chunk_size is None:
            chunk_size = self.chunk_size

        chunks = []
        paragraphs = context.split("\n\n")

        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)
            if current_size + para_size > chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks
