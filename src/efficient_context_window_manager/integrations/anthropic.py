"""
Anthropic Claude API adapter.

Integrates with Anthropic's Python client for Claude models.

Main class:
- AnthropicAdapter: Anthropic API integration

Used by: User code
Related: base.py (parent)
"""

from typing import List, Dict, Any, Optional
import asyncio
from .base import BaseAdapter
from ..types import Chunk


class AnthropicAdapter(BaseAdapter):
    """
    Adapter for Anthropic Claude API.

    Uses context window manager to handle long documents via intelligent chunking
    and compression.

    Supports: claude-3-opus, claude-3-sonnet, etc.

    Attributes:
        client: Anthropic client (lazy-loaded)
        model_name: Claude model name
    """

    def __init__(
        self,
        context_manager,
        model_name: str = "claude-3-opus-20250219",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Anthropic adapter.

        Args:
            context_manager: ContextWindowManager instance
            model_name: Claude model name
            api_key: Optional API key (uses env var if not provided)
            **kwargs: Additional config
        """
        super().__init__(context_manager, model_name, **kwargs)
        self._client = None
        self.api_key = api_key

    @property
    def client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic library required: pip install anthropic")
        return self._client

    def format_context(self, chunks: List[Chunk]) -> str:
        """
        Format chunks as context string.

        Simple format: concatenate chunk contents with separators.

        Args:
            chunks: List of chunks

        Returns:
            Formatted context string
        """
        if not chunks:
            return ""

        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(f"[Chunk {i+1}]\n{chunk.content}")

        return "\n\n---\n\n".join(parts)

    def create_message(
        self,
        system_prompt: str,
        context: str,
        user_query: str,
    ) -> List[Dict[str, str]]:
        """
        Create Anthropic-format message.

        Uses Messages API with system prompt.

        Args:
            system_prompt: System instructions
            context: Formatted context
            user_query: User's query

        Returns:
            List of message dicts
        """
        full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {user_query}"

        return [
            {
                "role": "user",
                "content": full_prompt
            }
        ]

    def call_api(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Call Anthropic API synchronously.

        Args:
            messages: Formatted messages
            max_tokens: Max tokens in response
            temperature: Response creativity (0-1)

        Returns:
            API response text
        """
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                system=messages[0].get("system", "You are a helpful assistant."),
                messages=messages,
                temperature=temperature,
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")

    async def call_api_async(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Call Anthropic API asynchronously.

        Args:
            messages: Formatted messages
            max_tokens: Max tokens in response
            temperature: Response creativity

        Returns:
            API response text
        """
        try:
            import anthropic
            async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
            response = await async_client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=messages,
                temperature=temperature,
            )
            return response.content[0].text
        except ImportError:
            raise ImportError("anthropic async support requires anthropic library")
        except Exception as e:
            raise RuntimeError(f"Anthropic async API error: {e}")

    def process_with_context(
        self,
        document: str,
        query: str,
        system_prompt: str = "You are a helpful assistant analyzing documents.",
    ) -> Dict[str, Any]:
        """
        End-to-end processing with context management.

        Args:
            document: Reference document
            query: User query
            system_prompt: System instructions

        Returns:
            Result dict with response and metrics
        """
        # Prepare request with context management
        request_info = self.prepare_request(document, query, system_prompt)

        # Extract message
        messages = request_info["message"]

        # Call API
        response = self.call_api(messages)

        return {
            "response": response,
            "chunks_used": request_info["chunks_used"],
            "tokens_used": request_info["tokens_used"],
            "window_utilization": request_info["window_utilization"],
            "model": self.model_name,
        }
