"""
OpenAI API adapter.

Integrates with OpenAI's Python client for GPT models.

Main class:
- OpenAIAdapter: OpenAI API integration

Used by: User code
Related: base.py (parent)
"""

from typing import List, Dict, Any, Optional
from .base import BaseAdapter
from ..types import Chunk


class OpenAIAdapter(BaseAdapter):
    """
    Adapter for OpenAI ChatCompletion API.

    Uses context window manager for intelligent chunking of large documents.

    Supports: gpt-3.5-turbo, gpt-4, gpt-4-turbo, etc.

    Attributes:
        client: OpenAI client (lazy-loaded)
        model_name: OpenAI model name
    """

    def __init__(
        self,
        context_manager,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI adapter.

        Args:
            context_manager: ContextWindowManager instance
            model_name: OpenAI model name
            api_key: Optional API key (uses env var if not provided)
            **kwargs: Additional config
        """
        super().__init__(context_manager, model_name, **kwargs)
        self._client = None
        self.api_key = api_key

    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai library required: pip install openai")
        return self._client

    def format_context(self, chunks: List[Chunk]) -> str:
        """
        Format chunks as context string.

        Args:
            chunks: List of chunks

        Returns:
            Formatted context string
        """
        if not chunks:
            return ""

        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(f"<chunk index=\"{i+1}\">\n{chunk.content}\n</chunk>")

        return "\n\n".join(parts)

    def create_message(
        self,
        system_prompt: str,
        context: str,
        user_query: str,
    ) -> List[Dict[str, str]]:
        """
        Create OpenAI ChatCompletion format message.

        Args:
            system_prompt: System instructions
            context: Formatted context
            user_query: User's query

        Returns:
            List of message dicts
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuery: {user_query}"
            }
        ]
        return messages

    def call_api(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Call OpenAI API synchronously.

        Args:
            messages: Formatted messages
            max_tokens: Max tokens in response
            temperature: Response creativity (0-2)

        Returns:
            API response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    async def call_api_async(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Call OpenAI API asynchronously.

        Args:
            messages: Formatted messages
            max_tokens: Max tokens in response
            temperature: Response creativity

        Returns:
            API response text
        """
        try:
            from openai import AsyncOpenAI
            async_client = AsyncOpenAI(api_key=self.api_key)
            response = await async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except ImportError:
            raise ImportError("openai async support requires openai library")
        except Exception as e:
            raise RuntimeError(f"OpenAI async API error: {e}")

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
        # Prepare request
        request_info = self.prepare_request(document, query, system_prompt)

        # Extract messages
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
