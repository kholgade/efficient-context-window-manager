"""
Ollama local LLM adapter.

Integrates with locally running Ollama instance for on-device inference.

Main class:
- OllamaAdapter: Ollama local LLM integration

Used by: User code
Related: base.py (parent)
"""

from typing import List, Dict, Any, Optional
import requests
import json
from .base import BaseAdapter
from ..types import Chunk


class OllamaAdapter(BaseAdapter):
    """
    Adapter for Ollama local LLM.

    Uses context window management for intelligent document processing
    with local models (Llama2, Mistral, etc.).

    Attributes:
        base_url: Ollama server URL (default: http://localhost:11434)
        model_name: Ollama model name
    """

    def __init__(
        self,
        context_manager,
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        """
        Initialize Ollama adapter.

        Args:
            context_manager: ContextWindowManager instance
            model_name: Ollama model name (e.g., 'llama2', 'mistral')
            base_url: Ollama server URL
            **kwargs: Additional config
        """
        super().__init__(context_manager, model_name, **kwargs)
        self.base_url = base_url.rstrip("/")

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
            parts.append(f"[{i+1}] {chunk.content}")

        return "\n\n".join(parts)

    def create_message(
        self,
        system_prompt: str,
        context: str,
        user_query: str,
    ) -> str:
        """
        Create prompt string for Ollama.

        Ollama uses text prompts, not structured messages.

        Args:
            system_prompt: System instructions
            context: Formatted context
            user_query: User's query

        Returns:
            Complete prompt string
        """
        prompt = f"""{system_prompt}

Context:
{context}

Question: {user_query}

Answer:"""
        return prompt

    def call_api(
        self,
        prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """
        Call Ollama API synchronously.

        Args:
            prompt: Complete prompt string
            temperature: Response creativity (0-1)

        Returns:
            API response text
        """
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
            }

            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Ensure Ollama is running: ollama serve"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")

    async def call_api_async(
        self,
        prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """
        Call Ollama API asynchronously.

        Args:
            prompt: Complete prompt string
            temperature: Response creativity

        Returns:
            API response text
        """
        try:
            import aiohttp
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=300) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        raise RuntimeError(f"Ollama returned status {response.status}")

        except ImportError:
            raise ImportError("aiohttp required for async: pip install aiohttp")
        except Exception as e:
            raise RuntimeError(f"Ollama async API error: {e}")

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

        # Extract prompt (message is actually a prompt string for Ollama)
        system = system_prompt
        context = self._format_context_from_message(request_info["message"])
        prompt = self.create_message(system, context, query)

        # Call API
        response = self.call_api(prompt)

        return {
            "response": response,
            "chunks_used": request_info["chunks_used"],
            "tokens_used": request_info["tokens_used"],
            "window_utilization": request_info["window_utilization"],
            "model": self.model_name,
        }

    def _format_context_from_message(self, message: str) -> str:
        """
        Extract context from message (for compatibility with base adapter).

        Args:
            message: Message string

        Returns:
            Context portion
        """
        if isinstance(message, list):
            # Extract from list of dicts
            for msg in message:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if "Context:" in content:
                        return content.split("Context:")[1].split("\n\nQuestion:")[0].strip()
        return ""
