"""
Base adapter interface for LLM integrations.

Abstracts away API-specific details while using context window management.

Main class:
- BaseAdapter: Abstract interface for LLM APIs

Subclasses: anthropic.py, openai.py, ollama.py
Used by: User code, integration-specific implementations
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..context_manager import ContextWindowManager
from ..types import Chunk, ContextWindow


class BaseAdapter(ABC):
    """
    Abstract adapter for LLM API integration.

    Handles:
    - API-specific request/response formatting
    - Token limit enforcement via context manager
    - Message formatting (system/user/assistant roles)
    - Error handling and retries

    Attributes:
        context_manager: Manages context windows
        model_name: LLM model name
        api_config: API-specific settings
    """

    def __init__(
        self,
        context_manager: ContextWindowManager,
        model_name: str,
        **api_config
    ):
        """
        Initialize adapter.

        Args:
            context_manager: ContextWindowManager instance
            model_name: Model identifier
            **api_config: API-specific configuration
        """
        self.context_manager = context_manager
        self.model_name = model_name
        self.api_config = api_config

    @abstractmethod
    def format_context(self, chunks: List[Chunk]) -> str:
        """
        Format chunks as context string.

        Args:
            chunks: Chunks to format

        Returns:
            Formatted context string for API
        """
        pass

    @abstractmethod
    def create_message(
        self,
        system_prompt: str,
        context: str,
        user_query: str,
    ) -> Dict[str, Any]:
        """
        Create API-specific message structure.

        Args:
            system_prompt: System instructions
            context: Formatted context from chunks
            user_query: User's actual query

        Returns:
            API-specific message dict
        """
        pass

    @abstractmethod
    async def call_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Make API call and get response.

        Args:
            messages: Formatted messages for API

        Returns:
            API response text

        Must be implemented by subclasses.
        """
        pass

    def prepare_request(
        self,
        document: str,
        query: str,
        system_prompt: str = "You are a helpful assistant.",
    ) -> Dict[str, Any]:
        """
        Prepare full request with context management.

        Steps:
        1. Chunk document
        2. Create context window
        3. Format for API
        4. Return ready-to-use request

        Args:
            document: Reference document
            query: User query
            system_prompt: Optional system prompt

        Returns:
            Prepared request dict ready for API
        """
        # Chunk document
        chunks = self.context_manager.process_document(document)

        # Create context window
        window = self.context_manager.create_context_window(chunks, query)

        # Format context
        context_str = self.format_context(window.chunks)

        # Create message
        message = self.create_message(system_prompt, context_str, query)

        return {
            "message": message,
            "chunks_used": len(window.chunks),
            "tokens_used": window.current_tokens,
            "window_utilization": window.utilization,
            "metadata": {
                "model": self.model_name,
                "context_strategy": self.context_manager.context_strategy.value,
                "chunking_strategy": self.context_manager.chunking_strategy.value,
            }
        }

    def estimate_request_tokens(
        self,
        system_prompt: str,
        context: str,
        query: str,
    ) -> int:
        """
        Estimate total tokens for request.

        Args:
            system_prompt: System prompt
            context: Context string
            query: User query

        Returns:
            Estimated token count
        """
        total = (
            self.context_manager.estimate_tokens(system_prompt) +
            self.context_manager.estimate_tokens(context) +
            self.context_manager.estimate_tokens(query) +
            10  # Overhead
        )
        return total
