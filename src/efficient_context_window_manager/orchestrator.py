"""
High-level orchestrator for efficient LLM calls with context window management.

Provides a simple function-based interface for making LLM calls while automatically
managing context windows, chunking, and compression.

Key function:
- efficient_llm_call: Make efficient LLM call with managed context

Design:
1. User creates context manager and accumulates context/documents
2. User calls efficient_llm_call with manager, query, and model
3. Function handles: chunking, context selection, API call, returns response + metrics

Related: integrations/ (adapters), context_manager.py
Used by: User code, high-level applications
"""

from typing import Dict, Any, Optional, List, Callable
from enum import Enum

from .context_manager import ContextWindowManager
from .types import Chunk, ContextWindow, ProcessingMode
from .rlm.orchestrator import RLMOrchestrator


class LLMProvider(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class EfficientLLMCall:
    """
    Orchestrator for efficient LLM calls with context management.

    Manages the complete pipeline:
    1. Chunk accumulated documents
    2. Select relevant chunks within token budget
    3. Format context for LLM
    4. Make API call
    5. Return response with metrics

    Attributes:
        context_manager: ContextWindowManager with accumulated context
        documents: Accumulated documents to process
        model_name: Target LLM model
        provider: LLM provider (auto-detected or specified)
        system_prompt: Instructions for the model
    """

    def __init__(
        self,
        context_manager: ContextWindowManager,
        model_name: str,
        provider: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant.",
    ):
        """
        Initialize efficient LLM call orchestrator.

        Args:
            context_manager: ContextWindowManager instance
            model_name: Target model name
            provider: LLM provider ('anthropic', 'openai', 'ollama')
                     If None, auto-detected from model_name
            system_prompt: System instructions for model
        """
        self.context_manager = context_manager
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.documents: List[str] = []
        self.chunks: List[Chunk] = []

        # Auto-detect provider if not specified
        if provider is None:
            provider = self._detect_provider(model_name)

        self.provider = self._parse_provider(provider)

    def add_context(self, document: str) -> "EfficientLLMCall":
        """
        Add document to context for processing.

        Can be called multiple times to accumulate context.

        Args:
            document: Text document to add

        Returns:
            Self for method chaining
        """
        if document and len(document) > 0:
            self.documents.append(document)
        return self

    def add_contexts(self, documents: List[str]) -> "EfficientLLMCall":
        """
        Add multiple documents to context.

        Args:
            documents: List of documents

        Returns:
            Self for method chaining
        """
        for doc in documents:
            self.add_context(doc)
        return self

    def prepare(self) -> "EfficientLLMCall":
        """
        Prepare context by chunking accumulated documents.

        Chunks all added documents using context manager's chunking strategy.

        Returns:
            Self for method chaining
        """
        self.chunks = []
        for doc in self.documents:
            chunks = self.context_manager.process_document(doc)
            self.chunks.extend(chunks)
        return self

    def call(
        self,
        query: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make efficient LLM call with managed context.

        Automatically selects between ContextWindowManager (chunking/compression)
        and RLMOrchestrator (recursive calls) based on document size and mode.

        Args:
            query: User query/prompt
            max_tokens: Max tokens in response
            temperature: Model creativity (0-2)
            **kwargs: Provider-specific arguments

        Returns:
            Dict with response, metrics, and mode used
        """
        # Get total document size
        total_size = sum(len(doc) for doc in self.documents)

        # Determine whether to use RLM or Manager
        use_rlm = self.context_manager.should_use_rlm(total_size)

        if use_rlm:
            return self._call_with_rlm(query, temperature, **kwargs)
        else:
            return self._call_with_manager(query, max_tokens, temperature, **kwargs)

    def _call_with_manager(
        self,
        query: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Use ContextWindowManager for context processing."""
        # Prepare if not already done
        if not self.chunks:
            self.prepare()

        # Create context window
        window = self.context_manager.create_context_window(self.chunks)

        # Format context
        context_str = self._format_context(window.chunks)

        # Make call based on provider
        if self.provider == LLMProvider.ANTHROPIC:
            result = self._call_anthropic(query, context_str, max_tokens, temperature, **kwargs)
        elif self.provider == LLMProvider.OPENAI:
            result = self._call_openai(query, context_str, max_tokens, temperature, **kwargs)
        elif self.provider == LLMProvider.OLLAMA:
            result = self._call_ollama(query, context_str, max_tokens, temperature, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        result["processing_mode"] = "manager"
        return result

    def _call_with_rlm(
        self,
        query: str,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Use RLM for recursive context processing."""
        # Create RLM orchestrator with appropriate LLM call function
        llm_call_fn = self._create_llm_callable(temperature, **kwargs)
        rlm = RLMOrchestrator(
            llm_call=llm_call_fn,
            model_name=self.model_name,
            max_recursion_depth=kwargs.get("max_recursion_depth", 5),
            chunk_size=kwargs.get("rlm_chunk_size", 2000),
        )

        # Combine all documents
        combined_doc = "\n\n---\n\n".join(self.documents)

        # Process with RLM
        rlm_result = rlm.process_document(
            document=combined_doc,
            query=query,
            strategy=kwargs.get("rlm_strategy", "auto"),
        )

        return {
            "response": rlm_result["response"],
            "tokens_used": sum(len(doc) for doc in self.documents),
            "model": self.model_name,
            "provider": self.provider.value,
            "processing_mode": "rlm",
            "metadata": {
                "llm_calls": rlm_result["llm_calls"],
                "strategy": rlm_result["strategy"],
                "strategies_used": rlm.get_stats()["strategies_used"],
            }
        }

    def _create_llm_callable(self, temperature: float, **kwargs) -> Callable:
        """Create a callable that invokes the appropriate LLM provider."""
        if self.provider == LLMProvider.ANTHROPIC:
            return self._create_anthropic_callable(temperature, **kwargs)
        elif self.provider == LLMProvider.OPENAI:
            return self._create_openai_callable(temperature, **kwargs)
        elif self.provider == LLMProvider.OLLAMA:
            return self._create_ollama_callable(temperature, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _create_anthropic_callable(self, temperature: float, **kwargs) -> Callable:
        """Create callable for Anthropic API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic library required: pip install anthropic")

        client = anthropic.Anthropic()

        def call_fn(prompt: str) -> str:
            response = client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", 1024),
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        return call_fn

    def _create_openai_callable(self, temperature: float, **kwargs) -> Callable:
        """Create callable for OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai library required: pip install openai")

        client = OpenAI()

        def call_fn(prompt: str) -> str:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 1024),
                temperature=temperature,
            )
            return response.choices[0].message.content

        return call_fn

    def _create_ollama_callable(self, temperature: float, **kwargs) -> Callable:
        """Create callable for Ollama API."""
        import requests

        base_url = kwargs.get("base_url", "http://localhost:11434")

        def call_fn(prompt: str) -> str:
            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=300,
            )
            response.raise_for_status()
            return response.json().get("response", "")

        return call_fn

    def _call_anthropic(
        self,
        query: str,
        context: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Make call to Anthropic Claude."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic library required: pip install anthropic")

        client = anthropic.Anthropic()

        # Format message
        full_prompt = self.system_prompt + "\n\nContext:\n" + context + f"\n\nQuery: {query}"

        response = client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": full_prompt}],
        )

        return {
            "response": response.content[0].text,
            "tokens_used": self.context_manager.estimate_tokens(full_prompt),
            "window_utilization": self.context_manager.last_window.utilization,
            "chunks_used": len(self.context_manager.last_window.chunks),
            "model": self.model_name,
            "provider": "anthropic",
            "metadata": {
                "strategy": self.context_manager.context_strategy.value,
                "chunking": self.context_manager.chunking_strategy.value,
            }
        }

    def _call_openai(
        self,
        query: str,
        context: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Make call to OpenAI GPT."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai library required: pip install openai")

        client = OpenAI()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuery: {query}"
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return {
            "response": response.choices[0].message.content,
            "tokens_used": self.context_manager.estimate_tokens(
                self.system_prompt + context + query
            ),
            "window_utilization": self.context_manager.last_window.utilization,
            "chunks_used": len(self.context_manager.last_window.chunks),
            "model": self.model_name,
            "provider": "openai",
            "metadata": {
                "strategy": self.context_manager.context_strategy.value,
                "chunking": self.context_manager.chunking_strategy.value,
            }
        }

    def _call_ollama(
        self,
        query: str,
        context: str,
        max_tokens: int,
        temperature: float,
        base_url: str = "http://localhost:11434",
        **kwargs
    ) -> Dict[str, Any]:
        """Make call to Ollama local LLM."""
        import requests

        prompt = self.system_prompt + "\n\nContext:\n" + context + f"\n\nQuery: {query}"

        try:
            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()

            return {
                "response": result.get("response", ""),
                "tokens_used": self.context_manager.estimate_tokens(prompt),
                "window_utilization": self.context_manager.last_window.utilization,
                "chunks_used": len(self.context_manager.last_window.chunks),
                "model": self.model_name,
                "provider": "ollama",
                "metadata": {
                    "strategy": self.context_manager.context_strategy.value,
                    "chunking": self.context_manager.chunking_strategy.value,
                }
            }
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {base_url}. "
                "Ensure Ollama is running: ollama serve"
            )

    def _format_context(self, chunks: List[Chunk]) -> str:
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

    def _detect_provider(self, model_name: str) -> str:
        """
        Auto-detect LLM provider from model name.

        Args:
            model_name: Model identifier

        Returns:
            Provider name
        """
        model_lower = model_name.lower()

        if "claude" in model_lower:
            return "anthropic"
        elif "gpt" in model_lower or "text-davinci" in model_lower:
            return "openai"
        else:
            return "ollama"  # Default to Ollama for other models

    def _parse_provider(self, provider_str: str) -> LLMProvider:
        """
        Parse provider string to enum.

        Args:
            provider_str: Provider name string

        Returns:
            LLMProvider enum
        """
        provider_lower = provider_str.lower()

        if provider_lower in ["anthropic", "claude"]:
            return LLMProvider.ANTHROPIC
        elif provider_lower in ["openai", "gpt"]:
            return LLMProvider.OPENAI
        elif provider_lower == "ollama":
            return LLMProvider.OLLAMA
        else:
            raise ValueError(f"Unknown provider: {provider_str}")

    def clear_context(self) -> "EfficientLLMCall":
        """
        Clear accumulated documents and chunks.

        Returns:
            Self for method chaining
        """
        self.documents = []
        self.chunks = []
        return self

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about accumulated context.

        Returns:
            Dict with document count, chunk count, total tokens
        """
        total_tokens = sum(c.token_count for c in self.chunks) if self.chunks else 0

        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "total_tokens": total_tokens,
            "model": self.model_name,
            "provider": self.provider.value,
        }


# Convenience function for simple usage
def efficient_llm_call(
    context_manager: ContextWindowManager,
    documents: List[str] | str,
    query: str,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    **kwargs
) -> Dict[str, Any]:
    """
    Make efficient LLM call with managed context (simple interface).

    One-shot function for making a single LLM call with context management.
    For more control, use EfficientLLMCall class directly.

    Args:
        context_manager: ContextWindowManager instance
        documents: Single document or list of documents to include as context
        query: User query/prompt
        model_name: Target model name (if None, uses context_manager's model)
        provider: LLM provider ('anthropic', 'openai', 'ollama')
                 If None, auto-detected
        system_prompt: System instructions
        max_tokens: Max response tokens
        temperature: Model creativity
        **kwargs: Provider-specific arguments

    Returns:
        Dict with response, metrics, and metadata

    Example:
        from efficient_context_window_manager import ContextWindowManager
        from efficient_context_window_manager.orchestrator import efficient_llm_call

        manager = ContextWindowManager(model_name="claude-3-opus-20250219")

        result = efficient_llm_call(
            context_manager=manager,
            documents=["doc1.txt content", "doc2.txt content"],
            query="What are the key findings?",
        )

        print(result['response'])
        print(f"Used {result['tokens_used']} tokens")
    """
    # Use context manager's model if not specified
    if model_name is None:
        model_name = context_manager.model_name

    # Normalize documents input
    if isinstance(documents, str):
        documents = [documents]

    # Create call orchestrator
    orchestrator = EfficientLLMCall(
        context_manager=context_manager,
        model_name=model_name,
        provider=provider,
        system_prompt=system_prompt,
    )

    # Add documents and call
    orchestrator.add_contexts(documents)
    result = orchestrator.call(query, max_tokens, temperature, **kwargs)

    return result


__all__ = ["EfficientLLMCall", "efficient_llm_call", "LLMProvider"]
