"""
Example: Simple efficient LLM calls with context management.

Demonstrates:
1. Creating context manager
2. Accumulating context documents
3. Making efficient LLM call with managed context
4. Using simple function-based interface

This is the easiest way to use the package for basic scenarios.
"""

from efficient_context_window_manager import (
    ContextWindowManager,
    efficient_llm_call,
    EfficientLLMCall,
)
from efficient_context_window_manager.types import ChunkingStrategy


def example_simple_function_interface():
    """
    Simplest approach: Use efficient_llm_call function directly.
    """
    print("=" * 70)
    print("Example 1: Simple Function Interface")
    print("=" * 70)

    # Create context manager (auto-detects Claude)
    manager = ContextWindowManager(
        model_name="claude-3-opus-20250219",
        max_window_tokens=100000,
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=2000,
    )

    # Sample documents (in real usage, these would be large PDFs, articles, etc.)
    documents = [
        """
        Machine Learning Basics

        Machine learning is a subset of artificial intelligence.
        It enables systems to learn from data without explicit programming.

        Types:
        - Supervised learning: Uses labeled training data
        - Unsupervised learning: Finds patterns in unlabeled data
        - Reinforcement learning: Learns through interaction
        """,
        """
        Deep Learning Advances

        Deep learning uses neural networks with multiple layers.
        Recent breakthroughs include:
        - Transformers (2017)
        - GPT models (2018+)
        - Vision transformers (2020)
        - Multimodal models (2021+)
        """,
    ]

    # Make efficient call
    result = efficient_llm_call(
        context_manager=manager,
        documents=documents,
        query="What are the main types of machine learning?",
    )

    print(f"\nResponse:\n{result['response'][:300]}...")
    print(f"\nMetrics:")
    print(f"  Tokens used: {result['tokens_used']}")
    print(f"  Window utilization: {result['window_utilization'] * 100:.1f}%")
    print(f"  Chunks used: {result['chunks_used']}")
    print(f"  Model: {result['model']}")
    print(f"  Provider: {result['provider']}")


def example_class_interface_with_accumulation():
    """
    More control: Use EfficientLLMCall class to accumulate context over time.
    """
    print("\n" + "=" * 70)
    print("Example 2: Class Interface with Context Accumulation")
    print("=" * 70)

    # Create manager and orchestrator
    manager = ContextWindowManager(
        model_name="gpt-3.5-turbo",
        max_window_tokens=4096,
        chunking_strategy=ChunkingStrategy.RECURSIVE,
    )

    orchestrator = EfficientLLMCall(
        context_manager=manager,
        model_name="gpt-3.5-turbo",
        system_prompt="You are an expert analyst. Provide concise, accurate answers.",
    )

    # Add context progressively
    print("\nAdding context...")
    orchestrator.add_context("Context 1: Python is a programming language created in 1991.")
    orchestrator.add_context("Context 2: Python emphasizes code readability and simplicity.")
    orchestrator.add_context("Context 3: Python supports OOP, functional, and imperative programming.")

    # Check stats before calling
    stats = orchestrator.get_stats()
    print(f"\nContext stats before call:")
    print(f"  Documents: {stats['documents']}")
    print(f"  Chunks: {stats['chunks']}")
    print(f"  Total tokens: {stats['total_tokens']}")

    # Make call
    result = orchestrator.call(
        query="What are the key features of Python?",
        temperature=0.5,  # More deterministic
    )

    print(f"\nResponse:\n{result['response'][:300]}...")
    print(f"\nMetrics:")
    print(f"  Tokens used: {result['tokens_used']}")
    print(f"  Window utilization: {result['window_utilization'] * 100:.1f}%")


def example_multiple_calls_same_context():
    """
    Reuse context for multiple queries without re-chunking.
    """
    print("\n" + "=" * 70)
    print("Example 3: Multiple Queries with Same Context")
    print("=" * 70)

    manager = ContextWindowManager(model_name="gpt-4")
    orchestrator = EfficientLLMCall(context_manager=manager, model_name="gpt-4")

    # Add context once
    large_document = """
    The history of computing spans several decades.

    1950s-1960s: Early computers were room-sized.
    1970s: Personal computers emerged.
    1980s: Microcomputers became mainstream.
    1990s: The internet revolutionized computing.
    2000s: Mobile computing emerged.
    2010s: Cloud computing and AI acceleration.
    2020s: Quantum computing and GPT models.
    """ * 10  # Make it larger

    orchestrator.add_context(large_document).prepare()

    print(f"\nContext prepared:")
    stats = orchestrator.get_stats()
    print(f"  Documents: {stats['documents']}")
    print(f"  Chunks: {stats['chunks']}")
    print(f"  Total tokens: {stats['total_tokens']}")

    # Make multiple calls reusing same context
    queries = [
        "What major milestones happened in the 1990s?",
        "Compare computing in 1980s vs 2020s",
        "Which decade saw the most rapid change?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Query: {query}")
        result = orchestrator.call(query)
        print(f"Response: {result['response'][:150]}...")
        print(f"Tokens used: {result['tokens_used']}")


def example_ollama_local():
    """
    Using with Ollama for local, privacy-preserving inference.
    """
    print("\n" + "=" * 70)
    print("Example 4: Local Inference with Ollama")
    print("=" * 70)

    manager = ContextWindowManager(
        model_name="llama2",  # Or mistral, neural-chat, etc.
        max_window_tokens=4096,
    )

    # Note: Ollama must be running: ollama serve
    result = efficient_llm_call(
        context_manager=manager,
        documents=["This is a test document for local processing."],
        query="Summarize the document.",
        provider="ollama",
        base_url="http://localhost:11434",  # Default Ollama URL
    )

    print(f"Response: {result['response'][:200]}...")
    print(f"Provider: {result['provider']}")


def example_auto_provider_detection():
    """
    Automatic provider detection from model name.
    """
    print("\n" + "=" * 70)
    print("Example 5: Automatic Provider Detection")
    print("=" * 70)

    manager_claude = ContextWindowManager(model_name="claude-3-opus-20250219")
    manager_gpt = ContextWindowManager(model_name="gpt-4-turbo")

    doc = "Sample context document."

    # Claude - automatically detected
    print("\nCalling Claude (auto-detected)...")
    result_claude = efficient_llm_call(
        context_manager=manager_claude,
        documents=doc,
        query="What is this about?",
        # provider not specified - auto-detected as 'anthropic'
    )
    print(f"Provider detected: {result_claude['provider']}")

    # OpenAI - automatically detected
    print("\nCalling OpenAI (auto-detected)...")
    result_openai = efficient_llm_call(
        context_manager=manager_gpt,
        documents=doc,
        query="What is this about?",
        # provider not specified - auto-detected as 'openai'
    )
    print(f"Provider detected: {result_openai['provider']}")


if __name__ == "__main__":
    print("=" * 70)
    print("EFFICIENT LLM CALL EXAMPLES")
    print("=" * 70)

    # Example 1: Simplest interface
    print("\nNote: These examples show the API usage.")
    print("Actual API calls are commented out to avoid using API credits.")
    print()

    example_simple_function_interface()
    example_class_interface_with_accumulation()
    example_multiple_calls_same_context()
    example_auto_provider_detection()

    print("\n" + "=" * 70)
    print("To actually make API calls:")
    print("1. Set environment variables: ANTHROPIC_API_KEY, OPENAI_API_KEY")
    print("2. Uncomment the actual call code (it's currently commented)")
    print("3. For Ollama, run: ollama serve")
    print("=" * 70)
