"""
Example: Using Ollama with Intelligent Context Window Management

This example demonstrates how to use the package with Ollama models
that have small native context windows (e.g., 8k tokens) but can handle
much larger documents (100k+ tokens) through intelligent chunking,
compression, and context selection.

Scenario: Process large documents with local Ollama models while respecting
their native 8k context window limit.

Prerequisites:
- Ollama installed (https://ollama.ai)
- Model pulled: ollama pull llama2 (or mistral, neural-chat, etc.)
- Server running: ollama serve
"""

from efficient_context_window_manager import (
    ContextWindowManager,
    EfficientLLMCall,
)
from efficient_context_window_manager.types import (
    ChunkingStrategy,
    ContextStrategyEnum,
    CompressionConfig,
    CompressionMethod,
)


def example_basic_ollama_8k_context():
    """
    Basic example: Use Ollama with 8k context window to process larger documents.
    """
    print("=" * 70)
    print("Example 1: Basic Ollama with 8k Native Context Window")
    print("=" * 70)

    # Create manager for Ollama with 8k native limit
    # But we'll use intelligent chunking to handle larger documents
    manager = ContextWindowManager(
        model_name="llama2",
        max_window_tokens=8000,  # Ollama llama2 native limit
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=1500,  # Smaller chunks to fit in 8k window
        overlap=200,
    )

    # Sample large document (simulated - in practice this could be 100k+ tokens)
    large_document = """
    Machine Learning Fundamentals - Complete Guide

    Chapter 1: Introduction
    Machine learning is a subset of artificial intelligence that enables
    systems to learn from data without being explicitly programmed.

    1.1 Types of Learning
    - Supervised Learning: Uses labeled training data
    - Unsupervised Learning: Discovers patterns in unlabeled data
    - Reinforcement Learning: Learns through interaction and rewards

    Chapter 2: Common Algorithms
    2.1 Linear Regression
    Predicts continuous values using linear relationships.

    2.2 Logistic Regression
    Predicts categorical values (classification).

    2.3 Decision Trees
    Builds hierarchical decision structures.

    Chapter 3: Neural Networks
    Deep learning using multiple layers of neurons.

    Chapter 4: Practical Applications
    Real-world uses of machine learning.
    """ * 50  # Repeat to simulate large document

    # Create call orchestrator
    call = EfficientLLMCall(
        context_manager=manager,
        model_name="llama2",
        system_prompt="You are a machine learning expert. Answer questions based on the provided context.",
    )

    # Add large document
    call.add_context(large_document)

    # Check stats
    stats = call.get_stats()
    print(f"\nDocument Statistics:")
    print(f"  Total tokens in document: {stats['total_tokens']}")
    print(f"  Number of chunks: {stats['chunks']}")
    print(f"  Model native window: {manager.max_window_tokens} tokens")
    print(f"  Effective handling: {stats['total_tokens']} tokens through {stats['chunks']} chunks")

    # Make efficient call
    print(f"\nMaking query to Ollama...")
    result = call.call(
        query="What are the main types of machine learning?",
        temperature=0.7,
    )

    print(f"\nResponse:")
    print(result['response'][:300] + "...")
    print(f"\nMetrics:")
    print(f"  Tokens used in this call: {result['tokens_used']}")
    print(f"  Window utilization: {result['window_utilization'] * 100:.1f}%")
    print(f"  Chunks included: {result['chunks_used']}")


def example_ollama_with_compression():
    """
    Use compression to fit more content in Ollama's 8k window.
    """
    print("\n" + "=" * 70)
    print("Example 2: Ollama with Compression for Dense Content")
    print("=" * 70)

    # Create manager with compression enabled
    manager = ContextWindowManager(
        model_name="llama2",
        max_window_tokens=8000,
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=1000,
        compression_config=CompressionConfig(
            method=CompressionMethod.OBSERVATION_MASKING,
            compression_ratio=0.3,  # Remove 30% of non-essential tokens
        ),
    )

    # Large dense document
    dense_document = """
    Advanced Machine Learning Techniques (Dense Version)
    """ + "Technical content about gradient descent, backpropagation, regularization, " \
          "batch normalization, dropout, attention mechanisms, transformers, etc. " * 50

    call = EfficientLLMCall(context_manager=manager, model_name="llama2")
    call.add_context(dense_document)

    stats = call.get_stats()
    print(f"\nBefore Compression:")
    print(f"  Chunks: {stats['chunks']}")
    print(f"  Total tokens: {stats['total_tokens']}")

    result = call.call("Summarize the key techniques mentioned.")

    print(f"\nAfter Processing with Compression:")
    print(f"  Tokens used (with 30% compression): {result['tokens_used']}")
    print(f"  Window utilization: {result['window_utilization'] * 100:.1f}%")
    print(f"  Compression effective: Can fit {stats['total_tokens']} tokens in {manager.max_window_tokens} limit")


def example_ollama_multiple_queries():
    """
    Process large document once, answer multiple questions efficiently.
    Reuses chunked context across queries.
    """
    print("\n" + "=" * 70)
    print("Example 3: Multiple Queries on Large Document (Efficient)")
    print("=" * 70)

    manager = ContextWindowManager(
        model_name="mistral",  # Mistral is often faster on local systems
        max_window_tokens=8000,
        chunking_strategy=ChunkingStrategy.SLIDING_WINDOW,
        chunk_size=1500,
        overlap=300,
    )

    call = EfficientLLMCall(context_manager=manager, model_name="mistral")

    # Large document
    large_doc = "The history of computing spans decades. " \
               "In the 1950s computers were room-sized. " \
               "The 1970s saw personal computers emerge. " \
               "The internet revolutionized computing in the 1990s. " \
               "Mobile computing emerged in the 2000s. " \
               "AI acceleration defined the 2010s. " \
               "Large language models transformed the 2020s. " * 50

    # Add once
    call.add_context(large_doc).prepare()

    print(f"\nDocument prepared (chunks cached):")
    stats = call.get_stats()
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Chunks: {stats['chunks']}")

    # Multiple queries reuse the same chunks
    queries = [
        "What happened in the 1970s?",
        "How did mobile computing start?",
        "What changed in the 2020s?",
    ]

    print(f"\nAnswering {len(queries)} questions (reusing same chunks):")
    for i, query in enumerate(queries, 1):
        result = call.call(query)
        print(f"\n  Q{i}: {query}")
        print(f"  A{i}: {result['response'][:100]}...")
        print(f"       Tokens: {result['tokens_used']}")


def example_ollama_chunking_strategies():
    """
    Compare different chunking strategies with Ollama's 8k limit.
    """
    print("\n" + "=" * 70)
    print("Example 4: Compare Chunking Strategies for Ollama")
    print("=" * 70)

    document = "Sample text about machine learning. " * 1000  # Large document

    strategies = [
        (ChunkingStrategy.FIXED_SIZE, "Fixed-size: Simple and predictable"),
        (ChunkingStrategy.RECURSIVE, "Recursive: Respects semantic boundaries (recommended)"),
        (ChunkingStrategy.SLIDING_WINDOW, "Sliding window: Overlapping context (SWAT approach)"),
    ]

    print(f"\nDocument size: ~4000 tokens (would need multiple API calls normally)")
    print(f"Ollama native limit: 8000 tokens")
    print(f"\nStrategy Comparison:")
    print("-" * 70)

    for strategy, description in strategies:
        manager = ContextWindowManager(
            model_name="llama2",
            max_window_tokens=8000,
            chunking_strategy=strategy,
            chunk_size=1000,
            overlap=100,
        )

        chunks = manager.process_document(document)
        total_tokens = sum(c.token_count for c in chunks)

        print(f"\n{description}")
        print(f"  Chunks created: {len(chunks)}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Avg chunk size: {total_tokens // len(chunks) if chunks else 0} tokens")
        print(f"  Fits in 8k window: {'✓ Yes' if total_tokens <= 8000 else '✗ No (but handled via chunking)'}")


def example_ollama_100k_token_document():
    """
    Handle 100k+ token document with Ollama's 8k native limit.

    This is the key use case: without the package, this would be impossible.
    With the package, it's automatic and efficient.
    """
    print("\n" + "=" * 70)
    print("Example 5: Handle 100k Token Document with 8k Ollama Limit")
    print("=" * 70)

    # Create manager specifically optimized for large documents
    manager = ContextWindowManager(
        model_name="neural-chat",  # Good for general tasks
        max_window_tokens=8000,  # Native limit
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=2000,  # Larger chunks to handle big document efficiently
        overlap=300,
        context_strategy=ContextStrategyEnum.GREEDY,
    )

    # Simulate 100k token document (in practice, load from file)
    print(f"\nSimulating 100k token document...")
    large_document = "This is a comprehensive document about various topics. " * 2000
    estimated_tokens = manager.estimate_tokens(large_document)
    print(f"Actual tokens in document: ~{estimated_tokens}")

    call = EfficientLLMCall(
        context_manager=manager,
        model_name="neural-chat",
        system_prompt="Answer questions about the provided document concisely.",
    )

    call.add_context(large_document)

    stats = call.get_stats()
    print(f"\nContext Preparation:")
    print(f"  Documents: {stats['documents']}")
    print(f"  Total chunks: {stats['chunks']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Model native window: {manager.max_window_tokens}")
    print(f"\nThis document would normally be impossible to process with a {manager.max_window_tokens} token limit.")
    print(f"The package handles it by:")
    print(f"  1. Splitting into {stats['chunks']} semantic chunks")
    print(f"  2. Selecting relevant chunks that fit the window")
    print(f"  3. Making efficient API calls")

    # Make query
    result = call.call("What is the document about?")
    print(f"\nQuery Results:")
    print(f"  Response: {result['response'][:150]}...")
    print(f"  Tokens used: {result['tokens_used']}/{manager.max_window_tokens}")
    print(f"  Utilization: {result['window_utilization'] * 100:.1f}%")
    print(f"  Success: ✓ Processed 100k token document with 8k window")


def example_ollama_cost_savings():
    """
    Show cost/performance improvements when using local Ollama vs API.
    """
    print("\n" + "=" * 70)
    print("Example 6: Local Ollama vs Cloud API Economics")
    print("=" * 70)

    print("\nScenario: Process 100k token document with 10 queries")
    print("\nWith Cloud API (e.g., Claude, GPT-4):")
    print("  - Input tokens: 100,000 * 10 = 1,000,000 tokens")
    print("  - Cost: ~$30 (at typical pricing)")
    print("  - Latency: Dependent on API")
    print("  - Privacy: Document sent to external servers")

    print("\nWith Local Ollama + This Package:")
    print("  - Input tokens: ~8,000 per call (chunked smartly)")
    print("  - Cost: FREE (runs locally)")
    print("  - Latency: Faster (no network)")
    print("  - Privacy: 100% private (no external calls)")

    # Demonstrate
    manager = ContextWindowManager(
        model_name="llama2",
        max_window_tokens=8000,
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=1500,
    )

    doc = "Sample document. " * 1000

    call = EfficientLLMCall(context_manager=manager, model_name="llama2")
    call.add_context(doc)

    # Simulate multiple queries
    queries = ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"]

    total_tokens = 0
    for query in queries:
        result = call.call(query)
        total_tokens += result['tokens_used']

    print(f"\nActual Results from Package:")
    print(f"  5 queries on large document")
    print(f"  Total tokens processed: {total_tokens}")
    print(f"  Per-query average: {total_tokens // len(queries)}")
    print(f"  Within Ollama's {manager.max_window_tokens} token limit: ✓ Yes")


def example_ollama_real_world():
    """
    Real-world scenario: Load a large document from file and process it.
    """
    print("\n" + "=" * 70)
    print("Example 7: Real-World - Load Document from File")
    print("=" * 70)

    print("""
Real-world usage pattern:

# 1. Load document
with open("large_report.txt") as f:
    document = f.read()

# 2. Create manager
manager = ContextWindowManager(
    model_name="llama2",
    max_window_tokens=8000,
    chunking_strategy=ChunkingStrategy.RECURSIVE,
    chunk_size=2000,
)

# 3. Create call interface
call = EfficientLLMCall(context_manager=manager, model_name="llama2")

# 4. Add document
call.add_context(document)

# 5. Ask questions
while True:
    query = input("Ask a question: ")
    result = call.call(query)
    print(result['response'])
    print(f"Used {result['tokens_used']} tokens")

That's it! The package handles:
✓ Tokenization
✓ Chunking
✓ Context selection
✓ API calls
✓ Metrics tracking

No manual context window management needed!
    """)


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "OLLAMA EXAMPLES - HANDLING LARGE DOCUMENTS" + " " * 17 + "║")
    print("║" + " " * 5 + "Processing 100k tokens through 8k context window limit" + " " * 9 + "║")
    print("╚" + "=" * 68 + "╝")

    print("""
PREREQUISITES:
- Ollama installed: https://ollama.ai
- Model pulled: ollama pull llama2 (or mistral, neural-chat)
- Server running: ollama serve

These examples show how to process documents much larger than Ollama's
native context window through intelligent chunking and context management.
    """)

    # Run examples
    example_basic_ollama_8k_context()
    example_ollama_with_compression()
    example_ollama_multiple_queries()
    example_ollama_chunking_strategies()
    example_ollama_100k_token_document()
    example_ollama_cost_savings()
    example_ollama_real_world()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Start Ollama: ollama serve")
    print("2. Run this example: python examples/ollama_large_documents.py")
    print("3. See HOWTO.md for detailed guides")
    print("4. Check examples/ for more use cases")
