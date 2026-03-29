"""
Example: Using context window manager with Anthropic Claude.

Demonstrates:
1. Creating context manager
2. Setting up Anthropic adapter
3. Processing long document with Claude
4. Viewing results and metrics
"""

import os
from efficient_context_window_manager import ContextWindowManager
from efficient_context_window_manager.integrations.anthropic import AnthropicAdapter
from efficient_context_window_manager.types import (
    ChunkingStrategy, ContextStrategyEnum, CompressionConfig, CompressionMethod
)


def example_anthropic_basic():
    """Basic example with Anthropic."""
    # Long document (simulated)
    document = """
    Machine Learning Fundamentals

    Chapter 1: Introduction to ML
    Machine learning is a subset of artificial intelligence that enables
    systems to learn from data without being explicitly programmed.

    Types of Learning:
    - Supervised: Learn from labeled examples
    - Unsupervised: Find patterns in unlabeled data
    - Reinforcement: Learn through interaction and feedback

    Chapter 2: Common Algorithms
    Linear Regression: Predict continuous values
    Classification: Predict categorical values
    Clustering: Group similar data points
    """ * 50  # Make it longer

    # Create context manager optimized for Claude
    context_manager = ContextWindowManager(
        model_name="claude-3-opus-20250219",
        max_window_tokens=100000,  # Claude has large context
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=2000,
        overlap=200,
    )

    # Create adapter
    adapter = AnthropicAdapter(
        context_manager=context_manager,
        model_name="claude-3-opus-20250219",
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    # Process document with query
    document_chunk = document
    query = "What are the main types of machine learning?"

    # Prepare request (without making actual API call)
    request_info = adapter.prepare_request(
        document=document_chunk,
        query=query,
        system_prompt="You are an AI expert. Answer questions based on the provided context.",
    )

    print("Request prepared for Anthropic API:")
    print(f"  Chunks used: {request_info['chunks_used']}")
    print(f"  Tokens used: {request_info['tokens_used']}")
    print(f"  Window utilization: {request_info['window_utilization'] * 100:.1f}%")
    print(f"  Model: {request_info['metadata']['model']}")
    print()

    # To actually call the API (uncomment):
    # result = adapter.process_with_context(document_chunk, query)
    # print(f"Response: {result['response'][:200]}...")


def example_with_compression():
    """Example with compression enabled."""
    document = "Sample text. " * 2000  # Very long document

    # Create manager with compression
    context_manager = ContextWindowManager(
        model_name="claude-3-opus-20250219",
        max_window_tokens=50000,
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=1000,
        compression_config=CompressionConfig(
            method=CompressionMethod.OBSERVATION_MASKING,
            compression_ratio=0.3,  # Compress 30%
        ),
    )

    adapter = AnthropicAdapter(
        context_manager=context_manager,
        model_name="claude-3-opus-20250219",
    )

    request_info = adapter.prepare_request(
        document=document,
        query="Summarize the key points.",
    )

    print("Request with compression:")
    print(f"  Original tokens (approx): {len(document) // 4}")
    print(f"  After compression: {request_info['tokens_used']}")
    print(f"  Compression ratio: {(1 - request_info['tokens_used'] / (len(document) // 4)) * 100:.1f}%")


def example_sliding_window():
    """Example with sliding window strategy."""
    document = """
    Document part 1: Introduction
    Document part 2: Methods
    Document part 3: Results
    Document part 4: Discussion
    """ * 100

    context_manager = ContextWindowManager(
        model_name="claude-3-opus-20250219",
        max_window_tokens=50000,
        chunking_strategy=ChunkingStrategy.SLIDING_WINDOW,
        chunk_size=1000,
        overlap=300,
        context_strategy=ContextStrategyEnum.SLIDING_WINDOW,
    )

    adapter = AnthropicAdapter(
        context_manager=context_manager,
        model_name="claude-3-opus-20250219",
    )

    request_info = adapter.prepare_request(
        document=document,
        query="What are the main sections?",
    )

    print("Sliding window strategy:")
    print(f"  Chunks used: {request_info['chunks_used']}")
    print(f"  Tokens used: {request_info['tokens_used']}")
    print(f"  Utilization: {request_info['window_utilization'] * 100:.1f}%")


if __name__ == "__main__":
    print("=" * 60)
    print("ANTHROPIC INTEGRATION EXAMPLES")
    print("=" * 60)
    print()

    print("1. Basic Anthropic Integration")
    print("-" * 60)
    example_anthropic_basic()

    print("\n2. With Compression")
    print("-" * 60)
    example_with_compression()

    print("\n3. Sliding Window Strategy")
    print("-" * 60)
    example_sliding_window()

    print("\nNote: To make actual API calls, set ANTHROPIC_API_KEY and uncomment call lines.")
