"""
Basic example: Chunking a document with different strategies.

Demonstrates:
1. Loading document
2. Creating context manager with different chunking strategies
3. Processing document into chunks
4. Viewing chunk statistics
"""

from efficient_context_window_manager import ContextWindowManager
from efficient_context_window_manager.tokenizers import AutoTokenizer
from efficient_context_window_manager.types import ChunkingStrategy


def example_basic_chunking():
    """Basic chunking example."""
    # Sample document
    document = """
    The history of artificial intelligence (AI) spans centuries.

    Early Concepts (1950s-1970s)
    The foundations of AI were laid in the 1950s when Alan Turing proposed the famous
    "Turing Test" as a measure of machine intelligence. This period saw the development
    of early AI programs like ELIZA and expert systems.

    The AI Winter (1980s-1990s)
    Despite initial enthusiasm, AI faced significant setbacks during the 1980s and 1990s.
    Limited computational power and overly ambitious goals led to reduced funding and
    research interest.

    Modern Era (2010s-present)
    The advent of deep learning and neural networks revolutionized AI. Large language models
    like GPT and BERT demonstrated remarkable capabilities in natural language processing.
    """ * 10  # Repeat to make longer

    # Create context manager with recursive chunking
    manager = ContextWindowManager(
        model_name="gpt-3.5-turbo",
        max_window_tokens=4096,
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=500,
        overlap=50,
    )

    # Process document
    chunks = manager.process_document(document)

    # Print results
    print(f"Document length: {len(document)} characters")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Total tokens: {sum(c.token_count for c in chunks)}")
    print()

    for i, chunk in enumerate(chunks[:3]):  # Show first 3
        print(f"Chunk {i+1}:")
        print(f"  Tokens: {chunk.token_count}")
        print(f"  Length: {len(chunk.content)} chars")
        print(f"  Preview: {chunk.content[:100]}...")
        print()


def example_different_strategies():
    """Compare different chunking strategies."""
    document = "Sample text. " * 1000  # Longer document

    tokenizer = AutoTokenizer("gpt-3.5-turbo")

    strategies = [
        ChunkingStrategy.FIXED_SIZE,
        ChunkingStrategy.RECURSIVE,
        ChunkingStrategy.SLIDING_WINDOW,
    ]

    print("Comparing chunking strategies:")
    print("-" * 60)

    for strategy in strategies:
        manager = ContextWindowManager(
            tokenizer=tokenizer,
            model_name="gpt-3.5-turbo",
            chunking_strategy=strategy,
            chunk_size=500,
            overlap=50,
        )

        chunks = manager.process_document(document)
        total_tokens = sum(c.token_count for c in chunks)

        print(f"{strategy.value:20} | Chunks: {len(chunks):3} | Total tokens: {total_tokens}")


def example_context_window_creation():
    """Create context windows from chunks."""
    document = """
    Context Window Management allows processing very large documents
    by breaking them into manageable pieces that fit a model's native context limits.
    """ * 20

    manager = ContextWindowManager(
        model_name="gpt-3.5-turbo",
        max_window_tokens=2048,
        chunk_size=512,
    )

    # Get chunks
    chunks = manager.process_document(document)

    # Create context window
    context_window = manager.create_context_window(chunks)

    print(f"Context window created:")
    print(f"  Total model capacity: {context_window.max_tokens} tokens")
    print(f"  Tokens used: {context_window.current_tokens} tokens")
    print(f"  Available: {context_window.available_tokens} tokens")
    print(f"  Utilization: {context_window.utilization * 100:.1f}%")
    print(f"  Chunks included: {len(context_window.chunks)}")


if __name__ == "__main__":
    print("=" * 60)
    print("BASIC USAGE EXAMPLES")
    print("=" * 60)
    print()

    print("1. Basic Chunking")
    print("-" * 60)
    example_basic_chunking()

    print("\n2. Comparing Strategies")
    print("-" * 60)
    example_different_strategies()

    print("\n3. Context Window Creation")
    print("-" * 60)
    example_context_window_creation()
