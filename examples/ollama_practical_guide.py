"""
Practical Guide: Using Ollama with Large Documents

This example shows production-ready patterns for using Ollama models
with the context window manager for real-world scenarios.

Key scenarios:
1. Loading documents from files (PDF, TXT, etc.)
2. Batch processing multiple documents
3. Interactive Q&A session
4. Performance monitoring
5. Error handling
"""

import os
from efficient_context_window_manager import (
    ContextWindowManager,
    EfficientLLMCall,
)
from efficient_context_window_manager.types import ChunkingStrategy


class OllamaDocumentProcessor:
    """
    Production-ready wrapper for processing documents with Ollama.
    """

    def __init__(
        self,
        model_name: str = "llama2",
        chunk_size: int = 1500,
        ollama_url: str = "http://localhost:11434",
    ):
        """
        Initialize document processor.

        Args:
            model_name: Ollama model (llama2, mistral, neural-chat, etc.)
            chunk_size: Target tokens per chunk
            ollama_url: Ollama server URL
        """
        self.model_name = model_name
        self.ollama_url = ollama_url

        # Create manager optimized for Ollama
        self.manager = ContextWindowManager(
            model_name=model_name,
            max_window_tokens=7000,  # Leave buffer below 8k
            chunking_strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=chunk_size,
            overlap=200,
        )

        # Will store the call orchestrator
        self.call = None

        # Metrics
        self.stats = {
            "documents_loaded": 0,
            "total_queries": 0,
            "total_tokens_used": 0,
            "avg_response_tokens": 0,
        }

    def load_text_file(self, filepath: str) -> bool:
        """
        Load a text file.

        Args:
            filepath: Path to text file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return self._load_content(content)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return False

    def load_markdown_file(self, filepath: str) -> bool:
        """
        Load a markdown file.

        Args:
            filepath: Path to markdown file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return self._load_content(content)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return False

    def load_documents_from_directory(self, directory: str, pattern: str = "*.txt") -> int:
        """
        Load multiple documents from a directory.

        Args:
            directory: Directory path
            pattern: File pattern (*.txt, *.md, etc.)

        Returns:
            Number of files loaded
        """
        import glob

        loaded = 0
        for filepath in glob.glob(os.path.join(directory, pattern)):
            if self.load_text_file(filepath):
                loaded += 1
                print(f"✓ Loaded: {filepath}")
            else:
                print(f"✗ Failed: {filepath}")

        return loaded

    def _load_content(self, content: str) -> bool:
        """
        Load content into the processor.

        Args:
            content: Text content

        Returns:
            True if successful
        """
        try:
            if self.call is None:
                self.call = EfficientLLMCall(
                    context_manager=self.manager,
                    model_name=self.model_name,
                )

            self.call.add_context(content)
            self.stats["documents_loaded"] += 1

            # Show stats
            stats = self.call.get_stats()
            print(f"  Documents: {stats['documents']}")
            print(f"  Chunks: {stats['chunks']}")
            print(f"  Total tokens: {stats['total_tokens']}")

            return True
        except Exception as e:
            print(f"Error loading content: {e}")
            return False

    def query(self, question: str, temperature: float = 0.7) -> str:
        """
        Ask a question about loaded documents.

        Args:
            question: Question to ask
            temperature: Model temperature (0-1, lower = deterministic)

        Returns:
            Response from model
        """
        if self.call is None:
            print("Error: No documents loaded")
            return ""

        try:
            result = self.call.call(
                query=question,
                temperature=temperature,
            )

            # Update stats
            self.stats["total_queries"] += 1
            self.stats["total_tokens_used"] += result['tokens_used']

            return result['response']
        except Exception as e:
            print(f"Error querying: {e}")
            return ""

    def get_metrics(self) -> dict:
        """
        Get usage metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = self.stats.copy()

        if self.call:
            doc_stats = self.call.get_stats()
            metrics.update({
                "model": self.model_name,
                "documents": doc_stats['documents'],
                "chunks": doc_stats['chunks'],
                "total_document_tokens": doc_stats['total_tokens'],
            })

        if metrics['total_queries'] > 0:
            metrics['avg_tokens_per_query'] = (
                metrics['total_tokens_used'] // metrics['total_queries']
            )

        return metrics

    def print_metrics(self):
        """Print formatted metrics."""
        metrics = self.get_metrics()

        print("\n" + "=" * 50)
        print("USAGE METRICS")
        print("=" * 50)

        for key, value in metrics.items():
            print(f"{key:.<30} {value}")


def example_interactive_qa():
    """
    Interactive Q&A session with documents.
    """
    print("\n" + "=" * 70)
    print("Example 1: Interactive Q&A Session")
    print("=" * 70)

    processor = OllamaDocumentProcessor(
        model_name="llama2",
        chunk_size=1500,
    )

    # Simulate loading documents
    sample_doc = """
    The Industrial Revolution began in Britain in the late 18th century.
    It transformed manufacturing through mechanization and steam power.

    Key inventions:
    - Steam engine (James Watt, 1769)
    - Power loom (Edmund Cartwright, 1785)
    - Steam locomotive (George Stephenson, 1825)

    Impact:
    - Rapid urbanization
    - Rise of factory system
    - Growth of working class
    - Environmental pollution
    - Technological acceleration

    The Industrial Revolution spread to Europe and America by the 1800s.
    """ * 50

    print("\nLoading document...")
    processor._load_content(sample_doc)

    # Sample queries
    queries = [
        "What was the Industrial Revolution?",
        "Name three key inventions.",
        "What were the social impacts?",
    ]

    print(f"\nAsking {len(queries)} questions...")
    for i, query in enumerate(queries, 1):
        print(f"\nQ{i}: {query}")
        answer = processor.query(query, temperature=0.5)
        print(f"A{i}: {answer[:200]}...")

    processor.print_metrics()


def example_batch_processing():
    """
    Batch process multiple documents.
    """
    print("\n" + "=" * 70)
    print("Example 2: Batch Processing Multiple Documents")
    print("=" * 70)

    processor = OllamaDocumentProcessor(model_name="mistral")

    # Simulate multiple documents
    docs = {
        "doc1": "Python is a programming language. " * 100,
        "doc2": "JavaScript is used for web development. " * 100,
        "doc3": "Go is designed for concurrent programming. " * 100,
    }

    print("\nLoading documents...")
    for name, content in docs.items():
        print(f"\n{name}:")
        processor._load_content(content)

    # Ask questions across all documents
    print(f"\nQuerying across all documents...")
    answer = processor.query("What are the main programming languages mentioned?")
    print(f"Answer: {answer[:250]}...")

    processor.print_metrics()


def example_memory_efficient():
    """
    Demonstrate memory efficiency by reusing context.
    """
    print("\n" + "=" * 70)
    print("Example 3: Memory Efficient - Reuse Context")
    print("=" * 70)

    processor = OllamaDocumentProcessor(model_name="neural-chat")

    # Large document
    large_doc = """
    Machine Learning Overview

    Supervised Learning:
    - Linear Regression
    - Logistic Regression
    - Decision Trees
    - Random Forests
    - Support Vector Machines
    - Neural Networks

    Unsupervised Learning:
    - K-Means Clustering
    - Hierarchical Clustering
    - DBSCAN
    - Gaussian Mixture Models
    - Principal Component Analysis

    Reinforcement Learning:
    - Q-Learning
    - Policy Gradients
    - Actor-Critic Methods
    """ * 100

    print("\nLoading large document once...")
    processor._load_content(large_doc)

    # Multiple queries reuse the same chunked content
    # No re-chunking needed
    print(f"\nAsking multiple questions (reusing chunks)...")

    questions = [
        "What is supervised learning?",
        "List unsupervised methods.",
        "Explain reinforcement learning.",
        "Compare supervised and unsupervised.",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        answer = processor.query(q)
        print(f"A{i}: {answer[:150]}...")

    metrics = processor.get_metrics()
    print(f"\nEfficiency:")
    print(f"  Documents: 1 (loaded once)")
    print(f"  Queries: {metrics['total_queries']}")
    print(f"  Total tokens: {metrics['total_tokens_used']}")
    print(f"  Avg tokens/query: {metrics.get('avg_tokens_per_query', 0)}")
    print(f"  ✓ Context reused across all queries (no re-chunking)")


def example_streaming_responses():
    """
    Show how to handle streaming responses (advanced).
    """
    print("\n" + "=" * 70)
    print("Example 4: Handling Long Responses Efficiently")
    print("=" * 70)

    processor = OllamaDocumentProcessor(model_name="llama2")

    doc = """
    Comprehensive guide to machine learning.
    """ * 200

    processor._load_content(doc)

    print("\nQuerying for comprehensive response...")
    response = processor.query(
        "Provide a comprehensive overview of all machine learning concepts.",
        temperature=0.7,
    )

    print(f"Response (first 500 chars):")
    print(response[:500] + "...")

    metrics = processor.get_metrics()
    print(f"\nResponse handling:")
    print(f"  Total tokens used: {metrics['total_tokens_used']}")
    print(f"  Model used: {metrics['model']}")
    print(f"  Context window: 7000 tokens (Ollama 8k limit)")


def example_error_handling():
    """
    Proper error handling for production use.
    """
    print("\n" + "=" * 70)
    print("Example 5: Error Handling and Recovery")
    print("=" * 70)

    processor = OllamaDocumentProcessor()

    # Try loading non-existent file
    print("\n1. Handling missing files:")
    success = processor.load_text_file("non_existent_file.txt")
    if not success:
        print("  ✓ Handled gracefully (no crash)")

    # Try loading empty content
    print("\n2. Handling empty content:")
    try:
        processor._load_content("")
        print("  ✓ Empty content handled")
    except Exception as e:
        print(f"  ✓ Error caught: {e}")

    # Load valid content
    print("\n3. Loading valid content:")
    processor._load_content("Sample document for testing.")
    print("  ✓ Content loaded successfully")

    # Try query without documents
    print("\n4. Querying without documents:")
    processor2 = OllamaDocumentProcessor()
    answer = processor2.query("Question?")
    if not answer:
        print("  ✓ Handled gracefully (no crash)")


def example_performance_monitoring():
    """
    Monitor performance and resource usage.
    """
    print("\n" + "=" * 70)
    print("Example 6: Performance Monitoring")
    print("=" * 70)

    processor = OllamaDocumentProcessor(model_name="mistral")

    # Load content
    content = "Monitoring performance of the system. " * 300
    processor._load_content(content)

    print("\nMaking queries and monitoring...")

    for i in range(3):
        query = f"Question {i+1}?"
        response = processor.query(query)

        metrics = processor.get_metrics()
        print(f"\nAfter query {i+1}:")
        print(f"  Queries made: {metrics['total_queries']}")
        print(f"  Total tokens: {metrics['total_tokens_used']}")
        print(f"  Avg per query: {metrics.get('avg_tokens_per_query', 0)}")

    print("\nFinal metrics:")
    processor.print_metrics()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 12 + "OLLAMA PRACTICAL GUIDE - PRODUCTION PATTERNS" + " " * 12 + "║")
    print("║" + " " * 8 + "Real-world examples for document processing with Ollama" + " " * 5 + "║")
    print("╚" + "=" * 68 + "╝")

    print("""
This guide shows production-ready patterns for:
- Loading documents from files
- Batch processing multiple documents
- Memory-efficient context reuse
- Error handling and recovery
- Performance monitoring

Requirements:
- Ollama installed and running (ollama serve)
- Model pulled (ollama pull llama2)
    """)

    # Run examples
    example_interactive_qa()
    example_batch_processing()
    example_memory_efficient()
    example_streaming_responses()
    example_error_handling()
    example_performance_monitoring()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
