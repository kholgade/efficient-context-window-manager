"""
Speed Benchmarks: With vs Without Context Window Manager

Measures execution time for:
1. Document chunking
2. Token estimation
3. Context window creation
4. Total pipeline execution
"""

import time
from typing import Tuple, List
from efficient_context_window_manager import (
    ContextWindowManager,
    efficient_llm_call,
)
from efficient_context_window_manager.types import ChunkingStrategy


class SpeedBenchmark:
    """Benchmark speed of package operations."""

    def __init__(self):
        """Initialize benchmark."""
        self.results = {}

    def benchmark_tokenization(
        self,
        text: str,
        num_iterations: int = 100,
    ) -> Tuple[float, dict]:
        """
        Benchmark token counting speed.

        Args:
            text: Text to tokenize
            num_iterations: How many times to run

        Returns:
            Average time and metadata
        """
        manager = ContextWindowManager(model_name="gpt-3.5-turbo")

        # Warmup
        manager.estimate_tokens(text)

        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            manager.estimate_tokens(text)
        end = time.time()

        avg_time = (end - start) / num_iterations
        return avg_time, {
            "total_iterations": num_iterations,
            "total_time": end - start,
            "avg_time_per_token_count": avg_time,
        }

    def benchmark_chunking(
        self,
        text: str,
        strategy: ChunkingStrategy,
        num_iterations: int = 10,
    ) -> Tuple[float, dict]:
        """
        Benchmark chunking speed.

        Args:
            text: Text to chunk
            strategy: Chunking strategy
            num_iterations: How many times to run

        Returns:
            Average time and metadata
        """
        manager = ContextWindowManager(
            model_name="gpt-3.5-turbo",
            chunking_strategy=strategy,
            chunk_size=1000,
        )

        # Warmup
        manager.process_document(text)

        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            manager.process_document(text)
        end = time.time()

        avg_time = (end - start) / num_iterations

        # Get actual chunks for reference
        chunks = manager.process_document(text)

        return avg_time, {
            "strategy": strategy.value,
            "total_iterations": num_iterations,
            "total_time": end - start,
            "avg_time_per_chunk": avg_time,
            "chunks_created": len(chunks),
        }

    def benchmark_context_window_creation(
        self,
        text: str,
        num_iterations: int = 10,
    ) -> Tuple[float, dict]:
        """
        Benchmark context window creation.

        Args:
            text: Document text
            num_iterations: How many times to run

        Returns:
            Average time and metadata
        """
        manager = ContextWindowManager(model_name="gpt-3.5-turbo")

        # Pre-chunk
        chunks = manager.process_document(text)

        # Benchmark window creation
        start = time.time()
        for _ in range(num_iterations):
            manager.create_context_window(chunks)
        end = time.time()

        avg_time = (end - start) / num_iterations

        return avg_time, {
            "total_iterations": num_iterations,
            "total_time": end - start,
            "avg_time_per_window_creation": avg_time,
        }

    def benchmark_full_pipeline(
        self,
        text: str,
        chunk_size: int = 1000,
        num_iterations: int = 5,
    ) -> Tuple[float, dict]:
        """
        Benchmark full pipeline (chunking + context creation).

        Args:
            text: Document text
            chunk_size: Target chunk size
            num_iterations: How many times to run

        Returns:
            Average time and metadata
        """
        manager = ContextWindowManager(
            model_name="gpt-3.5-turbo",
            chunk_size=chunk_size,
        )

        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            chunks = manager.process_document(text)
            manager.create_context_window(chunks)
        end = time.time()

        avg_time = (end - start) / num_iterations

        return avg_time, {
            "total_iterations": num_iterations,
            "total_time": end - start,
            "avg_time_full_pipeline": avg_time,
        }

    def benchmark_caching_effect(
        self,
        text: str,
    ) -> Tuple[float, float, dict]:
        """
        Benchmark caching efficiency.

        Args:
            text: Document text

        Returns:
            First call time, cached call time, and metadata
        """
        manager = ContextWindowManager(model_name="gpt-3.5-turbo")

        # First call (no cache)
        start1 = time.time()
        manager.estimate_tokens(text)
        time1 = time.time() - start1

        # Cached call
        start2 = time.time()
        manager.estimate_tokens(text)
        time2 = time.time() - start2

        speedup = time1 / time2 if time2 > 0 else 0

        return time1, time2, {
            "first_call_time": time1,
            "cached_call_time": time2,
            "speedup_factor": speedup,
            "cache_improvement": f"{(1 - time2/time1) * 100:.1f}%",
        }

    def run_all_benchmarks(self, text: str) -> dict:
        """
        Run all speed benchmarks.

        Args:
            text: Document text

        Returns:
            Dictionary of all results
        """
        print("\n" + "=" * 70)
        print("SPEED BENCHMARKS")
        print("=" * 70)

        results = {}

        # Tokenization
        print("\n1. Tokenization Speed...")
        avg_time, meta = self.benchmark_tokenization(text)
        results["tokenization"] = meta
        print(f"   Avg time: {avg_time*1000:.3f}ms")

        # Chunking strategies
        print("\n2. Chunking Speed (by strategy)...")
        results["chunking"] = {}
        for strategy in [
            ChunkingStrategy.FIXED_SIZE,
            ChunkingStrategy.RECURSIVE,
            ChunkingStrategy.SLIDING_WINDOW,
        ]:
            avg_time, meta = self.benchmark_chunking(text, strategy)
            results["chunking"][strategy.value] = meta
            print(f"   {strategy.value}: {avg_time*1000:.2f}ms")

        # Context window creation
        print("\n3. Context Window Creation...")
        avg_time, meta = self.benchmark_context_window_creation(text)
        results["context_window"] = meta
        print(f"   Avg time: {avg_time*1000:.3f}ms")

        # Full pipeline
        print("\n4. Full Pipeline...")
        avg_time, meta = self.benchmark_full_pipeline(text)
        results["full_pipeline"] = meta
        print(f"   Avg time: {avg_time*1000:.2f}ms")

        # Caching effect
        print("\n5. Caching Effect...")
        time1, time2, meta = self.benchmark_caching_effect(text)
        results["caching"] = meta
        print(f"   First call: {time1*1000:.3f}ms")
        print(f"   Cached call: {time2*1000:.3f}ms")
        print(f"   Speedup: {meta['speedup_factor']:.1f}x")

        self.results = results
        return results


if __name__ == "__main__":
    # Sample documents of different sizes
    small_doc = "Sample text. " * 100  # ~100 tokens
    medium_doc = "Sample text. " * 1000  # ~1000 tokens
    large_doc = "Sample text. " * 10000  # ~10k tokens

    benchmark = SpeedBenchmark()

    print("\n" + "=" * 70)
    print("SPEED BENCHMARKS - DIFFERENT DOCUMENT SIZES")
    print("=" * 70)

    print("\n--- SMALL DOCUMENT (~100 tokens) ---")
    benchmark.run_all_benchmarks(small_doc)

    print("\n--- MEDIUM DOCUMENT (~1000 tokens) ---")
    benchmark.run_all_benchmarks(medium_doc)

    print("\n--- LARGE DOCUMENT (~10k tokens) ---")
    benchmark.run_all_benchmarks(large_doc)

    print("\n" + "=" * 70)
    print("Speed benchmarks completed!")
    print("=" * 70)
