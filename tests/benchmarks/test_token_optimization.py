"""
Token Optimization Benchmarks: With vs Without Package

Compares:
1. Naive approach (no chunking, send entire document)
2. With package (intelligent chunking + compression)
3. Token savings and efficiency gains
"""

from efficient_context_window_manager import ContextWindowManager
from efficient_context_window_manager.types import (
    ChunkingStrategy,
    CompressionConfig,
    CompressionMethod,
)


class TokenOptimizationBenchmark:
    """Benchmark token optimization with and without package."""

    def __init__(self):
        """Initialize benchmark."""
        self.results = {}

    def naive_approach(self, text: str, model_name: str = "gpt-3.5-turbo") -> dict:
        """
        Naive approach: Count tokens for entire document.

        Args:
            text: Document text
            model_name: Model name

        Returns:
            Token statistics
        """
        manager = ContextWindowManager(model_name=model_name)
        total_tokens = manager.estimate_tokens(text)

        return {
            "approach": "naive",
            "total_tokens": total_tokens,
            "chunks": 1,
            "overhead_tokens": 0,
            "efficiency": "No optimization",
        }

    def smart_chunking_approach(
        self,
        text: str,
        model_name: str = "gpt-3.5-turbo",
        chunk_size: int = 1000,
    ) -> dict:
        """
        Smart approach: Chunk and select relevant portions.

        Args:
            text: Document text
            model_name: Model name
            chunk_size: Target chunk size

        Returns:
            Token statistics
        """
        manager = ContextWindowManager(
            model_name=model_name,
            chunking_strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=chunk_size,
        )

        chunks = manager.process_document(text)
        total_tokens = sum(c.token_count for c in chunks)

        # Estimate context window (assume 80% utilization)
        max_window = 4096  # GPT-3.5
        context_tokens = int(max_window * 0.8)
        selected_tokens = 0
        selected_chunks = 0

        for chunk in chunks:
            if selected_tokens + chunk.token_count <= context_tokens:
                selected_tokens += chunk.token_count
                selected_chunks += 1
            else:
                break

        return {
            "approach": "smart_chunking",
            "total_tokens": total_tokens,
            "chunks": len(chunks),
            "selected_tokens": selected_tokens,
            "selected_chunks": selected_chunks,
            "efficiency": f"{(selected_tokens / total_tokens * 100):.1f}% selected",
        }

    def compression_approach(
        self,
        text: str,
        model_name: str = "gpt-3.5-turbo",
        chunk_size: int = 1000,
        compression_ratio: float = 0.3,
    ) -> dict:
        """
        Compression approach: Chunk + compress + select.

        Args:
            text: Document text
            model_name: Model name
            chunk_size: Target chunk size
            compression_ratio: Target compression (0.3 = 30% reduction)

        Returns:
            Token statistics
        """
        manager = ContextWindowManager(
            model_name=model_name,
            chunking_strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=chunk_size,
            compression_config=CompressionConfig(
                method=CompressionMethod.OBSERVATION_MASKING,
                compression_ratio=compression_ratio,
            ),
        )

        chunks = manager.process_document(text)
        total_tokens_original = sum(c.token_count for c in chunks)

        # Simulate compression
        compressed_tokens = int(total_tokens_original * (1 - compression_ratio))
        avg_chunk_size_compressed = int(
            (total_tokens_original // len(chunks)) * (1 - compression_ratio)
        )

        # Estimate how many fit
        max_window = 4096
        context_tokens = int(max_window * 0.8)
        selected_chunks = context_tokens // max(avg_chunk_size_compressed, 1)

        return {
            "approach": "smart_chunking_compression",
            "total_tokens_original": total_tokens_original,
            "total_tokens_compressed": compressed_tokens,
            "chunks": len(chunks),
            "compression_ratio": f"{compression_ratio * 100:.0f}%",
            "tokens_saved": total_tokens_original - compressed_tokens,
            "selected_chunks_after_compression": selected_chunks,
            "efficiency": f"{(selected_chunks / len(chunks) * 100):.1f}% more chunks fit",
        }

    def compare_approaches(self, text: str, model_name: str = "gpt-3.5-turbo") -> dict:
        """
        Compare all approaches.

        Args:
            text: Document text
            model_name: Model name

        Returns:
            Comparison results
        """
        results = {
            "naive": self.naive_approach(text, model_name),
            "smart_chunking": self.smart_chunking_approach(text, model_name),
            "compression": self.compression_approach(text, model_name),
        }

        # Calculate improvements
        naive_tokens = results["naive"]["total_tokens"]
        smart_tokens = results["smart_chunking"]["selected_tokens"]
        compressed_tokens = results["compression"]["total_tokens_compressed"]

        results["improvement"] = {
            "smart_vs_naive": {
                "tokens_saved": naive_tokens - smart_tokens,
                "percentage": f"{(1 - smart_tokens / naive_tokens) * 100:.1f}%",
            },
            "compression_vs_naive": {
                "tokens_saved": naive_tokens - compressed_tokens,
                "percentage": f"{(1 - compressed_tokens / naive_tokens) * 100:.1f}%",
            },
        }

        return results

    def benchmark_different_sizes(self) -> dict:
        """
        Benchmark documents of different sizes.

        Returns:
            Results for different sizes
        """
        sizes = {
            "small": ("Sample text. " * 100, "~100 tokens"),
            "medium": ("Sample text. " * 500, "~500 tokens"),
            "large": ("Sample text. " * 2000, "~2000 tokens"),
            "xlarge": ("Sample text. " * 5000, "~5000 tokens"),
        }

        all_results = {}

        print("\n" + "=" * 70)
        print("TOKEN OPTIMIZATION - DIFFERENT DOCUMENT SIZES")
        print("=" * 70)

        for size_name, (text, description) in sizes.items():
            print(f"\n{size_name.upper()} ({description})")
            print("-" * 70)

            results = self.compare_approaches(text)
            all_results[size_name] = results

            naive = results["naive"]["total_tokens"]
            smart = results["smart_chunking"]["selected_tokens"]
            compressed = results["compression"]["total_tokens_compressed"]

            print(f"\nApproach Comparison:")
            print(f"  Naive (no optimization):     {naive:,} tokens")
            print(f"  Smart chunking:               {smart:,} tokens")
            print(f"  Smart + compression:          {compressed:,} tokens")
            print(f"\nSavings:")
            print(
                f"  Smart chunking saves:         {naive - smart:,} tokens "
                f"({(1 - smart/naive)*100:.1f}%)"
            )
            print(
                f"  Smart + compression saves:    {naive - compressed:,} tokens "
                f"({(1 - compressed/naive)*100:.1f}%)"
            )

        self.results = all_results
        return all_results

    def benchmark_chunking_strategies(self, text: str) -> dict:
        """
        Compare different chunking strategies for token efficiency.

        Args:
            text: Document text

        Returns:
            Comparison results
        """
        strategies = [
            ChunkingStrategy.FIXED_SIZE,
            ChunkingStrategy.RECURSIVE,
            ChunkingStrategy.SLIDING_WINDOW,
        ]

        results = {}

        print("\n" + "=" * 70)
        print("CHUNKING STRATEGY COMPARISON")
        print("=" * 70)

        for strategy in strategies:
            manager = ContextWindowManager(
                model_name="gpt-3.5-turbo",
                chunking_strategy=strategy,
                chunk_size=1000,
            )

            chunks = manager.process_document(text)
            total_tokens = sum(c.token_count for c in chunks)

            results[strategy.value] = {
                "strategy": strategy.value,
                "chunks": len(chunks),
                "total_tokens": total_tokens,
                "avg_chunk_size": total_tokens // len(chunks) if chunks else 0,
            }

            print(f"\n{strategy.value}:")
            print(f"  Chunks: {len(chunks)}")
            print(f"  Total tokens: {total_tokens:,}")
            print(f"  Avg chunk size: {total_tokens // len(chunks) if chunks else 0} tokens")

        return results


if __name__ == "__main__":
    benchmark = TokenOptimizationBenchmark()

    # Run comprehensive benchmarks
    print("\n" + "=" * 70)
    print("TOKEN OPTIMIZATION BENCHMARKS")
    print("=" * 70)

    benchmark.benchmark_different_sizes()
    benchmark.benchmark_chunking_strategies("Sample text. " * 2000)

    print("\n" + "=" * 70)
    print("Token optimization benchmarks completed!")
    print("=" * 70)
