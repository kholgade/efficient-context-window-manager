# Efficient Context Window Manager

Break large context windows into manageable chunks while honoring model's native limits and preserving semantic coherence.

## Problem

Most models have finite context windows:
- Llama 2: 4k-70k tokens
- GPT-3.5: 4k-16k tokens
- GPT-4: 8k-128k tokens
- Claude: 100k-200k tokens
- Ollama local models: varies

Even with large context windows, intelligent management reduces costs and improves latency.

## Solution

This package provides:

1. **Multiple chunking strategies** (recursive, sliding window, semantic, fixed-size)
2. **Token counting abstraction** (works with any model's tokenizer)
3. **Context window management** (RAG, summarization, compression)
4. **LLM integration adapters** (Anthropic, OpenAI, Ollama, custom)
5. **Compression strategies** (observation masking, summarization, relevance filtering)

## Installation

```bash
pip install efficient-context-window-manager

# Optional: for specific integrations
pip install efficient-context-window-manager[anthropic]
pip install efficient-context-window-manager[openai]
```

## Quick Start

### Basic Chunking

```python
from efficient_context_window_manager import ContextWindowManager
from efficient_context_window_manager.types import ChunkingStrategy

# Create manager
manager = ContextWindowManager(
    model_name="gpt-3.5-turbo",
    max_window_tokens=4096,
    chunking_strategy=ChunkingStrategy.RECURSIVE,
    chunk_size=1000,
    overlap=100,
)

# Process document
chunks = manager.process_document(long_document)

# Use chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk.token_count} tokens")
```

### With Anthropic Claude

```python
from efficient_context_window_manager import ContextWindowManager
from efficient_context_window_manager.integrations.anthropic import AnthropicAdapter

# Setup
context_manager = ContextWindowManager(
    model_name="claude-3-opus-20250219",
    max_window_tokens=100000,
    chunking_strategy=ChunkingStrategy.RECURSIVE,
)

adapter = AnthropicAdapter(
    context_manager=context_manager,
    model_name="claude-3-opus-20250219",
)

# Process document with query
result = adapter.process_with_context(
    document=long_document,
    query="What are the key findings?",
)

print(f"Response: {result['response']}")
print(f"Tokens used: {result['tokens_used']}")
```

### With OpenAI

```python
from efficient_context_window_manager.integrations.openai import OpenAIAdapter

adapter = OpenAIAdapter(
    context_manager=context_manager,
    model_name="gpt-4",
)

result = adapter.process_with_context(document, query)
```

### With Ollama (local)

```python
from efficient_context_window_manager.integrations.ollama import OllamaAdapter

adapter = OllamaAdapter(
    context_manager=context_manager,
    model_name="llama2",
    base_url="http://localhost:11434",
)

result = adapter.process_with_context(document, query)
```

## Architecture

### Core Modules

```
efficient_context_window_manager/
├── types.py                 # Core data structures
├── tokenizers/              # Token counting abstractions
│   ├── base.py              # BaseTokenizer interface
│   ├── openai_tokenizer.py  # OpenAI/GPT tokenization
│   ├── huggingface_tokenizer.py  # Local model tokenization
│   ├── anthropic_tokenizer.py   # Claude tokenization
│   └── auto.py              # AutoTokenizer factory
├── chunking/                # Chunking strategies
│   ├── base.py              # BaseChunker interface
│   ├── fixed_size.py        # Fixed-size chunks
│   ├── recursive.py         # Semantic recursive splitting
│   ├── sliding_window.py    # SWAT approach
│   ├── semantic.py          # Embedding-based splitting
│   └── factory.py           # Chunker factory
├── compression/             # Compression strategies
│   ├── base.py              # BaseCompressor interface
│   ├── masking.py           # Observation masking
│   └── summarization.py     # LLM summarization
├── context_manager.py       # Main orchestrator
└── integrations/            # LLM API adapters
    ├── base.py              # BaseAdapter interface
    ├── anthropic.py         # Anthropic Claude
    ├── openai.py            # OpenAI GPT
    └── ollama.py            # Ollama local
```

## Chunking Strategies

### 1. Recursive Chunking (Recommended)
- Splits hierarchically by semantic delimiters
- Preserves document structure
- Best for mixed content
- Uses: paragraphs → sentences → words → characters

```python
from efficient_context_window_manager.types import ChunkingStrategy

manager = ContextWindowManager(
    chunking_strategy=ChunkingStrategy.RECURSIVE,
    chunk_size=1000,
)
```

### 2. Sliding Window (SWAT)
- Creates overlapping windows with controlled stride
- Efficient for sequential processing
- Based on research paper: "Sliding Window Attention Training for Efficient LLMs"

```python
manager = ContextWindowManager(
    chunking_strategy=ChunkingStrategy.SLIDING_WINDOW,
    chunk_size=1000,
    overlap=300,  # 30% overlap
)
```

### 3. Fixed Size
- Simple deterministic chunking
- Predictable output
- No semantic awareness

```python
manager = ContextWindowManager(
    chunking_strategy=ChunkingStrategy.FIXED_SIZE,
    chunk_size=1000,
)
```

### 4. Semantic
- Uses embeddings to identify natural boundaries
- Highest quality but slower
- Requires embedding function

```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')

manager = ContextWindowManager(
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    embedding_fn=embedder.encode,
)
```

## Context Management Strategies

### Greedy (Default)
- Adds chunks in order until full
- Fast, simple
- Best for sequential documents

```python
from efficient_context_window_manager.types import ContextStrategyEnum

manager = ContextWindowManager(
    context_strategy=ContextStrategyEnum.GREEDY,
)
```

### Summarization
- Compresses older chunks when approaching limit
- Preserves recent context
- Good for long conversations

```python
manager = ContextWindowManager(
    context_strategy=ContextStrategyEnum.SUMMARIZE,
)
```

### Compression
- Pre-compresses chunks to fit more
- Uses observation masking or LLM summarization
- Reduces token usage

```python
from efficient_context_window_manager.types import (
    ContextStrategyEnum, CompressionConfig, CompressionMethod
)

manager = ContextWindowManager(
    context_strategy=ContextStrategyEnum.COMPRESS,
    compression_config=CompressionConfig(
        method=CompressionMethod.OBSERVATION_MASKING,
        compression_ratio=0.3,  # Remove 30% of tokens
    ),
)
```

### Sliding Window
- Maintains contiguous window of chunks
- Efficient memory usage
- Good for long sequences

```python
manager = ContextWindowManager(
    context_strategy=ContextStrategyEnum.SLIDING_WINDOW,
)
```

## Compression Methods

### Observation Masking (Recommended)
- Removes non-essential tokens (articles, filler words)
- Fast and effective
- Maintains coherence

```python
from efficient_context_window_manager.types import CompressionMethod

compression_config = CompressionConfig(
    method=CompressionMethod.OBSERVATION_MASKING,
    compression_ratio=0.3,  # Target 30% reduction
)
```

### Semantic Summarization
- Uses LLM to create summaries
- Better semantic preservation
- More expensive (requires API call)

```python
compression_config = CompressionConfig(
    method=CompressionMethod.SEMANTIC_SUMMARIZATION,
    compression_ratio=0.4,
)
```

## Tokenizer Support

Auto-detects tokenizer by model name:

```python
from efficient_context_window_manager.tokenizers import AutoTokenizer

# Automatically select right tokenizer
tokenizer = AutoTokenizer("gpt-3.5-turbo")      # OpenAI
tokenizer = AutoTokenizer("claude-3-opus-20250219")  # Anthropic
tokenizer = AutoTokenizer("meta-llama/Llama-2-7b")   # HuggingFace
```

## Examples

See `examples/` directory:
- `basic_usage.py` - Chunking strategies demo
- `with_anthropic.py` - Anthropic integration
- `with_openai.py` - OpenAI integration (coming soon)

## Key Concepts

### Chunk
Unit of text with metadata:
- `content`: Text
- `token_count`: Accurate token count
- `metadata`: Position, source, importance

### ContextWindow
Current processing state:
- `current_tokens`: Used tokens
- `max_tokens`: Model limit
- `chunks`: Selected chunks
- `utilization`: Percentage used

### CompressionConfig
Settings for compression:
- `method`: Compression strategy
- `compression_ratio`: Target token reduction
- `preserve_recent_tokens`: Keep recent context
- `preserve_important`: Mark important chunks

## Performance Considerations

1. **Token Counting**: Cached to avoid repeated computation
2. **Compression**: Optional, trades speed for tokens
3. **Chunking**: Recursive is slower but higher quality
4. **Semantic**: Requires embeddings (can be expensive)

## Research Background

The package implements techniques from:

1. **"Sliding Window Attention Training for Efficient Large Language Models"** (2502.18845)
   - SWAT approach for efficient context windows

2. **"Prompt Compression for Large Language Models: A Survey"** (2410.12388)
   - Observation masking, semantic compression

3. **"Characterizing Prompt Compression Methods"** (2407.08892)
   - Effective compression strategies comparison

4. **"Beyond the Limits: Context Length Extension Survey"** (2402.02244)
   - Comprehensive context management techniques

## API Reference

### ContextWindowManager

Main orchestrator class.

**Methods:**
- `process_document(document: str) -> List[Chunk]`
- `create_context_window(chunks: List[Chunk]) -> ContextWindow`
- `estimate_tokens(text: str) -> int`
- `get_metrics() -> Dict`

### Adapters

Base class: `BaseAdapter`

**Concrete:**
- `AnthropicAdapter`
- `OpenAIAdapter`
- `OllamaAdapter`

**Methods:**
- `process_with_context(document, query, system_prompt) -> Dict`
- `prepare_request(document, query) -> Dict`
- `call_api(messages) -> str`

## Contributing

Contributions welcome! Areas for enhancement:
- Additional compression methods
- New LLM integrations
- Performance optimizations
- Better semantic chunking
- Caching backends

## License

MIT

## Support

For issues and questions:
- GitHub: https://github.com/kholgade/efficient-context-window-manager
- Docs: See this README and examples/
