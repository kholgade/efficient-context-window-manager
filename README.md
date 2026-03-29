# Efficient Context Window Manager

**Break large context windows into manageable chunks while honoring model's native limits and preserving semantic coherence.**

A production-ready Python package that lets you work with context windows far larger than individual models support. Automatically chunks documents, selects relevant content, manages compression, and calls LLMs efficiently.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## ⚡ Quick Start

### The Simplest Way (3 lines)

```python
from efficient_context_window_manager import ContextWindowManager, efficient_llm_call

manager = ContextWindowManager(model_name="claude-3-opus-20250219")
result = efficient_llm_call(manager, documents=your_docs, query="What are the key points?")
print(result['response'])
```

### Accumulate Context Then Query

```python
from efficient_context_window_manager import ContextWindowManager, EfficientLLMCall

call = EfficientLLMCall(ContextWindowManager(model_name="gpt-4"), model_name="gpt-4")
call.add_context("Doc 1...").add_context("Doc 2...").add_context("Doc 3...")
result = call.call("Your question?")
```

## 🎯 The Problem

Most models have finite context windows - intelligent management is essential:

- **Claude**: 100k-200k tokens
- **GPT-4**: 128k tokens
- **GPT-3.5**: 16k tokens
- **Llama**: 4k-70k tokens
- **Local models**: Varies (often 4k-8k)

Without context management: `Document (500k tokens) → Model fails`

With this package: `Document → Smart chunks → Context window respecting model limits → LLM call`

## ✨ Features

| Feature | Description |
|---------|-------------|
| **4 Chunking Strategies** | Recursive (semantic-aware), Sliding Window, Fixed-Size, Semantic (embedding-based) |
| **Model-Agnostic** | Works with Claude, GPT, Llama, Ollama, or any model |
| **Auto Tokenization** | Automatically detects and uses the right tokenizer |
| **Smart Context Management** | Greedy, summarization, compression, sliding window strategies |
| **Compression Methods** | Observation masking (fast) or semantic summarization (high quality) |
| **LLM Integrations** | Anthropic Claude, OpenAI GPT, Ollama local, or custom APIs |
| **Production-Ready** | Token caching, error handling, metrics tracking |
| **Easy to Use** | Simple function-based interface or powerful class API |
| **Research-Backed** | Implements SWAT, LLMLingua, RAG best practices |

## 📖 Documentation

- **[HOWTO Guide](HOWTO.md)** ← Start here for comprehensive guide
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** ← Technical architecture
- **[Examples](examples/)** ← Working code samples
- **[API Reference](#api-reference)** ← Complete API docs

## 📦 Installation

```bash
# Basic installation
pip install efficient-context-window-manager

# With Claude support
pip install efficient-context-window-manager[anthropic]

# With OpenAI support
pip install efficient-context-window-manager[openai]

# From source
git clone https://github.com/kholgade/efficient-context-window-manager.git
cd efficient-context-window-manager
pip install -e .
```

### Environment Setup

Set API keys for the services you use:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# For Ollama (local), start the server:
ollama serve  # Runs on http://localhost:11434 by default
```

## Quick Start

### Simplest: One-Shot LLM Call

```python
from efficient_context_window_manager import ContextWindowManager, efficient_llm_call

# Create manager (auto-detects tokenizer)
manager = ContextWindowManager(model_name="claude-3-opus-20250219")

# Make efficient LLM call with context
result = efficient_llm_call(
    context_manager=manager,
    documents=["Your document 1", "Your document 2"],  # Add context
    query="What are the key points?",
)

print(result['response'])  # Claude's answer
print(f"Used {result['tokens_used']} tokens")
```

### With Context Accumulation

```python
from efficient_context_window_manager import ContextWindowManager, EfficientLLMCall

manager = ContextWindowManager(model_name="claude-3-opus-20250219")
call = EfficientLLMCall(context_manager=manager, model_name="claude-3-opus-20250219")

# Add context progressively
call.add_context("Document 1: ...")
call.add_context("Document 2: ...")
call.add_context("Document 3: ...")

# Make call
result = call.call(query="What are the key findings?")

# Make another call with same context
result2 = call.call(query="How do these relate to X?")

# Clear for new context
call.clear_context()
```

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

## 🏆 Why This Package?

### Simple vs Complex

```python
# Without this package - handle everything manually
chunks = []
for i in range(0, len(text), 1000):
    chunks.append(text[i:i+1000])
# ... estimate tokens ...
# ... handle API differences ...
# ... compress if needed ...

# With this package - automatic
result = efficient_llm_call(manager, text, query)
```

### Specialized for Context Windows

This package focuses specifically on **context window management** - something general frameworks handle as an afterthought:

| Feature | This Package | LangChain | LLaMA Index |
|---------|--------------|-----------|-------------|
| **Context Window Optimization** | ✅ Core focus | ⚠️ Secondary | ⚠️ Secondary |
| **Compression Strategies** | ✅ 2 methods | ❌ Limited | ❌ Limited |
| **Chunking Strategies** | ✅ 4 types | ✅ 2 types | ✅ 2 types |
| **Research-Based (SWAT/LLMLingua)** | ✅ Yes | ⚠️ Partial | ⚠️ Partial |
| **Simple API** | ✅ Yes | ❌ Complex | ❌ Complex |

---

## Architecture

### Core Modules

```
efficient_context_window_manager/
├── types.py                 # Core data structures
├── orchestrator.py          # High-level LLM call management ⭐ NEW
├── context_manager.py       # Main window orchestrator
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

## 💡 Common Use Cases

See [HOWTO.md](HOWTO.md) for detailed walkthroughs:

1. **PDF/Document Analysis** - Extract text and ask questions
2. **Multi-Document Analysis** - Query across multiple sources
3. **RAG Systems** - Combine retrieval with efficient context management
4. **Local Private Processing** - Use Ollama for privacy
5. **Chatbots** - Manage conversation history with large context
6. **Code Analysis** - Process large codebases with context

## 🎓 Examples

Working examples in `examples/` directory:

```bash
# Basic chunking strategies demo
python examples/basic_usage.py

# Anthropic Claude integration
python examples/with_anthropic.py

# Efficient LLM calls (recommended starting point)
python examples/efficient_llm_call_demo.py
```

Or check [HOWTO.md](HOWTO.md) for step-by-step guides.

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

## Orchestrator: Simple LLM Calls

### `efficient_llm_call()` - One-Shot Function

Simple function for making a single efficient LLM call:

```python
from efficient_context_window_manager import efficient_llm_call, ContextWindowManager

result = efficient_llm_call(
    context_manager=manager,
    documents=["doc1", "doc2"],
    query="Your question",
    model_name="claude-3-opus-20250219",  # Optional
    provider="anthropic",                  # Optional - auto-detected
)

# Returns: {response, tokens_used, window_utilization, chunks_used, ...}
```

### `EfficientLLMCall` - Class for Reuse

Class for accumulating context and making multiple queries:

```python
from efficient_context_window_manager import EfficientLLMCall, ContextWindowManager

call = EfficientLLMCall(
    context_manager=manager,
    model_name="gpt-4",
)

# Accumulate context
call.add_context("Document 1...")
call.add_context("Document 2...")

# Query 1
result1 = call.call("First question?")

# Query 2 - reuses same context
result2 = call.call("Second question?")

# Clear for new context
call.clear_context()

# Get stats
stats = call.get_stats()
```

**Features:**
- Method chaining: `call.add_context(...).add_context(...).call(...)`
- Auto-detects LLM provider from model name
- Prepares context on-demand
- Supports multiple consecutive queries
- Returns detailed metrics

---

## API Reference

### ContextWindowManager

Main context window orchestrator class.

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
