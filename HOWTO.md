# How to Use: Efficient Context Window Manager

Complete guide to using the package effectively for different scenarios.

## Table of Contents

1. [Installation](#installation)
2. [Basic Concepts](#basic-concepts)
3. [Getting Started (5 minutes)](#getting-started-5-minutes)
4. [Common Use Cases](#common-use-cases)
5. [Advanced Features](#advanced-features)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Basic Installation

```bash
pip install efficient-context-window-manager
```

### With API Support

```bash
# Install specific integrations you need
pip install efficient-context-window-manager[anthropic]  # Claude
pip install efficient-context-window-manager[openai]     # GPT
pip install efficient-context-window-manager[dev]        # For development
```

### From Source

```bash
git clone https://github.com/kholgade/efficient-context-window-manager.git
cd efficient-context-window-manager
pip install -e .
```

---

## Basic Concepts

### Context Window

The maximum number of tokens a model can process at once.

- Claude 3: 200,000 tokens
- GPT-4: 128,000 tokens
- GPT-3.5: 16,000 tokens
- Llama 2: 4,000-70,000 tokens

### The Problem This Solves

Without context management:
```
Document (500k tokens) → Model fails (exceeds limit)
```

With this package:
```
Document → Chunks → Context Window Manager → Model call
          ↓                                      ↓
      (semantically                         (respects
       coherent)                             model limits)
```

### Key Components

1. **Tokenizer**: Counts tokens in text (model-specific)
2. **Chunker**: Breaks documents into pieces
3. **Context Manager**: Selects chunks that fit the model
4. **Orchestrator**: Makes the actual LLM call
5. **Adapters**: Connect to specific LLM APIs

---

## Getting Started (5 minutes)

### Scenario 1: Simple Q&A on a Document

```python
from efficient_context_window_manager import (
    ContextWindowManager,
    efficient_llm_call,
)

# Step 1: Create manager (auto-detects your model)
manager = ContextWindowManager(
    model_name="claude-3-opus-20250219"
)

# Step 2: Read your document
with open("large_document.txt") as f:
    document = f.read()

# Step 3: Ask a question
result = efficient_llm_call(
    context_manager=manager,
    documents=document,
    query="What are the main points?",
)

# Step 4: Get answer
print(result['response'])
print(f"Tokens used: {result['tokens_used']}")
```

**That's it!** The package handles:
- ✓ Tokenizing your document
- ✓ Splitting it into chunks
- ✓ Selecting relevant chunks
- ✓ Calling Claude
- ✓ Returning response

### Scenario 2: Multiple Questions on Same Document

```python
from efficient_context_window_manager import ContextWindowManager, EfficientLLMCall

# Setup
manager = ContextWindowManager(model_name="gpt-4")
call = EfficientLLMCall(context_manager=manager, model_name="gpt-4")

# Add document once
call.add_context(large_document)

# Ask multiple questions (reuses same chunks)
q1 = call.call("What is the main topic?")
q2 = call.call("Who are the key people mentioned?")
q3 = call.call("What are the future implications?")

print(q1['response'])
print(q2['response'])
print(q3['response'])
```

---

## Common Use Cases

### Use Case 1: PDF Analysis

```python
import PyPDF2
from efficient_context_window_manager import efficient_llm_call, ContextWindowManager

# Extract text from PDF
with open("report.pdf", "rb") as f:
    reader = PyPDF2.PdfReader(f)
    text = "".join(page.extract_text() for page in reader.pages)

# Create manager
manager = ContextWindowManager(model_name="claude-3-opus-20250219")

# Ask questions
result = efficient_llm_call(
    context_manager=manager,
    documents=text,
    query="Summarize the key findings",
)

print(result['response'])
```

### Use Case 2: Multiple Document Analysis

```python
from efficient_context_window_manager import EfficientLLMCall, ContextWindowManager
import os

manager = ContextWindowManager(model_name="gpt-4")
call = EfficientLLMCall(context_manager=manager, model_name="gpt-4")

# Add multiple documents
for filename in os.listdir("documents/"):
    with open(f"documents/{filename}") as f:
        call.add_context(f.read())

# Analyze across all documents
result = call.call("What are the common themes?")
print(result['response'])
```

### Use Case 3: RAG (Retrieval-Augmented Generation)

```python
from efficient_context_window_manager import efficient_llm_call, ContextWindowManager

manager = ContextWindowManager(model_name="claude-3-opus-20250219")

# Your retrieval system returns top-K chunks
retrieved_chunks = semantic_search(query, chunk_database)

# Pass to LLM with context
result = efficient_llm_call(
    context_manager=manager,
    documents=retrieved_chunks,  # Only relevant chunks
    query=original_query,
)
```

### Use Case 4: Local Private Inference

```python
from efficient_context_window_manager import efficient_llm_call, ContextWindowManager

# Use local model (requires Ollama running: ollama serve)
manager = ContextWindowManager(model_name="llama2")

result = efficient_llm_call(
    context_manager=manager,
    documents=sensitive_document,
    query="Analyze this privately",
    provider="ollama",
    base_url="http://localhost:11434",
)

print(result['response'])  # Processed locally, no API calls
```

### Use Case 5: Comparing Chunking Strategies

```python
from efficient_context_window_manager import ContextWindowManager
from efficient_context_window_manager.types import ChunkingStrategy

document = open("large_doc.txt").read()

strategies = [
    ChunkingStrategy.FIXED_SIZE,
    ChunkingStrategy.RECURSIVE,
    ChunkingStrategy.SLIDING_WINDOW,
]

for strategy in strategies:
    manager = ContextWindowManager(
        model_name="gpt-3.5-turbo",
        chunking_strategy=strategy,
        chunk_size=1000,
    )

    chunks = manager.process_document(document)
    print(f"{strategy.value}: {len(chunks)} chunks, {sum(c.token_count for c in chunks)} tokens")
```

---

## Advanced Features

### Feature 1: Custom Chunking Strategy

```python
from efficient_context_window_manager import ContextWindowManager
from efficient_context_window_manager.types import ChunkingStrategy

# Use different chunking strategies
manager = ContextWindowManager(
    model_name="gpt-4",
    chunking_strategy=ChunkingStrategy.RECURSIVE,  # Semantic-aware
    chunk_size=2000,  # Target 2000 tokens per chunk
    overlap=300,  # 300-token overlap for context
)
```

**Available Strategies:**

| Strategy | Best For | Speed | Quality |
|----------|----------|-------|---------|
| FIXED_SIZE | Homogeneous data | ⚡⚡⚡ | ⭐⭐ |
| RECURSIVE | Mixed documents | ⚡⚡ | ⭐⭐⭐ |
| SLIDING_WINDOW | Sequential data | ⚡⚡ | ⭐⭐⭐ |
| SEMANTIC | High quality | ⚡ | ⭐⭐⭐⭐ |

### Feature 2: Compression

Reduce token usage when approaching model limits:

```python
from efficient_context_window_manager import ContextWindowManager
from efficient_context_window_manager.types import CompressionConfig, CompressionMethod

manager = ContextWindowManager(
    model_name="gpt-3.5-turbo",
    compression_config=CompressionConfig(
        method=CompressionMethod.OBSERVATION_MASKING,
        compression_ratio=0.3,  # Remove 30% of non-essential tokens
    ),
)
```

**Methods:**

| Method | Speed | Quality | Cost |
|--------|-------|---------|------|
| OBSERVATION_MASKING | Fast ⚡⚡⚡ | Good ⭐⭐⭐ | Low 💰 |
| SEMANTIC_SUMMARIZATION | Slow ⚡ | Great ⭐⭐⭐⭐ | High 💰💰 |

### Feature 3: Context Strategies

Control how chunks are selected:

```python
from efficient_context_window_manager import ContextWindowManager
from efficient_context_window_manager.types import ContextStrategyEnum

# GREEDY: Fill context with chunks in order (fastest)
manager = ContextWindowManager(
    context_strategy=ContextStrategyEnum.GREEDY,
)

# SUMMARIZE: Compress old chunks when full
manager = ContextWindowManager(
    context_strategy=ContextStrategyEnum.SUMMARIZE,
)

# SLIDING_WINDOW: Keep contiguous window
manager = ContextWindowManager(
    context_strategy=ContextStrategyEnum.SLIDING_WINDOW,
)
```

### Feature 4: Custom Tokenizer

```python
from efficient_context_window_manager import ContextWindowManager
from efficient_context_window_manager.tokenizers import AutoTokenizer

# Auto-detect (recommended)
tokenizer = AutoTokenizer("claude-3-opus-20250219")

# Or create manager with custom tokenizer
manager = ContextWindowManager(
    tokenizer=tokenizer,
    max_window_tokens=200000,
)
```

### Feature 5: Method Chaining

```python
from efficient_context_window_manager import EfficientLLMCall, ContextWindowManager

result = (
    EfficientLLMCall(ContextWindowManager(model_name="gpt-4"), model_name="gpt-4")
    .add_context("Document 1")
    .add_context("Document 2")
    .add_context("Document 3")
    .call("What's the summary?")
)
```

### Feature 6: Metrics and Monitoring

```python
from efficient_context_window_manager import EfficientLLMCall, ContextWindowManager

manager = ContextWindowManager(model_name="gpt-4")
call = EfficientLLMCall(context_manager=manager, model_name="gpt-4")

call.add_context(document)

# Check stats before calling
stats = call.get_stats()
print(f"Documents: {stats['documents']}")
print(f"Chunks: {stats['chunks']}")
print(f"Total tokens: {stats['total_tokens']}")

# After calling
result = call.call("Question?")
print(f"Tokens used: {result['tokens_used']}")
print(f"Window utilization: {result['window_utilization'] * 100:.1f}%")
```

---

## Best Practices

### 1. Choose the Right Chunking Strategy

```python
# For academic papers, books
ChunkingStrategy.RECURSIVE  # Respects paragraphs/sections

# For homogeneous data (logs, time series)
ChunkingStrategy.FIXED_SIZE  # Simple, predictable

# For sequences, conversations
ChunkingStrategy.SLIDING_WINDOW  # Overlapping windows

# For maximum quality (if budget allows)
ChunkingStrategy.SEMANTIC  # Embedding-based
```

### 2. Set Appropriate Chunk Sizes

```python
# Default: 1000 tokens
manager = ContextWindowManager(chunk_size=1000)

# Smaller chunks (more precision)
manager = ContextWindowManager(chunk_size=500)

# Larger chunks (more context)
manager = ContextWindowManager(chunk_size=2000)

# For large models with big context
manager = ContextWindowManager(chunk_size=4000)
```

### 3. Use Overlap Wisely

```python
# Default: 10% overlap
manager = ContextWindowManager(chunk_size=1000, overlap=100)

# More overlap = better continuity but more tokens
manager = ContextWindowManager(chunk_size=1000, overlap=300)

# Less overlap = fewer tokens but might lose context
manager = ContextWindowManager(chunk_size=1000, overlap=50)
```

### 4. Monitor Token Usage

```python
result = call.call("Query?")

# Check if you're using too many tokens
if result['window_utilization'] > 0.9:
    print("Warning: Using >90% of context window")
    # Consider: smaller chunks, compression, or fewer documents

if result['tokens_used'] > expected_budget:
    print("Exceeding token budget")
    # Consider: compression, different strategy
```

### 5. Set Proper Model Limits

```python
# Correct: Leave buffer for response
manager = ContextWindowManager(
    model_name="gpt-3.5-turbo",
    max_window_tokens=4096 - 1000,  # Leave 1000 for response
)

# Or use context strategy with compression
manager = ContextWindowManager(
    model_name="gpt-3.5-turbo",
    max_window_tokens=4096,
    context_strategy=ContextStrategyEnum.COMPRESS,
)
```

### 6. Handle Large Documents

```python
# Don't put everything at once
documents = load_all_documents()
call.add_contexts(documents)  # This will chunk all

# Instead, add selectively
important_docs = retrieve_relevant(query)
call.add_contexts(important_docs)
```

---

## Troubleshooting

### Problem: "Token limit exceeded"

**Solution:** Reduce chunk size or enable compression

```python
# Option 1: Smaller chunks
manager = ContextWindowManager(chunk_size=500)

# Option 2: Enable compression
from efficient_context_window_manager.types import CompressionConfig, CompressionMethod

manager = ContextWindowManager(
    compression_config=CompressionConfig(
        method=CompressionMethod.OBSERVATION_MASKING,
        compression_ratio=0.3,
    ),
)
```

### Problem: "Cannot connect to Ollama"

**Solution:** Start Ollama server

```bash
# In a terminal
ollama serve

# Then in code
result = efficient_llm_call(
    manager, documents, query,
    provider="ollama",
    base_url="http://localhost:11434",
)
```

### Problem: "ModuleNotFoundError: No module named 'anthropic'"

**Solution:** Install the integration

```bash
pip install anthropic
# or
pip install efficient-context-window-manager[anthropic]
```

### Problem: "Tokenizer not found for model"

**Solution:** Use AutoTokenizer or specify explicitly

```python
from efficient_context_window_manager.tokenizers import AutoTokenizer

# Auto-detection
tokenizer = AutoTokenizer("your-model-name")

# Or use context manager (auto-creates tokenizer)
manager = ContextWindowManager(model_name="your-model-name")
```

### Problem: "Poor response quality with chunking"

**Solution:** Increase overlap or use better chunking

```python
# More overlap = better context preservation
manager = ContextWindowManager(
    chunk_size=1000,
    overlap=200,  # 20% overlap
    chunking_strategy=ChunkingStrategy.RECURSIVE,
)

# Or use semantic chunking for highest quality
manager = ContextWindowManager(
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    embedding_fn=your_embedding_function,
)
```

### Problem: "Slow inference"

**Solution:** Reduce token count or use caching

```python
# Reduce tokens via compression
manager = ContextWindowManager(
    compression_config=CompressionConfig(
        compression_ratio=0.4,  # More aggressive
    ),
)

# Or fewer, larger chunks
manager = ContextWindowManager(
    chunk_size=2000,
    overlap=100,
)
```

---

## Performance Tips

### 1. Cache Tokenization

The package caches token counts automatically:
```python
# First call tokenizes and caches
manager.estimate_tokens(text)  # Caches result

# Subsequent calls are instant
manager.estimate_tokens(text)  # Returns from cache
```

### 2. Reuse Context Across Queries

```python
# Good: Prepare once, query multiple times
call = EfficientLLMCall(manager)
call.add_context(doc).prepare()
r1 = call.call("q1?")  # Reuses chunks
r2 = call.call("q2?")  # Reuses chunks

# Bad: Re-prepare for each query
for query in queries:
    result = efficient_llm_call(manager, doc, query)  # Chunks again
```

### 3. Use Appropriate Model Sizes

```python
# For large documents, use models with bigger contexts
manager = ContextWindowManager(model_name="claude-3-opus-20250219")  # 200k context

# For small documents, smaller models are fine
manager = ContextWindowManager(model_name="gpt-3.5-turbo")  # 16k context
```

---

## Examples

Run the included examples:

```bash
# Basic usage
python examples/basic_usage.py

# Anthropic integration
python examples/with_anthropic.py

# Efficient LLM calls
python examples/efficient_llm_call_demo.py
```

---

## Getting Help

- **Documentation**: See [README.md](README.md)
- **Examples**: Check [examples/](examples/) directory
- **Issues**: Report on [GitHub Issues](https://github.com/kholgade/efficient-context-window-manager/issues)
- **API Reference**: See [API reference section in README.md](README.md#api-reference)

---

## Summary

**Quick Decision Tree:**

```
I have a question about a document
  ↓
Use efficient_llm_call()
  ├─ Single document? → efficient_llm_call(manager, doc, query)
  └─ Multiple docs? → EfficientLLMCall(...).add_contexts(...).call(query)

I want to compare chunking strategies
  ↓
Loop through ChunkingStrategy enum and measure

I want maximum quality
  ↓
Use ChunkingStrategy.RECURSIVE or SEMANTIC

I want maximum speed
  ↓
Use ChunkingStrategy.FIXED_SIZE

I want to process privately
  ↓
Use provider="ollama" with local Llama model

I'm running out of tokens
  ↓
Enable compression or reduce chunk size
```

Happy chunking! 🚀
