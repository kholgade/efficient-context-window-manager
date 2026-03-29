# Using Ollama with Large Documents

Complete guide for using Ollama models with the context window manager to process documents larger than your model's native context window.

## The Problem This Solves

Ollama models typically have small context windows:
- **Llama2**: 4,096 tokens
- **Mistral**: 8,192 tokens
- **Neural-Chat**: 8,192 tokens
- **Zephyr**: 4,096 tokens

But you want to process documents of **100k+ tokens**.

**Without this package**: Impossible - documents exceed model limits

**With this package**: Automatic and efficient through intelligent chunking

---

## Quick Start

### 1. Install and Start Ollama

```bash
# Install Ollama (https://ollama.ai)
# Then start the server
ollama serve

# In another terminal, pull a model
ollama pull llama2  # Or: mistral, neural-chat, zephyr, etc.
```

### 2. Install the Package

```bash
pip install efficient-context-window-manager
```

### 3. Process Large Documents

```python
from efficient_context_window_manager import (
    ContextWindowManager,
    efficient_llm_call,
)

# Create manager
manager = ContextWindowManager(
    model_name="llama2",
    max_window_tokens=8000,  # Llama2's native limit
)

# Load large document
with open("large_document.txt") as f:
    document = f.read()

# Make efficient call
result = efficient_llm_call(
    context_manager=manager,
    documents=document,
    query="What are the key points?",
    provider="ollama",
)

print(result['response'])
print(f"Tokens used: {result['tokens_used']}")
```

That's it! The package handles:
✓ Chunking the document
✓ Managing context windows
✓ Calling Ollama efficiently

---

## How It Works

### Without the Package

```
100k token document
    ↓
[Exceeds 8k limit]
    ↓
❌ Error: Token limit exceeded
```

### With the Package

```
100k token document
    ↓
[Intelligent chunking]
    ↓
Chunk 1 (8k)  ← Call Ollama
Chunk 2 (8k)  ← Call Ollama
Chunk 3 (8k)  ← Call Ollama
...
    ↓
✓ All content processed
✓ Coherent response
```

---

## Common Models and Limits

| Model | Context | Speed | Quality | RAM |
|-------|---------|-------|---------|-----|
| **Llama2** | 4k | ⭐⭐ | ⭐⭐⭐ | 7GB |
| **Mistral** | 8k | ⚡⚡⚡ | ⭐⭐⭐ | 7GB |
| **Neural-Chat** | 8k | ⚡⚡⚡ | ⭐⭐⭐ | 7GB |
| **Zephyr** | 4k | ⚡⚡⚡ | ⭐⭐⭐ | 7GB |
| **Orca-Mini** | 4k | ⚡⚡⚡ | ⭐⭐ | 3GB |

Choose based on your hardware and use case.

---

## Examples

### Example 1: Basic Document Q&A

```python
from efficient_context_window_manager import (
    ContextWindowManager,
    EfficientLLMCall,
)

# Setup
manager = ContextWindowManager(
    model_name="mistral",
    max_window_tokens=8000,
    chunk_size=1500,  # Smaller chunks for 8k limit
)

call = EfficientLLMCall(context_manager=manager, model_name="mistral")

# Load document
with open("report.txt") as f:
    call.add_context(f.read())

# Ask question
result = call.call("What are the main findings?")
print(result['response'])
```

### Example 2: Multiple Queries (Efficient)

```python
call = EfficientLLMCall(context_manager=manager, model_name="mistral")
call.add_context(large_document)

# These all reuse the same chunks - no re-chunking
r1 = call.call("What is this about?")
r2 = call.call("Who are the main people?")
r3 = call.call("What happens next?")

# Efficient: chunked once, queried multiple times
```

### Example 3: Batch Processing

```python
import os

call = EfficientLLMCall(context_manager=manager, model_name="mistral")

# Load multiple documents
for filename in os.listdir("documents/"):
    with open(f"documents/{filename}") as f:
        call.add_context(f.read())

# Query across all
result = call.call("What's the common theme?")
```

### Example 4: With Compression

Fit even more content by removing non-essential tokens:

```python
from efficient_context_window_manager.types import (
    CompressionConfig,
    CompressionMethod,
)

manager = ContextWindowManager(
    model_name="llama2",
    max_window_tokens=4096,
    compression_config=CompressionConfig(
        method=CompressionMethod.OBSERVATION_MASKING,
        compression_ratio=0.3,  # Remove 30% of non-essential tokens
    ),
)

# Now fits more content in 4k window
```

---

## Configuration Guide

### Chunk Size Selection

Choose based on your model's context window:

```python
# For 4k models (Llama2)
manager = ContextWindowManager(
    model_name="llama2",
    max_window_tokens=4000,
    chunk_size=800,  # Leave room for queries
    overlap=100,
)

# For 8k models (Mistral)
manager = ContextWindowManager(
    model_name="mistral",
    max_window_tokens=8000,
    chunk_size=1500,  # More room
    overlap=200,
)

# For custom models
manager = ContextWindowManager(
    model_name="custom",
    max_window_tokens=YOUR_LIMIT,
    chunk_size=YOUR_LIMIT // 5,  # Use 1/5 for chunks
)
```

### Chunking Strategies

Choose based on content type:

```python
from efficient_context_window_manager.types import ChunkingStrategy

# For documents, papers, books (recommended)
manager = ContextWindowManager(
    chunking_strategy=ChunkingStrategy.RECURSIVE,
)

# For code, logs, structured data
manager = ContextWindowManager(
    chunking_strategy=ChunkingStrategy.FIXED_SIZE,
)

# For sequences, time series
manager = ContextWindowManager(
    chunking_strategy=ChunkingStrategy.SLIDING_WINDOW,
)
```

---

## Best Practices

### 1. Leave Token Buffer

```python
# Good: Leave room for response
manager = ContextWindowManager(
    model_name="mistral",
    max_window_tokens=7000,  # 8k limit, leave 1k for response
)

# Bad: Using full window
manager = ContextWindowManager(
    model_name="mistral",
    max_window_tokens=8192,  # No buffer
)
```

### 2. Monitor Utilization

```python
result = call.call(query)

if result['window_utilization'] > 0.9:
    print("Warning: Using >90% of context window")
    # Consider: reduce chunk size or use compression
```

### 3. Reuse Context

```python
# Good: Load once, query many times
call = EfficientLLMCall(manager)
call.add_context(doc)
for query in queries:
    call.call(query)  # Reuses chunks

# Bad: Reload for each query
for query in queries:
    result = efficient_llm_call(manager, doc, query)  # Re-chunks each time
```

### 4. Use Appropriate Models

```python
# For fast inference on local hardware
manager = ContextWindowManager(model_name="mistral")

# For better quality (if hardware allows)
manager = ContextWindowManager(model_name="neural-chat")

# For smaller hardware
manager = ContextWindowManager(model_name="orca-mini")
```

---

## Performance Tips

### 1. Adjust Ollama Parameters

```bash
# Use GPU for faster inference
OLLAMA_GPU=1 ollama serve

# Set number of threads
OLLAMA_NUM_THREAD=8 ollama serve

# Increase context (if model supports)
ollama pull llama2:13b  # Larger version
```

### 2. Optimize Chunk Size

```python
# Experiment with different sizes
for chunk_size in [500, 1000, 1500, 2000]:
    manager = ContextWindowManager(
        model_name="mistral",
        chunk_size=chunk_size,
    )
    chunks = manager.process_document(doc)
    print(f"Chunk size {chunk_size}: {len(chunks)} chunks")
```

### 3. Use Caching

The package caches token counts automatically:

```python
# First call tokenizes (slower)
result1 = call.call("Q1?")

# Subsequent calls use cache (faster)
result2 = call.call("Q2?")
```

---

## Troubleshooting

### "Cannot connect to Ollama"

**Error**: `Cannot connect to Ollama at http://localhost:11434`

**Solution**: Start Ollama server
```bash
ollama serve
```

### "Model not found"

**Error**: `Model 'llama2' not found`

**Solution**: Pull the model
```bash
ollama pull llama2
```

### "Token limit exceeded despite chunking"

**Solution**: Reduce chunk size or enable compression
```python
# Option 1: Smaller chunks
manager = ContextWindowManager(chunk_size=800)

# Option 2: Enable compression
manager = ContextWindowManager(
    compression_config=CompressionConfig(
        compression_ratio=0.3,
    ),
)

# Option 3: Use larger model
ollama pull llama2:13b
```

### "Slow responses"

**Solutions**:
```bash
# Use GPU if available
OLLAMA_GPU=1 ollama serve

# Use a faster model
ollama pull mistral  # Faster than llama2

# Reduce context window
manager = ContextWindowManager(
    max_window_tokens=4000,  # Faster processing
)
```

---

## Running the Examples

### Basic Examples

```bash
python examples/ollama_large_documents.py
```

This shows:
- Basic 8k window handling
- Compression for dense content
- Multiple queries
- Chunking strategy comparison
- 100k token document processing
- Cost savings analysis

### Practical Guide

```bash
python examples/ollama_practical_guide.py
```

This shows:
- Interactive Q&A
- Batch processing
- Memory efficiency
- Response handling
- Error handling
- Performance monitoring

---

## Architecture

How the package handles Ollama's 8k limit:

```
User provides: 100k token document
               ↓
        [Tokenizer]
        Counts tokens accurately
               ↓
        [Chunker]
        Splits into semantic pieces
        (e.g., 15 chunks of 1500 tokens each)
               ↓
        [Context Manager]
        Selects chunks that fit 8k window
        (e.g., 5 chunks = 7.5k tokens)
               ↓
        [Format for API]
        Prepares for Ollama
               ↓
        [Call Ollama]
        Sends to local server
               ↓
        [Response]
        Returns answer with metrics
```

---

## Key Advantages Over Manual Handling

| Task | Manual | This Package |
|------|--------|--------------|
| Tokenize document | Write regex | ✓ Automatic |
| Split into chunks | Manual/naive | ✓ Smart splitting |
| Respect context limit | Calculate manually | ✓ Automatic |
| Handle compression | Build yourself | ✓ Built-in |
| Call Ollama API | Use requests | ✓ Integrated |
| Track metrics | Manual logging | ✓ Automatic |

---

## FAQ

**Q: Can I use multiple Ollama models?**
A: Yes, just create separate managers:
```python
manager1 = ContextWindowManager(model_name="llama2")
manager2 = ContextWindowManager(model_name="mistral")
```

**Q: What if my document is >1M tokens?**
A: The package handles it! It chunks intelligently:
```python
# Works for any size document
result = efficient_llm_call(manager, huge_document, query)
```

**Q: Can I use streaming responses?**
A: Not yet, but you can get long responses:
```python
result = call.call("Give a very detailed answer")
print(result['response'])  # Full response, not streamed
```

**Q: How do I know if it's working?**
A: Check metrics:
```python
result = call.call(query)
print(f"Tokens used: {result['tokens_used']}")
print(f"Window utilization: {result['window_utilization'] * 100:.1f}%")
```

**Q: Can I use this with cloud Ollama?**
A: Yes, set the URL:
```python
result = efficient_llm_call(
    manager, doc, query,
    provider="ollama",
    base_url="http://remote-server:11434",
)
```

---

## Next Steps

1. **Install Ollama**: https://ollama.ai
2. **Run examples**: `python examples/ollama_large_documents.py`
3. **Read HOWTO.md**: For more use cases
4. **Check API Reference**: In README.md
5. **Process your documents**: Use the patterns shown

---

## Support

- **Issues**: GitHub Issues
- **Questions**: Check HOWTO.md
- **Examples**: See examples/ directory
- **Docs**: README.md and IMPLEMENTATION_SUMMARY.md

Happy processing! 🚀
