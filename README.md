# Efficient Context Window Manager

Intelligent context window management for large language models. Break large documents and conversations into model-native chunks, manage memory across short-term and long-term storage, and assemble optimal context for each API call.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## What It Does

Working with LLMs often means dealing with context window limits:
- Claude: 100k-200k tokens
- GPT-4: 128k tokens
- GPT-3.5: 16k tokens
- Llama: 4k-70k tokens
- Local models: varies

This package automatically:
1. **Chunks large documents** into model-native sizes
2. **Manages tiered memory** (recent, compressed, external)
3. **Intelligently assembles context** respecting token limits
4. **Compresses older messages** when approaching limits
5. **Integrates with RAG and tool results** with priority handling

## Key Capabilities

| Capability | Details |
|-----------|---------|
| **3 Memory Tiers** | Short-term (recent), Long-term (compressed), External (RAG/tools) |
| **Automatic Tiering** | FIFO eviction from short-term to long-term when full |
| **Semantic Compression** | Multiple compression methods for reducing token usage |
| **Token Budgeting** | Real-time tracking of token usage with threshold alerts |
| **Priority Assembly** | Tools > recent > compressed > external in context assembly |
| **4 Chunking Strategies** | Fixed-size, Recursive, Semantic, Sliding Window |
| **Framework Agnostic** | Works with any agent framework (AutoGen, CrewAI, LangGraph, custom) |
| **Model Agnostic** | Supports Claude, GPT, Ollama, and any LLM with token limits |
| **Production Ready** | Token caching, error handling, comprehensive status reporting |

## Installation

```bash
pip install efficient-context-window-manager
```

With optional dependencies:
```bash
# For Anthropic Claude
pip install efficient-context-window-manager[anthropic]

# For OpenAI
pip install efficient-context-window-manager[openai]

# For HuggingFace tokenizers
pip install efficient-context-window-manager[huggingface]
```

## Quick Start

### Document Processing

```python
from efficient_context_window_manager import ContextWindowManager, efficient_llm_call

manager = ContextWindowManager(model_name="gpt-4")
result = efficient_llm_call(
    context_manager=manager,
    documents=["Your large document..."],
    query="What are the key points?"
)
print(result['response'])
```

### Message-by-Message Context Management

```python
from efficient_context_window_manager import ContextManager
from efficient_context_window_manager.core.context_manager import ContextManagerConfig

config = ContextManagerConfig(model_name="gpt-4", max_tokens=8000)
manager = ContextManager(config)

# Add messages (auto-handled by memory tiers)
await manager.add_message("user", "First question")
await manager.add_message("assistant", "First answer")

# Add tool results (highest priority)
await manager.add_tool_result("search", "Search results...")

# Get optimized context for API call
context = await manager.assemble_context("search results")
for msg in context.messages:
    print(f"{msg.role}: {msg.content}")
```

### Framework Integration

```python
from efficient_context_window_manager.adapters import AutoGenAdapter

adapter = AutoGenAdapter(model_name="gpt-4", max_tokens=8000)
# adapter can now be used to manage context for AutoGen agents
```

## Documentation

- **[HOWTO Guide](HOWTO.md)** - Comprehensive guides, architecture, and how-to instructions
- **[API Reference](HOWTO.md#api-reference)** - Complete API documentation

## Architecture Overview

The package implements a tiered memory system for intelligent context management:

```
┌─────────────────────────────────────────────────────────────┐
│  Incoming Messages / Tool Results / RAG                      │
├─────────────────────────────────────────────────────────────┤
│                     ContextManager                           │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐ ┌─────────────┐ │
│  │  ShortTermMemory │  │  LongTermMemory  │ │  External   │ │
│  │  (Recent, FIFO)  │  │ (Compressed)     │ │ (RAG/Tools) │ │
│  └──────────────────┘  └──────────────────┘ └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│            ContextAssembler + TokenBudgeter                  │
├─────────────────────────────────────────────────────────────┤
│         Optimized Context Window (respecting limits)         │
├─────────────────────────────────────────────────────────────┤
│              LLM API Call (Claude/GPT/Ollama)                │
└─────────────────────────────────────────────────────────────┘
```

## Contributing

Contributions are welcome! Please ensure all tests pass and follow the existing code style.

## License

MIT License - See LICENSE file for details.

## Acknowledgments

Implements concepts from:
- Recursive Language Models (RLM) paper
- LLMLingua compression techniques
- SWAT sliding window approach
- RAG best practices
