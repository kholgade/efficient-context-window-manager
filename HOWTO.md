# How To Guide

Complete technical guide and how-to instructions for Efficient Context Window Manager.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Architecture](#architecture)
3. [Memory Management](#memory-management)
4. [Chunking Strategies](#chunking-strategies)
5. [Compression Methods](#compression-methods)
6. [Integration Guides](#integration-guides)
7. [Framework Adapters](#framework-adapters)
8. [Configuration Reference](#configuration-reference)
9. [Advanced Usage](#advanced-usage)
10. [API Reference](#api-reference)

---

## Getting Started

### Installation from Source

```bash
git clone https://github.com/kholgade/efficient-context-window-manager.git
cd efficient-context-window-manager
pip install -e .
```

### Environment Setup

```bash
# Set API keys for services you use
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# For local Ollama
ollama serve  # Runs on http://localhost:11434 by default
```

---

## Architecture

### High-Level Design

The package implements a tiered memory architecture for intelligent context management:

#### Three Memory Tiers

1. **ShortTermMemory**: Recent, uncompressed messages
   - FIFO eviction when capacity exceeded
   - Default: 4000 tokens
   - Used for: Recent conversation history, immediate context

2. **LongTermMemory**: Compressed, searchable history
   - Messages compressed using semantic or masking methods
   - Supports keyword and semantic search
   - Default: 10000 tokens
   - Used for: Older context, historical information

3. **ExternalRetrieval**: Tool results, RAG, real-time data
   - Highest priority in assembly
   - Separate management for tool outputs
   - Optional RAG engine integration
   - Real-time data sources
   - Used for: Function call results, external knowledge

#### Context Assembly Priority

```
Messages are assembled in this priority order:

1. TOOL RESULTS (highest priority)
   - Always preserved if space available
   - Allocation: tool_reserve_tokens (default 500)

2. RECENT MESSAGES
   - From ShortTermMemory
   - Allocation: recent_tokens_pct × available_budget (default 40%)

3. COMPRESSED HISTORY
   - From LongTermMemory
   - Retrieved via keyword/semantic search
   - Allocation: compressed_tokens_pct × available_budget (default 40%)

4. EXTERNAL/RAG
   - From RAG engine or real-time data
   - Allocation: external_tokens_pct × available_budget (default 20%)
```

#### Token Budgeting

The TokenBudgeter tracks token usage with state transitions:

```python
BudgetState:
  - HEALTHY: < 70% utilization      → No action
  - CAUTION: 70-85% utilization     → Monitor closely
  - CRITICAL: > 85% utilization     → Compression triggered
  - EXHAUSTED: >= 100% utilization  → Error state
```

When CRITICAL state is reached and auto_compress is enabled:
1. Compression is triggered
2. Oldest short-term messages are promoted to long-term
3. Messages are compressed using configured method
4. Budget is recalculated

---

## Memory Management

### Working with ShortTermMemory

```python
from efficient_context_window_manager.memory import ShortTermMemory

short_term = ShortTermMemory(max_tokens=4000)

# Add message (may evict if over capacity)
evicted = short_term.add(message)

if evicted:
    print(f"Message promoted to long-term: {evicted.content[:50]}...")

# Get messages
all_msgs = short_term.get_all()
recent_msgs = short_term.get_recent(max_messages=5)

# Clear when done
short_term.clear()
```

### Working with LongTermMemory

```python
from efficient_context_window_manager.memory import LongTermMemory

long_term = LongTermMemory(max_compressed_tokens=10000)

# Store messages (automatically compressed)
long_term.store([msg1, msg2, msg3])

# Retrieve relevant messages
query = "What was discussed about authentication?"
results = await long_term.retrieve(
    query=query,
    max_tokens=2000,
    strategy="hybrid"  # or "keyword" or "semantic"
)

# Clear
long_term.clear()
```

**Retrieval Strategies**:
- `keyword`: Simple word matching (fast)
- `semantic`: Vector similarity (requires vector_store)
- `hybrid`: Combines both

### Working with ExternalRetrieval

```python
from efficient_context_window_manager.memory import ExternalRetrieval

external = ExternalRetrieval()

# Add tool results (highest priority)
external.add_tool_result("search_web", "Found: ...")
external.add_tool_result("database_query", "Result: ...")

# Optional: Set RAG engine
external.add_rag_engine(rag_engine)

# Optional: Add real-time data
external.add_real_time_data("weather", {"temp": 72, "humidity": 65})

# Retrieve with priorities
results = await external.retrieve(
    query="weather data",
    max_tokens=2000,
    sources=["tools", "rag", "web"]  # Priority order
)
```

---

## Chunking Strategies

### Recursive Chunking (Recommended)

Splits hierarchically by semantic delimiters, preserving document structure.

```python
from efficient_context_window_manager import ContextWindowManager
from efficient_context_window_manager.types import ChunkingStrategy

manager = ContextWindowManager(
    chunking_strategy=ChunkingStrategy.RECURSIVE,
    chunk_size=1000,
    overlap=100,
)

chunks = manager.process_document(document)
```

**How it works**:
1. Split by paragraphs (separated by blank lines)
2. If too large, split by sentences
3. If still too large, split by words
4. If still too large, split by characters

**Best for**: Mixed content (blog posts, reports, code with comments)

### Sliding Window (SWAT)

Creates overlapping windows with controlled stride.

```python
manager = ContextWindowManager(
    chunking_strategy=ChunkingStrategy.SLIDING_WINDOW,
    chunk_size=1000,
    overlap=200,  # 20% overlap
)

chunks = manager.process_document(document)
```

**How it works**:
```
Document: [=========CHUNK1=========] [=========CHUNK2=========]
              ↑                ↑         ↑                ↑
              0              1000      800              1800
                    ← overlap →
```

**Best for**: Sequential documents, logs, time-series data

### Fixed Size

Simple deterministic chunking by character count.

```python
manager = ContextWindowManager(
    chunking_strategy=ChunkingStrategy.FIXED_SIZE,
    chunk_size=1000,
)
```

**Best for**: When predictability matters more than quality

### Semantic

Uses embeddings to identify natural boundaries.

```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')

manager = ContextWindowManager(
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    embedding_fn=embedder.encode,
)
```

**Best for**: High-quality chunking when speed is less critical

---

## Compression Methods

### Observation Masking

Fast masking-based compression that marks and removes non-essential tokens.

```python
from efficient_context_window_manager.compression.masking import ObservationMaskingCompressor
from efficient_context_window_manager.types import CompressionConfig

config = CompressionConfig(
    method=CompressionMethod.OBSERVATION_MASKING,
    compression_ratio=0.3,  # Target 30% reduction
)

compressor = ObservationMaskingCompressor(config)
compressed = compressor.compress(chunk)
```

**How it works**:
1. Identify "observation" tokens (detailed descriptions, examples)
2. Mark them for removal
3. Reconstruct with observation masks
4. Achieves 20-40% compression

**Speed**: Very fast (~50ms for typical chunk)

### Semantic Summarization

Uses LLM to semantically summarize content.

```python
from efficient_context_window_manager.compression.summarization import SemanticSummarizationCompressor

compressor = SemanticSummarizationCompressor(config)
compressed = compressor.compress(chunk)
```

**How it works**:
1. Prompt LLM to summarize key points
2. Replace original with summary
3. Achieves 40-60% compression
4. Preserves semantic meaning

**Speed**: Slower (~1-2s per chunk, requires API call)

---

## Integration Guides

### With Anthropic Claude

```python
from efficient_context_window_manager import ContextWindowManager
from efficient_context_window_manager.integrations.anthropic import AnthropicAdapter
from efficient_context_window_manager.types import ChunkingStrategy

# Create context manager
context_manager = ContextWindowManager(
    model_name="claude-3-opus-20250219",
    max_window_tokens=100000,
    chunking_strategy=ChunkingStrategy.RECURSIVE,
)

# Create adapter
adapter = AnthropicAdapter(
    context_manager=context_manager,
    model_name="claude-3-opus-20250219",
)

# Process document
result = adapter.process_with_context(
    document=large_document,
    query="What are the key findings?",
)

print(result['response'])
print(f"Tokens used: {result['tokens_used']}")
```

### With OpenAI GPT

```python
from efficient_context_window_manager.integrations.openai import OpenAIAdapter

adapter = OpenAIAdapter(
    context_manager=context_manager,
    model_name="gpt-4",
)

result = adapter.process_with_context(large_document, query)
```

### With Ollama (Local)

```python
from efficient_context_window_manager.integrations.ollama import OllamaAdapter

adapter = OllamaAdapter(
    context_manager=context_manager,
    model_name="llama2",
    base_url="http://localhost:11434",
)

result = adapter.process_with_context(large_document, query)
```

---

## Framework Adapters

### AutoGen Integration

```python
from efficient_context_window_manager.adapters import AutoGenAdapter
from autogen import AssistantAgent

# Create adapter
adapter = AutoGenAdapter(model_name="gpt-4", max_tokens=8000)

# Create agent
agent = AssistantAgent(name="assistant", llm_config={"model": "gpt-4"})

# Attach adapter to agent
adapter.attach_to_agent(agent)

# Now messages are automatically managed
await adapter.add_message_from_autogen("user", "What can you help with?")
context = await adapter.get_assembled_context_for_api_call(query="help")
```

### CrewAI Integration

```python
from efficient_context_window_manager.adapters import CrewAIAdapter

# Create adapter
adapter = CrewAIAdapter(model_name="gpt-4", max_tokens=8000)

# Track tasks
await adapter.start_task("research")
await adapter.add_agent_response("researcher", "Found relevant papers...")
await adapter.add_tool_output("arxiv_search", "3 papers found", "researcher")

# Get context for next agent
context = await adapter.get_context_for_agent(
    agent_name="writer",
    task_name="research"
)

await adapter.end_task("research", result="Research complete")
```

### LangGraph Integration

```python
from efficient_context_window_manager.adapters import LangGraphAdapter
from langgraph.graph import StateGraph

# Create adapter
adapter = LangGraphAdapter(model_name="gpt-4", max_tokens=8000)

# Use in graph nodes
async def agent_node(state):
    # Add state messages to manager
    await adapter.add_messages_from_state(state.get("messages", []))

    # Get optimized context
    context = await adapter.get_context_for_state(state, query="process")

    # Use context in API call
    # ... make API call with context ...

    return {"messages": [...]}

# Build graph
graph = StateGraph(...)
graph.add_node("agent", agent_node)
```

---

## Configuration Reference

### ContextManagerConfig

```python
from efficient_context_window_manager.core.context_manager import ContextManagerConfig

config = ContextManagerConfig(
    # Required
    model_name="gpt-4",                    # Target model name
    max_tokens=8000,                       # Model's context window

    # Memory limits
    short_term_tokens=4000,                # Recent messages buffer
    long_term_tokens=10000,                # Compressed history buffer

    # Strategy
    strategy=ContextStrategyEnum.HYBRID,   # Assembly strategy

    # Compression
    auto_compress=True,                    # Auto-trigger compression
    compression_threshold=0.85,            # Trigger at 85% utilization
    compressor=None,                       # Custom compressor (None = default)

    # Assembly
    assembly_config=None,                  # Custom assembly config

    # Utilization threshold (0.9 = 90% of model's max)
    utilization_threshold=0.9,
)

manager = ContextManager(config)
```

### AssemblyConfig

```python
from efficient_context_window_manager.assembly.assembler import AssemblyConfig

config = AssemblyConfig(
    strategy=ContextStrategyEnum.HYBRID,
    tool_reserve_tokens=500,               # Always reserve for tools
    recent_tokens_pct=0.4,                 # 40% for recent
    compressed_tokens_pct=0.4,             # 40% for compressed
    external_tokens_pct=0.2,               # 20% for external
    preserve_recent_count=5,               # Always keep 5 most recent
)
```

### CompressionConfig

```python
from efficient_context_window_manager.types import CompressionConfig, CompressionMethod

config = CompressionConfig(
    method=CompressionMethod.OBSERVATION_MASKING,
    compression_ratio=0.3,                 # Target 30% reduction
    preserve_recent_tokens=1000,           # Keep 1000 tokens recent
    preserve_important=True,               # Keep marked important chunks
    use_cache=True,                        # Cache compression results
)
```

---

## Advanced Usage

### Custom Tokenizer

```python
from efficient_context_window_manager.tokenizers.base import BaseTokenizer

class CustomTokenizer(BaseTokenizer):
    def count_tokens(self, text: str) -> int:
        # Your tokenization logic
        return len(text.split())

manager = ContextWindowManager(
    tokenizer=CustomTokenizer(),
    model_name="custom-model"
)
```

### Custom Compressor

```python
from efficient_context_window_manager.compression.base import BaseCompressor
from efficient_context_window_manager.types import Chunk, CompressionConfig

class CustomCompressor(BaseCompressor):
    def compress(self, chunk: Chunk) -> Chunk:
        # Your compression logic
        return Chunk(
            content=compressed_content,
            token_count=reduced_tokens,
            ...
        )

config = ContextManagerConfig(
    model_name="gpt-4",
    max_tokens=8000,
    compressor=CustomCompressor(CompressionConfig())
)
```

### Metrics and Monitoring

```python
manager = ContextManager(config)

# Monitor status
status = manager.get_status()
print(f"Budget state: {status['budget']['state']}")
print(f"Utilization: {status['budget']['utilization_pct']:.1f}%")
print(f"Short-term messages: {status['short_term_messages']}")
print(f"Total compressions: {status['metadata']['compressions_triggered']}")

# Get budget details
budget = manager.budgeter.get_usage_report()
print(f"Used: {budget['current_tokens']}/{budget['max_tokens']}")
print(f"By source: {budget['breakdown']}")
```

---

## API Reference

### ContextManager

Main orchestrator for context management.

```python
class ContextManager:
    async def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Message]

    async def add_tool_result(
        self,
        tool_name: str,
        result: str,
    ) -> None

    async def assemble_context(
        self,
        query: str,
        max_tokens: Optional[int] = None,
    ) -> AssembledContext

    def get_status(self) -> Dict[str, Any]

    def clear(self) -> None

    def update_strategy(self, strategy: ContextStrategyEnum) -> None
```

### ShortTermMemory

Recent message buffer with FIFO eviction.

```python
class ShortTermMemory:
    def add(self, message: Message) -> Optional[Message]

    def get_all(self) -> List[Message]

    def get_recent(self, max_messages: int) -> List[Message]

    def clear(self) -> None
```

### LongTermMemory

Compressed message storage with semantic retrieval.

```python
class LongTermMemory:
    def store(self, messages: List[Message]) -> None

    async def retrieve(
        self,
        query: str,
        max_tokens: int,
        strategy: str = "keyword",
    ) -> List[Message]

    def clear(self) -> None
```

### ExternalRetrieval

RAG, tool results, and real-time data management.

```python
class ExternalRetrieval:
    async def retrieve(
        self,
        query: str,
        max_tokens: int,
        sources: Optional[List[str]] = None,
    ) -> List[Message]

    def add_tool_result(self, tool_name: str, result: str) -> None

    def add_rag_engine(self, engine: Any) -> None

    def add_real_time_data(self, key: str, data: Any) -> None

    def clear_tool_results(self) -> None
```

### TokenBudgeter

Token usage tracking and state management.

```python
class TokenBudgeter:
    def can_fit(self, tokens: int, safety_margin: float = 0.05) -> bool

    def allocate(self, tokens: int, source: str = "unknown") -> bool

    def deallocate(self, tokens: int, source: str = "unknown") -> bool

    def should_compress(self) -> bool

    def compression_target(self) -> int

    def get_usage_report(self) -> Dict[str, any]

    @property
    def state(self) -> BudgetState

    @property
    def utilization(self) -> float
```

### ContextAssembler

Intelligent assembly of context from memory sources.

```python
class ContextAssembler:
    async def assemble(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        external: ExternalRetrieval,
        query: str,
        max_tokens: int,
        utilization_threshold: float = 0.9,
    ) -> AssembledContext

    def update_config(self, **kwargs) -> None
```

---

## Troubleshooting

### Messages not being evicted from short-term

**Issue**: Messages stay in short-term memory even when limit exceeded

**Solution**: Verify that message token_count is being set correctly:
```python
msg = Message(role="user", content="...", token_count=len(content)//4)
```

### Token count inaccuracy

**Issue**: Token count doesn't match API's actual count

**Solution**: Use the model-specific tokenizer:
```python
from efficient_context_window_manager.tokenizers import AutoTokenizer

tokenizer = AutoTokenizer("gpt-4")
tokens = tokenizer.count_tokens(text)
```

### Compression not triggering

**Issue**: `auto_compress=True` but compression never happens

**Solution**: Check that budget state reaches CRITICAL:
```python
status = manager.get_status()
print(f"Budget state: {status['budget']['state']}")
```

Compression triggers only when state is CRITICAL (>85%) or EXHAUSTED (>=100%).

---

## Performance Tips

1. **Use semantic chunking for quality**: Slower but better chunk boundaries
2. **Use keyword search for speed**: Simple word matching is fast
3. **Cache compression results**: `use_cache=True` (default)
4. **Batch operations**: Add multiple messages before assembling context
5. **Monitor compression ratio**: Check if compression is helping
6. **Adjust thresholds**: Tune tool_reserve_tokens and percentages based on use case
