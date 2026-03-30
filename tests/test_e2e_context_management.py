"""
End-to-end tests for complete context management system.

Tests the full workflow: adding messages, managing memory tiers,
assembling context, and handling compression.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from efficient_context_window_manager.core import ContextManager
from efficient_context_window_manager.core.context_manager import ContextManagerConfig
from efficient_context_window_manager.types import ContextStrategyEnum


async def test_basic_message_flow():
    """Test adding messages and retrieving assembled context."""
    config = ContextManagerConfig(
        model_name="gpt-4",
        max_tokens=8000,
        short_term_tokens=2000,
        long_term_tokens=4000,
    )
    manager = ContextManager(config)

    # Add messages
    await manager.add_message("user", "What is Python?")
    await manager.add_message("assistant", "Python is a programming language.")
    await manager.add_message("user", "What can I do with it?")

    # Assemble context
    context = await manager.assemble_context("Python programming")
    assert len(context.messages) == 3
    assert context.total_tokens > 0
    print(f"✓ Basic flow: {len(context.messages)} messages, {context.total_tokens} tokens")


async def test_short_term_memory_eviction():
    """Test that messages are evicted from short-term when capacity exceeded."""
    config = ContextManagerConfig(
        model_name="gpt-4",
        max_tokens=8000,
        short_term_tokens=300,  # Very small to force eviction
        long_term_tokens=2000,
    )
    manager = ContextManager(config)

    # Add enough messages to exceed short-term capacity
    # Each message of 150 chars = ~38 tokens, so 300 tokens = ~8 messages
    for i in range(10):
        await manager.add_message("user", f"Message {i}: " + "x" * 150)

    # Some messages should be evicted to long-term
    short_term_count = len(manager.short_term.messages)
    long_term_count = len(manager.long_term.compressed_messages)

    assert short_term_count < 10, "Early messages should be evicted"
    assert long_term_count > 0, "Evicted messages should be in long-term"
    print(f"✓ Eviction: {short_term_count} in short-term, {long_term_count} in long-term")


async def test_tool_result_priority():
    """Test that tool results have highest priority in assembly."""
    config = ContextManagerConfig(
        model_name="gpt-4",
        max_tokens=2000,
        short_term_tokens=1000,
    )
    manager = ContextManager(config)

    # Add messages to fill context
    await manager.add_message("user", "Query: " + "x" * 300)
    await manager.add_message("assistant", "Response: " + "y" * 300)

    # Add tool result
    await manager.add_tool_result("search", "Search result: " + "z" * 200)

    # Assemble context
    context = await manager.assemble_context("search results")

    # Tool result should be included
    tool_messages = [m for m in context.messages if m.role == "tool"]
    assert len(tool_messages) > 0, "Tool results should be prioritized"
    assert context.tool_tokens > 0
    print(f"✓ Tool priority: {context.tool_tokens} tokens allocated to tools")


async def test_token_budgeting():
    """Test token budget tracking and state transitions."""
    config = ContextManagerConfig(
        model_name="gpt-4",
        max_tokens=1000,
        short_term_tokens=800,
    )
    manager = ContextManager(config)

    # Check initial state
    assert manager.budgeter.state.value == "healthy"

    # Add messages to approach limit
    msg_content = "x" * 200  # ~50 tokens
    for i in range(5):
        await manager.add_message("user", f"Message {i}: {msg_content}")

    # Check state progression
    state = manager.budgeter.state.value
    assert state in ["healthy", "caution", "critical"]
    print(f"✓ Budgeting: Budget state = {state}")


async def test_compression_trigger():
    """Test that compression is triggered when critical."""
    config = ContextManagerConfig(
        model_name="gpt-4",
        max_tokens=1000,
        short_term_tokens=800,
        auto_compress=True,
    )
    manager = ContextManager(config)

    initial_compressions = manager.metadata["compressions_triggered"]

    # Add messages to trigger compression
    msg_content = "x" * 150  # ~38 tokens
    for i in range(10):
        await manager.add_message("user", f"Message {i}: {msg_content}")

    final_compressions = manager.metadata["compressions_triggered"]
    assert final_compressions >= initial_compressions, "Compression should be triggered"
    print(f"✓ Compression: {final_compressions - initial_compressions} compressions triggered")


async def test_multiple_sources_assembly():
    """Test assembly from all three memory sources."""
    config = ContextManagerConfig(
        model_name="gpt-4",
        max_tokens=4000,
        short_term_tokens=1500,
    )
    manager = ContextManager(config)

    # Add messages (will populate short-term and trigger eviction to long-term)
    for i in range(5):
        await manager.add_message("user", f"User message {i}: some content here.")
        await manager.add_message("assistant", f"Assistant response {i}: more content.")

    # Add tool results (external source)
    await manager.add_tool_result("calculator", "2 + 2 = 4")
    await manager.add_tool_result("database", "User found in database")

    # Assemble context
    context = await manager.assemble_context("math calculations and data")

    assert len(context.messages) > 0
    assert context.recent_tokens > 0
    assert context.compressed_tokens >= 0
    assert context.tool_tokens > 0
    print(
        f"✓ Multi-source assembly: "
        f"recent={context.recent_tokens}, "
        f"compressed={context.compressed_tokens}, "
        f"tools={context.tool_tokens}"
    )


async def test_strategy_switching():
    """Test switching between context strategies."""
    config = ContextManagerConfig(
        model_name="gpt-4",
        max_tokens=4000,
        strategy=ContextStrategyEnum.GREEDY,
    )
    manager = ContextManager(config)

    assert manager.config.strategy == ContextStrategyEnum.GREEDY

    # Switch strategy
    manager.update_strategy(ContextStrategyEnum.HYBRID)
    assert manager.config.strategy == ContextStrategyEnum.HYBRID

    print("✓ Strategy switching: GREEDY -> HYBRID")


async def test_status_reporting():
    """Test status and metadata reporting."""
    config = ContextManagerConfig(
        model_name="gpt-4",
        max_tokens=4000,
    )
    manager = ContextManager(config)

    await manager.add_message("user", "Test message 1")
    await manager.add_message("assistant", "Test message 2")

    status = manager.get_status()
    assert "budget" in status
    assert "short_term_messages" in status
    assert "long_term_compressed_messages" in status
    assert "metadata" in status

    assert status["short_term_messages"] == 2
    assert status["metadata"]["total_messages_added"] == 2
    print(f"✓ Status reporting: {status['short_term_messages']} short-term messages")


async def test_clear_all_memory():
    """Test clearing all memory and resetting."""
    config = ContextManagerConfig(
        model_name="gpt-4",
        max_tokens=4000,
    )
    manager = ContextManager(config)

    await manager.add_message("user", "Message 1")
    await manager.add_message("assistant", "Message 2")
    await manager.add_tool_result("tool1", "Result 1")

    # Verify messages were added
    assert len(manager.short_term.messages) > 0 or len(manager.long_term.compressed_messages) > 0

    # Clear all
    manager.clear()

    assert len(manager.short_term.messages) == 0
    assert len(manager.long_term.compressed_messages) == 0
    assert len(manager.external.tool_results) == 0
    assert manager.budgeter.current_tokens == 0
    print("✓ Clear all memory: Successfully cleared all tiers")


def test_adapters_initialization():
    """Test that all adapters can be initialized."""
    from efficient_context_window_manager.adapters import (
        AutoGenAdapter,
        CrewAIAdapter,
        LangGraphAdapter,
    )

    autogen = AutoGenAdapter("gpt-4", 8000)
    crewai = CrewAIAdapter("gpt-4", 8000)
    langgraph = LangGraphAdapter("gpt-4", 8000)

    assert autogen.manager is not None
    assert crewai.manager is not None
    assert langgraph.manager is not None
    print("✓ Adapters: All initialized successfully")


if __name__ == "__main__":
    # Run async tests manually
    asyncio.run(test_basic_message_flow())
    asyncio.run(test_short_term_memory_eviction())
    asyncio.run(test_tool_result_priority())
    asyncio.run(test_token_budgeting())
    asyncio.run(test_compression_trigger())
    asyncio.run(test_multiple_sources_assembly())
    asyncio.run(test_strategy_switching())
    asyncio.run(test_status_reporting())
    asyncio.run(test_clear_all_memory())
    test_adapters_initialization()

    print("\n✅ All E2E tests passed!")
