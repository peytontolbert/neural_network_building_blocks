"""Tests for memory management components."""

import torch
import pytest
from nncore.memory import (
    EpisodicMemory,
    WorkingMemoryBuffer,
    HierarchicalMemory,
    SharedSwarmMemory
)

def test_episodic_memory():
    """Test EpisodicMemory functionality."""
    batch_size = 16
    memory_size = 100
    memory_dim = 256
    query_dim = 128
    
    memory = EpisodicMemory(
        memory_size=memory_size,
        memory_dim=memory_dim,
        query_dim=query_dim,
        compression_ratio=0.5
    )
    
    # Test basic retrieval
    query = torch.randn(batch_size, query_dim)
    retrieved, attention, importance = memory(query)
    
    assert retrieved.shape == (batch_size, memory_dim)
    assert attention.shape[1] == memory.attention.num_heads
    assert importance.shape == (batch_size, memory_size)
    
    # Test memory update
    memory.train()
    old_memory = memory.memory.data.clone()
    _ = memory(query, update=True)
    
    assert not torch.equal(memory.memory.data, old_memory)
    assert memory.memory.data.shape == (1, memory_size, memory_dim)
    
    # Test without update
    memory.train()
    old_memory = memory.memory.data.clone()
    _ = memory(query, update=False)
    
    assert torch.equal(memory.memory.data, old_memory)

def test_working_memory_buffer():
    """Test WorkingMemoryBuffer functionality."""
    batch_size = 32
    buffer_size = 50
    memory_dim = 128
    seq_length = 10
    
    buffer = WorkingMemoryBuffer(
        buffer_size=buffer_size,
        memory_dim=memory_dim
    )
    
    # Test writing to buffer
    input_data = torch.randn(batch_size, seq_length, memory_dim)
    priorities = torch.rand(batch_size, seq_length)
    
    buffer.write(input_data, priorities)
    assert not torch.all(buffer.buffer == 0)
    assert not torch.all(buffer.priorities == 0)
    
    # Test reading with query
    query = torch.randn(batch_size, memory_dim)
    retrieved, attention = buffer.read(query)
    
    assert retrieved.shape == (batch_size, memory_dim)
    assert attention.shape[1] == buffer.attention.num_heads
    
    # Test reading top-k
    top_k = 5
    retrieved_topk, attention_topk = buffer.read(query, top_k=top_k)
    
    assert retrieved_topk.shape == (batch_size, memory_dim)
    assert attention_topk.shape[-1] == top_k

def test_hierarchical_memory():
    """Test HierarchicalMemory functionality."""
    batch_size = 16
    num_levels = 3
    level_sizes = [100, 50, 25]
    memory_dim = 256
    query_dim = 128
    
    memory = HierarchicalMemory(
        num_levels=num_levels,
        level_sizes=level_sizes,
        memory_dim=memory_dim,
        query_dim=query_dim
    )
    
    # Test basic retrieval
    query = torch.randn(batch_size, query_dim)
    combined, level_retrievals, level_weights = memory(query)
    
    assert combined.shape == (batch_size, memory_dim)
    assert len(level_retrievals) == num_levels
    assert level_weights.shape == (batch_size, num_levels)
    assert torch.allclose(level_weights.sum(dim=1), torch.ones(batch_size))
    
    for retrieved in level_retrievals:
        assert retrieved.shape == (batch_size, memory_dim)
    
    # Test memory update
    memory.train()
    old_memories = [level.memory.data.clone() for level in memory.levels]
    _ = memory(query, update=True)
    
    for i, level in enumerate(memory.levels):
        assert not torch.equal(level.memory.data, old_memories[i])
    
    # Test without update
    memory.train()
    old_memories = [level.memory.data.clone() for level in memory.levels]
    _ = memory(query, update=False)
    
    for i, level in enumerate(memory.levels):
        assert torch.equal(level.memory.data, old_memories[i])

def test_shared_swarm_memory():
    """Test SharedSwarmMemory functionality."""
    batch_size = 8
    num_agents = 4
    memory_size = 50
    memory_dim = 128
    query_dim = 64
    
    memory = SharedSwarmMemory(
        num_agents=num_agents,
        memory_size=memory_size,
        memory_dim=memory_dim,
        query_dim=query_dim
    )
    
    # Test basic retrieval
    queries = [torch.randn(batch_size, query_dim) for _ in range(num_agents)]
    retrievals, consensus, attention_weights = memory(queries)
    
    assert len(retrievals) == num_agents
    assert consensus.shape == (batch_size, memory_dim)
    assert len(attention_weights) == num_agents
    
    for retrieved in retrievals:
        assert retrieved.shape == (batch_size, memory_dim)
    
    # Test memory update
    memory.train()
    old_memory = memory.shared_memory.data.clone()
    _ = memory(queries, update=True)
    
    assert not torch.equal(memory.shared_memory.data, old_memory)
    
    # Test without update
    memory.train()
    old_memory = memory.shared_memory.data.clone()
    _ = memory(queries, update=False)
    
    assert torch.equal(memory.shared_memory.data, old_memory)
    
    # Test consensus mechanism
    retrievals1, consensus1, _ = memory(queries)
    retrievals2, consensus2, _ = memory([q + 0.1 for q in queries])
    
    assert not torch.allclose(consensus1, consensus2)

if __name__ == "__main__":
    pytest.main([__file__]) 