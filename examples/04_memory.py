"""Example demonstrating the usage of memory components from the nncore package."""

import torch
import torch.nn as nn
from src.nncore.memory import (
    EpisodicMemory,
    WorkingMemoryBuffer,
    HierarchicalMemory,
    SharedSwarmMemory
)
from src.nncore.utils import DeviceManager

# Set random seed for reproducibility
torch.manual_seed(42)

def demonstrate_episodic_memory():
    """Demonstrate EpisodicMemory functionality."""
    print("\n=== EpisodicMemory Demo ===")
    
    # Create an EpisodicMemory
    batch_size = 16
    memory_size = 100
    memory_dim = 256
    query_dim = 128
    
    memory = EpisodicMemory(
        memory_size=memory_size,
        memory_dim=memory_dim,
        query_dim=query_dim,
        num_heads=8,
        compression_ratio=0.5
    )
    
    # Generate sample query
    query = torch.randn(batch_size, query_dim)
    
    # Retrieve from memory
    retrieved, attention_weights, importance = memory(query)
    
    print(f"Query shape: {query.shape}")
    print(f"Retrieved memory shape: {retrieved.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Memory importance scores shape: {importance.shape}")
    
    # Test memory update
    print("\nTesting memory update:")
    memory.train()
    old_memory = memory.memory.data.clone()
    retrieved, _, _ = memory(query, update=True)
    print(f"Memory updated: {not torch.equal(memory.memory.data, old_memory)}")

def demonstrate_working_memory_buffer():
    """Demonstrate WorkingMemoryBuffer functionality."""
    print("\n=== WorkingMemoryBuffer Demo ===")
    
    # Create a WorkingMemoryBuffer
    batch_size = 32
    buffer_size = 50
    memory_dim = 128
    seq_length = 10
    
    buffer = WorkingMemoryBuffer(
        buffer_size=buffer_size,
        memory_dim=memory_dim
    )
    
    # Generate sample data
    input_data = torch.randn(batch_size, seq_length, memory_dim)
    priorities = torch.rand(batch_size, seq_length)
    
    # Write to buffer
    buffer.write(input_data, priorities)
    print(f"Input data shape: {input_data.shape}")
    print(f"Buffer state shape: {buffer.buffer.shape}")
    
    # Read from buffer
    query = torch.randn(batch_size, memory_dim)
    retrieved, attention = buffer.read(query)
    print(f"\nQuery shape: {query.shape}")
    print(f"Retrieved data shape: {retrieved.shape}")
    print(f"Attention weights shape: {attention.shape}")
    
    # Test top-k retrieval
    print("\nTesting top-k retrieval:")
    top_k = 5
    retrieved_topk, attention_topk = buffer.read(query, top_k=top_k)
    print(f"Top-{top_k} retrieved shape: {retrieved_topk.shape}")
    print(f"Top-{top_k} attention shape: {attention_topk.shape}")

def demonstrate_hierarchical_memory():
    """Demonstrate HierarchicalMemory functionality."""
    print("\n=== HierarchicalMemory Demo ===")
    
    # Create a HierarchicalMemory
    batch_size = 16
    num_levels = 3
    level_sizes = [100, 50, 25]
    memory_dim = 256
    query_dim = 128
    
    memory = HierarchicalMemory(
        num_levels=num_levels,
        level_sizes=level_sizes,
        memory_dim=memory_dim,
        query_dim=query_dim,
        compression_ratios=[0.5, 0.3, 0.2]
    )
    
    # Generate sample query
    query = torch.randn(batch_size, query_dim)
    
    # Access memory
    combined, level_retrievals, level_weights = memory(query)
    
    print(f"Query shape: {query.shape}")
    print(f"Combined output shape: {combined.shape}")
    print(f"Level weights shape: {level_weights.shape}")
    
    print("\nLevel retrievals shapes:")
    for i, retrieval in enumerate(level_retrievals):
        print(f"Level {i}: {retrieval.shape}")
    
    # Test memory update
    print("\nTesting memory update:")
    memory.train()
    old_memories = [level.memory.data.clone() for level in memory.levels]
    _ = memory(query, update=True)
    
    for i, (old, new) in enumerate(zip(old_memories, [level.memory.data for level in memory.levels])):
        print(f"Level {i} updated: {not torch.equal(old, new)}")

def demonstrate_shared_swarm_memory():
    """Demonstrate SharedSwarmMemory functionality."""
    print("\n=== SharedSwarmMemory Demo ===")
    
    # Create a SharedSwarmMemory
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
    
    # Generate sample queries for each agent
    queries = [torch.randn(batch_size, query_dim) for _ in range(num_agents)]
    
    # Access shared memory
    retrievals, consensus, attention_weights = memory(queries)
    
    print(f"Number of agents: {num_agents}")
    print(f"Consensus features shape: {consensus.shape}")
    
    print("\nPer-agent retrievals shapes:")
    for i, retrieval in enumerate(retrievals):
        print(f"Agent {i}: {retrieval.shape}")
    
    print("\nPer-agent attention weights shapes:")
    for i, weights in enumerate(attention_weights):
        print(f"Agent {i}: {weights.shape}")
    
    # Test memory update
    print("\nTesting memory update:")
    memory.train()
    old_memory = memory.shared_memory.data.clone()
    _ = memory(queries, update=True)
    print(f"Shared memory updated: {not torch.equal(memory.shared_memory.data, old_memory)}")

class MemoryAugmentedNetwork(nn.Module):
    """Example network combining different memory components."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        memory_size: int = 100,
        num_agents: int = 4
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Memory components
        self.episodic = EpisodicMemory(
            memory_size=memory_size,
            memory_dim=hidden_dim,
            query_dim=hidden_dim
        )
        
        self.working = WorkingMemoryBuffer(
            buffer_size=memory_size // 2,
            memory_dim=hidden_dim
        )
        
        self.hierarchical = HierarchicalMemory(
            num_levels=3,
            level_sizes=[memory_size, memory_size // 2, memory_size // 4],
            memory_dim=hidden_dim,
            query_dim=hidden_dim
        )
        
        self.swarm = SharedSwarmMemory(
            num_agents=num_agents,
            memory_size=memory_size // num_agents,
            memory_dim=hidden_dim,
            query_dim=hidden_dim
        )
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project input
        hidden = self.input_proj(x)
        
        # Access episodic memory
        episodic_out, _, _ = self.episodic(hidden)
        
        # Update and access working memory
        self.working.write(hidden.unsqueeze(1))
        working_out, _ = self.working.read(hidden)
        
        # Access hierarchical memory
        hierarchical_out, _, _ = self.hierarchical(hidden)
        
        # Access shared swarm memory
        swarm_queries = [hidden] * self.swarm.num_agents
        swarm_retrievals, swarm_consensus, _ = self.swarm(swarm_queries)
        
        # Combine memory outputs
        combined = torch.cat([
            episodic_out,
            working_out,
            hierarchical_out,
            swarm_consensus
        ], dim=-1)
        
        # Classify
        output = self.classifier(combined)
        return output

def demonstrate_memory_augmented_network():
    """Demonstrate the complete memory-augmented network."""
    print("\n=== MemoryAugmentedNetwork Demo ===")
    
    # Create network
    batch_size = 16
    input_dim = 128
    hidden_dim = 256
    num_classes = 10
    
    network = MemoryAugmentedNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    
    # Generate sample input
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    output = network(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with device management
    network = DeviceManager.to_device(network)
    x = DeviceManager.to_device(x)
    
    output = network(x)
    print(f"\nDevice management:")
    print(f"Model device: {next(network.parameters()).device}")
    print(f"Input device: {x.device}")
    print(f"Output device: {output.device}")

def main():
    """Run all demonstrations."""
    demonstrate_episodic_memory()
    demonstrate_working_memory_buffer()
    demonstrate_hierarchical_memory()
    demonstrate_shared_swarm_memory()
    demonstrate_memory_augmented_network()

if __name__ == "__main__":
    main() 