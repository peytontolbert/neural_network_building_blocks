"""Memory management components for neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from .attention import MultiHeadAttention
from .core_layers import SmartDense
from .utils import DeviceManager, TensorOps, WeightInitializer

class EpisodicMemory(nn.Module):
    """Episodic memory with attention-based retrieval and compression."""
    
    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        query_dim: int,
        num_heads: int = 8,
        compression_ratio: float = 0.5,
        dropout: float = 0.1,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.compression_ratio = compression_ratio
        
        # Memory slots
        self.memory = nn.Parameter(TensorOps.create_tensor(
            torch.randn(1, memory_size, memory_dim),
            device=device,
            dtype=dtype
        ))
        
        # Query transformation
        self.query_proj = nn.Linear(query_dim, memory_dim, device=device, dtype=dtype)
        WeightInitializer['xavier_uniform'](self.query_proj.weight)
        
        # Memory attention
        self.attention = MultiHeadAttention(
            embed_dim=memory_dim,
            num_heads=num_heads,
            dropout=dropout,
            device=device,
            dtype=dtype
        )
        
        # Memory compression
        compressed_dim = int(memory_dim * compression_ratio)
        self.compressor = nn.Sequential(
            nn.Linear(memory_dim, compressed_dim, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(compressed_dim, memory_dim, device=device, dtype=dtype)
        )
        WeightInitializer['xavier_uniform'](self.compressor[0].weight)
        WeightInitializer['xavier_uniform'](self.compressor[2].weight)
        
        # Memory update mechanism
        self.update_gate = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim, device=device, dtype=dtype),
            nn.Sigmoid()
        )
        WeightInitializer['xavier_uniform'](self.update_gate[0].weight)
        
        # Importance scoring
        self.importance_scorer = nn.Sequential(
            nn.Linear(memory_dim, 1, device=device, dtype=dtype),
            nn.Sigmoid()
        )
        WeightInitializer['xavier_uniform'](self.importance_scorer[0].weight)
    
    def forward(
        self,
        query: torch.Tensor,
        update: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with memory retrieval and optional update.
        
        Args:
            query: Query tensor [batch_size, query_dim]
            update: Whether to update memory
            
        Returns:
            retrieved: Retrieved memory content
            attention_weights: Attention weights
            importance_scores: Memory slot importance scores
        """
        # Move input to correct device
        query = DeviceManager.to_device(query)
        
        batch_size = query.size(0)
        memory = self.memory.expand(batch_size, -1, -1)
        
        # Transform query
        query = self.query_proj(query).unsqueeze(1)  # [B, 1, memory_dim]
        
        # Retrieve from memory
        retrieved, attention_weights = self.attention(query, memory, memory)
        retrieved = retrieved.squeeze(1)
        
        # Score memory importance
        importance_scores = self.importance_scorer(memory).squeeze(-1)
        
        if update and self.training:
            # Compress and update memory
            compressed = self.compressor(retrieved)
            update_gate = self.update_gate(torch.cat([memory, compressed.unsqueeze(1).expand_as(memory)], dim=-1))
            
            # Update memory based on importance
            new_memory = (1 - update_gate) * memory + update_gate * compressed.unsqueeze(1)
            
            # Sort and prune based on importance
            _, indices = importance_scores.sort(descending=True)
            keep_size = int(self.memory_size * (1 - self.compression_ratio))
            keep_indices = indices[:, :keep_size]
            
            # Update memory parameter
            with torch.no_grad():
                self.memory.data = new_memory[0, keep_indices[0]].unsqueeze(0)
        
        return retrieved, attention_weights, importance_scores

class WorkingMemoryBuffer(nn.Module):
    """Working memory buffer with priority-based access."""
    
    def __init__(
        self,
        buffer_size: int,
        memory_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.memory_dim = memory_dim
        
        # Buffer slots
        self.register_buffer('buffer', torch.zeros(1, buffer_size, memory_dim, device=device, dtype=dtype))
        self.register_buffer('priorities', torch.zeros(1, buffer_size, device=device, dtype=dtype))
        
        # Buffer attention
        self.attention = MultiHeadAttention(
            embed_dim=memory_dim,
            num_heads=num_heads,
            dropout=dropout,
            device=device,
            dtype=dtype
        )
        
        # Priority update network
        self.priority_net = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(memory_dim, 1, device=device, dtype=dtype),
            nn.Sigmoid()
        )
        WeightInitializer['xavier_uniform'](self.priority_net[0].weight)
        WeightInitializer['xavier_uniform'](self.priority_net[2].weight)
    
    def write(self, input_data: torch.Tensor, priority: Optional[torch.Tensor] = None):
        """Write data to buffer with priority."""
        input_data = DeviceManager.to_device(input_data)
        batch_size = input_data.size(0)
        
        if priority is None:
            # Compute priority based on content
            priority = self.priority_net(
                torch.cat([input_data, input_data], dim=-1)
            ).squeeze(-1)
        
        # Find lowest priority slots
        _, indices = self.priorities.expand(batch_size, -1).topk(
            k=input_data.size(1),
            dim=1,
            largest=False
        )
        
        # Update buffer and priorities
        for b in range(batch_size):
            self.buffer[0, indices[b]] = input_data[b]
            self.priorities[0, indices[b]] = priority[b]
    
    def read(
        self,
        query: torch.Tensor,
        top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read from buffer using attention.
        
        Args:
            query: Query tensor [batch_size, query_dim]
            top_k: Number of highest priority items to consider
            
        Returns:
            retrieved: Retrieved content
            attention_weights: Attention weights
        """
        query = DeviceManager.to_device(query)
        batch_size = query.size(0)
        
        if top_k is not None:
            # Select top-k priority items
            _, indices = self.priorities.expand(batch_size, -1).topk(k=top_k, dim=1)
            buffer = self.buffer.expand(batch_size, -1, -1).gather(
                1,
                indices.unsqueeze(-1).expand(-1, -1, self.memory_dim)
            )
        else:
            buffer = self.buffer.expand(batch_size, -1, -1)
        
        # Retrieve using attention
        retrieved, attention_weights = self.attention(
            query.unsqueeze(1),
            buffer,
            buffer
        )
        
        return retrieved.squeeze(1), attention_weights

class HierarchicalMemory(nn.Module):
    """Hierarchical memory system with multiple levels."""
    
    def __init__(
        self,
        num_levels: int,
        level_sizes: List[int],
        memory_dim: int,
        query_dim: int,
        compression_ratios: Optional[List[float]] = None,
        device=None,
        dtype=None
    ):
        super().__init__()
        assert len(level_sizes) == num_levels, "Must specify size for each level"
        
        if compression_ratios is None:
            compression_ratios = [0.5] * num_levels
        
        # Create memory levels
        self.levels = nn.ModuleList([
            EpisodicMemory(
                memory_size=size,
                memory_dim=memory_dim,
                query_dim=query_dim,
                compression_ratio=ratio,
                device=device,
                dtype=dtype
            ) for size, ratio in zip(level_sizes, compression_ratios)
        ])
        
        # Level selection network
        self.level_selector = nn.Sequential(
            nn.Linear(query_dim, memory_dim, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(memory_dim, num_levels, device=device, dtype=dtype)
        )
        WeightInitializer['xavier_uniform'](self.level_selector[0].weight)
        WeightInitializer['xavier_uniform'](self.level_selector[2].weight)
        
        # Level combination
        self.level_combiner = nn.Sequential(
            nn.Linear(memory_dim * num_levels, memory_dim, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim, device=device, dtype=dtype)
        )
        WeightInitializer['xavier_uniform'](self.level_combiner[0].weight)
        WeightInitializer['xavier_uniform'](self.level_combiner[2].weight)
    
    def forward(
        self,
        query: torch.Tensor,
        update: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through hierarchical memory.
        
        Args:
            query: Query tensor [batch_size, query_dim]
            update: Whether to update memories
            
        Returns:
            combined: Combined memory retrieval
            level_retrievals: List of retrievals from each level
            level_weights: Level selection weights
        """
        query = DeviceManager.to_device(query)
        
        # Get level selection weights
        level_weights = F.softmax(self.level_selector(query), dim=-1)
        
        # Retrieve from each level
        level_retrievals = []
        level_attentions = []
        level_importances = []
        
        for i, level in enumerate(self.levels):
            retrieved, attention, importance = level(query, update=update)
            level_retrievals.append(retrieved)
            level_attentions.append(attention)
            level_importances.append(importance)
        
        # Combine retrievals weighted by level selection
        stacked_retrievals = torch.stack(level_retrievals, dim=1)  # [B, L, D]
        weighted_retrievals = stacked_retrievals * level_weights.unsqueeze(-1)
        combined = self.level_combiner(weighted_retrievals.flatten(1))
        
        return combined, level_retrievals, level_weights

class SharedSwarmMemory(nn.Module):
    """Shared memory system for swarm coordination."""
    
    def __init__(
        self,
        num_agents: int,
        memory_size: int,
        memory_dim: int,
        query_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.num_agents = num_agents
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        # Shared memory
        self.shared_memory = nn.Parameter(TensorOps.create_tensor(
            torch.randn(1, memory_size, memory_dim),
            device=device,
            dtype=dtype
        ))
        
        # Query projection for each agent
        self.query_projs = nn.ModuleList([
            nn.Linear(query_dim, memory_dim, device=device, dtype=dtype)
            for _ in range(num_agents)
        ])
        for proj in self.query_projs:
            WeightInitializer['xavier_uniform'](proj.weight)
        
        # Memory attention
        self.attention = MultiHeadAttention(
            embed_dim=memory_dim,
            num_heads=num_heads,
            dropout=dropout,
            device=device,
            dtype=dtype
        )
        
        # Memory update network
        self.update_net = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim, device=device, dtype=dtype)
        )
        WeightInitializer['xavier_uniform'](self.update_net[0].weight)
        WeightInitializer['xavier_uniform'](self.update_net[2].weight)
        
        # Consensus mechanism
        self.consensus = nn.Sequential(
            nn.Linear(memory_dim * num_agents, memory_dim, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim, device=device, dtype=dtype)
        )
        WeightInitializer['xavier_uniform'](self.consensus[0].weight)
        WeightInitializer['xavier_uniform'](self.consensus[2].weight)
    
    def forward(
        self,
        queries: List[torch.Tensor],
        update: bool = True
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with shared memory access.
        
        Args:
            queries: List of query tensors for each agent [batch_size, query_dim]
            update: Whether to update shared memory
            
        Returns:
            agent_retrievals: List of retrieved memories for each agent
            consensus_memory: Consensus-based memory update
            attention_weights: List of attention weights for each agent
        """
        # Move queries to correct device
        queries = [DeviceManager.to_device(q) for q in queries]
        assert len(queries) == self.num_agents, "Must provide query for each agent"
        
        batch_size = queries[0].size(0)
        memory = self.shared_memory.expand(batch_size, -1, -1)
        
        # Process each agent's query
        agent_retrievals = []
        attention_weights = []
        
        for i, (query, proj) in enumerate(zip(queries, self.query_projs)):
            # Project query
            projected = proj(query).unsqueeze(1)  # [B, 1, D]
            
            # Retrieve from memory
            retrieved, attention = self.attention(projected, memory, memory)
            agent_retrievals.append(retrieved.squeeze(1))
            attention_weights.append(attention)
        
        if update and self.training:
            # Compute consensus update
            stacked_retrievals = torch.stack(agent_retrievals, dim=1)  # [B, A, D]
            consensus_features = self.consensus(stacked_retrievals.flatten(1))
            
            # Update shared memory
            update_features = self.update_net(
                torch.cat([memory, consensus_features.unsqueeze(1).expand_as(memory)], dim=-1)
            )
            
            # Update memory parameter
            with torch.no_grad():
                self.shared_memory.data = update_features[0].unsqueeze(0)
        
        return agent_retrievals, consensus_features, attention_weights 