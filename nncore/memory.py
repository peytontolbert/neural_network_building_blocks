"""Memory modules for neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from .utils import DeviceManager
from .attention import MultiHeadAttention


class EpisodicMemory(nn.Module):
    """Memory module with episodic storage and retrieval."""

    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        query_dim: int,
        compression_ratio: float = 0.5,
        device=None,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.query_dim = query_dim

        # Get device
        if device is None:
            device = DeviceManager.get_default_device()
        self.device = device

        # Initialize memory with shape [1, memory_size, memory_dim]
        self.register_parameter(
            "memory",
            nn.Parameter(
                torch.randn(1, memory_size, memory_dim, device=device)
                / (memory_dim**0.5)
            ),
        )

        # Query projection
        self.query_proj = nn.Linear(query_dim, memory_dim).to(device)

        # Multi-head attention mechanism
        self.attention = MultiHeadAttention(
            embed_dim=memory_dim, num_heads=8, dropout=0.1, device=device
        )

        # Move to device and initialize
        self.to(device)
        self = DeviceManager.initialize_module(self)

    def forward(
        self, query: torch.Tensor, update: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve from memory using query."""
        query = DeviceManager.to_device(query, self.device)
        batch_size = query.size(0)

        # Project query
        query = self.query_proj(query)  # [B, memory_dim]
        query = query.unsqueeze(1)  # [B, 1, memory_dim]

        # Prepare memory for attention
        memory = self.memory.expand(batch_size, -1, -1)  # [B, memory_size, memory_dim]

        # Use multi-head attention
        retrieved, attention = self.attention(
            query=query,  # [B, 1, memory_dim]
            key=memory,  # [B, memory_size, memory_dim]
            value=memory,  # [B, memory_size, memory_dim]
        )

        # Reshape outputs
        retrieved = retrieved.squeeze(1)  # [B, memory_dim]
        attention = attention.squeeze(1)  # [B, num_heads, memory_size]

        # Compute importance scores
        importance = torch.norm(self.memory.squeeze(0), dim=1)  # [memory_size]

        # Update memory if in training mode and update is True
        if self.training and update:
            with torch.no_grad():
                # Average attention across heads and batch
                avg_attention = attention.mean(dim=1)  # [B, memory_size]
                avg_attention = avg_attention.mean(dim=0)  # [memory_size]

                # Average retrieved vectors across batch
                avg_retrieved = retrieved.mean(dim=0)  # [memory_dim]

                # Compute memory update using matrix multiplication
                memory_update = torch.matmul(
                    avg_attention.view(self.memory_size, 1),  # [memory_size, 1]
                    avg_retrieved.view(1, self.memory_dim),  # [1, memory_dim]
                )  # [memory_size, memory_dim]

                self.memory.data = (
                    self.memory.data * 0.9 + memory_update.unsqueeze(0) * 0.1
                )

        return retrieved, attention, importance.expand(batch_size, -1)


class WorkingMemoryBuffer(nn.Module):
    """Short-term memory buffer with priority-based updates."""

    def __init__(self, buffer_size: int, memory_dim: int, device=None):
        super().__init__()
        self.buffer_size = buffer_size
        self.memory_dim = memory_dim

        # Get device
        if device is None:
            device = DeviceManager.get_default_device()
        self.device = device

        # Initialize buffer and priorities
        self.register_parameter(
            "buffer", nn.Parameter(torch.zeros(buffer_size, memory_dim, device=device))
        )
        self.register_parameter(
            "priorities", nn.Parameter(torch.zeros(buffer_size, device=device))
        )

        # Priority network
        self.priority_net = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
            nn.Sigmoid(),
        ).to(device)

        # Attention mechanism
        self.attention = MultiHeadAttention(
            embed_dim=memory_dim, num_heads=8, dropout=0.1, device=device
        )

        # Move to device and initialize
        self.to(device)
        self = DeviceManager.initialize_module(self)

    def write(self, input_data: torch.Tensor, priority: Optional[torch.Tensor] = None):
        """Write data to buffer with priority."""
        input_data = DeviceManager.to_device(input_data, self.device)
        if priority is not None:
            priority = DeviceManager.to_device(priority, self.device)

        batch_size = input_data.size(0)
        seq_length = input_data.size(1) if input_data.dim() > 2 else 1

        if priority is None:
            # Compute priority based on content
            priority = self.priority_net(
                torch.cat([input_data, input_data], dim=-1)
            ).squeeze(-1)

        # Find lowest priority slots
        with torch.no_grad():
            _, indices = self.priorities.expand(batch_size, -1).topk(
                k=int(seq_length), dim=1, largest=False  # Ensure k is an integer
            )

            # Update buffer and priorities
            for b in range(batch_size):
                self.buffer.data[indices[b]] = input_data[b]
                self.priorities.data[indices[b]] = priority[b]

    def read(
        self, query: torch.Tensor, top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read from buffer using attention and optionally select top-k items."""
        query = DeviceManager.to_device(query, self.device)
        batch_size = query.size(0)

        # Reshape query for attention
        query = query.unsqueeze(1)  # [B, 1, D]

        if top_k is not None:
            # Get top-k by priority
            with torch.no_grad():
                _, indices = self.priorities.topk(
                    k=int(top_k)
                )  # Ensure k is an integer
                buffer_subset = self.buffer[indices]  # [K, D]
                buffer_subset = buffer_subset.unsqueeze(0).expand(
                    batch_size, -1, -1
                )  # [B, K, D]
        else:
            buffer_subset = self.buffer.unsqueeze(0).expand(
                batch_size, -1, -1
            )  # [B, N, D]

        # Compute attention
        retrieved, attention = self.attention(
            query=query,  # [B, 1, D]
            key=buffer_subset,  # [B, K/N, D]
            value=buffer_subset,  # [B, K/N, D]
        )

        return retrieved.squeeze(1), attention.squeeze(1)


class HierarchicalMemory(nn.Module):
    """Multi-level hierarchical memory system."""

    def __init__(
        self,
        num_levels: int,
        level_sizes: List[int],
        memory_dim: int,
        query_dim: int,
        device=None,
    ):
        super().__init__()
        assert (
            len(level_sizes) == num_levels
        ), f"Expected {num_levels} level sizes, got {len(level_sizes)}"
        self.num_levels = num_levels
        self.memory_dim = memory_dim

        # Get device
        if device is None:
            device = DeviceManager.get_default_device()
        self.device = device

        # Create memory levels
        self.levels = nn.ModuleList(
            [
                EpisodicMemory(
                    memory_size=size,
                    memory_dim=memory_dim,
                    query_dim=query_dim,
                    device=device,
                )
                for size in level_sizes
            ]
        ).to(device)

        # Level selection network
        self.level_selector = nn.Sequential(
            nn.Linear(query_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, num_levels),
        ).to(device)

        # Move everything to device and initialize
        self.to(device)
        self = DeviceManager.initialize_module(self)

    def forward(
        self, query: torch.Tensor, update: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Query all memory levels and combine results."""
        # Move query to device and ensure it's contiguous
        query = DeviceManager.to_device(query, self.device).contiguous()
        batch_size = query.size(0)

        # Get level weights
        level_selector_output = self.level_selector(query)  # [B, num_levels]
        level_weights = F.softmax(level_selector_output, dim=-1)

        # Query each level
        retrievals = []

        for i, level in enumerate(self.levels):
            # Get retrieval from each level
            retrieved, _, _ = level(query, update=update)  # [B, memory_dim]
            retrievals.append(retrieved)

        # Stack retrievals
        retrievals_tensor = torch.stack(
            retrievals, dim=1
        )  # [B, num_levels, memory_dim]

        # Combine retrievals using level weights
        combined = torch.sum(
            retrievals_tensor * level_weights.unsqueeze(-1), dim=1
        )  # [B, memory_dim]

        # Convert retrievals tensor back to list
        level_retrievals = [retrievals_tensor[:, i, :] for i in range(self.num_levels)]

        # Move level_weights to CPU for test comparison
        level_weights = level_weights.cpu()

        return combined, level_retrievals, level_weights


class SharedSwarmMemory(nn.Module):
    """Shared memory system for swarm agents."""

    def __init__(
        self,
        num_agents: int,
        memory_size: int,
        memory_dim: int,
        query_dim: int,
        device=None,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.memory_size = memory_size
        self.memory_dim = memory_dim

        # Get device
        if device is None:
            device = DeviceManager.get_default_device()
        self.device = device

        # Shared memory
        self.register_parameter(
            "shared_memory",
            nn.Parameter(
                torch.randn(1, memory_size, memory_dim, device=device)
                / (memory_dim**0.5)
            ),
        )

        # Query projections for each agent
        self.query_projs = nn.ModuleList(
            [nn.Linear(query_dim, memory_dim).to(device) for _ in range(num_agents)]
        )

        # Move to device and initialize
        self.to(device)
        self = DeviceManager.initialize_module(self)

    def forward(
        self, queries: List[torch.Tensor], update: bool = False
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """Process queries from all agents."""
        assert (
            len(queries) == self.num_agents
        ), f"Expected {self.num_agents} queries, got {len(queries)}"

        # Move queries to device and get batch size
        queries = [DeviceManager.to_device(q, self.device) for q in queries]
        batch_size = queries[0].size(0)

        # Project queries
        projected_queries = []
        for i, (query, proj) in enumerate(zip(queries, self.query_projs)):
            projected = proj(query).unsqueeze(1)  # [B, 1, memory_dim]
            projected_queries.append(projected)

        # Stack queries and prepare memory
        stacked_queries = torch.cat(
            projected_queries, dim=1
        )  # [B, num_agents, memory_dim]
        memory = self.shared_memory.expand(
            batch_size, -1, -1
        )  # [B, memory_size, memory_dim]

        # Compute attention for each agent
        retrievals = []
        attention_weights = []

        for i in range(self.num_agents):
            agent_query = stacked_queries[:, i : i + 1, :]  # [B, 1, memory_dim]
            attention = torch.matmul(
                agent_query, memory.transpose(-2, -1)
            )  # [B, 1, memory_size]
            attention = F.softmax(attention / (self.memory_dim**0.5), dim=-1)
            retrieved = torch.matmul(attention, memory)  # [B, 1, memory_dim]

            retrievals.append(retrieved.squeeze(1))
            attention_weights.append(attention.squeeze(1))

        # Compute consensus
        consensus = torch.mean(torch.stack(retrievals, dim=0), dim=0)  # [B, memory_dim]

        # Update memory if in training mode and update is True
        if self.training and update:
            with torch.no_grad():
                # Compute consensus update
                attention_mean = torch.mean(
                    torch.stack(attention_weights, dim=0), dim=0
                )  # [B, memory_size]
                memory_update = torch.matmul(
                    attention_mean.transpose(0, 1), consensus
                )  # [memory_size, memory_dim]
                self.shared_memory.data = (
                    self.shared_memory.data * 0.9 + memory_update.unsqueeze(0) * 0.1
                )

        return retrievals, consensus, attention_weights
