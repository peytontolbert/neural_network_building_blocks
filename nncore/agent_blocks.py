"""Agent-oriented neural network components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
from .attention import MultiHeadAttention
from .core_layers import SmartDense
from .utils import DeviceManager

class MemoryModule(nn.Module):
    """Advanced memory module with attention-based retrieval."""
    
    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        query_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        
        # Learnable memory slots
        self.memory = nn.Parameter(torch.randn(1, memory_size, memory_dim))
        
        # Memory attention
        self.attention = MultiHeadAttention(
            embed_dim=memory_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=memory_dim,
            vdim=memory_dim
        )
        
        # Query transformation
        self.query_proj = nn.Linear(query_dim, memory_dim)
        
        # Memory update mechanism
        self.update_gate = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.Sigmoid()
        )
        
        self.candidate = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.Tanh()
        )
        
        # Move all components to the same device
        self.to(DeviceManager.get_device())
    
    def forward(
        self,
        query: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with memory retrieval and update.
        
        Args:
            query: Input query tensor [batch_size, query_dim]
            mask: Optional mask for memory slots [batch_size, memory_size]
            
        Returns:
            retrieved: Retrieved memory content
            attention_weights: Attention weights over memory slots
        """
        # Move inputs to the same device as the module
        query = DeviceManager.to_device(query, self.memory.device)
        batch_size = query.size(0)
        
        if mask is not None:
            mask = DeviceManager.to_device(mask, self.memory.device)
            # Convert boolean mask to attention mask
            # Shape: [batch_size, num_heads, 1, memory_size]
            attn_mask = torch.zeros(batch_size, self.num_heads, 1, self.memory_size, device=mask.device)
            attn_mask = attn_mask.masked_fill(
                mask.unsqueeze(1).unsqueeze(1).expand(-1, self.num_heads, 1, -1),
                float('-inf')
            )
            
        # Expand memory for batch
        memory = self.memory.expand(batch_size, -1, -1)
        
        # Transform query
        query = self.query_proj(query).unsqueeze(1)  # [B, 1, memory_dim]
        
        # Retrieve from memory
        retrieved, attention_weights = self.attention(
            query=query,
            key=memory,
            value=memory,
            attn_mask=attn_mask if mask is not None else None
        )
        
        # Update memory
        if self.training:
            concat = torch.cat([memory, retrieved.expand_as(memory)], dim=-1)
            update_gate = self.update_gate(concat)
            candidate = self.candidate(concat)
            
            memory = (1 - update_gate) * memory + update_gate * candidate
            self.memory.data = memory.mean(dim=0, keepdim=True).detach()
        
        return retrieved.squeeze(1), attention_weights

class DecisionLayer(nn.Module):
    """Decision layer with uncertainty estimation."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_samples: int = 10,
        temperature: float = 0.1,  # Reduced temperature for more conservative sampling
        min_std: float = 0.01,    # Tighter bounds on standard deviation
        max_std: float = 0.3      # Tighter bounds on standard deviation
    ):
        super().__init__()
        self.num_samples = num_samples
        self.temperature = temperature
        self.min_std = min_std
        self.max_std = max_std
        
        # Main network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean and log variance heads
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_var = nn.Linear(hidden_dim, output_dim)
        
        # Initialize log variance to start with small values
        self.log_var.bias.data.fill_(-5.0)
        
        # Move to appropriate device
        self.to(DeviceManager.get_device())
    
    def forward(
        self,
        x: torch.Tensor,
        return_distribution: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            x: Input tensor
            return_distribution: Whether to return distribution parameters
            
        Returns:
            If return_distribution:
                decision: Sampled decision
                mean: Mean of the distribution
                log_var: Log variance of the distribution
            Else:
                decision: Sampled decision
        """
        # Move input to the same device
        x = DeviceManager.to_device(x, next(self.parameters()).device)
        
        hidden = self.network(x)
        mean = self.mean(hidden)
        log_var = self.log_var(hidden)
        
        # Clamp standard deviation
        std = torch.clamp(
            torch.exp(0.5 * log_var),
            min=self.min_std,
            max=self.max_std
        )
        
        if self.training or return_distribution:
            # Use reparameterization trick with temperature
            eps = torch.randn_like(std) * self.temperature
            decision = mean + eps * std
        else:
            decision = mean
        
        if return_distribution:
            return decision, mean, log_var
        return decision

class StateEncoder(nn.Module):
    """State encoder with attention-based processing."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.num_heads = num_heads
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout
                ),
                'norm1': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout)
                ),
                'norm2': nn.LayerNorm(hidden_dim)
            }) for _ in range(num_layers)
        ])
        
        # Move all components to the same device
        self.to(DeviceManager.get_device())
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with self-attention processing."""
        # Move inputs to the same device as the module
        x = DeviceManager.to_device(x, next(self.parameters()).device)
        if mask is not None:
            mask = DeviceManager.to_device(mask, next(self.parameters()).device)
            # Expand mask for multi-head attention
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            
        x = self.input_proj(x)
        
        for layer in self.layers:
            # Self-attention
            attended, _ = layer['attention'](x, x, x, attn_mask=mask)
            x = layer['norm1'](x + attended)
            
            # Feed-forward
            x = layer['norm2'](x + layer['ffn'](x))
            
        return x

class PolicyNetwork(nn.Module):
    """Policy network with state-dependent action distribution."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        discrete: bool = False,
        min_std: float = 1e-4,
        max_std: float = 1.0
    ):
        super().__init__()
        self.discrete = discrete
        self.min_std = min_std
        self.max_std = max_std
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        if discrete:
            self.action_head = nn.Linear(hidden_dim, action_dim)
        else:
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass returning action distribution.
        
        Args:
            state: Input state tensor
            deterministic: Whether to return deterministic action
            
        Returns:
            If discrete:
                action_probs: Action probabilities
            Else:
                action: Sampled action
                log_prob: Log probability of the action
        """
        features = self.network(state)
        
        if self.discrete:
            action_logits = self.action_head(features)
            action_probs = F.softmax(action_logits, dim=-1)
            
            if deterministic:
                return action_probs
            else:
                return torch.distributions.Categorical(action_probs)
        else:
            mean = self.mean(features)
            std = torch.clamp(
                torch.exp(self.log_std),
                min=self.min_std,
                max=self.max_std
            )
            
            if deterministic:
                return mean
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.rsample()
                return action, dist.log_prob(action).sum(-1)

class ValueEstimator(nn.Module):
    """Value estimation network with uncertainty."""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        ensemble_size: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Ensemble of value heads
        self.value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(ensemble_size)
        ])
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with ensemble prediction.
        
        Returns:
            mean_value: Mean predicted value
            std_value: Standard deviation of predicted values
        """
        features = self.feature_extractor(state)
        
        # Get predictions from all ensemble members
        values = torch.stack([head(features) for head in self.value_heads], dim=0)
        
        mean_value = values.mean(dim=0)
        std_value = values.std(dim=0)
        
        return mean_value, std_value 