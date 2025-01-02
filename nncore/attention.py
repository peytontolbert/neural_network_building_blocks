"""Advanced attention mechanisms for neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from .utils import DeviceManager, TensorOps, WeightInitializer

class MultiHeadAttention(nn.Module):
    """Multi-head attention with optimizations."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        device=None,
        dtype=None
    ):
        super().__init__()
        
        # Initialize device if not provided
        device = device or DeviceManager.get_default_device()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        
        # Use TensorOps for parameter creation
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        
        if add_bias_kv:
            self.bias_k = nn.Parameter(TensorOps.create_tensor(
                (1, 1, embed_dim),
                device=device,
                dtype=dtype
            ))
            self.bias_v = nn.Parameter(TensorOps.create_tensor(
                (1, 1, embed_dim),
                device=device,
                dtype=dtype
            ))
        else:
            self.bias_k = self.bias_v = None
            
        self.add_zero_attn = add_zero_attn
        self.reset_parameters()
        
        # Move to specified device
        self.to(device)
        
    def reset_parameters(self):
        """Initialize parameters."""
        # Use WeightInitializer for initialization
        WeightInitializer['xavier_uniform'](self.q_proj.weight, gain=1/math.sqrt(2))
        WeightInitializer['xavier_uniform'](self.k_proj.weight, gain=1/math.sqrt(2))
        WeightInitializer['xavier_uniform'](self.v_proj.weight, gain=1/math.sqrt(2))
        WeightInitializer['xavier_uniform'](self.out_proj.weight)
        
        if self.bias_k is not None:
            WeightInitializer['xavier_normal'](self.bias_k)
        if self.bias_v is not None:
            WeightInitializer['xavier_normal'](self.bias_v)
            
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor of shape (N, L, E)
            key: Key tensor of shape (N, S, E)
            value: Value tensor of shape (N, S, E)
            key_padding_mask: Mask for padded elements
            attn_mask: Attention mask for masked attention
            
        Returns:
            output: Attention output
            attention_weights: Attention weights
        """
        # Move inputs to correct device
        device = self.q_proj.weight.device
        query = DeviceManager.to_device(query, device)
        key = DeviceManager.to_device(key, device)
        value = DeviceManager.to_device(value, device)
        if key_padding_mask is not None:
            key_padding_mask = DeviceManager.to_device(key_padding_mask, device)
        if attn_mask is not None:
            attn_mask = DeviceManager.to_device(attn_mask, device)
        
        N, L, E = query.shape
        S = key.shape[1]
        
        scaling = float(self.head_dim) ** -0.5
        
        # Project and reshape
        q = self.q_proj(query) * scaling  # [N, L, E]
        k = self.k_proj(key)              # [N, S, E]
        v = self.v_proj(value)            # [N, S, E]
        
        # Reshape to multi-head format
        q = q.contiguous().view(N, L, self.num_heads, self.head_dim).transpose(1, 2)  # [N, H, L, D]
        k = k.contiguous().view(N, S, self.num_heads, self.head_dim).transpose(1, 2)  # [N, H, S, D]
        v = v.contiguous().view(N, S, self.num_heads, self.head_dim).transpose(1, 2)  # [N, H, S, D]
        
        # Compute attention scores with improved numerical stability
        attn_output_weights = torch.matmul(q, k.transpose(-2, -1))  # [N, H, L, S]
        attn_output_weights = attn_output_weights / math.sqrt(self.head_dim)
        
        # Apply masks if provided
        if attn_mask is not None:
            attn_output_weights = attn_output_weights + attn_mask
            
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # [N, 1, 1, S]
                float('-inf')
            )
        
        # Apply softmax with improved numerical stability
        attn_output_weights = F.softmax(attn_output_weights, dim=-1, dtype=torch.float32)
        
        # Apply dropout
        if self.dropout > 0:
            attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)
        
        # Convert back to original dtype
        attn_output_weights = attn_output_weights.to(query.dtype)
        
        # Compute attention output
        attn_output = torch.matmul(attn_output_weights, v)  # [N, H, L, D]
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(N, L, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output, attn_output_weights

class LinearAttention(nn.Module):
    """Linear attention variant with O(N) complexity."""
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        device=None,
        dtype=None
    ):
        super().__init__()
        
        # Initialize device if not provided
        device = device or DeviceManager.get_default_device()
        
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        # Use TensorOps for parameter creation
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False, device=device, dtype=dtype)
        WeightInitializer['xavier_uniform'](self.to_qkv.weight)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, device=device, dtype=dtype),
            nn.Dropout(dropout)
        )
        WeightInitializer['xavier_uniform'](self.to_out[0].weight)
        
        # Move to specified device
        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with linear attention computation."""
        # Move input to correct device
        x = DeviceManager.to_device(x, self.to_qkv.weight.device)
        
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, self.heads, -1).transpose(1, 2), qkv)
        
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        
        context = torch.matmul(k.transpose(-2, -1), v)
        out = torch.matmul(q, context)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)