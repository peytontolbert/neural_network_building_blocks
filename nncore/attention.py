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
        dtype=None,
    ):
        super().__init__()

        # Initialize device if not provided
        device = device or DeviceManager.get_default_device()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.scaling = float(self.head_dim) ** -0.5

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        # Use TensorOps for parameter creation
        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.k_proj = nn.Linear(
            self.kdim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            self.vdim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

        if add_bias_kv:
            self.bias_k = nn.Parameter(
                TensorOps.create_tensor((1, 1, embed_dim), device=device, dtype=dtype)
            )
            self.bias_v = nn.Parameter(
                TensorOps.create_tensor((1, 1, embed_dim), device=device, dtype=dtype)
            )
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.reset_parameters()

        # Move to specified device
        self.to(device)

    def reset_parameters(self):
        """Initialize parameters."""
        WeightInitializer["xavier_uniform"](self.q_proj.weight, gain=1 / math.sqrt(2))
        WeightInitializer["xavier_uniform"](self.k_proj.weight, gain=1 / math.sqrt(2))
        WeightInitializer["xavier_uniform"](self.v_proj.weight, gain=1 / math.sqrt(2))
        WeightInitializer["xavier_uniform"](self.out_proj.weight)

        if self.bias_k is not None:
            WeightInitializer["xavier_normal"](self.bias_k)
        if self.bias_v is not None:
            WeightInitializer["xavier_normal"](self.bias_v)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of multi-head attention."""
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

        # Project and reshape
        q = self.q_proj(query).view(N, L, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(N, S, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(N, S, self.num_heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # [N, H, L, D]
        k = k.transpose(1, 2)  # [N, H, S, D]
        v = v.transpose(1, 2)  # [N, H, S, D]

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # [N, H, L, S]
        attn_weights = attn_weights * self.scaling

        # Apply attention mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # Apply softmax with improved numerical stability
        attn_weights = attn_weights.float()

        # Subtract max for numerical stability
        attn_max = torch.max(attn_weights, dim=-1, keepdim=True)[0].detach()
        exp_weights = torch.exp(attn_weights - attn_max)

        # Compute denominator and handle potential zeros
        denom = exp_weights.sum(dim=-1, keepdim=True)
        denom = torch.where(denom > 0, denom, torch.ones_like(denom))

        # Normalize attention weights
        attn_weights = exp_weights / denom

        # Double-check normalization
        attn_weights = F.normalize(attn_weights, p=1, dim=-1)

        # Apply dropout if specified
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)
            # Renormalize after dropout
            attn_weights = F.normalize(attn_weights, p=1, dim=-1)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)  # [N, H, L, D]

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(N, L, E)
        output = self.out_proj(attn_output)

        return output, attn_weights


class LinearAttention(nn.Module):
    """Linear attention variant with O(N) complexity."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # Initialize device if not provided
        device = device or DeviceManager.get_default_device()

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        # Use TensorOps for parameter creation
        self.to_qkv = nn.Linear(
            dim, inner_dim * 3, bias=False, device=device, dtype=dtype
        )
        WeightInitializer["xavier_uniform"](self.to_qkv.weight)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, device=device, dtype=dtype), nn.Dropout(dropout)
        )
        WeightInitializer["xavier_uniform"](self.to_out[0].weight)

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
