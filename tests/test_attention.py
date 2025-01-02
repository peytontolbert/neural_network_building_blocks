"""Tests for attention mechanisms."""

import torch
import pytest
from nncore.attention import MultiHeadAttention, LinearAttention

def test_multi_head_attention():
    """Test MultiHeadAttention layer basic functionality."""
    batch_size = 8
    seq_length = 16
    embed_dim = 256
    num_heads = 8
    
    layer = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.1
    )
    
    query = torch.randn(batch_size, seq_length, embed_dim)
    key = torch.randn(batch_size, seq_length, embed_dim)
    value = torch.randn(batch_size, seq_length, embed_dim)
    
    output, attention_weights = layer(query, key, value)
    
    # Check output shape
    assert output.shape == (batch_size, seq_length, embed_dim)
    
    # Check attention weights shape
    expected_attn_shape = (batch_size, num_heads, seq_length, seq_length)
    assert attention_weights.shape == expected_attn_shape
    
    # Check if attention weights sum to 1 along the correct dimension
    attn_sum = attention_weights.sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-6)

def test_linear_attention():
    """Test LinearAttention layer basic functionality."""
    batch_size = 8
    seq_length = 16
    dim = 256
    
    layer = LinearAttention(
        dim=dim,
        heads=8,
        dim_head=32,
        dropout=0.1
    )
    
    x = torch.randn(batch_size, seq_length, dim)
    output = layer(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_length, dim)
    
    # Test with different sequence lengths
    x_longer = torch.randn(batch_size, seq_length * 2, dim)
    output_longer = layer(x_longer)
    assert output_longer.shape == (batch_size, seq_length * 2, dim)

if __name__ == "__main__":
    pytest.main([__file__]) 