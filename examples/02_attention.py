"""Example demonstrating the usage of attention mechanisms from the nncore package."""

import torch
import torch.nn as nn
from src.nncore.attention import MultiHeadAttention, LinearAttention
from src.nncore.utils import DeviceManager

# Set random seed for reproducibility
torch.manual_seed(42)


def demonstrate_multi_head_attention():
    """Demonstrate MultiHeadAttention functionality."""
    print("\n=== MultiHeadAttention Demo ===")

    # Create a MultiHeadAttention layer
    batch_size = 16
    seq_length = 32
    embed_dim = 256
    num_heads = 8

    attention = MultiHeadAttention(
        embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, bias=True
    )

    # Generate sample input
    query = torch.randn(batch_size, seq_length, embed_dim)
    key = torch.randn(batch_size, seq_length, embed_dim)
    value = torch.randn(batch_size, seq_length, embed_dim)

    # Forward pass
    output, attention_weights = attention(query, key, value)

    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Demonstrate self-attention
    print("\nSelf-attention demo:")
    output_self, attention_weights_self = attention(query, query, query)
    print(f"Self-attention output shape: {output_self.shape}")

    # Demonstrate attention masking
    print("\nMasked attention demo:")
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

    output_masked, attention_weights_masked = attention(
        query, key, value, attention_mask=mask
    )
    print(f"Masked attention output shape: {output_masked.shape}")


def demonstrate_linear_attention():
    """Demonstrate LinearAttention functionality."""
    print("\n=== LinearAttention Demo ===")

    # Create a LinearAttention layer
    batch_size = 16
    seq_length = 64
    embed_dim = 256

    attention = LinearAttention(embed_dim=embed_dim, eps=1e-6, causal=False)

    # Generate sample input
    x = torch.randn(batch_size, seq_length, embed_dim)

    # Forward pass
    output = attention(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Demonstrate causal attention
    print("\nCausal linear attention demo:")
    causal_attention = LinearAttention(embed_dim=embed_dim, causal=True)
    output_causal = causal_attention(x)
    print(f"Causal attention output shape: {output_causal.shape}")


class TransformerBlock(nn.Module):
    """Simple Transformer block using attention mechanisms."""

    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1
    ):
        super().__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )

        # Linear attention for local processing
        self.local_attention = LinearAttention(embed_dim=embed_dim)

        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention with residual
        attended, _ = self.attention(x, x, x, attention_mask=mask)
        x = self.norm1(x + self.dropout(attended))

        # Linear attention with residual
        local = self.local_attention(x)
        x = self.norm2(x + self.dropout(local))

        # Feed-forward with residual
        ff = self.ff_network(x)
        x = self.norm3(x + self.dropout(ff))

        return x


def demonstrate_transformer_block():
    """Demonstrate a Transformer block using both attention mechanisms."""
    print("\n=== TransformerBlock Demo ===")

    # Create a TransformerBlock
    batch_size = 8
    seq_length = 32
    embed_dim = 256
    num_heads = 8
    ff_dim = 1024

    block = TransformerBlock(
        embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=0.1
    )

    # Generate sample input
    x = torch.randn(batch_size, seq_length, embed_dim)

    # Create attention mask (optional)
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

    # Forward pass
    output = block(x, mask)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test with device management
    block = DeviceManager.to_device(block)
    x = DeviceManager.to_device(x)
    mask = DeviceManager.to_device(mask)

    output = block(x, mask)
    print(f"\nDevice management:")
    print(f"Model device: {next(block.parameters()).device}")
    print(f"Input device: {x.device}")
    print(f"Output device: {output.device}")


def main():
    """Run all demonstrations."""
    demonstrate_multi_head_attention()
    demonstrate_linear_attention()
    demonstrate_transformer_block()


if __name__ == "__main__":
    main()
