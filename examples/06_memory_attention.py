"""Example demonstrating the combination of memory and attention mechanisms."""

import torch
import torch.nn as nn
from src.nncore.memory import EpisodicMemory, WorkingMemoryBuffer
from src.nncore.attention import MultiHeadAttention, LinearAttention
from src.nncore.utils import DeviceManager

# Set random seed for reproducibility
torch.manual_seed(42)


class MemoryAttentionBlock(nn.Module):
    """Block combining memory and attention mechanisms."""

    def __init__(
        self,
        input_dim: int,
        memory_size: int = 100,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Memory components
        self.episodic = EpisodicMemory(
            memory_size=memory_size, memory_dim=input_dim, query_dim=input_dim
        )

        self.working = WorkingMemoryBuffer(
            buffer_size=memory_size // 2, memory_dim=input_dim
        )

        # Attention components
        self.self_attention = MultiHeadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout
        )

        self.memory_attention = MultiHeadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout
        )

        self.local_attention = LinearAttention(embed_dim=input_dim)

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size = x.size(0)

        # Self-attention
        attended, _ = self.self_attention(x, x, x, attention_mask=mask)
        x = self.norm1(x + self.dropout(attended))

        # Memory access and attention
        episodic_mem, _, _ = self.episodic(x)
        self.working.write(x.unsqueeze(1))
        working_mem, _ = self.working.read(x)

        # Combine memory states
        memory_states = torch.stack([episodic_mem, working_mem], dim=1)

        # Attend to memory
        memory_context, _ = self.memory_attention(
            x.unsqueeze(1), memory_states, memory_states
        )
        memory_context = memory_context.squeeze(1)
        x = self.norm2(x + self.dropout(memory_context))

        # Local processing
        local = self.local_attention(x)
        x = self.norm3(x + self.dropout(local))

        return x


class MemoryAugmentedTransformer(nn.Module):
    """Transformer architecture augmented with memory mechanisms."""

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 6,
        memory_size: int = 100,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, input_dim)

        # Memory-attention layers
        self.layers = nn.ModuleList(
            [
                MemoryAttentionBlock(
                    input_dim=input_dim,
                    memory_size=memory_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Output normalization
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Embed input
        x = self.input_embedding(x)

        # Process through layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final normalization
        x = self.norm(x)

        return x


def demonstrate_memory_attention_block():
    """Demonstrate MemoryAttentionBlock functionality."""
    print("\n=== MemoryAttentionBlock Demo ===")

    # Create block
    batch_size = 16
    seq_length = 32
    input_dim = 256

    block = MemoryAttentionBlock(input_dim=input_dim, memory_size=100, num_heads=8)

    # Generate sample input
    x = torch.randn(batch_size, seq_length, input_dim)

    # Optional attention mask
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

    # Forward pass
    output = block(x, mask)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test with different sequence lengths
    x_short = torch.randn(batch_size, seq_length // 2, input_dim)
    mask_short = mask[:, : seq_length // 2, : seq_length // 2]
    output_short = block(x_short, mask_short)

    print(f"\nVariable sequence length:")
    print(f"Short input shape: {x_short.shape}")
    print(f"Short output shape: {output_short.shape}")


def demonstrate_memory_augmented_transformer():
    """Demonstrate MemoryAugmentedTransformer functionality."""
    print("\n=== MemoryAugmentedTransformer Demo ===")

    # Create model
    batch_size = 8
    seq_length = 64
    input_dim = 512

    model = MemoryAugmentedTransformer(
        input_dim=input_dim, num_layers=6, memory_size=100, num_heads=8
    )

    # Generate sample input
    x = torch.randn(batch_size, seq_length, input_dim)

    # Create attention mask
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

    # Forward pass
    output = model(x, mask)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test with device management
    model = DeviceManager.to_device(model)
    x = DeviceManager.to_device(x)
    mask = DeviceManager.to_device(mask)

    output = model(x, mask)
    print(f"\nDevice management:")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Input device: {x.device}")
    print(f"Output device: {output.device}")


def main():
    """Run all demonstrations."""
    demonstrate_memory_attention_block()
    demonstrate_memory_augmented_transformer()


if __name__ == "__main__":
    main()
