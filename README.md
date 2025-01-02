# Neural Network Building Blocks

A comprehensive library of reusable neural network components designed for maximum composability, with special consideration for agent-based architectures and swarm intelligence systems.

## Features

### Core Components
- Smart Dense layers with dynamic capacity
- Advanced Convolution suite with attention
- Enhanced residual and dense connections
- Multi-head and linear attention mechanisms
- Advanced normalization and regularization
- Dynamic routing modules
- Feature pyramid networks

### Agent & Swarm Features
- Memory modules with attention-based retrieval
- Policy networks for agent decision-making
- Agent-to-agent attention mechanisms
- Swarm coordination components
- Shared memory systems

### Advanced Capabilities
- Layer composition utilities
- Neural architecture search
- Cross-modal processing
- Self-improvement components
- Dynamic parameter adaptation

## Installation

```bash
git clone https://github.com/peytontolbert/neural-network-building-blocks.git
cd neural_network_building_blocks
pip install -r requirements.txt
```

## Quick Start

### Basic Layers

```python
from nncore.core_layers import SmartDense, AdvancedConv2d

# Create a smart dense layer with dynamic capacity
layer = SmartDense(
    in_features=256,
    out_features=128,
    dynamic_growth=True,
    growth_factor=1.5
)

# Create an advanced convolution layer with attention
conv = AdvancedConv2d(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    attention=True
)
```

### Attention Mechanisms

```python
from nncore.attention import MultiHeadAttention, LinearAttention

# Multi-head attention with optimizations
mha = MultiHeadAttention(
    embed_dim=512,
    num_heads=8,
    dropout=0.1
)

# Linear attention for O(N) complexity
linear_attn = LinearAttention(
    dim=512,
    heads=8,
    dim_head=64
)
```

### Advanced Components

```python
from nncore.advanced_blocks import EnhancedResidualBlock, DenseBlock
from nncore.agent_blocks import MemoryModule

# Create an enhanced residual block
residual = EnhancedResidualBlock(
    channels=256,
    expansion=4,
    use_attention=True
)

# Create a memory module for agents
memory = MemoryModule(
    memory_size=1000,
    memory_dim=256,
    query_dim=128
)
```

### Layer Composition

```python
from nncore.composition import LayerFactory, BlockComposer, ArchitectureGenerator

# Create components using factory
factory = LayerFactory()
composer = BlockComposer(factory)

# Generate architecture automatically
generator = ArchitectureGenerator(factory, composer)
model = generator.generate_architecture(
    input_shape=(32, 3, 224, 224),
    target_shape=(32, 1000),
    constraints={'max_layers': 5}
)
```

## Documentation

For detailed documentation, see the `docs` directory:

- [Core Layers](docs/core_layers.md)
- [Attention Mechanisms](docs/attention.md)
- [Agent Components](docs/agent_components.md)
- [Advanced Blocks](docs/advanced_blocks.md)
- [Composition Tools](docs/composition.md)

## Integration with Day 1's PyTorch Basics Library

This library seamlessly integrates with the PyTorch Basics Library from Day 1, utilizing its:
- Device management utilities
- Enhanced tensor operations
- Weight initialization schemes
- Performance monitoring tools

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 