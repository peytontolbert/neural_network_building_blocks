# Neural Network Building Blocks Examples

This directory contains example scripts demonstrating the usage of various components from the `nncore` package.

## Examples Overview

1. **Core Layers** (`01_core_layers.py`)
   - Demonstrates enhanced layer implementations
   - Shows usage of `SmartDense` and `AdvancedConv2d`
   - Includes example of building a simple network
   - Demonstrates device management integration

2. **Attention Mechanisms** (`02_attention.py`)
   - Shows implementation of attention-based components
   - Demonstrates `MultiHeadAttention` and `LinearAttention`
   - Includes example of building a Transformer block
   - Shows masking and self-attention patterns

3. **Multimodal Processing** (`03_multimodal.py`)
   - Demonstrates multimodal component usage
   - Shows `SpeechEncoder`, `VisionProcessor`, `TextProcessor`
   - Demonstrates cross-modal fusion
   - Includes complete multimodal classifier example

4. **Memory Management** (`04_memory.py`)
   - Shows various memory system implementations
   - Demonstrates `EpisodicMemory`, `WorkingMemoryBuffer`
   - Shows `HierarchicalMemory` and `SharedSwarmMemory`
   - Includes memory-augmented network example

5. **Adaptive Components** (`05_adaptive.py`)
   - Demonstrates self-improving components
   - Shows `AdaptiveComputation`, `MetaLearningModule`
   - Demonstrates `EvolutionaryLayer` and population-based training
   - Includes complete adaptive network example

6. **Memory-Attention Integration** (`06_memory_attention.py`)
   - Shows how to combine memory and attention mechanisms
   - Demonstrates memory-augmented attention blocks
   - Includes memory-augmented Transformer implementation
   - Shows handling of variable sequence lengths

7. **Adaptive Multimodal Processing** (`07_adaptive_multimodal.py`)
   - Demonstrates integration of adaptive and multimodal components
   - Shows adaptive modality-specific processing
   - Includes adaptive multimodal fusion
   - Demonstrates cross-modal adaptation capabilities

## Running the Examples

Each example can be run independently. Make sure you have installed the package and its dependencies first.

```bash
# Install the package
pip install -e ..

# Run individual examples
python 01_core_layers.py
python 02_attention.py
python 03_multimodal.py
python 04_memory.py
python 05_adaptive.py
python 06_memory_attention.py
python 07_adaptive_multimodal.py
```

## Example Features

### Core Layers Example
- Layer initialization and configuration
- Forward pass demonstrations
- Different activation functions
- Device management
- Building composite networks

### Attention Example
- Attention mechanism implementations
- Self-attention patterns
- Attention masking
- Transformer block construction

### Multimodal Example
- Processing different data modalities
- Cross-modal feature fusion
- Handling missing modalities
- Multimodal classification

### Memory Example
- Memory system implementations
- Memory writing and retrieval
- Hierarchical memory access
- Shared memory for multi-agent systems

### Adaptive Example
- Dynamic computation paths
- Meta-learning and adaptation
- Evolutionary neural components
- Population-based training

### Memory-Attention Example
- Memory-augmented attention
- Combining episodic and working memory
- Memory-based context enhancement
- Variable sequence length handling

### Adaptive Multimodal Example
- Modality-specific adaptation
- Cross-modal feature evolution
- Adaptive fusion mechanisms
- Multi-modal meta-learning

## Advanced Usage Examples

### Combining Memory and Attention
```python
# Create a memory-attention block
block = MemoryAttentionBlock(
    input_dim=256,
    memory_size=100,
    num_heads=8
)

# Process sequence with memory augmentation
output = block(sequence, attention_mask=mask)
```

### Adaptive Multimodal Processing
```python
# Create an adaptive multimodal network
network = AdaptiveMultimodalNetwork(
    speech_dim=80,
    vision_dim=2048,
    text_dim=512,
    hidden_dim=512,
    num_classes=10
)

# Process multiple modalities
output = network(
    speech_input=speech,
    vision_input=vision,
    text_input=text
)

# Adapt to new data
network.adapt(
    support_speech=speech_support,
    support_vision=vision_support,
    support_text=text_support,
    support_y=labels_support
)
```

## Notes

- All examples include proper error handling and device management
- Each component is demonstrated both individually and in combination
- Examples show both basic usage and advanced features
- Device management is consistently demonstrated across all examples
- Advanced examples show how to combine different components effectively

## Additional Resources

- See the main README.md for complete package documentation
- Check individual component docstrings for detailed API information
- Refer to the tests directory for additional usage examples
- See the tutorial notebooks for step-by-step guides

## Contributing

Feel free to contribute additional examples or improvements to existing ones. Please follow these guidelines:
1. Include comprehensive docstrings
2. Demonstrate both basic and advanced usage
3. Include proper error handling
4. Show device management integration
5. Add appropriate comments explaining key concepts
6. Show how components can be combined effectively