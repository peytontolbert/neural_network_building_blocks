"""
Getting Started with Neural Network Building Blocks

This tutorial introduces the key concepts and components of the nncore package.
We'll cover:
1. Core layers and utilities
2. Attention mechanisms
3. Multimodal processing
4. Memory management
5. Adaptive components
"""

import torch
import torch.nn as nn
from src.nncore import (
    # Core layers
    SmartDense,
    AdvancedConv2d,
    
    # Attention
    MultiHeadAttention,
    LinearAttention,
    
    # Multimodal
    SpeechEncoder,
    VisionProcessor,
    TextProcessor,
    CrossModalFusion,
    
    # Memory
    EpisodicMemory,
    WorkingMemoryBuffer,
    HierarchicalMemory,
    
    # Adaptive
    AdaptiveComputation,
    MetaLearningModule,
    EvolutionaryLayer
)
from src.nncore.utils import DeviceManager

# Set random seed for reproducibility
torch.manual_seed(42)

def demonstrate_core_layers():
    """Demonstrate core layer functionality."""
    print("\n=== Core Layers ===")
    print("The package provides enhanced versions of standard neural network layers.")
    
    # Create a SmartDense layer
    dense = SmartDense(
        in_features=64,
        out_features=32,
        activation='relu',
        dropout=0.1,
        initialization='xavier_uniform'
    )
    
    # Create an AdvancedConv2d layer
    conv = AdvancedConv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        padding='same',
        use_spectral_norm=True
    )
    
    # Test the layers
    x_dense = torch.randn(16, 64)
    x_conv = torch.randn(16, 3, 32, 32)
    
    out_dense = dense(x_dense)
    out_conv = conv(x_conv)
    
    print(f"Dense output shape: {out_dense.shape}")
    print(f"Conv output shape: {out_conv.shape}")

def demonstrate_attention():
    """Demonstrate attention mechanisms."""
    print("\n=== Attention Mechanisms ===")
    print("The package includes state-of-the-art attention mechanisms.")
    
    # Create attention layers
    mha = MultiHeadAttention(
        embed_dim=256,
        num_heads=8
    )
    
    linear_attn = LinearAttention(
        embed_dim=256
    )
    
    # Test attention
    x = torch.randn(16, 32, 256)
    
    # Multi-head attention
    out_mha, attn_weights = mha(x, x, x)
    print(f"Multi-head attention output shape: {out_mha.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Linear attention
    out_linear = linear_attn(x)
    print(f"Linear attention output shape: {out_linear.shape}")

def demonstrate_multimodal():
    """Demonstrate multimodal processing."""
    print("\n=== Multimodal Processing ===")
    print("The package supports processing multiple modalities.")
    
    # Create multimodal processors
    speech = SpeechEncoder(
        input_dim=80,
        hidden_dim=256
    )
    
    vision = VisionProcessor(
        image_size=(224, 224),
        hidden_dim=256
    )
    
    text = TextProcessor(
        vocab_size=30000,
        hidden_dim=256
    )
    
    fusion = CrossModalFusion(
        hidden_dim=256
    )
    
    # Test multimodal processing
    speech_input = torch.randn(8, 1000, 80)
    vision_input = torch.randn(8, 3, 224, 224)
    text_input = torch.randint(0, 30000, (8, 50))
    
    speech_features, _ = speech(speech_input)
    vision_features, _ = vision(vision_input)
    text_features, _ = text(text_input)
    
    fused = fusion(
        speech_features=speech_features,
        vision_features=vision_features,
        text_features=text_features
    )
    
    print(f"Fused features shape: {fused.shape}")

def demonstrate_memory():
    """Demonstrate memory management."""
    print("\n=== Memory Management ===")
    print("The package provides various memory mechanisms.")
    
    # Create memory components
    episodic = EpisodicMemory(
        memory_size=100,
        memory_dim=256,
        query_dim=256
    )
    
    working = WorkingMemoryBuffer(
        buffer_size=50,
        memory_dim=256
    )
    
    hierarchical = HierarchicalMemory(
        num_levels=3,
        level_sizes=[100, 50, 25],
        memory_dim=256,
        query_dim=256
    )
    
    # Test memory operations
    query = torch.randn(16, 256)
    
    # Episodic memory
    retrieved, attn, importance = episodic(query)
    print(f"Retrieved from episodic memory: {retrieved.shape}")
    
    # Working memory
    working.write(query.unsqueeze(1))
    read, read_attn = working.read(query)
    print(f"Retrieved from working memory: {read.shape}")
    
    # Hierarchical memory
    combined, level_retrievals, level_weights = hierarchical(query)
    print(f"Combined hierarchical memory: {combined.shape}")

def demonstrate_adaptive():
    """Demonstrate adaptive components."""
    print("\n=== Adaptive Components ===")
    print("The package includes self-improving components.")
    
    # Create adaptive components
    adaptive = AdaptiveComputation(
        input_dim=128,
        hidden_dim=256
    )
    
    meta = MetaLearningModule(
        input_dim=128,
        hidden_dim=256,
        output_dim=64
    )
    
    evolutionary = EvolutionaryLayer(
        input_dim=128,
        hidden_dim=256
    )
    
    # Test adaptive processing
    x = torch.randn(16, 128)
    
    # Adaptive computation
    out_adaptive, halting_probs, steps = adaptive(x)
    print(f"Adaptive computation output: {out_adaptive.shape} in {steps} steps")
    
    # Meta-learning
    out_meta = meta(x)
    print(f"Meta-learning output: {out_meta.shape}")
    
    # Evolution
    out_evo = evolutionary(x)
    evolutionary.evolve()
    print(f"Evolutionary output: {out_evo.shape}")

def demonstrate_device_management():
    """Demonstrate device management."""
    print("\n=== Device Management ===")
    print("All components work seamlessly with different compute devices.")
    
    # Check available devices
    print(f"Available devices: {DeviceManager.available_devices()}")
    
    # Move components to appropriate device
    model = nn.Sequential(
        SmartDense(128, 256),
        MultiHeadAttention(256, 8),
        EpisodicMemory(100, 256, 256),
        AdaptiveComputation(256, 128)
    )
    
    model = DeviceManager.to_device(model)
    print(f"Model device: {next(model.parameters()).device}")

def main():
    """Run all demonstrations."""
    print("Welcome to the Neural Network Building Blocks Tutorial!")
    print("This tutorial introduces the key components of the package.")
    
    demonstrate_core_layers()
    demonstrate_attention()
    demonstrate_multimodal()
    demonstrate_memory()
    demonstrate_adaptive()
    demonstrate_device_management()
    
    print("\nNext Steps:")
    print("1. Check out the examples directory for more complete implementations")
    print("2. See the other tutorials for advanced topics:")
    print("   - Building Complex Architectures")
    print("   - Advanced Memory Systems")
    print("   - Multimodal Applications")
    print("   - Self-Improving Networks")

if __name__ == "__main__":
    main() 