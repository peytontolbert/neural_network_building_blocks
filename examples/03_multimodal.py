"""Example demonstrating the usage of multimodal components from the nncore package."""

import torch
import torch.nn as nn
from src.nncore.multimodal import (
    SpeechEncoder,
    VisionProcessor,
    TextProcessor,
    CrossModalFusion
)
from src.nncore.utils import DeviceManager

# Set random seed for reproducibility
torch.manual_seed(42)

def demonstrate_speech_encoder():
    """Demonstrate SpeechEncoder functionality."""
    print("\n=== SpeechEncoder Demo ===")
    
    # Create a SpeechEncoder
    batch_size = 8
    seq_length = 1000  # 1 second of audio at 1kHz
    num_mels = 80
    hidden_dim = 256
    
    encoder = SpeechEncoder(
        input_dim=num_mels,
        hidden_dim=hidden_dim,
        num_layers=3,
        dropout=0.1
    )
    
    # Generate sample mel spectrogram input
    x = torch.randn(batch_size, seq_length, num_mels)
    
    # Forward pass
    output, hidden = encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden state shape: {hidden.shape}")
    
    # Test with different sequence lengths
    print("\nTesting variable sequence lengths:")
    lengths = torch.randint(100, seq_length, (batch_size,))
    output_packed, hidden_packed = encoder(x, lengths)
    print(f"Packed output shape: {output_packed.shape}")

def demonstrate_vision_processor():
    """Demonstrate VisionProcessor functionality."""
    print("\n=== VisionProcessor Demo ===")
    
    # Create a VisionProcessor
    batch_size = 16
    channels = 3
    height = 224
    width = 224
    patch_size = 16
    hidden_dim = 768
    
    processor = VisionProcessor(
        image_size=(height, width),
        patch_size=patch_size,
        in_channels=channels,
        hidden_dim=hidden_dim,
        num_layers=6
    )
    
    # Generate sample image input
    x = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    output, patch_embeddings = processor(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Patch embeddings shape: {patch_embeddings.shape}")
    
    # Test attention map visualization
    attention_maps = processor.get_attention_maps()
    print(f"\nAttention maps shapes:")
    for i, attn_map in enumerate(attention_maps):
        print(f"Layer {i}: {attn_map.shape}")

def demonstrate_text_processor():
    """Demonstrate TextProcessor functionality."""
    print("\n=== TextProcessor Demo ===")
    
    # Create a TextProcessor
    batch_size = 32
    seq_length = 50
    vocab_size = 30000
    hidden_dim = 512
    
    processor = TextProcessor(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=4,
        max_seq_length=100
    )
    
    # Generate sample token input
    tokens = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass
    output, embeddings = processor(tokens)
    
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test with attention mask
    print("\nTesting with attention mask:")
    mask = torch.ones_like(tokens).bool()
    mask[:, seq_length//2:] = False  # Mask out second half
    output_masked, _ = processor(tokens, attention_mask=mask)
    print(f"Masked output shape: {output_masked.shape}")

def demonstrate_cross_modal_fusion():
    """Demonstrate CrossModalFusion functionality."""
    print("\n=== CrossModalFusion Demo ===")
    
    # Create components
    batch_size = 8
    hidden_dim = 512
    
    speech_encoder = SpeechEncoder(
        input_dim=80,
        hidden_dim=hidden_dim
    )
    
    vision_processor = VisionProcessor(
        image_size=(224, 224),
        hidden_dim=hidden_dim
    )
    
    text_processor = TextProcessor(
        vocab_size=30000,
        hidden_dim=hidden_dim
    )
    
    fusion = CrossModalFusion(
        hidden_dim=hidden_dim,
        num_heads=8,
        dropout=0.1
    )
    
    # Generate sample inputs
    speech_input = torch.randn(batch_size, 1000, 80)
    vision_input = torch.randn(batch_size, 3, 224, 224)
    text_input = torch.randint(0, 30000, (batch_size, 50))
    
    # Process each modality
    speech_features, _ = speech_encoder(speech_input)
    vision_features, _ = vision_processor(vision_input)
    text_features, _ = text_processor(text_input)
    
    # Fuse modalities
    fused_features = fusion(
        speech_features=speech_features,
        vision_features=vision_features,
        text_features=text_features
    )
    
    print(f"Speech features shape: {speech_features.shape}")
    print(f"Vision features shape: {vision_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    print(f"Fused features shape: {fused_features.shape}")
    
    # Test with missing modalities
    print("\nTesting with missing modalities:")
    fused_vision_text = fusion(
        vision_features=vision_features,
        text_features=text_features
    )
    print(f"Vision-Text fusion shape: {fused_vision_text.shape}")

class MultimodalClassifier(nn.Module):
    """Example multimodal classifier combining all components."""
    
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Modality processors
        self.speech_encoder = SpeechEncoder(
            input_dim=80,
            hidden_dim=hidden_dim
        )
        
        self.vision_processor = VisionProcessor(
            image_size=(224, 224),
            hidden_dim=hidden_dim
        )
        
        self.text_processor = TextProcessor(
            vocab_size=30000,
            hidden_dim=hidden_dim
        )
        
        # Cross-modal fusion
        self.fusion = CrossModalFusion(
            hidden_dim=hidden_dim,
            num_heads=8,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes)
        )
    
    def forward(
        self,
        speech_input=None,
        vision_input=None,
        text_input=None
    ):
        # Process available modalities
        speech_features = None
        vision_features = None
        text_features = None
        
        if speech_input is not None:
            speech_features, _ = self.speech_encoder(speech_input)
        
        if vision_input is not None:
            vision_features, _ = self.vision_processor(vision_input)
        
        if text_input is not None:
            text_features, _ = self.text_processor(text_input)
        
        # Fuse modalities
        fused_features = self.fusion(
            speech_features=speech_features,
            vision_features=vision_features,
            text_features=text_features
        )
        
        # Classify
        output = self.classifier(fused_features)
        return output

def demonstrate_multimodal_classifier():
    """Demonstrate the complete multimodal classifier."""
    print("\n=== MultimodalClassifier Demo ===")
    
    # Create classifier
    batch_size = 8
    num_classes = 10
    hidden_dim = 512
    
    classifier = MultimodalClassifier(
        num_classes=num_classes,
        hidden_dim=hidden_dim
    )
    
    # Generate sample inputs
    speech_input = torch.randn(batch_size, 1000, 80)
    vision_input = torch.randn(batch_size, 3, 224, 224)
    text_input = torch.randint(0, 30000, (batch_size, 50))
    
    # Test with all modalities
    output_all = classifier(
        speech_input=speech_input,
        vision_input=vision_input,
        text_input=text_input
    )
    print(f"Output with all modalities shape: {output_all.shape}")
    
    # Test with subset of modalities
    output_vision_text = classifier(
        vision_input=vision_input,
        text_input=text_input
    )
    print(f"Output with vision-text only shape: {output_vision_text.shape}")
    
    # Move to appropriate device
    classifier = DeviceManager.to_device(classifier)
    speech_input = DeviceManager.to_device(speech_input)
    vision_input = DeviceManager.to_device(vision_input)
    text_input = DeviceManager.to_device(text_input)
    
    output = classifier(
        speech_input=speech_input,
        vision_input=vision_input,
        text_input=text_input
    )
    
    print(f"\nDevice management:")
    print(f"Model device: {next(classifier.parameters()).device}")
    print(f"Input devices: {speech_input.device}, {vision_input.device}, {text_input.device}")
    print(f"Output device: {output.device}")

def main():
    """Run all demonstrations."""
    demonstrate_speech_encoder()
    demonstrate_vision_processor()
    demonstrate_text_processor()
    demonstrate_cross_modal_fusion()
    demonstrate_multimodal_classifier()

if __name__ == "__main__":
    main() 