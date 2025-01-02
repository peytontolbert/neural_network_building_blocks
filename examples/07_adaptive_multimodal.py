"""Example demonstrating the combination of adaptive and multimodal components."""

import torch
import torch.nn as nn
from src.nncore.multimodal import (
    SpeechEncoder,
    VisionProcessor,
    TextProcessor,
    CrossModalFusion,
)
from src.nncore.adaptive import (
    AdaptiveComputation,
    MetaLearningModule,
    EvolutionaryLayer,
)
from src.nncore.utils import DeviceManager

# Set random seed for reproducibility
torch.manual_seed(42)


class AdaptiveModalityProcessor(nn.Module):
    """Adaptive processor for a single modality."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_evolutionary_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Adaptive computation for dynamic processing depth
        self.adaptive = AdaptiveComputation(input_dim=input_dim, hidden_dim=hidden_dim)

        # Meta-learning for fast adaptation
        self.meta = MetaLearningModule(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim
        )

        # Evolutionary layers for self-modification
        self.evolutionary = nn.ModuleList(
            [
                EvolutionaryLayer(input_dim=hidden_dim, hidden_dim=hidden_dim)
                for _ in range(num_evolutionary_layers)
            ]
        )

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamic computation
        x, _, _ = self.adaptive(x)

        # Meta-learned processing
        x = self.meta(x)

        # Evolutionary processing
        for layer in self.evolutionary:
            x = layer(x)

        # Final projection
        x = self.output(x)

        return x

    def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor):
        """Adapt the processor using support data."""
        # Meta-learning adaptation
        self.meta.adapt(support_x, support_y)

        # Evolve layers
        for layer in self.evolutionary:
            layer.evolve()


class AdaptiveMultimodalNetwork(nn.Module):
    """Network combining adaptive processing with multimodal fusion."""

    def __init__(
        self,
        speech_dim: int = 80,
        vision_dim: int = 2048,
        text_dim: int = 512,
        hidden_dim: int = 512,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Modality-specific processors
        self.speech_encoder = SpeechEncoder(input_dim=speech_dim, hidden_dim=hidden_dim)

        self.vision_processor = VisionProcessor(
            image_size=(224, 224), hidden_dim=hidden_dim
        )

        self.text_processor = TextProcessor(vocab_size=30000, hidden_dim=hidden_dim)

        # Adaptive processors
        self.speech_adaptive = AdaptiveModalityProcessor(
            input_dim=hidden_dim, hidden_dim=hidden_dim
        )

        self.vision_adaptive = AdaptiveModalityProcessor(
            input_dim=hidden_dim, hidden_dim=hidden_dim
        )

        self.text_adaptive = AdaptiveModalityProcessor(
            input_dim=hidden_dim, hidden_dim=hidden_dim
        )

        # Cross-modal fusion
        self.fusion = CrossModalFusion(
            hidden_dim=hidden_dim, num_heads=8, dropout=dropout
        )

        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(
        self,
        speech_input: Optional[torch.Tensor] = None,
        vision_input: Optional[torch.Tensor] = None,
        text_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Process each modality if available
        speech_features = None
        vision_features = None
        text_features = None

        if speech_input is not None:
            speech_features, _ = self.speech_encoder(speech_input)
            speech_features = self.speech_adaptive(speech_features)

        if vision_input is not None:
            vision_features, _ = self.vision_processor(vision_input)
            vision_features = self.vision_adaptive(vision_features)

        if text_input is not None:
            text_features, _ = self.text_processor(text_input)
            text_features = self.text_adaptive(text_features)

        # Fuse modalities
        fused_features = self.fusion(
            speech_features=speech_features,
            vision_features=vision_features,
            text_features=text_features,
        )

        # Classify
        output = self.classifier(fused_features)
        return output

    def adapt(
        self,
        support_speech: Optional[torch.Tensor] = None,
        support_vision: Optional[torch.Tensor] = None,
        support_text: Optional[torch.Tensor] = None,
        support_y: torch.Tensor = None,
    ):
        """Adapt all modality processors."""
        if support_speech is not None:
            speech_features, _ = self.speech_encoder(support_speech)
            self.speech_adaptive.adapt(speech_features, support_y)

        if support_vision is not None:
            vision_features, _ = self.vision_processor(support_vision)
            self.vision_adaptive.adapt(vision_features, support_y)

        if support_text is not None:
            text_features, _ = self.text_processor(support_text)
            self.text_adaptive.adapt(text_features, support_y)


def demonstrate_adaptive_modality_processor():
    """Demonstrate AdaptiveModalityProcessor functionality."""
    print("\n=== AdaptiveModalityProcessor Demo ===")

    # Create processor
    batch_size = 16
    input_dim = 256
    hidden_dim = 512

    processor = AdaptiveModalityProcessor(input_dim=input_dim, hidden_dim=hidden_dim)

    # Generate sample data
    x = torch.randn(batch_size, input_dim)
    support_x = torch.randn(batch_size, input_dim)
    support_y = torch.randn(batch_size, hidden_dim)

    # Initial forward pass
    output = processor(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test adaptation
    print("\nTesting adaptation:")
    processor.adapt(support_x, support_y)
    output_adapted = processor(x)
    print(f"Adaptation changed output: {not torch.equal(output, output_adapted)}")


def demonstrate_adaptive_multimodal_network():
    """Demonstrate AdaptiveMultimodalNetwork functionality."""
    print("\n=== AdaptiveMultimodalNetwork Demo ===")

    # Create network
    batch_size = 8
    speech_length = 1000
    num_classes = 10

    network = AdaptiveMultimodalNetwork(num_classes=num_classes)

    # Generate sample inputs
    speech_input = torch.randn(batch_size, speech_length, 80)
    vision_input = torch.randn(batch_size, 3, 224, 224)
    text_input = torch.randint(0, 30000, (batch_size, 50))

    # Test with all modalities
    output_all = network(
        speech_input=speech_input, vision_input=vision_input, text_input=text_input
    )
    print(f"Output with all modalities shape: {output_all.shape}")

    # Test with subset of modalities
    output_vision_text = network(vision_input=vision_input, text_input=text_input)
    print(f"Output with vision-text only shape: {output_vision_text.shape}")

    # Test adaptation
    print("\nTesting adaptation:")
    support_y = torch.randn(batch_size, num_classes)
    network.adapt(
        support_speech=speech_input,
        support_vision=vision_input,
        support_text=text_input,
        support_y=support_y,
    )

    output_adapted = network(
        speech_input=speech_input, vision_input=vision_input, text_input=text_input
    )
    print(f"Adaptation changed output: {not torch.equal(output_all, output_adapted)}")

    # Test with device management
    network = DeviceManager.to_device(network)
    speech_input = DeviceManager.to_device(speech_input)
    vision_input = DeviceManager.to_device(vision_input)
    text_input = DeviceManager.to_device(text_input)

    output = network(
        speech_input=speech_input, vision_input=vision_input, text_input=text_input
    )

    print(f"\nDevice management:")
    print(f"Model device: {next(network.parameters()).device}")
    print(
        f"Input devices: {speech_input.device}, {vision_input.device}, {text_input.device}"
    )
    print(f"Output device: {output.device}")


def main():
    """Run all demonstrations."""
    demonstrate_adaptive_modality_processor()
    demonstrate_adaptive_multimodal_network()


if __name__ == "__main__":
    main()
