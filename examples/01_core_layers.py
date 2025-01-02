"""Example demonstrating the usage of core layers from the nncore package."""

import torch
import torch.nn as nn
from src.nncore.core_layers import SmartDense, AdvancedConv2d
from src.nncore.utils import DeviceManager

# Set random seed for reproducibility
torch.manual_seed(42)


def demonstrate_smart_dense():
    """Demonstrate SmartDense layer functionality."""
    print("\n=== SmartDense Layer Demo ===")

    # Create a SmartDense layer
    input_dim = 64
    output_dim = 32
    batch_size = 16

    layer = SmartDense(
        in_features=input_dim,
        out_features=output_dim,
        bias=True,
        dropout=0.1,
        activation="relu",
        initialization="xavier_uniform",
    )

    # Generate sample input
    x = torch.randn(batch_size, input_dim)

    # Forward pass
    output = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test different activation functions
    print("\nTesting different activation functions:")
    activations = ["relu", "gelu", "silu", "tanh"]

    for activation in activations:
        layer = SmartDense(
            in_features=input_dim, out_features=output_dim, activation=activation
        )
        output = layer(x)
        print(f"{activation.upper()} activation output stats:")
        print(f"Mean: {output.mean():.4f}")
        print(f"Std: {output.std():.4f}\n")


def demonstrate_advanced_conv2d():
    """Demonstrate AdvancedConv2d layer functionality."""
    print("\n=== AdvancedConv2d Layer Demo ===")

    # Create an AdvancedConv2d layer
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    batch_size = 8
    height = 32
    width = 32

    conv_layer = AdvancedConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding="same",
        use_spectral_norm=True,
        activation="relu",
    )

    # Generate sample input
    x = torch.randn(batch_size, in_channels, height, width)

    # Forward pass
    output = conv_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test different padding modes
    print("\nTesting different padding modes:")
    padding_modes = ["same", "valid", "reflect", "replicate"]

    for padding in padding_modes:
        conv_layer = AdvancedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        output = conv_layer(x)
        print(f"Padding mode '{padding}'")
        print(f"Output shape: {output.shape}\n")


class SimpleNet(nn.Module):
    """Simple network combining SmartDense and AdvancedConv2d layers."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            AdvancedConv2d(3, 32, kernel_size=3, padding="same", activation="relu"),
            nn.MaxPool2d(2),
            AdvancedConv2d(32, 64, kernel_size=3, padding="same", activation="relu"),
            nn.MaxPool2d(2),
            AdvancedConv2d(64, 128, kernel_size=3, padding="same", activation="relu"),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            SmartDense(128 * 4 * 4, 512, activation="relu", dropout=0.5),
            SmartDense(512, num_classes, activation=None),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def demonstrate_simple_net():
    """Demonstrate a simple network using our custom layers."""
    print("\n=== SimpleNet Demo ===")

    batch_size = 8

    # Create model and test
    model = SimpleNet()
    x = torch.randn(batch_size, 3, 32, 32)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


def demonstrate_device_management():
    """Demonstrate device management functionality."""
    print("\n=== Device Management Demo ===")

    # Check available devices
    print(f"Available devices: {DeviceManager.available_devices()}")

    # Create model and data
    model = SimpleNet()
    x = torch.randn(8, 3, 32, 32)

    # Move model and data to appropriate device
    model = DeviceManager.to_device(model)
    x = DeviceManager.to_device(x)

    # Run inference
    with torch.no_grad():
        output = model(x)

    print(f"Model device: {next(model.parameters()).device}")
    print(f"Input device: {x.device}")
    print(f"Output device: {output.device}")


def main():
    """Run all demonstrations."""
    demonstrate_smart_dense()
    demonstrate_advanced_conv2d()
    demonstrate_simple_net()
    demonstrate_device_management()


if __name__ == "__main__":
    main()
