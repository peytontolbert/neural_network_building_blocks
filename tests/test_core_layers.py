"""Tests for core neural network layers."""

import torch
import pytest
from nncore.core_layers import SmartDense, AdvancedConv2d


def test_smart_dense():
    """Test SmartDense layer basic functionality."""
    batch_size = 32
    in_features = 20
    out_features = 10

    layer = SmartDense(in_features, out_features)
    x = torch.randn(batch_size, in_features)

    output = layer(x)
    assert output.shape == (batch_size, out_features)

    # Test with activation
    layer_with_activation = SmartDense(
        in_features, out_features, activation=torch.nn.ReLU()
    )
    output_activated = layer_with_activation(x)
    assert output_activated.shape == (batch_size, out_features)
    assert torch.all(output_activated >= 0)  # ReLU check


def test_advanced_conv2d():
    """Test AdvancedConv2d layer basic functionality."""
    batch_size = 16
    in_channels = 3
    out_channels = 64
    height = 32
    width = 32

    layer = AdvancedConv2d(in_channels, out_channels, kernel_size=3, attention=True)
    x = torch.randn(batch_size, in_channels, height, width)

    output = layer(x)
    assert output.shape == (batch_size, out_channels, height, width)

    # Test separable convolution
    sep_layer = AdvancedConv2d(in_channels, out_channels, kernel_size=3, separable=True)
    output_sep = sep_layer(x)
    assert output_sep.shape == (batch_size, out_channels, height, width)


if __name__ == "__main__":
    pytest.main([__file__])
