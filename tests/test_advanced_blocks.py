"""Tests for advanced neural network blocks."""

import torch
import torch.nn as nn
import pytest
from nncore.advanced_blocks import (
    EnhancedResidualBlock,
    DenseBlock,
    FeaturePyramidBlock,
    DynamicRoutingBlock,
)
from nncore.utils import DeviceManager


def test_enhanced_residual_block():
    """Test EnhancedResidualBlock functionality."""
    device = DeviceManager.get_default_device()
    batch_size = 8
    channels = 64
    height = 32
    width = 32

    # Test basic block
    block = EnhancedResidualBlock(channels, device=device)
    x = torch.randn(batch_size, channels, height, width, device=device)
    output = block(x)

    assert output.shape == (batch_size, channels * 4, height, width)

    # Test with attention and dropout
    block_with_features = EnhancedResidualBlock(
        channels, use_attention=True, dropout=0.1, device=device
    )
    output_with_features = block_with_features(x)

    assert output_with_features.shape == (batch_size, channels * 4, height, width)

    # Test with stride
    block_stride = EnhancedResidualBlock(
        channels,
        stride=2,
        downsample=nn.Conv2d(
            channels, channels * 4, kernel_size=1, stride=2, device=device
        ),
        device=device,
    )
    output_stride = block_stride(x)

    assert output_stride.shape == (batch_size, channels * 4, height // 2, width // 2)


def test_dense_block():
    """Test DenseBlock functionality."""
    device = DeviceManager.get_default_device()
    batch_size = 8
    in_channels = 64
    growth_rate = 32
    num_layers = 4
    height = 32
    width = 32

    block = DenseBlock(
        in_channels, growth_rate=growth_rate, num_layers=num_layers, device=device
    )
    x = torch.randn(batch_size, in_channels, height, width, device=device)
    output = block(x)

    expected_channels = in_channels + growth_rate * num_layers
    assert output.shape == (batch_size, expected_channels, height, width)

    # Test with pruning
    block_with_pruning = DenseBlock(
        in_channels,
        growth_rate=growth_rate,
        num_layers=num_layers,
        pruning_threshold=0.1,
        device=device,
    )
    output_pruned = block_with_pruning(x)

    assert output_pruned.shape == (batch_size, expected_channels, height, width)
    assert (output_pruned == 0).any()  # Some values should be pruned


def test_feature_pyramid_block():
    """Test FeaturePyramidBlock functionality."""
    device = DeviceManager.get_default_device()
    batch_size = 8
    in_channels = [64, 128, 256]
    out_channels = 256

    # Create input features with different spatial dimensions
    features = [
        torch.randn(batch_size, ch, 32 // (2**i), 32 // (2**i), device=device)
        for i, ch in enumerate(in_channels)
    ]

    block = FeaturePyramidBlock(in_channels, out_channels, device=device)
    outputs = block(features)

    assert len(outputs) == len(features)
    for i, output in enumerate(outputs):
        assert output.shape == (batch_size, out_channels, 32 // (2**i), 32 // (2**i))

    # Test without residual connections
    block_no_residual = FeaturePyramidBlock(
        in_channels, out_channels, use_residual=False, device=device
    )
    outputs_no_residual = block_no_residual(features)

    assert len(outputs_no_residual) == len(features)


def test_dynamic_routing_block():
    """Test DynamicRoutingBlock functionality."""
    device = DeviceManager.get_default_device()
    batch_size = 8
    channels = 64
    height = 32
    width = 32
    num_experts = 4

    block = DynamicRoutingBlock(channels, num_experts=num_experts, device=device)
    x = torch.randn(batch_size, channels, height, width, device=device)
    output, routing_weights = block(x)

    assert output.shape == (batch_size, channels, height, width)
    assert routing_weights.shape == (batch_size, num_experts)

    # Test routing weights sum to 1
    routing_sum = routing_weights.sum(dim=1)
    assert torch.allclose(routing_sum, torch.ones_like(routing_sum))

    # Test with different temperature
    block_temp = DynamicRoutingBlock(
        channels, num_experts=num_experts, temperature=0.5, device=device
    )
    output_temp, routing_weights_temp = block_temp(x)

    assert output_temp.shape == (batch_size, channels, height, width)
    assert not torch.equal(
        routing_weights, routing_weights_temp
    )  # Different temperature should give different weights


if __name__ == "__main__":
    pytest.main([__file__])
