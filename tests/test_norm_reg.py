"""Tests for normalization and regularization components."""

import torch
import pytest
from nncore.norm_reg import (
    AdaptiveLayerNorm,
    PopulationBatchNorm,
    StructuredDropout,
    SpectralNorm,
)


def test_adaptive_layer_norm():
    """Test AdaptiveLayerNorm functionality."""
    batch_size = 32
    seq_length = 20
    hidden_size = 64

    layer = AdaptiveLayerNorm(hidden_size)
    x = torch.randn(batch_size, seq_length, hidden_size)

    output = layer(x)
    assert output.shape == (batch_size, seq_length, hidden_size)

    # Test mean and variance
    mean = output.mean(dim=-1)
    var = output.var(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-6)

    # Test without adaptive parameters
    layer_no_adapt = AdaptiveLayerNorm(hidden_size, adaptive_elementwise_affine=False)
    output_no_adapt = layer_no_adapt(x)
    assert output_no_adapt.shape == (batch_size, seq_length, hidden_size)


def test_population_batch_norm():
    """Test PopulationBatchNorm functionality."""
    batch_size = 16
    channels = 32
    height = 28
    width = 28

    layer = PopulationBatchNorm(channels)
    x = torch.randn(batch_size, channels, height, width)

    # Test training mode
    layer.train()
    output_train = layer(x)
    assert output_train.shape == (batch_size, channels, height, width)

    # Test eval mode
    layer.eval()
    output_eval = layer(x)
    assert output_eval.shape == (batch_size, channels, height, width)

    # Test running stats
    assert layer.running_mean is not None
    assert layer.running_var is not None
    assert layer.num_batches_tracked.item() == 1


def test_structured_dropout():
    """Test StructuredDropout functionality."""
    batch_size = 32
    channels = 64
    height = 32
    width = 32

    layer = StructuredDropout(p=0.5, structured_dim=1)
    x = torch.randn(batch_size, channels, height, width)

    # Test training mode
    layer.train()
    output_train = layer(x)
    assert output_train.shape == (batch_size, channels, height, width)

    # Test eval mode (should return input unchanged)
    layer.eval()
    output_eval = layer(x)
    assert torch.equal(output_eval, x)

    # Test adaptive dropout
    adaptive_layer = StructuredDropout(p=0.5, adaptive=True)
    output_adaptive = adaptive_layer(x)
    assert output_adaptive.shape == (batch_size, channels, height, width)


def test_spectral_norm():
    """Test SpectralNorm functionality."""
    batch_size = 16
    in_features = 32
    out_features = 64

    linear = torch.nn.Linear(in_features, out_features)
    layer = SpectralNorm(linear)

    x = torch.randn(batch_size, in_features)
    output = layer(x)

    assert output.shape == (batch_size, out_features)

    # Test power iteration
    u_before = getattr(linear, "weight_u").clone()
    _ = layer(x)
    u_after = getattr(linear, "weight_u")

    # Check that power iteration updates u
    assert not torch.equal(u_before, u_after)


if __name__ == "__main__":
    pytest.main([__file__])
