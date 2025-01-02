"""Tests for imported utility functions."""

import torch
import torch.nn as nn
import numpy as np
import pytest
from nncore.utils import DeviceManager, TensorOps, WeightInitializer


def test_device_manager():
    """Test imported DeviceManager functionality."""
    # Test device selection
    device = DeviceManager.get_device()
    assert isinstance(device, torch.device)

    # Test moving objects to device
    tensor = torch.randn(10, 10)
    moved_tensor = DeviceManager.to_device(tensor)
    assert moved_tensor.device == device

    # Test memory stats
    stats = DeviceManager.get_memory_stats()
    if torch.cuda.is_available():
        assert isinstance(stats, dict)
        assert "allocated" in stats
    else:
        assert isinstance(stats, dict)


def test_tensor_ops():
    """Test imported TensorOps functionality."""
    # Test tensor creation
    data = [1, 2, 3, 4]
    tensor = TensorOps.create_tensor(data)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.device == DeviceManager.get_device()

    # Test normalization
    x = torch.randn(10, 5)
    normalized = TensorOps.normalize(x)
    assert torch.allclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(normalized.std(), torch.tensor(1.0), atol=1e-6)


def test_weight_initializer():
    """Test imported WeightInitializer functionality."""

    # Create a simple model for testing
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 5)

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return x

    model = TestModel()

    # Test initialization
    initial_weights = {
        name: param.clone()
        for name, param in model.named_parameters()
        if "weight" in name
    }

    # Initialize with Kaiming normal
    model = WeightInitializer["initialize_model"](
        model, WeightInitializer["kaiming_normal"]
    )

    # Check that weights have changed
    for name, param in model.named_parameters():
        if "weight" in name:
            assert not torch.equal(param, initial_weights[name])


if __name__ == "__main__":
    pytest.main([__file__])
