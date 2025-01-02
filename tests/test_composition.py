"""Tests for layer composition utilities."""

import torch
import pytest
from nncore.composition import (
    LayerConfig,
    LayerFactory,
    BlockComposer,
    ArchitectureGenerator,
    SwarmComposer,
)


def test_layer_factory():
    """Test LayerFactory functionality."""
    factory = LayerFactory()

    # Test dense layer creation
    dense_config = LayerConfig(
        layer_type="dense", params={"in_features": 64, "out_features": 32}
    )
    dense_layer = factory.create_layer(dense_config)
    x = torch.randn(8, 64)
    output = dense_layer(x)
    assert output.shape == (8, 32)

    # Test sequential creation
    configs = [
        LayerConfig("dense", {"in_features": 64, "out_features": 32}, "fc1"),
        LayerConfig("dense", {"in_features": 32, "out_features": 16}, "fc2"),
    ]
    sequential = factory.create_sequential(configs)
    x = torch.randn(8, 64)
    output = sequential(x)
    assert output.shape == (8, 16)

    # Test invalid layer type
    with pytest.raises(ValueError):
        invalid_config = LayerConfig("invalid", {})
        factory.create_layer(invalid_config)


def test_block_composer():
    """Test BlockComposer functionality."""
    factory = LayerFactory()
    composer = BlockComposer(factory)

    # Test simple sequential composition
    configs = [
        LayerConfig("dense", {"in_features": 64, "out_features": 32}),
        LayerConfig("dense", {"in_features": 32, "out_features": 16}),
    ]
    block = composer.compose_block(configs)
    x = torch.randn(8, 64)
    output = block(x)
    assert output.shape == (8, 16)

    # Test custom connections
    configs = [
        LayerConfig("dense", {"in_features": 64, "out_features": 32}),
        LayerConfig("dense", {"in_features": 96, "out_features": 16}),  # 64 + 32 = 96
    ]
    connections = [
        (0, 1),
        (0, 2),
        (1, 2),
    ]  # Input -> Layer1, Input -> Layer2, Layer1 -> Layer2
    block = composer.compose_block(configs, connections)
    x = torch.randn(8, 64)
    output = block(x)
    assert output.shape == (8, 16)

    # Test block caching
    composer.cache_block("test_block", block)
    cached_block = composer.get_cached_block("test_block")
    assert cached_block is block


def test_architecture_generator():
    """Test ArchitectureGenerator functionality."""
    factory = LayerFactory()
    composer = BlockComposer(factory)
    generator = ArchitectureGenerator(factory, composer)

    # Test image architecture generation
    input_shape = (32, 3, 224, 224)  # [B, C, H, W]
    target_shape = (32, 1000)  # [B, num_classes]
    constraints = {"max_layers": 5}

    model = generator.generate_architecture(input_shape, target_shape, constraints)
    x = torch.randn(*input_shape)
    output = model(x)
    assert output.shape == target_shape

    # Test sequence architecture generation
    input_shape = (32, 50, 256)  # [B, L, D]
    target_shape = (32, 10)  # [B, num_classes]

    model = generator.generate_architecture(input_shape, target_shape, constraints)
    x = torch.randn(*input_shape)
    output = model(x)
    assert output.shape == target_shape


def test_swarm_composer():
    """Test SwarmComposer functionality."""
    factory = LayerFactory()
    composer = BlockComposer(factory)
    swarm_composer = SwarmComposer(factory, composer)

    batch_size = 16
    num_agents = 8
    agent_dim = 32
    communication_dim = 64

    network = swarm_composer.create_swarm_network(
        num_agents=num_agents, agent_dim=agent_dim, communication_dim=communication_dim
    )

    x = torch.randn(batch_size, num_agents, agent_dim)
    output = network(x)

    # Check output shape
    assert output.shape == (batch_size, num_agents, agent_dim)

    # Check that different agents produce different outputs
    agent_outputs = output[0]  # Take first batch
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            assert not torch.allclose(agent_outputs[i], agent_outputs[j])


if __name__ == "__main__":
    pytest.main([__file__])
