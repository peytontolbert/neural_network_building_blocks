"""Layer composition and architecture generation utilities."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from collections import OrderedDict
from .core_layers import SmartDense, AdvancedConv2d
from .attention import MultiHeadAttention, LinearAttention
from .advanced_blocks import (
    EnhancedResidualBlock,
    DenseBlock,
    FeaturePyramidBlock,
    DynamicRoutingBlock,
)
from .utils import DeviceManager


@dataclass
class LayerConfig:
    """Configuration for layer creation."""

    layer_type: str
    params: Dict[str, Any]
    name: Optional[str] = None


class LayerFactory:
    """Smart layer creation with configuration management."""

    def __init__(self):
        self.layer_registry = {
            "dense": SmartDense,
            "conv2d": AdvancedConv2d,
            "attention": MultiHeadAttention,
            "linear_attention": LinearAttention,
            "residual": EnhancedResidualBlock,
            "dense_block": DenseBlock,
            "pyramid": FeaturePyramidBlock,
            "dynamic_routing": DynamicRoutingBlock,
        }

    def register_layer(self, name: str, layer_class: type):
        """Register a new layer type."""
        self.layer_registry[name] = layer_class

    def create_layer(self, config: LayerConfig) -> nn.Module:
        """Create a layer from configuration."""
        if config.layer_type not in self.layer_registry:
            raise ValueError(f"Unknown layer type: {config.layer_type}")

        layer_class = self.layer_registry[config.layer_type]
        return layer_class(**config.params)

    def create_sequential(self, configs: List[LayerConfig]) -> nn.Sequential:
        """Create a sequential model from configurations."""
        has_named_layers = any(config.name for config in configs)

        if has_named_layers:
            layers_dict = OrderedDict()
            for config in configs:
                layer = self.create_layer(config)
                name = config.name if config.name else f"layer_{len(layers_dict)}"
                layers_dict[name] = layer
            return nn.Sequential(layers_dict)
        else:
            layers = [self.create_layer(config) for config in configs]
            return nn.Sequential(*layers)


class BlockComposer:
    """Advanced block composition with dependency management."""

    def __init__(self, factory: LayerFactory):
        self.factory = factory
        self.block_cache = {}

    def compose_block(
        self,
        configs: List[LayerConfig],
        connections: Optional[List[Tuple[int, int]]] = None,
    ) -> nn.Module:
        """Compose a block with custom connections."""
        if not connections:
            return self.factory.create_sequential(configs)

        class CustomBlock(nn.Module):
            def __init__(self, layers, connections):
                super().__init__()
                self.layers = nn.ModuleList(layers)
                self.connections = connections

            def forward(self, x):
                outputs = [None] * (len(self.layers) + 1)
                outputs[0] = x
                device = x.device

                for i, layer in enumerate(self.layers, 1):
                    inputs = [outputs[j] for j, k in self.connections if k == i]
                    if len(inputs) == 1:
                        outputs[i] = layer(inputs[0])
                    else:
                        # Ensure all tensors are on the same device
                        inputs = [inp.to(device) for inp in inputs]
                        outputs[i] = layer(torch.cat(inputs, dim=1))

                return outputs[-1]

        layers = [self.factory.create_layer(config) for config in configs]
        return CustomBlock(layers, connections)

    def get_cached_block(self, cache_key: str) -> Optional[nn.Module]:
        """Retrieve a cached block."""
        return self.block_cache.get(cache_key)

    def cache_block(self, cache_key: str, block: nn.Module):
        """Cache a block for reuse."""
        self.block_cache[cache_key] = block


class ArchitectureGenerator:
    """Architecture generation with constraints."""

    def __init__(self, factory: LayerFactory, composer: Optional[BlockComposer] = None):
        self.factory = factory
        self.composer = composer

    def generate_architecture(
        self,
        input_shape: Tuple[int, ...],
        target_shape: Tuple[int, ...],
        constraints: Dict[str, Any],
    ) -> nn.Module:
        """Generate an architecture meeting the constraints."""
        layers = []
        device = DeviceManager.get_default_device()

        # Add initial convolution if dealing with image data
        if len(input_shape) == 4:  # [B, C, H, W]
            # Add convolution layers
            layers.append(
                self.factory.create_layer(
                    LayerConfig(
                        "conv2d",
                        {
                            "in_channels": input_shape[1],
                            "out_channels": 64,
                            "kernel_size": 7,
                            "stride": 2,
                            "padding": 3,
                        },
                    )
                )
            )

            layers.append(
                self.factory.create_layer(
                    LayerConfig(
                        "conv2d",
                        {
                            "in_channels": 64,
                            "out_channels": 128,
                            "kernel_size": 3,
                            "stride": 2,
                            "padding": 1,
                        },
                    )
                )
            )

            layers.append(
                self.factory.create_layer(
                    LayerConfig(
                        "conv2d",
                        {
                            "in_channels": 128,
                            "out_channels": 256,
                            "kernel_size": 3,
                            "stride": 2,
                            "padding": 1,
                        },
                    )
                )
            )

            # Add flatten layer
            layers.append(nn.Flatten(1))

            # Calculate flattened dimension
            h = input_shape[2] // 8  # After 3 stride-2 convolutions
            w = input_shape[3] // 8
            flattened_dim = 256 * h * w

            # Add final dense layer
            layers.append(
                self.factory.create_layer(
                    LayerConfig(
                        "dense",
                        {"in_features": flattened_dim, "out_features": target_shape[1]},
                    )
                )
            )

        # Add attention for sequence data
        elif len(input_shape) == 3:  # [B, L, D]
            # First transform sequence features
            layers.append(
                self.factory.create_layer(
                    LayerConfig(
                        "dense",
                        {
                            "in_features": input_shape[2],
                            "out_features": target_shape[1],
                        },
                    )
                )
            )

            # Add global pooling to reduce sequence length
            class GlobalPool(nn.Module):
                def forward(self, x):
                    return x.mean(dim=1)

            layers.append(GlobalPool())

        # Create sequential model and move to device
        model = nn.Sequential(*layers)
        return DeviceManager.initialize_module(model, device)


class SwarmNetwork(nn.Module):
    def __init__(self, agent_block, communication_block, num_agents):
        super().__init__()
        self.agent_block = agent_block
        self.communication_block = communication_block
        self.num_agents = num_agents

    def forward(self, x):
        # x shape: [batch_size, num_agents, agent_dim]
        device = x.device
        batch_size = x.size(0)

        # Process each agent's state
        agent_states = []
        for i in range(self.num_agents):
            agent_state = self.agent_block(x[:, i])  # [B, agent_dim]
            agent_states.append(agent_state)
        agent_states = torch.stack(agent_states, dim=1)  # [B, N, D]

        # Communication phase
        # Reshape for attention
        query = agent_states.view(batch_size * self.num_agents, 1, -1)  # [B*N, 1, D]
        key_value = agent_states.view(batch_size, self.num_agents, -1)  # [B, N, D]
        key_value = key_value.unsqueeze(1).expand(
            -1, self.num_agents, -1, -1
        )  # [B, N, N, D]
        key_value = key_value.reshape(
            batch_size * self.num_agents, self.num_agents, -1
        )  # [B*N, N, D]

        # Apply attention
        comm_tensor, _ = self.communication_block(
            query=query, key=key_value, value=key_value
        )

        # Reshape back
        comm_tensor = comm_tensor.view(batch_size, self.num_agents, -1)

        # Aggregate communication
        final_states = agent_states + comm_tensor
        return final_states


class SwarmComposer:
    """Composer for swarm-based architectures."""

    def __init__(self, factory: LayerFactory, composer: BlockComposer):
        self.factory = factory
        self.composer = composer

    def create_swarm_network(
        self, num_agents: int, agent_dim: int, communication_dim: int
    ) -> nn.Module:
        """Create a network for swarm-based computation."""
        # Get device
        device = DeviceManager.get_default_device()

        class SwarmNetwork(nn.Module):
            def __init__(self, num_agents, agent_dim, communication_dim, device):
                super().__init__()
                self.num_agents = num_agents
                self.device = device

                # Create separate agent blocks for each agent
                self.agent_blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(agent_dim, communication_dim),
                            nn.ReLU(),
                            nn.LayerNorm(communication_dim),
                        ).to(device)
                        for _ in range(num_agents)
                    ]
                )

                # Create communication block
                self.communication_block = MultiHeadAttention(
                    embed_dim=communication_dim, num_heads=4, dropout=0.1, device=device
                ).to(device)

                # Create post-communication blocks
                self.post_comm_blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(communication_dim, agent_dim),
                            nn.ReLU(),
                            nn.LayerNorm(agent_dim),
                        ).to(device)
                        for _ in range(num_agents)
                    ]
                )

                # Initialize weights differently for each agent
                for i, (agent_block, post_block) in enumerate(
                    zip(self.agent_blocks, self.post_comm_blocks)
                ):
                    scale = (
                        1.0 + (i - num_agents // 2) * 0.1
                    )  # Different scales for diversity
                    for module in agent_block.modules():
                        if isinstance(module, nn.Linear):
                            nn.init.xavier_uniform_(module.weight, gain=scale)
                            if module.bias is not None:
                                nn.init.uniform_(module.bias, -0.1 * scale, 0.1 * scale)
                    for module in post_block.modules():
                        if isinstance(module, nn.Linear):
                            nn.init.xavier_uniform_(module.weight, gain=scale)
                            if module.bias is not None:
                                nn.init.uniform_(module.bias, -0.1 * scale, 0.1 * scale)

                # Move entire network to device
                self.to(device)

            def forward(self, x):
                # x shape: [batch_size, num_agents, agent_dim]
                x = x.to(self.device)
                batch_size = x.size(0)

                # Process each agent's state with its unique block
                agent_states = []
                for i in range(self.num_agents):
                    agent_state = self.agent_blocks[i](x[:, i])  # [B, comm_dim]
                    agent_states.append(agent_state)
                agent_states = torch.stack(agent_states, dim=1)  # [B, N, D]

                # Communication phase
                # All tensors are already on the correct device
                comm_output, _ = self.communication_block(
                    query=agent_states, key=agent_states, value=agent_states
                )

                # Post-communication processing with unique blocks
                outputs = []
                for i in range(self.num_agents):
                    agent_output = self.post_comm_blocks[i](
                        comm_output[:, i]
                    )  # [B, agent_dim]
                    outputs.append(agent_output)

                return torch.stack(outputs, dim=1)  # [B, N, agent_dim]

        # Create and return the network
        network = SwarmNetwork(num_agents, agent_dim, communication_dim, device)
        return DeviceManager.initialize_module(network, device)
