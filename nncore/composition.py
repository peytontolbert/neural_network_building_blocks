"""Layer composition and architecture generation utilities."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from .core_layers import SmartDense, AdvancedConv2d
from .attention import MultiHeadAttention, LinearAttention
from .advanced_blocks import (
    EnhancedResidualBlock,
    DenseBlock,
    FeaturePyramidBlock,
    DynamicRoutingBlock
)

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
            'dense': SmartDense,
            'conv2d': AdvancedConv2d,
            'attention': MultiHeadAttention,
            'linear_attention': LinearAttention,
            'residual': EnhancedResidualBlock,
            'dense_block': DenseBlock,
            'pyramid': FeaturePyramidBlock,
            'dynamic_routing': DynamicRoutingBlock
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
        layers = []
        for config in configs:
            layer = self.create_layer(config)
            if config.name:
                layers.append((config.name, layer))
            else:
                layers.append(layer)
        return nn.Sequential(*layers)

class BlockComposer:
    """Advanced block composition with dependency management."""
    
    def __init__(self, factory: LayerFactory):
        self.factory = factory
        self.block_cache = {}
    
    def compose_block(
        self,
        configs: List[LayerConfig],
        connections: Optional[List[Tuple[int, int]]] = None
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
                
                for i, layer in enumerate(self.layers, 1):
                    inputs = [outputs[j] for j, k in self.connections if k == i]
                    if len(inputs) == 1:
                        outputs[i] = layer(inputs[0])
                    else:
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
    """Neural architecture search and generation."""
    
    def __init__(self, factory: LayerFactory, composer: BlockComposer):
        self.factory = factory
        self.composer = composer
        
    def generate_architecture(
        self,
        input_shape: Tuple[int, ...],
        target_shape: Tuple[int, ...],
        constraints: Dict[str, Any]
    ) -> nn.Module:
        """Generate an architecture meeting the constraints."""
        # This is a placeholder for a more sophisticated architecture search
        configs = []
        
        # Add initial convolution if dealing with image data
        if len(input_shape) == 4:  # [B, C, H, W]
            configs.append(LayerConfig(
                'conv2d',
                {
                    'in_channels': input_shape[1],
                    'out_channels': 64,
                    'kernel_size': 3,
                    'padding': 1
                }
            ))
            
        # Add attention for sequence data
        elif len(input_shape) == 3:  # [B, L, D]
            configs.append(LayerConfig(
                'attention',
                {
                    'embed_dim': input_shape[2],
                    'num_heads': 8
                }
            ))
            
        # Add final layer to match target shape
        if len(target_shape) == 2:  # [B, D]
            configs.append(LayerConfig(
                'dense',
                {
                    'in_features': 64,
                    'out_features': target_shape[1]
                }
            ))
            
        return self.composer.compose_block(configs)

class SwarmComposer:
    """Swarm network generation and composition."""
    
    def __init__(self, factory: LayerFactory, composer: BlockComposer):
        self.factory = factory
        self.composer = composer
        
    def create_swarm_network(
        self,
        num_agents: int,
        agent_dim: int,
        communication_dim: int
    ) -> nn.Module:
        """Create a swarm neural network."""
        class SwarmNetwork(nn.Module):
            def __init__(self, agent_block, communication_block):
                super().__init__()
                self.agent_block = agent_block
                self.communication_block = communication_block
                
            def forward(self, x):
                # x shape: [batch_size, num_agents, agent_dim]
                agent_states = self.agent_block(x)
                
                # Communication phase
                batch_size = x.size(0)
                comm_shape = (batch_size, num_agents, num_agents, communication_dim)
                comm_tensor = agent_states.unsqueeze(2).expand(-1, -1, num_agents, -1)
                comm_tensor = self.communication_block(comm_tensor)
                
                # Aggregate communication
                final_states = agent_states + comm_tensor.mean(dim=2)
                return final_states
                
        # Create agent processing block
        agent_configs = [
            LayerConfig('dense', {'in_features': agent_dim, 'out_features': communication_dim}),
            LayerConfig('dense', {'in_features': communication_dim, 'out_features': agent_dim})
        ]
        agent_block = self.composer.compose_block(agent_configs)
        
        # Create communication block
        comm_configs = [
            LayerConfig('attention', {
                'embed_dim': communication_dim,
                'num_heads': 4,
                'dropout': 0.1
            })
        ]
        communication_block = self.composer.compose_block(comm_configs)
        
        return SwarmNetwork(agent_block, communication_block) 