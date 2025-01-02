from typing import Dict, Any
import torch.nn as nn
from .basic import DenseLayer, DropoutLayer, NormalizationLayer
from .attention import MultiHeadAttention, SelfAttention

LAYER_REGISTRY = {
    'dense': DenseLayer,
    'dropout': DropoutLayer,
    'norm': NormalizationLayer,
    'multihead_attention': MultiHeadAttention,
    'self_attention': SelfAttention
}

def get_layer(layer_type: str, **kwargs: Any) -> nn.Module:
    """
    Factory function to create neural network layers.
    
    Args:
        layer_type: Type of layer to create
        **kwargs: Arguments to pass to layer constructor
    
    Returns:
        Instantiated layer
    """
    if layer_type not in LAYER_REGISTRY:
        raise ValueError(f"Unknown layer type: {layer_type}")
    
    return LAYER_REGISTRY[layer_type](**kwargs) 