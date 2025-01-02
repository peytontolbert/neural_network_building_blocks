import torch
import torch.nn as nn
from typing import Optional, Union, Callable

class DenseLayer(nn.Module):
    """Configurable dense layer with activation and normalization."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Union[str, Callable] = 'relu',
        dropout: float = 0.0,
        batch_norm: bool = False
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Linear transformation
        self.layers.append(nn.Linear(in_features, out_features))
        
        # Batch normalization
        if batch_norm:
            self.layers.append(nn.BatchNorm1d(out_features))
        
        # Activation
        if isinstance(activation, str):
            activation = getattr(nn, activation.upper())()
        self.layers.append(activation)
        
        # Dropout
        if dropout > 0:
            self.layers.append(nn.Dropout(dropout))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DropoutLayer(nn.Module):
    """Enhanced dropout layer with optional spatial dropout."""
    
    def __init__(self, p: float = 0.5, spatial: bool = False):
        super().__init__()
        self.p = p
        self.spatial = spatial
        self.dropout = (nn.Dropout2d if spatial else nn.Dropout)(p)
    
    def forward(self, x):
        return self.dropout(x)

class NormalizationLayer(nn.Module):
    """Flexible normalization layer supporting multiple types."""
    
    def __init__(
        self,
        num_features: int,
        norm_type: str = 'batch',
        **kwargs
    ):
        super().__init__()
        
        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(num_features, **kwargs)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(num_features, **kwargs)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm1d(num_features, **kwargs)
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")
    
    def forward(self, x):
        return self.norm(x) 