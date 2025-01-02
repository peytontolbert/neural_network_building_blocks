import torch
import torch.nn as nn
from typing import List, Callable, Any

class ResidualBlock(nn.Module):
    """Residual connection wrapper for any layer."""
    
    def __init__(
        self,
        layer_fn: Callable[..., nn.Module],
        *args: Any,
        **kwargs: Any
    ):
        super().__init__()
        self.layer = layer_fn(*args, **kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer(x)

class SequentialBlock(nn.Module):
    """Enhanced sequential block with skip connections."""
    
    def __init__(
        self,
        layers: List[nn.Module],
        residual: bool = False
    ):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.residual = residual
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        for layer in self.layers:
            x = layer(x)
        
        if self.residual:
            x = x + identity
            
        return x 