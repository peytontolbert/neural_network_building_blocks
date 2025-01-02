"""Utility functions imported from PyTorch Basics Library."""

import torch
import numpy as np
from typing import Optional, Union, List, Any, Dict
from pytorch_basics_library.device_management import device_manager
from pytorch_basics_library.tensor_utils import tensor_ops
from pytorch_basics_library.initializers import (
    xavier_uniform, xavier_normal,
    kaiming_normal, kaiming_uniform,
    orthogonal, initialize_model
)

class DeviceManager:
    """Device management utilities."""
    
    _instance = None
    _default_device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
        return cls._instance
    
    @staticmethod
    def get_default_device():
        """Get the default device (CPU or CUDA)."""
        if DeviceManager._default_device is None:
            DeviceManager._default_device = device_manager.get_device()
        return DeviceManager._default_device
        
    @staticmethod
    def to_device(tensor: Any, device=None) -> Any:
        """Move tensor or module to specified device."""
        if device is None:
            device = DeviceManager.get_default_device()
            
        # Handle device string/index
        if isinstance(device, (str, int)):
            device = torch.device(device)
            
        # Move tensor to device
        if isinstance(tensor, (torch.Tensor, torch.nn.Module)):
            return tensor.to(device)
        elif isinstance(tensor, dict):
            return {k: DeviceManager.to_device(v, device) for k, v in tensor.items()}
        elif isinstance(tensor, (list, tuple)):
            return type(tensor)(DeviceManager.to_device(t, device) for t in tensor)
        return tensor
        
    @staticmethod
    def get_device():
        """Get current device."""
        return DeviceManager.get_default_device()
        
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get GPU memory statistics."""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'cached': 0}
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**2,
            'cached': torch.cuda.memory_reserved() / 1024**2
        }

class TensorOps:
    """Tensor operation utilities."""
    
    @staticmethod
    def create_tensor(shape_or_data, device=None, dtype=None, requires_grad=None):
        """Create a tensor with specified configuration."""
        if device is None:
            device = DeviceManager.get_default_device()
            
        # Handle different input types
        if isinstance(shape_or_data, (int, tuple, list)):
            if isinstance(shape_or_data, int):
                shape_or_data = [shape_or_data]
            tensor = torch.zeros(shape_or_data, dtype=dtype)
        elif isinstance(shape_or_data, (torch.Tensor, np.ndarray)):
            tensor = torch.as_tensor(shape_or_data, dtype=dtype)
        elif isinstance(shape_or_data, (float, int)):
            tensor = torch.tensor(shape_or_data, dtype=dtype)
        else:
            raise TypeError("Data must be a shape (int/tuple/list), tensor-like object, or scalar")
            
        # Move to device and set requires_grad
        tensor = DeviceManager.to_device(tensor, device)
        if requires_grad is not None:
            tensor.requires_grad_(requires_grad)
        return tensor
        
    @staticmethod
    def normalize(tensor, dim=-1, eps=1e-8):
        """Normalize tensor along dimension."""
        mean = tensor.mean(dim=dim, keepdim=True)
        std = tensor.std(dim=dim, keepdim=True, unbiased=False)
        return (tensor - mean) / (std + eps)

class Initializer:
    """Base class for weight initializers."""
    def __init__(self, fn, **kwargs):
        self.fn = fn
        self.kwargs = kwargs
        
    def __call__(self, tensor):
        if tensor is None:
            return None
        device = tensor.device
        tensor = tensor.cpu()
        tensor = self.fn(tensor, **self.kwargs)
        return tensor.to(device)
        
    def initialize(self, tensor):
        return self(tensor)

def wrap_initializer(fn):
    """Wrap initialization function to provide consistent interface."""
    def wrapper(*args, **kwargs):
        return Initializer(fn, **kwargs)
    wrapper.initialize = lambda tensor, **kwargs: Initializer(fn, **kwargs)(tensor)
    return wrapper

# Map initializer functions with proper parameter handling
WeightInitializer = {
    'xavier_uniform': wrap_initializer(xavier_uniform),
    'xavier_normal': wrap_initializer(xavier_normal),
    'kaiming_normal': wrap_initializer(kaiming_normal),
    'kaiming_uniform': wrap_initializer(kaiming_uniform),
    'orthogonal': wrap_initializer(orthogonal),
    'initialize_model': initialize_model
}

__all__ = ['DeviceManager', 'TensorOps', 'WeightInitializer'] 