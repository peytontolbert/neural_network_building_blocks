"""
Neural Network Building Blocks - A comprehensive library of composable neural network components.

This package provides advanced neural network components designed for maximum composability,
with special consideration for agent-based architectures and swarm intelligence systems.
"""

__version__ = "0.1.0"

from . import core_layers
from . import attention
from . import norm_reg
from . import advanced_blocks
from . import composition
from . import agent_blocks
from . import multimodal
from . import memory
from . import adaptive

__all__ = [
    "core_layers",
    "attention",
    "norm_reg",
    "advanced_blocks",
    "composition",
    "agent_blocks",
    "multimodal",
    "memory",
    "adaptive",
]
