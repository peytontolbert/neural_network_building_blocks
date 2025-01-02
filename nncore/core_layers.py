"""Core neural network layers with advanced functionality and dynamic capabilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from .utils import DeviceManager, TensorOps, WeightInitializer


class SmartDense(nn.Module):
    """Dense layer with dynamic capacity and advanced features."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dynamic_growth: bool = False,
        growth_factor: float = 1.5,
        activation: Optional[nn.Module] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dynamic_growth = dynamic_growth
        self.growth_factor = growth_factor

        # Initialize device if not provided
        device = device or DeviceManager.get_default_device()

        # Use TensorOps for tensor creation
        self.weight = nn.Parameter(
            TensorOps.create_tensor(
                (out_features, in_features), device=device, dtype=dtype
            )
        )

        if bias:
            self.bias = nn.Parameter(
                TensorOps.create_tensor((out_features,), device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.activation = activation
        self.reset_parameters()

        # Move to specified device
        self.to(device)

    def reset_parameters(self):
        """Initialize weights using Kaiming initialization."""
        WeightInitializer["kaiming_normal"](
            self.weight, mode="fan_out", nonlinearity="relu"
        )
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional dynamic growth."""
        # Ensure input is on correct device
        x = DeviceManager.to_device(x, self.weight.device)

        output = F.linear(x, self.weight, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        if self.dynamic_growth and self.training:
            # Implement dynamic growth logic here
            # This is a placeholder for future implementation
            pass

        return output


class AdvancedConv2d(nn.Module):
    """Enhanced 2D convolution with additional features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, int] = "same",
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        separable: bool = False,
        attention: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # Initialize device if not provided
        device = device or DeviceManager.get_default_device()

        if separable:
            # Depthwise convolution
            self.conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False,
                device=device,
                dtype=dtype,
            )
            # Pointwise convolution
            self.pointwise = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=bias,
                device=device,
                dtype=dtype,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                device=device,
                dtype=dtype,
            )
            self.pointwise = None

        # Initialize weights using initializers
        WeightInitializer["kaiming_normal"](
            self.conv.weight, mode="fan_out", nonlinearity="relu"
        )
        if self.pointwise is not None:
            WeightInitializer["kaiming_normal"](
                self.pointwise.weight, mode="fan_out", nonlinearity="relu"
            )

        self.attention = None
        if attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(
                    out_channels, out_channels // 4, 1, device=device, dtype=dtype
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels // 4, out_channels, 1, device=device, dtype=dtype
                ),
                nn.Sigmoid(),
            )
            # Initialize attention weights
            for m in self.attention.modules():
                if isinstance(m, nn.Conv2d):
                    WeightInitializer["kaiming_normal"](
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )

        # Move to specified device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional channel attention."""
        # Ensure input is on correct device
        x = DeviceManager.to_device(x, self.conv.weight.device)

        out = self.conv(x)
        if self.pointwise is not None:
            out = self.pointwise(out)

        if self.attention is not None:
            attention_weights = self.attention(out)
            out = out * attention_weights

        return out
