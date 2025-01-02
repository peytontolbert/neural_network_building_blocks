"""Advanced neural network building blocks and architectural components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
from .attention import MultiHeadAttention
from .norm_reg import AdaptiveLayerNorm, SpectralNorm
from .utils import DeviceManager, TensorOps, WeightInitializer


class EnhancedResidualBlock(nn.Module):
    """Enhanced residual block with additional features."""

    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        downsample: Optional[nn.Module] = None,
        base_width: int = 64,
        use_attention: bool = True,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        width = int(channels * (base_width / 64.0)) * groups
        self.expansion = expansion

        # Initialize all convolutions with Kaiming initialization
        self.conv1 = nn.Conv2d(
            channels, width, kernel_size=1, bias=False, device=device, dtype=dtype
        )
        WeightInitializer["kaiming_normal"](
            self.conv1.weight, mode="fan_out", nonlinearity="relu"
        )

        self.bn1 = nn.BatchNorm2d(width, device=device, dtype=dtype)

        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
            device=device,
            dtype=dtype,
        )
        WeightInitializer["kaiming_normal"](
            self.conv2.weight, mode="fan_out", nonlinearity="relu"
        )

        self.bn2 = nn.BatchNorm2d(width, device=device, dtype=dtype)

        self.conv3 = nn.Conv2d(
            width,
            channels * expansion,
            kernel_size=1,
            bias=False,
            device=device,
            dtype=dtype,
        )
        WeightInitializer["kaiming_normal"](
            self.conv3.weight, mode="fan_out", nonlinearity="relu"
        )

        self.bn3 = nn.BatchNorm2d(channels * expansion, device=device, dtype=dtype)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample or nn.Conv2d(
            channels,
            channels * expansion,
            kernel_size=1,
            stride=stride,
            device=device,
            dtype=dtype,
        )
        self.stride = stride

        if use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(
                    channels * expansion,
                    channels * expansion // 16,
                    1,
                    device=device,
                    dtype=dtype,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    channels * expansion // 16,
                    channels * expansion,
                    1,
                    device=device,
                    dtype=dtype,
                ),
                nn.Sigmoid(),
            )
            # Initialize attention convolutions
            WeightInitializer["kaiming_normal"](
                self.attention[1].weight, mode="fan_out", nonlinearity="relu"
            )
            WeightInitializer["kaiming_normal"](
                self.attention[3].weight, mode="fan_out", nonlinearity="sigmoid"
            )
        else:
            self.attention = None

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with enhanced residual connections."""
        # Move input to correct device
        x = DeviceManager.to_device(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.attention is not None:
            attention_weights = self.attention(out)
            out = out * attention_weights

        if self.dropout is not None:
            out = self.dropout(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DenseBlock(nn.Module):
    """Dense block with growth rate and optional pruning."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int = 32,
        num_layers: int = 4,
        bn_size: int = 4,
        dropout: float = 0.0,
        pruning_threshold: float = 0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.pruning_threshold = pruning_threshold

        for i in range(num_layers):
            self.layers.append(
                DenseLayer(
                    in_channels + i * growth_rate,
                    growth_rate,
                    bn_size,
                    dropout,
                    device=device,
                    dtype=dtype,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dense connections and optional pruning."""
        # Move input to correct device
        x = DeviceManager.to_device(x)
        features = [x]

        for layer in self.layers:
            out = layer(torch.cat(features, 1))

            if self.pruning_threshold > 0:
                # Simple magnitude-based pruning
                mask = (torch.abs(out) > self.pruning_threshold).float()
                out = out * mask

            features.append(out)

        return torch.cat(features, 1)


class DenseLayer(nn.Module):
    """Single layer for DenseBlock."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        dropout: float,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels, device=device, dtype=dtype)
        self.conv1 = nn.Conv2d(
            in_channels,
            bn_size * growth_rate,
            kernel_size=1,
            bias=False,
            device=device,
            dtype=dtype,
        )
        WeightInitializer["kaiming_normal"](
            self.conv1.weight, mode="fan_out", nonlinearity="relu"
        )

        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate, device=device, dtype=dtype)
        self.conv2 = nn.Conv2d(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=3,
            padding=1,
            bias=False,
            device=device,
            dtype=dtype,
        )
        WeightInitializer["kaiming_normal"](
            self.conv2.weight, mode="fan_out", nonlinearity="relu"
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for dense layer."""
        # Move input to correct device
        x = DeviceManager.to_device(x)

        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)

        if self.dropout is not None:
            out = self.dropout(out)

        return out


class FeaturePyramidBlock(nn.Module):
    """Feature pyramid network block for multi-scale feature processing."""

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        use_residual: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.use_residual = use_residual

        # Lateral connections
        self.lateral = nn.ModuleList(
            [
                nn.Conv2d(
                    in_ch, out_channels, kernel_size=1, device=device, dtype=dtype
                )
                for in_ch in in_channels
            ]
        )
        # Initialize lateral convolutions
        for conv in self.lateral:
            WeightInitializer["kaiming_normal"](
                conv.weight, mode="fan_out", nonlinearity="relu"
            )

        # Top-down connections
        self.top_down = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        device=device,
                        dtype=dtype,
                    ),
                    nn.BatchNorm2d(out_channels, device=device, dtype=dtype),
                    nn.ReLU(inplace=True),
                )
                for _ in range(len(in_channels) - 1)
            ]
        )
        # Initialize top-down convolutions
        for block in self.top_down:
            WeightInitializer["kaiming_normal"](
                block[0].weight, mode="fan_out", nonlinearity="relu"
            )

        if use_residual:
            self.residual = nn.ModuleList(
                [
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in range(len(in_channels))
                ]
            )
            # Initialize residual convolutions
            for conv in self.residual:
                WeightInitializer["kaiming_normal"](
                    conv.weight, mode="fan_out", nonlinearity="relu"
                )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass for feature pyramid."""
        # Move inputs to correct device
        features = [DeviceManager.to_device(f) for f in features]

        # Convert input features to same channel dimension
        laterals = [conv(feature) for feature, conv in zip(features, self.lateral)]

        # Top-down pathway
        outputs = [laterals[-1]]
        for i in range(len(features) - 2, -1, -1):
            top_down = F.interpolate(
                outputs[-1], size=laterals[i].shape[-2:], mode="nearest"
            )
            top_down = self.top_down[i](top_down)

            if self.use_residual:
                residual = self.residual[i](laterals[i])
                outputs.append(top_down + residual)
            else:
                outputs.append(top_down + laterals[i])

        return outputs[::-1]  # Return in original order (fine to coarse)


class DynamicRoutingBlock(nn.Module):
    """Dynamic routing module for adaptive computation paths."""

    def __init__(
        self,
        channels: int,
        num_experts: int = 4,
        routing_dim: int = 64,
        temperature: float = 1.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature

        # Router network
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, routing_dim, 1, device=device, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Conv2d(routing_dim, num_experts, 1, device=device, dtype=dtype),
        )
        # Initialize router convolutions
        WeightInitializer["kaiming_normal"](
            self.router[1].weight, mode="fan_out", nonlinearity="relu"
        )
        WeightInitializer["kaiming_normal"](
            self.router[3].weight, mode="fan_out", nonlinearity="linear"
        )

        # Expert networks
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels, channels, 3, padding=1, device=device, dtype=dtype
                    ),
                    nn.BatchNorm2d(channels, device=device, dtype=dtype),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        channels, channels, 3, padding=1, device=device, dtype=dtype
                    ),
                    nn.BatchNorm2d(channels, device=device, dtype=dtype),
                )
                for _ in range(num_experts)
            ]
        )
        # Initialize expert convolutions
        for expert in self.experts:
            WeightInitializer["kaiming_normal"](
                expert[0].weight, mode="fan_out", nonlinearity="relu"
            )
            WeightInitializer["kaiming_normal"](
                expert[3].weight, mode="fan_out", nonlinearity="relu"
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with dynamic routing."""
        # Move input to correct device
        x = DeviceManager.to_device(x)

        # Get routing weights
        routing_weights = self.router(x)
        routing_weights = routing_weights.squeeze(-1).squeeze(
            -1
        )  # Remove spatial dimensions
        routing_weights = F.softmax(routing_weights / self.temperature, dim=1)

        # Apply experts
        outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            outputs.append(expert_out.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # [B, num_experts, C, H, W]

        # Combine expert outputs
        routing_weights = routing_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        output = (outputs * routing_weights).sum(dim=1)

        return output, routing_weights.squeeze(-1).squeeze(-1).squeeze(-1)
