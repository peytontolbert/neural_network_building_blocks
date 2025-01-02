"""Advanced normalization and regularization components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from .utils import DeviceManager, TensorOps, WeightInitializer
import math


class AdaptiveLayerNorm(nn.Module):
    """Layer normalization with adaptive parameters."""

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        adaptive_elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.normalized_shape = (
            tuple(normalized_shape)
            if isinstance(normalized_shape, (list, tuple))
            else (normalized_shape,)
        )
        self.eps = eps
        self.adaptive_elementwise_affine = adaptive_elementwise_affine

        if adaptive_elementwise_affine:
            self.weight = nn.Parameter(
                TensorOps.create_tensor(
                    torch.ones(normalized_shape), device=device, dtype=dtype
                )
            )
            self.bias = nn.Parameter(
                TensorOps.create_tensor(
                    torch.zeros(normalized_shape), device=device, dtype=dtype
                )
            )
            self.gate = nn.Parameter(
                TensorOps.create_tensor(
                    torch.zeros(normalized_shape), device=device, dtype=dtype
                )
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            self.register_parameter("gate", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive normalization."""
        device = self.weight.device if self.weight is not None else None
        x = DeviceManager.to_device(x, device)

        # Calculate mean and variance with Bessel's correction
        mean = x.mean(dim=-1, keepdim=True)
        n = x.size(-1)
        correction = n / (n - 1) if n > 1 else 1.0
        var = x.var(dim=-1, keepdim=True, unbiased=True) * correction

        # Normalize with numerical stability
        x_centered = x - mean
        x_norm = x_centered / torch.sqrt(var + self.eps)

        # Apply correction factor to ensure exact unit variance
        correction_factor = 1.0 / torch.sqrt(
            x_norm.var(dim=-1, keepdim=True, unbiased=True) + self.eps
        )
        x_norm = x_norm * correction_factor

        if self.adaptive_elementwise_affine:
            gate = torch.sigmoid(self.gate)
            scale = self.weight * gate + (1.0 - gate)
            return x_norm * scale + self.bias
        return x_norm


class PopulationBatchNorm(nn.Module):
    """Batch normalization with population statistics tracking."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(
                TensorOps.create_tensor(
                    torch.ones(num_features), device=device, dtype=dtype
                )
            )
            self.bias = nn.Parameter(
                TensorOps.create_tensor(
                    torch.zeros(num_features), device=device, dtype=dtype
                )
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer(
                "running_mean",
                TensorOps.create_tensor(
                    torch.zeros(num_features), device=device, dtype=dtype
                ),
            )
            self.register_buffer(
                "running_var",
                TensorOps.create_tensor(
                    torch.ones(num_features), device=device, dtype=dtype
                ),
            )
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long, device=device)
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with population statistics."""
        # Move input to correct device
        x = DeviceManager.to_device(x)

        if self.training and self.track_running_stats:
            self.num_batches_tracked.add_(1)

        if self.training or not self.track_running_stats:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            n = x.numel() / x.size(1)

            with torch.no_grad():
                if self.track_running_stats:
                    if self.num_batches_tracked == 1:
                        self.running_mean.copy_(mean)
                        self.running_var.copy_(var)
                    else:
                        self.running_mean.lerp_(mean, self.momentum)
                        self.running_var.lerp_(var * n / (n - 1), self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean[None, :, None, None]) / torch.sqrt(
            var[None, :, None, None] + self.eps
        )

        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return x


class StructuredDropout(nn.Module):
    """Structured dropout with adaptation capabilities."""

    def __init__(
        self,
        p: float = 0.5,
        structured_dim: int = 1,
        adaptive: bool = False,
        inplace: bool = False,
        device=None,
    ):
        super().__init__()
        self.p = p
        self.structured_dim = structured_dim
        self.adaptive = adaptive
        self.inplace = inplace

        if adaptive:
            self.dropout_prob = nn.Parameter(
                TensorOps.create_tensor(torch.tensor(p), device=device)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with structured dropout."""
        if not self.training:
            return x

        # Move input to correct device
        device = self.dropout_prob.device if self.adaptive else None
        x = DeviceManager.to_device(x, device)

        p = float(self.dropout_prob if self.adaptive else self.p)
        p = max(0.0, min(1.0, p))  # Clamp between 0 and 1

        if self.structured_dim < 0:
            self.structured_dim = x.dim() + self.structured_dim

        shape = list(x.shape)
        shape[self.structured_dim] = 1

        mask = torch.bernoulli(torch.full(shape, 1 - p, device=x.device))
        mask = mask.expand_as(x)

        if self.inplace:
            x.mul_(mask).div_(1 - p)
            return x
        else:
            return x * mask / (1 - p)


class SpectralNorm(nn.Module):
    """Spectral normalization for weight matrices."""

    def __init__(
        self,
        module: nn.Module,
        name: str = "weight",
        n_power_iterations: int = 1,
        eps: float = 1e-12,
        dim: int = 0,
    ):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.dim = dim

        if not self._made_params():
            self._make_params()

    def _make_params(self):
        """Initialize spectral normalization parameters."""
        w = getattr(self.module, self.name)
        height = w.data.shape[self.dim]

        # Create u and v vectors on same device as weight
        device = w.device
        u = torch.randn(height, device=device)
        v = torch.randn(w.data.shape[0], device=device)

        # Normalize the vectors
        u = F.normalize(u, dim=0, eps=self.eps)
        v = F.normalize(v, dim=0, eps=self.eps)

        # Register as buffers instead of parameters
        self.module.register_buffer(self.name + "_u", u)
        self.module.register_buffer(self.name + "_v", v)

        return True

    def _made_params(self):
        """Check if parameters are already initialized."""
        return hasattr(self.module, self.name + "_u") and hasattr(
            self.module, self.name + "_v"
        )

    def _update_u_v(self):
        """Update u and v using power iteration."""
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name)

        height = w.data.shape[self.dim]

        for _ in range(self.n_power_iterations):
            v = F.normalize(torch.mv(w.view(height, -1).t(), u), dim=0, eps=self.eps)
            u = F.normalize(torch.mv(w.view(height, -1), v), dim=0, eps=self.eps)

        sigma = torch.dot(u, torch.mv(w.view(height, -1), v))

        return u, v, sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with spectral normalization."""
        # Move input to correct device
        x = DeviceManager.to_device(x, getattr(self.module, self.name).device)

        u, v, sigma = self._update_u_v()

        # Update u and v buffers
        self.module.register_buffer(self.name + "_u", u)
        self.module.register_buffer(self.name + "_v", v)

        # Apply spectral normalization
        w = getattr(self.module, self.name)
        w_sn = w / sigma

        # Temporarily replace weight
        w_orig = w.data.clone()
        w.data = w_sn.data

        # Forward pass
        out = self.module(x)

        # Restore original weight
        w.data = w_orig

        return out
