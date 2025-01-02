"""Multimodal processing components for handling different data modalities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from .attention import MultiHeadAttention
from .core_layers import SmartDense
from .utils import DeviceManager, TensorOps, WeightInitializer

class SpeechEncoder(nn.Module):
    """Speech encoder with multi-scale processing."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        kernel_sizes: List[int] = [3, 5, 7],
        dropout: float = 0.1,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim, device=device, dtype=dtype)
        WeightInitializer['xavier_uniform'](self.input_proj.weight)
        
        # Multi-scale convolutions
        self.conv_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim if i == 0 else hidden_dim * len(kernel_sizes),
                        hidden_dim,
                        kernel_size=k,
                        padding=k // 2,
                        device=device,
                        dtype=dtype
                    ),
                    nn.BatchNorm1d(hidden_dim, device=device, dtype=dtype),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ) for k in kernel_sizes
            ]) for i in range(num_layers)
        ])
        
        # Initialize convolutions
        for layer in self.conv_layers:
            for conv_block in layer:
                WeightInitializer['kaiming_normal'](conv_block[0].weight, mode='fan_out', nonlinearity='relu')
        
        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(
                embed_dim=hidden_dim * len(kernel_sizes),
                num_heads=8,
                dropout=dropout,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with multi-scale processing.
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_dim]
            mask: Optional mask tensor [batch_size, sequence_length]
            
        Returns:
            Processed tensor [batch_size, sequence_length, hidden_dim * num_kernels]
        """
        # Move input to correct device
        x = DeviceManager.to_device(x)
        
        # Project input
        x = self.input_proj(x)  # [B, L, H]
        x = x.transpose(1, 2)  # [B, H, L]
        
        # Process through layers
        for conv_list, attention in zip(self.conv_layers, self.attention_layers):
            # Multi-scale convolution
            conv_outputs = []
            for conv in conv_list:
                conv_outputs.append(conv(x if len(conv_outputs) == 0 else torch.cat(conv_outputs, dim=1)))
            x = torch.cat(conv_outputs, dim=1)  # [B, H*K, L]
            
            # Self-attention
            x = x.transpose(1, 2)  # [B, L, H*K]
            x, _ = attention(x, x, x, key_padding_mask=mask)
            x = x.transpose(1, 2)  # [B, H*K, L]
        
        return x.transpose(1, 2)  # [B, L, H*K]

class VisionProcessor(nn.Module):
    """Vision processor with hierarchical attention."""
    
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        num_scales: int = 3,
        attention_heads: int = 8,
        dropout: float = 0.1,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.num_scales = num_scales
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, device=device, dtype=dtype),
            nn.BatchNorm2d(base_channels, device=device, dtype=dtype),
            nn.ReLU(inplace=True)
        )
        WeightInitializer['kaiming_normal'](self.init_conv[0].weight, mode='fan_out', nonlinearity='relu')
        
        # Hierarchical processing
        self.stages = nn.ModuleList()
        curr_channels = base_channels
        
        for i in range(num_scales):
            stage = nn.Sequential(
                # Downsample
                nn.Conv2d(curr_channels, curr_channels * 2, kernel_size=3, stride=2, padding=1, device=device, dtype=dtype),
                nn.BatchNorm2d(curr_channels * 2, device=device, dtype=dtype),
                nn.ReLU(inplace=True),
                
                # Process
                nn.Conv2d(curr_channels * 2, curr_channels * 2, kernel_size=3, padding=1, device=device, dtype=dtype),
                nn.BatchNorm2d(curr_channels * 2, device=device, dtype=dtype),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            )
            
            # Initialize convolutions
            WeightInitializer['kaiming_normal'](stage[0].weight, mode='fan_out', nonlinearity='relu')
            WeightInitializer['kaiming_normal'](stage[3].weight, mode='fan_out', nonlinearity='relu')
            
            self.stages.append(stage)
            curr_channels *= 2
        
        # Spatial attention at each scale
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(
                embed_dim=ch,
                num_heads=attention_heads,
                dropout=dropout,
                device=device,
                dtype=dtype
            ) for ch in [base_channels * 2**i for i in range(1, num_scales + 1)]
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass with hierarchical processing.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            List of processed features at different scales
        """
        # Move input to correct device
        x = DeviceManager.to_device(x)
        
        # Initial processing
        x = self.init_conv(x)
        features = []
        
        # Process through stages
        for stage, attention in zip(self.stages, self.attention_layers):
            # Apply stage processing
            x = stage(x)
            
            # Apply spatial attention
            B, C, H, W = x.shape
            x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
            x_attn, _ = attention(x_flat, x_flat, x_flat)
            x = x_attn.transpose(1, 2).view(B, C, H, W)
            
            features.append(x)
        
        return features

class TextProcessor(nn.Module):
    """Text processor with tokenization and contextual processing."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Token and position embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim, device=device, dtype=dtype)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_length, embed_dim, device=device, dtype=dtype))
        
        # Initialize embeddings
        WeightInitializer['xavier_uniform'](self.token_embed.weight)
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Processing layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    device=device,
                    dtype=dtype
                ),
                'norm1': nn.LayerNorm(embed_dim, device=device, dtype=dtype),
                'ff': nn.Sequential(
                    nn.Linear(embed_dim, ff_dim, device=device, dtype=dtype),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_dim, embed_dim, device=device, dtype=dtype),
                    nn.Dropout(dropout)
                ),
                'norm2': nn.LayerNorm(embed_dim, device=device, dtype=dtype)
            }) for _ in range(num_layers)
        ])
        
        # Initialize feed-forward layers
        for layer in self.layers:
            WeightInitializer['xavier_uniform'](layer['ff'][0].weight)
            WeightInitializer['xavier_uniform'](layer['ff'][3].weight)
    
    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with contextual processing.
        
        Args:
            tokens: Input token indices [batch_size, sequence_length]
            mask: Optional attention mask [batch_size, sequence_length]
            
        Returns:
            Contextual embeddings [batch_size, sequence_length, embed_dim]
        """
        # Move input to correct device
        tokens = DeviceManager.to_device(tokens)
        
        # Get embeddings
        x = self.token_embed(tokens)
        x = x + self.pos_embed[:, :x.size(1)]
        
        # Process through layers
        for layer in self.layers:
            # Self-attention
            attended, _ = layer['attention'](x, x, x, attn_mask=mask)
            x = layer['norm1'](x + attended)
            
            # Feed-forward
            x = layer['norm2'](x + layer['ff'](x))
            
        return x

class CrossModalFusion(nn.Module):
    """Cross-modal fusion with attention-based alignment."""
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        fusion_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.modality_dims = modality_dims
        
        # Projection layers for each modality
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, fusion_dim, device=device, dtype=dtype)
            for name, dim in modality_dims.items()
        })
        
        # Initialize projections
        for proj in self.projections.values():
            WeightInitializer['xavier_uniform'](proj.weight)
        
        # Cross-modal attention
        self.cross_attention = nn.ModuleDict({
            f"{m1}_to_{m2}": MultiHeadAttention(
                embed_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
                device=device,
                dtype=dtype
            )
            for m1 in modality_dims.keys()
            for m2 in modality_dims.keys()
            if m1 != m2
        })
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * len(modality_dims), fusion_dim, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim, device=device, dtype=dtype)
        )
        
        # Initialize fusion layers
        WeightInitializer['xavier_uniform'](self.fusion[0].weight)
        WeightInitializer['xavier_uniform'](self.fusion[3].weight)
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass with cross-modal attention and fusion.
        
        Args:
            inputs: Dictionary of input tensors for each modality
            masks: Optional dictionary of attention masks
            
        Returns:
            Fused representation
        """
        # Move inputs to correct device
        inputs = {k: DeviceManager.to_device(v) for k, v in inputs.items()}
        if masks:
            masks = {k: DeviceManager.to_device(v) for k, v in masks.items()}
        
        # Project each modality
        projected = {
            name: self.projections[name](x)
            for name, x in inputs.items()
        }
        
        # Cross-modal attention
        attended = {}
        for m1 in self.modality_dims.keys():
            aligned_views = []
            for m2 in self.modality_dims.keys():
                if m1 != m2:
                    mask = None if not masks else masks.get(f"{m1}_to_{m2}")
                    attended_view, _ = self.cross_attention[f"{m1}_to_{m2}"](
                        projected[m1],
                        projected[m2],
                        projected[m2],
                        key_padding_mask=mask
                    )
                    aligned_views.append(attended_view)
            
            # Combine aligned views
            if aligned_views:
                attended[m1] = sum(aligned_views) / len(aligned_views)
            else:
                attended[m1] = projected[m1]
        
        # Combine all modalities
        combined = torch.cat(list(attended.values()), dim=-1)
        fused = self.fusion(combined)
        
        return fused 