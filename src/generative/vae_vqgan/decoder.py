"""
Decoder for VQ-VAEGAN.

Reconstructs images from latent representations using residual blocks,
non-local attention, and nearest-neighbor upsampling.

Source: https://github.com/dome272/VQGAN-pytorch/blob/main/decoder.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from vae_vqgan.common import GroupNorm, NonLocalBlock, ResidualBlock, UpsampleBlock


class Decoder(nn.Module):
    """Convolutional decoder with residual blocks and optional attention.

    Args:
        img_channels: Number of output image channels.
        latent_channels: Number of latent input channels.
        latent_size: Spatial size of latent input (used for attention placement).
        intermediate_channels: Channel sizes for each decoder stage (will be reversed).
        num_residual_blocks: Residual blocks per stage.
        dropout: Dropout probability.
        attention_resolution: Spatial sizes at which to insert attention blocks.
        use_checkpointing: Enable gradient checkpointing to save memory.
    """

    def __init__(
        self,
        img_channels: int = 1,
        latent_channels: int = 256,
        latent_size: int = 16,
        intermediate_channels: list = [128, 128, 256, 256, 512],
        num_residual_blocks: int = 3,
        dropout: float = 0.0,
        attention_resolution: list = [16],
        use_checkpointing: bool = True,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        # Defensive copy then reverse
        channels = list(intermediate_channels)[::-1]

        in_channels = channels[0]
        layers = [
            nn.Conv2d(latent_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels, in_channels, dropout=dropout),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels, dropout=dropout),
        ]

        # Track resolution independently from the parameter
        current_resolution = latent_size

        for n in range(len(channels)):
            out_channels = channels[n]

            for _ in range(num_residual_blocks):
                layers.append(ResidualBlock(in_channels, out_channels, dropout=dropout))
                in_channels = out_channels
                if current_resolution in attention_resolution:
                    layers.append(NonLocalBlock(in_channels))

            if n != 0:
                layers.append(UpsampleBlock(in_channels=in_channels))
                current_resolution *= 2

        layers.extend([
            GroupNorm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, img_channels, kernel_size=3, stride=1, padding=1),
        ])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            return checkpoint.checkpoint(self.model, x, use_reentrant=False)
        return self.model(x)
