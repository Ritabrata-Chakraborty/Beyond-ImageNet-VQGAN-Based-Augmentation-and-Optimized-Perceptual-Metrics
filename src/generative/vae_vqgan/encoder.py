"""
Encoder for VQ-VAEGAN.

Downsamples input images to a latent representation using residual blocks,
non-local attention, and strided convolutions.

Source: https://github.com/dome272/VQGAN-pytorch/blob/main/encoder.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from vae_vqgan.common import DownsampleBlock, GroupNorm, NonLocalBlock, ResidualBlock


class Encoder(nn.Module):
    """Convolutional encoder with residual blocks and optional attention.

    Args:
        img_channels: Number of input image channels.
        image_size: Spatial size of input images (used for attention placement).
        latent_channels: Number of output latent channels.
        intermediate_channels: Channel sizes for each encoder stage.
        num_residual_blocks: Residual blocks per stage.
        dropout: Dropout probability.
        attention_resolution: Spatial sizes at which to insert attention blocks.
        use_checkpointing: Enable gradient checkpointing to save memory.
    """

    def __init__(
        self,
        img_channels: int = 1,
        image_size: int = 256,
        latent_channels: int = 256,
        intermediate_channels: list = [128, 128, 256, 256, 512],
        num_residual_blocks: int = 2,
        dropout: float = 0.0,
        attention_resolution: list = [16],
        use_checkpointing: bool = True,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        # Defensive copy to avoid mutating caller's list
        channels = [intermediate_channels[0]] + list(intermediate_channels)

        layers = [
            nn.Conv2d(img_channels, channels[0], kernel_size=3, stride=1, padding=1),
        ]

        # Track resolution independently from the parameter
        current_resolution = image_size

        for n in range(len(channels) - 1):
            in_channels = channels[n]
            out_channels = channels[n + 1]

            for _ in range(num_residual_blocks):
                layers.append(ResidualBlock(in_channels, out_channels, dropout=dropout))
                in_channels = out_channels
                if current_resolution in attention_resolution:
                    layers.append(NonLocalBlock(in_channels))

            if n != len(channels) - 2:
                layers.append(DownsampleBlock(in_channels=channels[n + 1]))
                current_resolution //= 2

        in_channels = channels[-1]
        layers.extend([
            ResidualBlock(in_channels, in_channels, dropout=dropout),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels, dropout=dropout),
            GroupNorm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, latent_channels, kernel_size=3, stride=1, padding=1),
        ])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            return checkpoint.checkpoint(self.model, x, use_reentrant=False)
        return self.model(x)
