"""
Shared building blocks for the VQGAN encoder and decoder.

Source: https://github.com/dome272/VQGAN-pytorch/blob/main/helper.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


# --- Constants ---
GROUPNORM_NUM_GROUPS = 32


# --- Blocks ---


class GroupNorm(nn.Module):
    """Group normalization wrapper with fixed 32 groups.

    Args:
        in_channels: Number of input channels.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(
            num_groups=GROUPNORM_NUM_GROUPS, num_channels=in_channels, eps=1e-06, affine=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.group_norm(x)


class ResidualBlock(nn.Module):
    """Residual block: GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Dropout -> Conv + skip.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        dropout: Dropout probability.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            GroupNorm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            GroupNorm(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_channels != self.out_channels:
            return self.conv_shortcut(x) + self.block(x)
        return x + self.block(x)


class DownsampleBlock(nn.Module):
    """Downsample by 2x via asymmetric padding + stride-2 convolution.

    Args:
        in_channels: Number of input channels.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), value=0)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class UpsampleBlock(nn.Module):
    """Upsample by 2x via nearest interpolation + convolution.

    Args:
        in_channels: Number of input channels.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2.0)
        return self.conv(x)


class NonLocalBlock(nn.Module):
    """Self-attention block for CNNs.

    Reference: https://arxiv.org/abs/1805.08318

    Args:
        in_channels: Number of input channels.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.norm = GroupNorm(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.k = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.v = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.project_out = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.size()
        x_norm = self.norm(x)

        q = self.q(x_norm).reshape(batch, self.in_channels, height * width)
        k = self.k(x_norm).reshape(batch, self.in_channels, height * width)
        v = self.v(x_norm).reshape(batch, self.in_channels, height * width)

        # Attention: softmax(Q^T K / sqrt(d)) V
        q = q.permute(0, 2, 1)
        scores = torch.bmm(q, k) * (self.in_channels**-0.5)
        weights = self.softmax(scores).permute(0, 2, 1)
        attention = torch.bmm(v, weights)

        attention = attention.reshape(batch, self.in_channels, height, width)
        attention = self.project_out(attention)
        return x + attention
