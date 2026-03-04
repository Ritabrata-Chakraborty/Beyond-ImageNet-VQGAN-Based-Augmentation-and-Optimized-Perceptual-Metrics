"""
PatchGAN Discriminator with optional AC-GAN classification head.

Two output heads:
1. Adversarial: Patch-level real/fake scores (30x30 output).
2. Classification: Fault type logits (optional, enabled when num_classes > 0).

References:
    PatchGAN: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    AC-GAN: Odena et al., "Conditional Image Synthesis With Auxiliary Classifier GANs", ICML 2017
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """PatchGAN discriminator with optional auxiliary classifier.

    Args:
        image_channels: Number of input image channels.
        num_filters_last: Base filter count.
        n_layers: Number of convolutional layers.
        num_classes: Number of classes for AC-GAN head. 0 disables classification.
        classifier_dropout: Dropout rate for classification head.
    """

    def __init__(
        self,
        image_channels: int = 1,
        num_filters_last: int = 64,
        n_layers: int = 3,
        num_classes: int = 0,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Shared backbone
        layers = [
            nn.Conv2d(image_channels, num_filters_last, 4, 2, 1),
            nn.LeakyReLU(0.2),
        ]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2**i, 8)
            layers += [
                nn.Conv2d(
                    num_filters_last * num_filters_mult_last,
                    num_filters_last * num_filters_mult,
                    4,
                    2 if i < n_layers else 1,
                    1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True),
            ]

        self.backbone = nn.Sequential(*layers)
        self.feature_dim = num_filters_last * num_filters_mult

        # Adversarial head (PatchGAN output)
        self.adv_head = nn.Conv2d(self.feature_dim, 1, 4, 1, 1)

        # Classification head (optional AC-GAN)
        if num_classes > 0:
            self.cls_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.feature_dim, 256),
                nn.LeakyReLU(0.2),
                nn.Dropout(classifier_dropout),
                nn.Linear(256, num_classes),
            )
        else:
            self.cls_head = None

    def forward(self, x: torch.Tensor, return_cls: bool = True):
        """Forward pass.

        Args:
            x: Input images [B, C, H, W].
            return_cls: If True and classifier enabled, return (adv_out, cls_out).

        Returns:
            (adv_out, cls_out) if classifier enabled and return_cls=True, else adv_out.
        """
        features = self.backbone(x)
        adv_out = self.adv_head(features)

        if self.cls_head is not None and return_cls:
            cls_out = self.cls_head(features)
            return adv_out, cls_out

        return adv_out
