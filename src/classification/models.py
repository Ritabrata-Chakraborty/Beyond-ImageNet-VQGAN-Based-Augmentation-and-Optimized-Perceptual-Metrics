"""
2D LiteFormer Models with Multiple Variants
Adapted from 1D LiteFormer for 256x256 image inputs with 10 classes

Variants:
- A: Base 2D LiteFormer
- B: LiteFormer + CWCL (Continuous Wavelet Convolution Layer)
- C: LiteFormer + CWMS-GAN (Multi-scale GAN-inspired features)
- D: LiteFormer + DWT (Discrete Wavelet Transform)
- E: CWT-LiteFormer Fusion (CNN + Transformer with cross-modal fusion and attention consistency loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PatchEmbedding2D(nn.Module):
    """2D Patch Embedding for LiteFormer"""
    def __init__(self, in_channels=1, embed_dim=64, patch_size=4, stride=4):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        
        # 2D convolution for patch embedding
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=stride, 
            bias=False
        )
        
    def forward(self, x):
        # x: [B, 1, 32, 32] -> [B, 64, 8, 8]
        return self.proj(x)


class LiteFormerBlock2D(nn.Module):
    """2D LiteFormer Block adapted from 1D version"""
    def __init__(self, embed_dim=64, kernel_size=7, ffn_ratio=4, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Depthwise convolution (adapted from 1D)
        self.dconv = nn.Conv2d(
            embed_dim, embed_dim, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size//2,  # Add padding to maintain spatial size
            groups=embed_dim,  # Depthwise
            bias=False
        )
        self.norm1 = nn.BatchNorm2d(embed_dim)
        
        # FFN (same as 1D version)
        self.norm2 = nn.LayerNorm(embed_dim)
        ffn_dim = int(embed_dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x: [B, 64, H, W]
        B, C, H, W = x.shape
        
        # Depthwise convolution with residual
        identity = x
        x = self.norm1(self.dconv(x))
        x = identity + x
        
        # FFN with residual
        identity = x
        # Reshape for LayerNorm: [B, C, H, W] -> [B, H*W, C]
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        x_flat = self.norm2(x_flat)
        x_flat = self.ffn(x_flat)
        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        x = x_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = identity + x
        
        return x


class SequencePooling2D(nn.Module):
    """2D Sequence Pooling - pool spatial dimensions to single embedding"""
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable pooling weights
        self.pool_weights = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        # x: [B, 64, H, W] -> [B, 64, 1, 1]
        B, C, H, W = x.shape
        # Global average pooling
        x_pooled = F.adaptive_avg_pool2d(x, 1)  # [B, 64, 1, 1]
        return x_pooled.squeeze(-1).squeeze(-1)  # [B, 64]


class MLPHead(nn.Module):
    """MLP Head for classification. Optional dropout before the linear for regularization (e.g. few samples)."""
    def __init__(self, embed_dim=64, num_classes=10, dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [B, 64] -> [B, 10]
        return self.head(self.drop(x))


# ============================================================================
# VARIANT A: Base 2D LiteFormer
# ============================================================================

class LiteFormer2D_Base(nn.Module):
    """Base 2D LiteFormer - direct adaptation from 1D version"""
    def __init__(self, in_channels=1, embed_dim=64, patch_size=4, stride=4,
                 num_blocks=7, kernel_size=7, ffn_ratio=4, dropout=0.2, num_classes=10, head_dropout=None):
        super().__init__()
        if head_dropout is None:
            head_dropout = dropout
        # Patch embedding
        self.patch_embed = PatchEmbedding2D(in_channels, embed_dim, patch_size, stride)
        # LiteFormer blocks
        self.blocks = nn.ModuleList([
            LiteFormerBlock2D(embed_dim, kernel_size, ffn_ratio, dropout)
            for _ in range(num_blocks)
        ])
        # Sequence pooling
        self.sequence_pool = SequencePooling2D(embed_dim)
        # MLP head (optional head dropout for few-sample regularization)
        self.mlp_head = MLPHead(embed_dim, num_classes, dropout=head_dropout)
        
    def forward(self, x):
        # x: [B, 1, 32, 32]
        x = self.patch_embed(x)  # [B, 64, 8, 8]
        
        for block in self.blocks:
            x = block(x)  # [B, 64, 8, 8]
            
        x = self.sequence_pool(x)  # [B, 64]
        x = self.mlp_head(x)  # [B, 10]
        
        return x


# ============================================================================
# Dual-branch shared components (Variants B, C, D)
# ============================================================================

class ConcatFusion(nn.Module):
    """Channel concatenation + 1x1 conv to restore embed_dim. No attention. Optional dropout for regularization."""
    def __init__(self, embed_dim, aux_channels, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv2d(embed_dim + aux_channels, embed_dim, 1)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, main_feat, aux_feat):
        # main_feat: [B, embed_dim, H, W], aux_feat: [B, aux_channels, H_aux, W_aux]
        if aux_feat.shape[2:] != main_feat.shape[2:]:
            aux_feat = F.interpolate(aux_feat, size=main_feat.shape[2:], mode="bilinear", align_corners=False)
        fused = torch.cat([main_feat, aux_feat], dim=1)
        return self.drop(self.proj(fused))


def _downsample_conv(embed_dim, stride=2):
    """Stride-2 conv to halve spatial size; keeps embed_dim."""
    return nn.Sequential(
        nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=stride, padding=1, groups=embed_dim),
        nn.Conv2d(embed_dim, embed_dim, 1),
    )


class StagedLiteFormerBackbone(nn.Module):
    """Staged LiteFormer: patch_embed + multiple stages (blocks + downsample).
    Used by dual-branch variants B/C/D. Fusion is applied externally at each stage input."""
    def __init__(self, in_channels=1, embed_dim=64, patch_size=4, stride=4,
                 num_blocks=7, kernel_size=7, ffn_ratio=4, dropout=0.2, num_stages=3):
        super().__init__()
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding2D(in_channels, embed_dim, patch_size, stride)
        # Split blocks across stages: e.g. 2+2+3 for 3 stages
        blocks_per_stage = self._split_blocks(num_blocks, num_stages)
        self.stage_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        for i, n in enumerate(blocks_per_stage):
            self.stage_blocks.append(nn.ModuleList([
                LiteFormerBlock2D(embed_dim, kernel_size, ffn_ratio, dropout)
                for _ in range(n)
            ]))
            self.downsamplers.append(
                _downsample_conv(embed_dim) if i < num_stages - 1 else nn.Identity()
            )

    @staticmethod
    def _split_blocks(total, num_stages):
        base, extra = divmod(total, num_stages)
        return [base + (1 if i < extra else 0) for i in range(num_stages)]

    def forward_stage(self, x, stage_idx):
        """Run one stage: blocks then downsample."""
        for block in self.stage_blocks[stage_idx]:
            x = block(x)
        x = self.downsamplers[stage_idx](x)
        return x


class AuxDownsamplePath(nn.Module):
    """Auxiliary branch: after preprocessing, apply stride-2 convs then return
    num_stages feature maps interpolated to target_sizes. Optional dropout for regularization (few samples)."""
    def __init__(self, in_channels, aux_channels=32, num_stages=3, dropout=0.0):
        super().__init__()
        self.num_stages = num_stages
        self.convs = nn.ModuleList()
        c = in_channels
        drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        for _ in range(5):
            self.convs.append(nn.Sequential(
                nn.Conv2d(c, aux_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(aux_channels),
                nn.ReLU(inplace=True),
                drop,
            ))
            c = aux_channels
        self.aux_channels = aux_channels

    def forward(self, x, target_sizes=None):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        outputs = []
        for conv in self.convs:
            x = conv(x)
            outputs.append(x)
        if target_sizes is not None:
            # Return one tensor per stage, interpolated to target_sizes
            result = []
            for i, (th, tw) in enumerate(target_sizes):
                if i >= self.num_stages:
                    break
                # Use output at index i (in order of decreasing resolution); interpolate to (th, tw)
                src = outputs[min(i, len(outputs) - 1)]
                result.append(F.interpolate(src, size=(th, tw), mode="bilinear", align_corners=False))
            return result[: self.num_stages]
        # Legacy: first_8 index for 32/16 input
        first_8 = max(0, int(math.log2(max(H, 8) // 8)) - 1)
        indices = list(range(first_8, min(first_8 + self.num_stages, len(outputs))))
        return [outputs[i] for i in indices]


# ============================================================================
# VARIANT B: LiteFormer + CWCL (Continuous Wavelet Convolution Layer)
# ============================================================================

class GaborKernel1D(nn.Module):
    """1D Gabor kernel generator"""
    def __init__(self, kernel_size, sigma, freq, phase=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.freq = freq
        self.phase = phase
        
        # Create Gabor kernel
        half = kernel_size // 2
        x = torch.arange(-half, half + 1, dtype=torch.float32)
        gauss = torch.exp(-0.5 * (x / sigma) ** 2)
        cos = torch.cos(2 * math.pi * freq * x + phase)
        kernel = gauss * cos
        kernel = kernel / (kernel.abs().sum() + 1e-9)
        
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))
        
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # Apply 1D convolution along width dimension
        x_flat = x.view(B * C, 1, H * W)
        x_conv = F.conv1d(x_flat, self.kernel, padding=self.kernel_size//2)
        return x_conv.view(B, C, H, W)


class CWCL(nn.Module):
    """Continuous Wavelet Convolution Layer"""
    def __init__(self, in_channels=1, out_channels=64, kernel_size=9):
        super().__init__()
        self.out_channels = out_channels
        
        # Multiple Gabor kernels with different scales and frequencies
        scales = [0.8, 1.6, 3.2]
        freqs = [0.05, 0.08, 0.12]
        
        self.gabor_kernels = nn.ModuleList([
            GaborKernel1D(kernel_size, sigma, freq)
            for sigma, freq in zip(scales, freqs)
        ])
        
        # Learnable combination weights
        self.combination = nn.Conv2d(len(scales), out_channels, 1)
        
    def forward(self, x):
        # x: [B, 1, 32, 32]
        gabor_outputs = []
        for kernel in self.gabor_kernels:
            gabor_outputs.append(kernel(x))
        
        # Stack and combine
        x_combined = torch.cat(gabor_outputs, dim=1)  # [B, 3, 32, 32]
        x_out = self.combination(x_combined)  # [B, 64, 32, 32]
        return x_out


class LiteFormer2D_CWCL(nn.Module):
    """Dual-branch: Branch 1 = original image through staged LiteFormer;
    Branch 2 = CWCL (Gabor) -> downsample path; concat fusion at each stage (no attention)."""
    def __init__(self, in_channels=1, embed_dim=64, patch_size=4, stride=4,
                 num_blocks=7, kernel_size=7, ffn_ratio=4, dropout=0.2, num_classes=10, num_fusion_stages=3,
                 head_dropout=None, aux_dropout=0.0):
        super().__init__()
        if head_dropout is None:
            head_dropout = dropout
        self.num_fusion_stages = num_fusion_stages
        # Branch 2: Gabor preprocessing
        self.cwcl = CWCL(in_channels, embed_dim)
        self.aux_path = AuxDownsamplePath(embed_dim, aux_channels=32, num_stages=num_fusion_stages, dropout=aux_dropout)
        # Branch 1: staged LiteFormer (raw image)
        self.backbone = StagedLiteFormerBackbone(
            in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size, stride=stride,
            num_blocks=num_blocks, kernel_size=kernel_size, ffn_ratio=ffn_ratio, dropout=dropout,
            num_stages=num_fusion_stages,
        )
        self.fusions = nn.ModuleList([
            ConcatFusion(embed_dim, self.aux_path.aux_channels, dropout=aux_dropout)
            for _ in range(num_fusion_stages)
        ])
        self.sequence_pool = SequencePooling2D(embed_dim)
        self.mlp_head = MLPHead(embed_dim, num_classes, dropout=head_dropout)

    def forward(self, x):
        # x: [B, 1, H, W]
        x_main = self.backbone.patch_embed(x)  # [B, 64, H/4, W/4]
        x_aux = self.cwcl(x)  # [B, 64, H, W]
        h, w = x_main.shape[2], x_main.shape[3]
        target_sizes = [(max(1, h // (2 ** i)), max(1, w // (2 ** i))) for i in range(self.num_fusion_stages)]
        aux_list = self.aux_path(x_aux, target_sizes=target_sizes)
        for i in range(self.num_fusion_stages):
            x_main = self.fusions[i](x_main, aux_list[i])
            x_main = self.backbone.forward_stage(x_main, i)
        x_main = self.sequence_pool(x_main)
        return self.mlp_head(x_main)


# ============================================================================
# VARIANT C: LiteFormer + CWMS-GAN (Multi-scale GAN-inspired features)
# ============================================================================

class MultiScaleConvBlock(nn.Module):
    """Multi-scale convolution block inspired by GAN discriminators"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Multiple parallel convolutions with different kernel sizes
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 3, 1, 1),
            nn.BatchNorm2d(out_channels//4),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 5, 1, 2),
            nn.BatchNorm2d(out_channels//4),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 7, 1, 3),
            nn.BatchNorm2d(out_channels//4),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 9, 1, 4),
            nn.BatchNorm2d(out_channels//4),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        # x: [B, C, H, W]
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        
        # Concatenate multi-scale features
        x_out = torch.cat([x1, x2, x3, x4], dim=1)  # [B, C, H, W]
        return x_out


class CWMS_GAN(nn.Module):
    """CWMS-GAN inspired multi-scale feature extractor"""
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        
        # Multi-scale feature extraction
        self.ms_conv1 = MultiScaleConvBlock(in_channels, 32)
        self.ms_conv2 = MultiScaleConvBlock(32, 64)
        
        # Final projection
        self.proj = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        # x: [B, 1, 32, 32]
        x = self.ms_conv1(x)  # [B, 32, 32, 32]
        x = self.ms_conv2(x)  # [B, 64, 32, 32]
        x = self.proj(x)  # [B, 64, 32, 32]
        return x


class LiteFormer2D_CWMS_GAN(nn.Module):
    """Dual-branch: Branch 1 = original image through staged LiteFormer;
    Branch 2 = CWMS-GAN (multi-scale) -> downsample path; concat fusion at each stage (no attention)."""
    def __init__(self, in_channels=1, embed_dim=64, patch_size=4, stride=4,
                 num_blocks=7, kernel_size=7, ffn_ratio=4, dropout=0.2, num_classes=10, num_fusion_stages=3,
                 head_dropout=None, aux_dropout=0.0):
        super().__init__()
        if head_dropout is None:
            head_dropout = dropout
        self.num_fusion_stages = num_fusion_stages
        self.cwms_gan = CWMS_GAN(in_channels, embed_dim)
        self.aux_path = AuxDownsamplePath(embed_dim, aux_channels=32, num_stages=num_fusion_stages, dropout=aux_dropout)
        self.backbone = StagedLiteFormerBackbone(
            in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size, stride=stride,
            num_blocks=num_blocks, kernel_size=kernel_size, ffn_ratio=ffn_ratio, dropout=dropout,
            num_stages=num_fusion_stages,
        )
        self.fusions = nn.ModuleList([
            ConcatFusion(embed_dim, self.aux_path.aux_channels, dropout=aux_dropout)
            for _ in range(num_fusion_stages)
        ])
        self.sequence_pool = SequencePooling2D(embed_dim)
        self.mlp_head = MLPHead(embed_dim, num_classes, dropout=head_dropout)

    def forward(self, x):
        x_main = self.backbone.patch_embed(x)
        x_aux = self.cwms_gan(x)
        h, w = x_main.shape[2], x_main.shape[3]
        target_sizes = [(max(1, h // (2 ** i)), max(1, w // (2 ** i))) for i in range(self.num_fusion_stages)]
        aux_list = self.aux_path(x_aux, target_sizes=target_sizes)
        for i in range(self.num_fusion_stages):
            x_main = self.fusions[i](x_main, aux_list[i])
            x_main = self.backbone.forward_stage(x_main, i)
        x_main = self.sequence_pool(x_main)
        return self.mlp_head(x_main)


# ============================================================================
# VARIANT D: LiteFormer + DWT (Discrete Wavelet Transform)
# ============================================================================

class HaarDWT2D(nn.Module):
    """2D Haar Discrete Wavelet Transform"""
    def __init__(self):
        super().__init__()
        
        # Haar filters
        ll = torch.tensor([0.5, 0.5], dtype=torch.float32)
        hh = torch.tensor([0.5, -0.5], dtype=torch.float32)
        
        # 2D filters
        LL = ll[:, None] @ ll[None, :]
        LH = ll[:, None] @ hh[None, :]
        HL = hh[:, None] @ ll[None, :]
        HH = hh[:, None] @ hh[None, :]
        
        # Stack filters
        filters = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)
        self.register_buffer('filters', filters)  # [4, 1, 2, 2]
        
    def forward(self, x):
        # x: [B, 1, 32, 32]
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "H and W must be divisible by 2"
        
        # Apply DWT
        weight = self.filters.repeat(1, C, 1, 1).view(4*C, 1, 2, 2)
        out = F.conv2d(x, weight=weight, bias=None, stride=2, padding=0, groups=C)
        
        # Reshape: [B, 4*C, H/2, W/2] -> [B, C, 4, H/2, W/2]
        out = out.view(B, C, 4, H//2, W//2).permute(0, 2, 1, 3, 4).contiguous()
        out = out.view(B, 4*C, H//2, W//2)
        
        return out


class LiteFormer2D_DWT(nn.Module):
    """Dual-branch: Branch 1 = original image through staged LiteFormer;
    Branch 2 = Haar DWT -> downsample path; concat fusion at each stage (no attention)."""
    def __init__(self, in_channels=1, embed_dim=64, patch_size=4, stride=4,
                 num_blocks=7, kernel_size=7, ffn_ratio=4, dropout=0.2, num_classes=10, num_fusion_stages=3,
                 head_dropout=None, aux_dropout=0.0):
        super().__init__()
        if head_dropout is None:
            head_dropout = dropout
        self.num_fusion_stages = num_fusion_stages
        self.dwt = HaarDWT2D()
        # DWT outputs 4 channels at H/2 x W/2 (e.g. 16x16)
        self.aux_path = AuxDownsamplePath(4, aux_channels=32, num_stages=num_fusion_stages, dropout=aux_dropout)
        self.backbone = StagedLiteFormerBackbone(
            in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size, stride=stride,
            num_blocks=num_blocks, kernel_size=kernel_size, ffn_ratio=ffn_ratio, dropout=dropout,
            num_stages=num_fusion_stages,
        )
        self.fusions = nn.ModuleList([
            ConcatFusion(embed_dim, self.aux_path.aux_channels, dropout=aux_dropout)
            for _ in range(num_fusion_stages)
        ])
        self.sequence_pool = SequencePooling2D(embed_dim)
        self.mlp_head = MLPHead(embed_dim, num_classes, dropout=head_dropout)

    def forward(self, x):
        x_main = self.backbone.patch_embed(x)
        x_aux = self.dwt(x)  # [B, 4, H/2, W/2]
        h, w = x_main.shape[2], x_main.shape[3]
        target_sizes = [(max(1, h // (2 ** i)), max(1, w // (2 ** i))) for i in range(self.num_fusion_stages)]
        aux_list = self.aux_path(x_aux, target_sizes=target_sizes)
        for i in range(self.num_fusion_stages):
            x_main = self.fusions[i](x_main, aux_list[i])
            x_main = self.backbone.forward_stage(x_main, i)
        x_main = self.sequence_pool(x_main)
        return self.mlp_head(x_main)


# ============================================================================
# VARIANT E: CWT-LiteFormer Fusion (CNN + Transformer, attention consistency)
# ============================================================================

class _FusionCNNBranch(nn.Module):
    """CNN branch for local time-frequency pattern extraction (variant E)."""
    def __init__(self, in_channels=1, hidden_channels=64, dropout=0.1):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels if i == 0 else hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=d,
                    dilation=d,
                    groups=in_channels if i == 0 else hidden_channels,
                ),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
            )
            for i, d in enumerate([1, 2, 4])
        ])
        self.feature_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)

    def forward(self, x):
        for i, stage in enumerate(self.stages):
            x = stage(x) if i == 0 else x + stage(x)
        return self.feature_proj(x)


class _FusionTransformerBranch(nn.Module):
    """Transformer branch using LiteFormer blocks; stores attention-like maps (variant E)."""
    def __init__(self, in_channels=1, embed_dim=64, num_blocks=7, patch_size=4, kernel_size=7, ffn_ratio=4, dropout=0.2):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.blocks = nn.ModuleList([
            LiteFormerBlock2D(embed_dim, kernel_size, ffn_ratio, dropout)
            for _ in range(num_blocks)
        ])
        self.attention_maps = []

    def forward(self, x):
        tokens = self.patch_embed(x)
        B, C, H, W = tokens.shape
        inp_h, inp_w = x.shape[2], x.shape[3]
        self.attention_maps.clear()
        for block in self.blocks:
            tokens = block(tokens)
            with torch.no_grad():
                attn_map = torch.mean(torch.abs(tokens), dim=1, keepdim=True)
                attn_map = F.interpolate(attn_map, size=(inp_h, inp_w), mode="bilinear", align_corners=False)
                self.attention_maps.append(attn_map)
        return tokens


class _CrossModalFusion(nn.Module):
    """Cross-modal attention fusion (variant E). Caps spatial size for attention to avoid OOM on 256x256."""
    FUSION_MAX_SPATIAL = 32  # max side length for attention (32*32 = 1024 positions)

    def __init__(self, dim=64, hidden_dim=32):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Conv2d(dim, hidden_dim, 1)
        self.kv_proj = nn.Conv2d(dim, hidden_dim * 2, 1)
        self.out_proj = nn.Conv2d(hidden_dim, dim, 1)
        self.scale = hidden_dim ** -0.5

    def forward(self, local_feats, global_feats):
        target_size = local_feats.shape[-2:]
        H, W = target_size[0], target_size[1]
        # Cap resolution for attention to avoid OOM (e.g. 256x256 -> 65536^2 matrix)
        max_side = self.FUSION_MAX_SPATIAL
        if H > max_side or W > max_side:
            scale = min(max_side / H, max_side / W)
            fusion_h, fusion_w = max(1, int(H * scale)), max(1, int(W * scale))
            local_f = F.interpolate(local_feats, size=(fusion_h, fusion_w), mode="bilinear", align_corners=False)
            global_f = F.interpolate(global_feats, size=(fusion_h, fusion_w), mode="bilinear", align_corners=False)
        else:
            local_f = local_feats
            fusion_h, fusion_w = H, W
            global_f = F.interpolate(global_feats, size=(fusion_h, fusion_w), mode="bilinear", align_corners=False)

        Q = self.q_proj(global_f)
        K, V = self.kv_proj(local_f).chunk(2, dim=1)
        B, C, fh, fw = Q.shape
        Q = Q.flatten(2)
        K = K.flatten(2)
        V = V.flatten(2)
        attn = torch.softmax((Q.transpose(-2, -1) @ K) * self.scale, dim=-1)
        fused = (attn @ V.transpose(-2, -1)).transpose(-2, -1)
        fused = fused.view(B, self.hidden_dim, fh, fw)
        fused = self.out_proj(fused) + global_f
        if (fh, fw) != target_size:
            fused = F.interpolate(fused, size=target_size, mode="bilinear", align_corners=False)
        return fused


class _AttentionConsistencyLoss(nn.Module):
    """Attention consistency loss for variant E (fault bands, temporal, cross-scale)."""
    def __init__(self, lambda_temp=0.5, lambda_scale=0.3, lambda_fault=1.0):
        super().__init__()
        self.lambda_temp = lambda_temp
        self.lambda_scale = lambda_scale
        self.lambda_fault = lambda_fault

    def _create_fault_mask(self, H, W, device):
        mask = torch.zeros(1, 1, H, W, device=device)
        mask[:, :, int(0.75 * H) :, :] = 1.0
        mask[:, :, int(0.5 * H) : int(0.75 * H), :] = 0.7
        return mask

    def forward(self, attention_maps, device=None):
        if not attention_maps or len(attention_maps) == 0:
            return torch.tensor(0.0, device=device)
        total_loss = 0.0
        last_attention = attention_maps[-1]
        dev = last_attention.device
        H, W = last_attention.shape[2], last_attention.shape[3]
        fault_mask = self._create_fault_mask(H, W, dev)
        fault_attention = (last_attention * fault_mask).sum()
        total_attention = last_attention.sum() + 1e-8
        loss_fault = 1.0 - (fault_attention / total_attention)
        temporal_diffs = [
            (attn_map[:, :, :, 1:] - attn_map[:, :, :, :-1]).abs().mean()
            for attn_map in attention_maps
        ]
        loss_temporal = sum(temporal_diffs) / len(temporal_diffs)
        loss_scale = 0.0
        if len(attention_maps) > 1:
            scale_diffs = [
                (attention_maps[i] - attention_maps[i + 1]).pow(2).mean()
                for i in range(len(attention_maps) - 1)
            ]
            loss_scale = sum(scale_diffs) / len(scale_diffs)
        return self.lambda_fault * loss_fault + self.lambda_temp * loss_temporal + self.lambda_scale * loss_scale


class LiteFormer2D_Fusion(nn.Module):
    """CWT-LiteFormer Fusion: CNN + Transformer with cross-modal fusion and attention consistency (variant E)."""
    def __init__(
        self,
        in_channels=1,
        embed_dim=64,
        num_blocks=7,
        kernel_size=7,
        ffn_ratio=4,
        dropout=0.2,
        num_classes=10,
        head_dropout=None,
        cnn_dropout=0.1,
        **_
    ):
        super().__init__()
        if head_dropout is None:
            head_dropout = dropout
        self.cnn_branch = _FusionCNNBranch(in_channels, embed_dim, dropout=cnn_dropout)
        self.transformer_branch = _FusionTransformerBranch(
            in_channels, embed_dim, num_blocks,
            kernel_size=kernel_size, ffn_ratio=ffn_ratio, dropout=dropout,
        )
        self.fusion_module = _CrossModalFusion(embed_dim)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )
        self.attention_loss = _AttentionConsistencyLoss()

    def forward(self, x, return_attention_maps=False):
        local_features = self.cnn_branch(x)
        global_features = self.transformer_branch(x)
        fused = self.fusion_module(local_features, global_features)
        pooled = self.global_pool(fused)
        logits = self.classifier(pooled)
        if return_attention_maps:
            return logits, self.transformer_branch.attention_maps
        return logits

    def get_attention_loss(self, attention_maps=None):
        if attention_maps is None:
            attention_maps = self.transformer_branch.attention_maps
        return self.attention_loss(attention_maps, device=next(self.parameters()).device)


# ============================================================================
# Model Factory
# ============================================================================

VARIANT_CLASSES = {
    'A': LiteFormer2D_Base,
    'B': LiteFormer2D_CWCL,
    'C': LiteFormer2D_CWMS_GAN,
    'D': LiteFormer2D_DWT,
    'E': LiteFormer2D_Fusion,
}


def create_liteformer_2d_variant(variant='A', **kwargs):
    """Factory: create LiteFormer 2D variants. Defaults are small (embed_dim=32, num_blocks=4).
    For a larger model pass e.g. embed_dim=64, num_blocks=7, ffn_ratio=4.
    For very few samples, increase dropout, head_dropout, aux_dropout and/or weight_decay (in optimizer)."""
    default_kwargs = {
        'in_channels': 1,
        'embed_dim': 32,
        'patch_size': 4,
        'stride': 4,
        'num_blocks': 4,
        'kernel_size': 7,
        'ffn_ratio': 2,
        'dropout': 0.2,
        'head_dropout': None,
        'num_classes': 10,
    }
    default_kwargs.update(kwargs)
    v = variant.upper()
    if v not in VARIANT_CLASSES:
        raise ValueError(f"Unknown variant: {variant}. Choose from A, B, C, D, E")
    return VARIANT_CLASSES[v](**default_kwargs)


# ============================================================================
# Model Information
# ============================================================================

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(variant='A'):
    """Get model information for a variant"""
    model = create_liteformer_2d_variant(variant)
    param_count = count_parameters(model)
    dummy_input = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)
    return {
        'variant': variant,
        'parameters': param_count,
        'input_shape': dummy_input.shape,
        'output_shape': output.shape
    }


if __name__ == "__main__":
    variants = ['A', 'B', 'C', 'D', 'E']
    
    print("LiteFormer 2D Variants - Model Information")
    print("=" * 50)
    
    for variant in variants:
        info = get_model_info(variant)
        print(f"Variant {variant}:")
        print(f"  Parameters: {info['parameters']:,}")
        print(f"  Input: {info['input_shape']}")
        print(f"  Output: {info['output_shape']}")
        print()
