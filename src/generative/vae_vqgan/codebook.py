"""
Vector quantization codebook for VQ-VAEGAN.

Supports two update modes (config: codebook_ema):
- use_ema=True: Exponential Moving Average (EMA) codebook updates (no gradient).
- use_ema=False: Codebook updated by embedding loss gradient (||sg[z] - z_q||^2).

Also: dead code detection and reset, utilization and perplexity tracking.

Source: https://github.com/dome272/VQGAN-pytorch/blob/main/codebook.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CodeBook(nn.Module):
    """Vector quantization codebook with optional EMA or embedding-loss updates.

    Args:
        num_codebook_vectors: Number of embedding vectors in the codebook.
        latent_dim: Dimensionality of each embedding vector.
        beta: Weight for the commitment loss term.
        use_ema: If True, update codebook via EMA; if False, use embedding loss gradient.
        ema_decay: EMA decay rate (only when use_ema=True).
        ema_epsilon: Epsilon for Laplace smoothing in EMA (only when use_ema=True).
    """

    def __init__(
        self,
        num_codebook_vectors: int = 1024,
        latent_dim: int = 256,
        beta: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
    ):
        super().__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.latent_dim = latent_dim
        self.beta = beta
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon

        self.codebook = nn.Embedding(num_codebook_vectors, latent_dim)
        self.codebook.weight.data.uniform_(
            -1 / num_codebook_vectors, 1 / num_codebook_vectors
        )

        # EMA buffers (only used when use_ema=True)
        self.register_buffer("ema_cluster_size", torch.zeros(num_codebook_vectors))
        self.register_buffer("ema_embedding_sum", self.codebook.weight.data.clone())
        self.register_buffer("code_usage", torch.zeros(num_codebook_vectors))

    def forward(self, z: torch.Tensor) -> tuple:
        """Quantize encoder output to nearest codebook vectors.

        Args:
            z: Encoder output [B, C, H, W].

        Returns:
            z_q: Quantized tensor [B, C, H, W] (with straight-through gradient).
            indices: Indices of nearest codebook vectors [B*H*W].
            codebook_loss: Commitment loss (always); + embedding loss when use_ema=False.
            stats: Dict with utilization_pct, active_codes, perplexity, embedding_loss, commitment_loss.
        """
        z_permuted = z.permute(0, 2, 3, 1).contiguous()

        with torch.no_grad():
            z_flattened = z_permuted.view(-1, self.latent_dim)
            distance = (
                torch.sum(z_flattened**2, dim=1, keepdim=True)
                + torch.sum(self.codebook.weight**2, dim=1)
                - 2 * torch.matmul(z_flattened, self.codebook.weight.t())
            )
            indices = torch.argmin(distance, dim=1)
            z_q_nograd = self.codebook(indices).view(z_permuted.shape)
            encodings = F.one_hot(indices, self.num_codebook_vectors).float()
            batch_cluster_size = encodings.sum(0)

            if self.training:
                if self.use_ema:
                    self._ema_update(z_flattened, encodings)
                else:
                    self.code_usage.add_(batch_cluster_size)

            active_codes = torch.unique(indices).numel()
            utilization_pct = 100.0 * active_codes / self.num_codebook_vectors
            avg_probs = encodings.mean(0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            embedding_loss_val = torch.mean((z_permuted - z_q_nograd) ** 2).item()
            del distance, encodings, z_flattened

        # Commitment loss: beta * ||z - sg[z_q]||^2 (trains encoder).
        commitment_loss = self.beta * torch.mean((z_q_nograd.detach() - z_permuted) ** 2)

        if self.use_ema:
            codebook_loss = commitment_loss
            z_q_out = z_permuted + (z_q_nograd - z_permuted).detach()
            stats = {
                "active_codes": active_codes,
                "utilization_pct": utilization_pct,
                "perplexity": perplexity.item(),
                "embedding_loss": embedding_loss_val,
                "commitment_loss": commitment_loss.item(),
            }
        else:
            # Embedding loss ||sg[z] - z_q||^2 trains the codebook (gradient flows to codebook).
            z_q = self.codebook(indices).view(z_permuted.shape)
            embedding_loss = torch.mean((z_permuted.detach() - z_q) ** 2)
            codebook_loss = commitment_loss + embedding_loss
            z_q_out = z_permuted + (z_q - z_permuted).detach()
            stats = {
                "active_codes": active_codes,
                "utilization_pct": utilization_pct,
                "perplexity": perplexity.item(),
                "embedding_loss": embedding_loss.item(),
                "commitment_loss": commitment_loss.item(),
            }

        z_q_out = z_q_out.permute(0, 3, 1, 2)
        return z_q_out, indices, codebook_loss, stats

    def _ema_update(self, z_flattened: torch.Tensor, encodings: torch.Tensor):
        """Update codebook vectors via EMA."""
        batch_cluster_size = encodings.sum(0)

        # EMA cluster size update
        self.ema_cluster_size.mul_(self.ema_decay).add_(
            batch_cluster_size, alpha=1 - self.ema_decay
        )

        # Laplace smoothing for numerical stability
        n = self.ema_cluster_size.sum()
        smoothed_size = (
            (self.ema_cluster_size + self.ema_epsilon)
            / (n + self.num_codebook_vectors * self.ema_epsilon)
            * n
        )

        # EMA embedding sum update
        dw = encodings.t() @ z_flattened
        self.ema_embedding_sum.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

        # Update codebook weights from EMA statistics
        self.codebook.weight.data.copy_(
            self.ema_embedding_sum / smoothed_size.unsqueeze(1)
        )

        # Track code usage for epoch-based dead code detection
        self.code_usage.add_(batch_cluster_size)

    def reset_dead_codes(self, z_flattened: torch.Tensor, usage_threshold_pct: float = 0.1) -> int:
        """Reset codes below usage threshold to random encoder outputs.

        Only called from VAEVQGANTrainer during Phase 1 (VQGAN training). Never run
        during transformer prior training (Phase 2).

        Args:
            z_flattened: Recent encoder outputs to sample from [N, latent_dim].
            usage_threshold_pct: Threshold percentage (e.g., 0.1 means < 0.1% usage).

        Returns:
            Number of codes reset.
        """
        total_usage = self.code_usage.sum()
        if total_usage == 0:
            return 0  # No data processed yet
        
        usage_pct = (self.code_usage / total_usage) * 100.0
        dead_mask = usage_pct < usage_threshold_pct
        n_dead = dead_mask.sum().item()
        
        if n_dead > 0:
            # Reset to random encoder outputs
            random_idx = torch.randint(
                0, z_flattened.shape[0], (n_dead,), device=z_flattened.device
            )
            new_vectors = z_flattened[random_idx].detach().clone()
            self.codebook.weight.data[dead_mask] = new_vectors
            self.ema_embedding_sum.data[dead_mask] = new_vectors
            self.ema_cluster_size.data[dead_mask] = 1.0
        
        # Reset usage tracker
        self.code_usage.zero_()
        return n_dead
