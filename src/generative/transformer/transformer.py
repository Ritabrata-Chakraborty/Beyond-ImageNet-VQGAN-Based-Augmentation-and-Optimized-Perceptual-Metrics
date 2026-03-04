"""
Autoregressive transformer prior for VQGAN sampling.

Wraps a GPT model to predict codebook index sequences. Only applicable
when VQ mode is enabled (requires discrete codebook indices).

Source: https://github.com/dome272/VQGAN-pytorch/blob/main/transformer.py
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.mingpt import GPT


# --- Model ---


class VQGANTransformer(nn.Module):
    """Autoregressive transformer prior for VQGAN.

    Args:
        model: VAEVQGAN model instance (must have vq_mode=True).
        device: Device string.
        sos_token: Start-of-sequence token index (used when num_classes=0).
        pkeep: Probability of keeping original indices during training (mask ratio).
        block_size: Maximum sequence length for GPT.
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        sos_token: int = 0,
        pkeep: float = 0.5,
        block_size: int = 512,
        n_layer: int = 12,
        n_head: int = 16,
        n_embd: int = 1024,
        **kwargs,
    ):
        super().__init__()
        if not hasattr(model, 'vq_mode') or not model.vq_mode:
            raise ValueError("VQGANTransformer requires VAEVQGAN with vq_mode=True")
        self.sos_token = sos_token
        self.device = torch.device(device) if isinstance(device, str) else device
        self.vaevqgan = model
        self.pkeep = pkeep
        self.num_classes = getattr(model, 'num_classes', 0)

        # Expand vocab to include class-specific SOS tokens at indices
        # [num_codebook_vectors .. num_codebook_vectors + num_classes - 1]
        vocab_size = self.vaevqgan.num_codebook_vectors
        if self.num_classes > 0:
            self.sos_token_offset = vocab_size
            vocab_size += self.num_classes

        self.transformer = GPT(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
        )

    @torch.no_grad()
    def encode_to_z(self, x: torch.Tensor) -> tuple:
        """Encode images to flattened codebook indices."""
        quant_z, indices, _, _ = self.vaevqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(self, indices: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """Decode codebook indices back to images."""
        # Codebook only has indices [0, num_codebook_vectors); clamp to avoid gather OOB
        num_codes = self.vaevqgan.num_codebook_vectors
        indices = indices.clamp(0, num_codes - 1)
        latent_size = self.vaevqgan.latent_size
        ix_to_vectors = self.vaevqgan.codebook.codebook(indices).reshape(
            indices.shape[0], latent_size, latent_size, self.vaevqgan.latent_channels
        )
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        return self.vaevqgan.decode(ix_to_vectors, labels=labels)

    def _get_sos_tokens(self, batch_size: int, labels: torch.Tensor = None) -> torch.Tensor:
        """Get SOS tokens: class-dependent if num_classes > 0, else fixed."""
        if self.num_classes > 0 and labels is not None:
            # Clamp labels so SOS = offset+label stays in [offset, offset+num_classes-1] for GPT embedding
            labels = labels.clamp(0, self.num_classes - 1)
            return (labels + self.sos_token_offset).unsqueeze(1).long().to(self.device)
        return (torch.ones(batch_size, 1) * self.sos_token).long().to(self.device)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> tuple:
        """Forward pass: encode, mask, predict codebook indices.

        Args:
            x: Input images [B, C, H, W].
            labels: Class labels [B] for class-conditioned SOS tokens (optional).

        Returns:
            (logits, target_indices)
        """
        _, indices = self.encode_to_z(x)
        # Clamp so indices are valid for codebook and for GPT vocab (body tokens 0..num_codes-1)
        num_codes = self.vaevqgan.num_codebook_vectors
        indices = indices.clamp(0, num_codes - 1)

        sos_tokens = self._get_sos_tokens(x.shape[0], labels)

        # Random masking for training
        mask = torch.bernoulli(
            self.pkeep * torch.ones(indices.shape, device=indices.device)
        ).round().to(dtype=torch.int64)

        random_indices = torch.randint_like(indices, high=self.vaevqgan.num_codebook_vectors)
        new_indices = mask * indices + (1 - mask) * random_indices
        new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        target = indices
        logits, _ = self.transformer(new_indices[:, :-1])
        return logits, target

    def top_k_logits(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Zero out logits below the top-k values."""
        v, _ = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        steps: int = 256,
        temperature: float = 1.0,
        top_k: int = 100,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """Autoregressively sample codebook index sequences.

        Args:
            x: Starting indices [B, T] (can be empty for full sampling).
            c: SOS tokens [B, 1] (ignored if labels provided and num_classes > 0).
            steps: Number of autoregressive steps.
            temperature: Sampling temperature.
            top_k: Top-k filtering value.
            labels: Class labels [B] for class-conditioned SOS tokens (optional).
        """
        self.transformer.eval()
        if self.num_classes > 0 and labels is not None:
            c = self._get_sos_tokens(x.shape[0], labels)
        x = torch.cat((c, x), dim=1)
        # Only sample codebook indices [0, num_codebook_vectors); class SOS tokens
        # (>= num_codebook_vectors) must not appear in the sequence passed to z_to_image.
        num_codes = self.vaevqgan.num_codebook_vectors
        for _ in range(steps):
            logits, _ = self.transformer(x)
            logits = logits[:, -1, :] / temperature
            # Mask out class-token logits so we only sample codebook indices
            if logits.shape[-1] > num_codes:
                logits = logits.clone()
                logits[..., num_codes:] = -float("inf")
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, ix), dim=1)
        x = x[:, c.shape[1]:]
        self.transformer.train()
        return x

    @torch.no_grad()
    def log_images(self, x: torch.Tensor, labels: torch.Tensor = None) -> tuple:
        """Generate reconstruction + sampling visualizations for logging.

        Args:
            x: Input images [B, C, H, W].
            labels: Class labels [B] for class-conditioned generation (optional).
        """
        _, indices = self.encode_to_z(x)
        sos_tokens = self._get_sos_tokens(x.shape[0], labels)

        # Half-context sample
        start_indices = indices[:, :indices.shape[1] // 2]
        sample_indices = self.sample(
            start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1],
            labels=labels,
        )
        half_sample = self.z_to_image(sample_indices, labels=labels)

        # Full sample (no context)
        start_indices = indices[:, :0]
        sample_indices = self.sample(
            start_indices, sos_tokens, steps=indices.shape[1], labels=labels,
        )
        full_sample = self.z_to_image(sample_indices, labels=labels)

        x_rec = self.z_to_image(indices, labels=labels)
        log = {"input": x, "rec": x_rec, "half_sample": half_sample, "full_sample": full_sample}
        return log, torch.concat((x, x_rec, half_sample, full_sample))

    def load_checkpoint(self, path: str, device: str | torch.device | None = None) -> None:
        """Load model weights from checkpoint.

        Args:
            path: Path to checkpoint file.
            device: Device to map tensors to (e.g. 'cpu'). None uses save-time device.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        map_location = device if device is not None else None
        self.load_state_dict(torch.load(path, map_location=map_location))

    def save_checkpoint(self, path: str) -> None:
        """Save model weights to checkpoint."""
        torch.save(self.state_dict(), path)
