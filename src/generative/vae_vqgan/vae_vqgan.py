"""
Unified VQ-VAEGAN model.

- VQ mode (vq_mode=True): Discrete latent space via vector quantization (CodeBook).
- VAE mode (vq_mode=False): Continuous latent space via reparameterization (mu + logvar).

Both modes share: Encoder, Decoder, and post-quantization convolution.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from vae_vqgan import Encoder, Decoder, CodeBook

# --- Constants ---
LAMBDA_GRAD_EPS = 1e-4
LAMBDA_CLAMP_MAX = 10.0


# --- Model ---


class VAEVQGAN(nn.Module):
    """Unified VAE-VQGAN model with configurable latent space.

    Args:
        img_channels: Number of input image channels.
        img_size: Spatial size of input images.
        latent_channels: Number of latent space channels.
        latent_size: Spatial size of latent representation.
        intermediate_channels: Channel sizes for encoder/decoder layers.
        num_residual_blocks_encoder: Residual blocks per encoder stage.
        num_residual_blocks_decoder: Residual blocks per decoder stage.
        dropout: Dropout probability.
        attention_resolution: Resolutions at which to apply attention.
        vq_mode: If True, use codebook (VQGAN). If False, use reparameterization (VAEGAN).
        num_classes: Number of classes for conditional generation (0 = unconditional).
        num_codebook_vectors: Number of codebook entries (VQ mode only).
        codebook_ema: If True, update codebook via EMA; if False, use embedding loss gradient (VQ mode only).
        codebook_ema_decay: EMA decay rate (VQ mode only, when codebook_ema=True).
        codebook_ema_epsilon: Epsilon for Laplace smoothing in EMA (VQ mode only, when codebook_ema=True).
    """

    def __init__(
        self,
        img_channels: int = 1,
        img_size: int = 256,
        latent_channels: int = 256,
        latent_size: int = 16,
        intermediate_channels: list = [128, 128, 256, 256, 512],
        num_residual_blocks_encoder: int = 2,
        num_residual_blocks_decoder: int = 3,
        dropout: float = 0.0,
        attention_resolution: list = [16],
        vq_mode: bool = False,
        num_classes: int = 0,
        num_codebook_vectors: int = 1024,
        codebook_ema: bool = True,
        codebook_ema_decay: float = 0.99,
        codebook_ema_epsilon: float = 1e-5,
    ):
        super().__init__()

        self.img_channels = img_channels
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.vq_mode = vq_mode
        self.num_classes = num_classes
        self.num_codebook_vectors = num_codebook_vectors

        self.encoder = Encoder(
            img_channels=img_channels,
            image_size=img_size,
            latent_channels=latent_channels,
            intermediate_channels=intermediate_channels[:],
            num_residual_blocks=num_residual_blocks_encoder,
            dropout=dropout,
            attention_resolution=attention_resolution,
        )

        self.decoder = Decoder(
            img_channels=img_channels,
            latent_channels=latent_channels,
            latent_size=latent_size,
            intermediate_channels=intermediate_channels[:],
            num_residual_blocks=num_residual_blocks_decoder,
            dropout=dropout,
            attention_resolution=attention_resolution,
        )

        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

        if vq_mode:
            self.codebook = CodeBook(
                num_codebook_vectors=num_codebook_vectors,
                latent_dim=latent_channels,
                use_ema=codebook_ema,
                ema_decay=codebook_ema_decay,
                ema_epsilon=codebook_ema_epsilon,
            )
            self.quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        else:
            self.fc_mu = nn.Conv2d(latent_channels, latent_channels, 1)
            self.fc_logvar = nn.Conv2d(latent_channels, latent_channels, 1)

        # Class conditioning (shared between VQ and VAE modes)
        if num_classes > 0:
            self.class_embedding = nn.Embedding(num_classes, latent_channels)
        else:
            self.class_embedding = None

    def _add_class_conditioning(self, z: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """Add class embedding to latent tensor z [B, C, H, W]."""
        if self.class_embedding is None or labels is None:
            return z
        emb = self.class_embedding(labels)            # [B, latent_channels]
        emb = emb.unsqueeze(-1).unsqueeze(-1)         # [B, latent_channels, 1, 1]
        return z + emb.expand_as(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || N(0, I)). Sum over latent dims [C,H,W], mean over batch."""
        kl_per_sample = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]
        )
        return kl_per_sample.mean()

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> tuple:
        """Forward pass.

        Args:
            x: Input images [B, C, H, W].
            labels: Class labels [B] for conditional generation (optional).

        Returns:
            VQ mode: (decoded_images, codebook_indices, codebook_loss, utilization_stats)
            VAE mode: (decoded_images, mu, logvar, kl_loss)
        """
        encoded_images = self.encoder(x)

        if self.vq_mode:
            quant_x = self.quant_conv(encoded_images)
            z_q, codebook_indices, codebook_loss, utilization_stats = self.codebook(quant_x)
            post_quant_x = self.post_quant_conv(z_q)
            post_quant_x = self._add_class_conditioning(post_quant_x, labels)
            decoded_images = self.decoder(post_quant_x)
            return decoded_images, codebook_indices, codebook_loss, utilization_stats
        else:
            mu = self.fc_mu(encoded_images)
            logvar = self.fc_logvar(encoded_images)
            z = self.reparameterize(mu, logvar)
            post_quant_x = self.post_quant_conv(z)
            post_quant_x = self._add_class_conditioning(post_quant_x, labels)
            decoded_images = self.decoder(post_quant_x)
            kl_loss = self.kl_divergence(mu, logvar)
            return decoded_images, mu, logvar, kl_loss

    def encode(self, x: torch.Tensor) -> tuple:
        """Encode input images into the latent space.

        Returns:
            VQ mode: (z_q, codebook_indices, codebook_loss, utilization_stats)
            VAE mode: (z, mu, logvar)
        """
        x = self.encoder(x)

        if self.vq_mode:
            quant_x = self.quant_conv(x)
            z_q, codebook_indices, codebook_loss, utilization_stats = self.codebook(quant_x)
            return z_q, codebook_indices, codebook_loss, utilization_stats
        else:
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar

    def decode(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """Decode latent representation to image.

        Args:
            x: Latent tensor [B, C, H, W].
            labels: Class labels [B] for conditional generation (optional).
        """
        x = self.post_quant_conv(x)
        x = self._add_class_conditioning(x, labels)
        return self.decoder(x)

    def sample(self, n_samples: int, device: str | torch.device = "cuda", labels: torch.Tensor = None) -> torch.Tensor:
        """Generate images by sampling z ~ N(0, I). VAE mode only.

        Args:
            n_samples: Number of images to generate.
            device: Device for generated tensors.
            labels: Class labels [n_samples] for conditional generation (optional).
        """
        if self.vq_mode:
            raise NotImplementedError(
                "VQ mode sampling requires the transformer prior. "
                "Use VQGANTransformer.sample() instead."
            )
        z = torch.randn(
            n_samples, self.latent_channels, self.latent_size, self.latent_size,
            device=torch.device(device) if isinstance(device, str) else device,
        )
        return self.decode(z, labels=labels)

    def calculate_lambda(self, perceptual_loss, gan_loss):
        """Adaptive weight for balancing perceptual and GAN losses (Eq. 7 in paper).

        Falls back to 1.0 if the decoder structure is unexpected or the losses
        are not connected to the decoder's computation graph.
        """
        try:
            last_layer = self.decoder.model[-1]
            last_layer_weight = last_layer.weight

            perceptual_loss_grads = torch.autograd.grad(
                perceptual_loss, last_layer_weight, retain_graph=True
            )[0]
            gan_loss_grads = torch.autograd.grad(
                gan_loss, last_layer_weight, retain_graph=True
            )[0]
        except (IndexError, AttributeError, RuntimeError):
            return torch.tensor(1.0, device=perceptual_loss.device)

        lmda = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + LAMBDA_GRAD_EPS)
        lmda = torch.clamp(lmda, 0, LAMBDA_CLAMP_MAX).detach()
        return 0.8 * lmda

    @staticmethod
    def adopt_weight(disc_factor: float, i: int, threshold: int, value: float = 0.0) -> float:
        """Delay discriminator activation until step >= threshold."""
        if i < threshold:
            disc_factor = value
        return disc_factor

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
