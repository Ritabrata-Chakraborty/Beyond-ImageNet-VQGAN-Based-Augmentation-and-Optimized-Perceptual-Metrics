"""
VAE-VQGAN trainer: configurable latent handling (VQ or VAE) and optional AC-GAN.

- VQ mode: embedding + commitment loss (EMA codebook); both logged separately.
- VAE mode: KL divergence loss (sum over latent dims, mean over batch).
- Scalars logged every step; images logged at epoch end.
- Class-wise reconstruction grid uses class_display_order (9 faulty classes).
- Perceptual loss uses custom LPIPS from PerceptualSimilarity (configurable backbone, layers, linear layers).
"""

from __future__ import annotations

import os
import re

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from utils import weights_init, get_optimizer, get_scheduler
from vae_vqgan import Discriminator

from lpips.lpips import LPIPS as CustomLPIPS
from lpips import pretrained_networks as _pn


# --- Helpers ---


def _remap_lpips_slice_keys_to_features(
    state: dict,
    net_type: str,
    layer_indices: list[int],
) -> dict:
    """Remap net.sliceN.M.* to net.features.M.* so backbone loads; handles slice-only and dual-key checkpoints."""
    if not layer_indices:
        return state
    if net_type in ("vgg", "vgg16"):
        endpoints = [_pn.VGG16_LAYER_ENDPOINTS[i] for i in layer_indices]
    elif net_type == "alex":
        endpoints = [_pn.ALEXNET_LAYER_ENDPOINTS[i] for i in layer_indices]
    elif net_type == "squeeze":
        endpoints = [_pn.SQUEEZENET_LAYER_ENDPOINTS[i] for i in layer_indices]
    else:
        return state
    starts = [0] + endpoints[:-1]
    pattern = re.compile(r"^net\.slice(\d+)\.(\d+)\.(.+)$")
    new_state = dict(state)
    for key in list(new_state.keys()):
        m = pattern.match(key)
        if m is None:
            continue
        n_slice, m_mod, suffix = int(m.group(1)), int(m.group(2)), m.group(3)
        if n_slice < 1 or n_slice > len(starts):
            continue
        # Bounds check: m_mod must be a valid global index within this slice
        if not (starts[n_slice - 1] <= m_mod < endpoints[n_slice - 1]):
            continue
        # m_mod is already the global feature index (not a local offset)
        new_key = f"net.features.{m_mod}.{suffix}"
        feature_key_already_present = new_key in new_state
        if not feature_key_already_present:
            # Slice-only checkpoint: create the feature alias and remove the slice
            # key so it does not become an unexpected key in a features-only model.
            new_state[new_key] = new_state[key]
            del new_state[key]
        # else: both keys present (current format) - keep the slice key intact;
        # the model expects it too and the feature key already has the right value.
    return new_state


# --- Trainer class ---


class VAEVQGANTrainer:
    """Trainer for VAEVQGAN with optional AC-GAN classification.

    Args:
        model: VAEVQGAN model instance.
        run: wandb run object (or None to disable logging).
        vq_mode: If True, VQGAN mode; if False, VAEGAN mode.
        device: Device string or torch.device.
        learning_rate: Learning rate for both generator and discriminator.
        beta1: Adam beta1.
        beta2: Adam beta2.
        perceptual_loss_factor: Weight for perceptual loss.
        rec_loss_factor: Weight for L1 reconstruction loss.
        kl_factor: KL loss weight after kl_start (VAE mode only).
        kl_start: Step at which to activate KL loss; before that KL weight is 0 (VAE mode only).
        disc_factor: Discriminator loss weight.
        disc_start: Step at which to activate discriminator.
        classifier_enabled: Enable AC-GAN classification head.
        num_classes: Number of classes (if classifier enabled).
        cls_loss_weight: Weight for classification loss.
        classifier_dropout: Dropout for classification head.
        normal_class_idx: Index of the normal/healthy class.
        experiment_dir: Directory for saving outputs.
        perceptual_model: LPIPS backbone ("vgg", "vgg16", "alex", or "squeeze").
        perceptual_layers: Optional list of backbone layer indices to use; None = default set.
        perceptual_backbone_path: Optional path to backbone checkpoint (overrides ImageNet trunk).
        perceptual_model_path: Optional path to linear-layer weights; None = use pretrained v0.1 if use_linear.
        perceptual_use_linear: If True, use learned linear layers on top of features; if False, baseline (average).
        logging_config: Dict with logging interval settings.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        run,
        vq_mode: bool = False,
        device: str = "cuda",
        learning_rate: float = 2.25e-05,
        beta1: float = 0.5,
        beta2: float = 0.9,
        perceptual_loss_factor: float = 1.0,
        rec_loss_factor: float = 1.0,
        kl_factor: float = 1.0,
        kl_start: int = 3000,
        disc_factor: float = 1.0,
        disc_start: int = 100,
        classifier_enabled: bool = False,
        num_classes: int = 9,
        cls_loss_weight: float = 1.0,
        classifier_dropout: float = 0.3,
        normal_class_idx: int | None = None,
        experiment_dir: str = "./experiments",
        perceptual_model: str = "vgg",
        perceptual_layers: list = None,
        perceptual_backbone_path: str = None,
        perceptual_model_path: str = None,
        perceptual_use_linear: bool = True,
        logging_config: dict = None,
        class_display_order: list = None,
        optimizer: str = "adam",
        scheduler: str = "none",
        weight_decay: float = 0.0,
        scheduler_kwargs: dict = None,
        codebook_reset_steps: int = 100,
        codebook_usage_threshold: float = 0.1,
        use_amp: bool = True,
        grad_clip_norm: float = 1.0,
        disc_label_smooth_real: float = 0.9,
        disc_label_smooth_fake: float = -0.9,
        disc_lr_factor: float = 1.0,
        r1_gamma: float = 0.0,
        disc_ramp_steps: int = 0,
        disc_n_layers: int = 3,
    ):
        self.run = run
        self.device = device
        self.vq_mode = vq_mode
        self.model = model
        
        # Mixed precision training for memory savings
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Logging: scalars every step, images every N steps
        log_cfg = logging_config or {}
        self.print_every_n_steps = log_cfg.get("print_every_n_steps", 10)
        self.image_log_every_n_steps = log_cfg.get("image_log_every_n_steps", 50)
        self.n_display_images = log_cfg.get("n_display_images", 8)
        self.save_every_n_epochs = log_cfg.get("save_every_n_epochs", 20)
        
        # Codebook reset configuration (VQ mode only)
        self.codebook_reset_steps = codebook_reset_steps
        self.codebook_usage_threshold = codebook_usage_threshold
        self.steps_since_codebook_reset = 0

        # Classifier and class display order (9 faulty classes when normal_class_idx is None)
        self.classifier_enabled = classifier_enabled
        self.num_classes = num_classes if classifier_enabled else 0
        self.cls_loss_weight = cls_loss_weight
        self.normal_class_idx = normal_class_idx if (classifier_enabled and normal_class_idx is not None) else None
        self.class_display_order = (class_display_order or list(range(num_classes))) if classifier_enabled else []

        # Discriminator with optional classifier head
        self.discriminator = Discriminator(
            image_channels=self.model.img_channels,
            num_classes=self.num_classes,
            classifier_dropout=classifier_dropout,
            n_layers=disc_n_layers,
        ).to(self.device)
        self.discriminator.apply(weights_init)

        if self.classifier_enabled:
            self.cls_criterion = nn.CrossEntropyLoss()

        # Perceptual loss: custom LPIPS from PerceptualSimilarity (backbone, layers, linear layers configurable)
        # When model_path is set, load it in the trainer so we can verify keys and handle DataParallel prefix.
        load_custom_path = bool(perceptual_model_path)
        self.perceptual_loss = CustomLPIPS(
            net=perceptual_model,
            layers=perceptual_layers,
            backbone_path=perceptual_backbone_path,
            model_path=None if load_custom_path else perceptual_model_path,
            lpips=perceptual_use_linear,
            pretrained=perceptual_use_linear and not load_custom_path,
            eval_mode=True,
            verbose=False,
        )
        if load_custom_path:
            state = torch.load(perceptual_model_path, map_location="cpu")
            if not isinstance(state, dict):
                raise ValueError(f"LPIPS checkpoint must be a state dict, got {type(state).__name__}")
            # Checkpoint saved with DataParallel may have "module." prefix
            if any(k.startswith("module.") for k in state):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            # Checkpoint may have backbone params under net.slice*.*; our model expects net.features.*.
            state = _remap_lpips_slice_keys_to_features(
                state,
                perceptual_model,
                perceptual_layers or [],
            )
            load_result = self.perceptual_loss.load_state_dict(state, strict=False)
            missing, unexpected = load_result.missing_keys, load_result.unexpected_keys
            if missing:
                print(f"[WARNING] LPIPS checkpoint missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"[WARNING] LPIPS checkpoint unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
            if not missing and not unexpected:
                print(f"[INFO] LPIPS loaded from {perceptual_model_path} (strict match).")
        self.perceptual_loss = self.perceptual_loss.to(self.device)

        # Optimizers and schedulers
        self.optimizer_name = optimizer.lower()
        self.scheduler_name = (scheduler or "none").lower()
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.weight_decay = weight_decay
        disc_lr = learning_rate * disc_lr_factor
        self.opt_model, self.opt_disc = self._configure_optimizers(
            learning_rate=learning_rate, disc_learning_rate=disc_lr, beta1=beta1, beta2=beta2
        )

        # Hyperparameters
        self.disc_factor = disc_factor
        self.disc_start = disc_start
        self.perceptual_loss_factor = perceptual_loss_factor
        self.rec_loss_factor = rec_loss_factor
        self.kl_factor = kl_factor
        self.kl_start = kl_start
        self.grad_clip_norm = grad_clip_norm
        self.disc_label_smooth_real = disc_label_smooth_real
        self.disc_label_smooth_fake = disc_label_smooth_fake
        self.disc_lr_factor = disc_lr_factor
        self.r1_gamma = r1_gamma
        self.disc_ramp_steps = disc_ramp_steps

        # State
        self.experiment_dir = experiment_dir
        self.global_step = 0
        self.current_epoch = 0
        self.gif_images = []
        self.gif_frame_limit = 50

    def _get_model_params(self) -> list:
        """Collect generator parameters (for gradient clipping)."""
        params = (
            list(self.model.encoder.parameters())
            + list(self.model.decoder.parameters())
            + list(self.model.post_quant_conv.parameters())
        )
        if self.vq_mode:
            params += list(self.model.codebook.parameters())
            params += list(self.model.quant_conv.parameters())
        else:
            params += list(self.model.fc_mu.parameters())
            params += list(self.model.fc_logvar.parameters())
        if self.model.class_embedding is not None:
            params += list(self.model.class_embedding.parameters())
        return params

    def _configure_optimizers(
        self, learning_rate: float, disc_learning_rate: float, beta1: float, beta2: float
    ) -> tuple:
        """Build optimizers for generator and discriminator."""
        params = self._get_model_params()

        opt_model = get_optimizer(
            self.optimizer_name,
            params,
            learning_rate=learning_rate,
            betas=(beta1, beta2),
            weight_decay=self.weight_decay,
            eps=1e-08,
        )
        opt_disc = get_optimizer(
            self.optimizer_name,
            self.discriminator.parameters(),
            learning_rate=disc_learning_rate,
            betas=(beta1, beta2),
            weight_decay=self.weight_decay,
            eps=1e-08,
        )
        return opt_model, opt_disc

    def _create_schedulers(self, epochs: int, steps_per_epoch: int) -> tuple:
        """Create LR schedulers for model and discriminator. Step both every batch."""
        sk = self.scheduler_kwargs
        sched_model = get_scheduler(
            self.scheduler_name,
            self.opt_model,
            T_0=sk.get("T_0", 10),
            T_mult=sk.get("T_mult", 2),
            eta_min=sk.get("eta_min", 1.0e-6),
            steps_per_epoch=steps_per_epoch,
        )
        sched_disc = get_scheduler(
            self.scheduler_name,
            self.opt_disc,
            T_0=sk.get("T_0", 10),
            T_mult=sk.get("T_mult", 2),
            eta_min=sk.get("eta_min", 1.0e-6),
            steps_per_epoch=steps_per_epoch,
        )
        return sched_model, sched_disc

    def step(self, imgs: torch.Tensor, labels: torch.Tensor = None) -> tuple:
        """Perform a single training step.

        Args:
            imgs: Input images [B, C, H, W].
            labels: Class labels [B] (required if classifier_enabled).

        Returns:
            (decoded_images, total_loss, total_disc_loss)
        """
        # Forward pass with mixed precision (mode-dependent)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            if self.vq_mode:
                decoded_images, codebook_indices, codebook_loss, utilization_stats = self.model(imgs, labels)
            else:
                decoded_images, mu, logvar, kl_loss = self.model(imgs, labels)

            # Perceptual + Reconstruction Loss
            if imgs.shape[1] == 1:
                imgs_3ch = imgs.repeat(1, 3, 1, 1)
                decoded_3ch = decoded_images.repeat(1, 3, 1, 1)
                perceptual_loss = self.perceptual_loss(imgs_3ch, decoded_3ch)
                del imgs_3ch, decoded_3ch
            else:
                perceptual_loss = self.perceptual_loss(imgs, decoded_images)
            # LPIPS can be slightly negative (custom weights or numerical); clamp so loss is non-negative.
            perceptual_loss = torch.clamp(perceptual_loss, min=0.0)

            rec_loss = torch.abs(imgs - decoded_images)
            perceptual_rec_loss = (
                self.perceptual_loss_factor * perceptual_loss
                + self.rec_loss_factor * rec_loss
            ).mean()

            # Discriminator factor (delayed activation, optional linear ramp)
            disc_factor_base = self.model.adopt_weight(
                self.disc_factor, self.global_step, threshold=self.disc_start
            )
            if self.disc_ramp_steps > 0 and disc_factor_base > 0:
                steps_into_ramp = self.global_step - self.disc_start
                ramp = min(1.0, max(0.0, steps_into_ramp) / self.disc_ramp_steps)
                disc_factor = disc_factor_base * ramp
            else:
                disc_factor = disc_factor_base

            # Generator loss
            if self.classifier_enabled:
                disc_fake_for_gen, _ = self.discriminator(decoded_images, return_cls=True)
            else:
                disc_fake_for_gen = self.discriminator(decoded_images, return_cls=False)
            g_loss = -torch.mean(disc_fake_for_gen)

            lmda = self.model.calculate_lambda(perceptual_rec_loss, g_loss)

            # NaN-safe GAN contribution: if lambda or g_loss is NaN/inf, fall back to 0
            gan_contribution = disc_factor * lmda * g_loss
            if torch.isnan(gan_contribution) or torch.isinf(gan_contribution):
                gan_contribution = torch.tensor(0.0, device=self.device)

            # Mode-specific total loss
            if self.vq_mode:
                total_loss = perceptual_rec_loss + codebook_loss + gan_contribution
            else:
                # KL weight: 0 until kl_start, then kl_factor (same pattern as disc_factor/disc_start)
                current_kl_weight = self.model.adopt_weight(
                    self.kl_factor, self.global_step, threshold=self.kl_start
                )
                total_loss = perceptual_rec_loss + current_kl_weight * kl_loss + gan_contribution

            # Discriminator + Classifier loss (all inputs detached -- independent graph)
            if self.classifier_enabled:
                disc_real, cls_real = self.discriminator(imgs.detach(), return_cls=True)
                disc_fake_for_disc, _ = self.discriminator(decoded_images.detach(), return_cls=True)
                cls_loss = self.cls_criterion(cls_real, labels)
            else:
                disc_real = self.discriminator(imgs.detach(), return_cls=False)
                disc_fake_for_disc = self.discriminator(decoded_images.detach(), return_cls=False)
                cls_loss = torch.tensor(0.0, device=self.device)

            # Hinge loss with optional label smoothing
            d_loss_real = torch.mean(F.relu(self.disc_label_smooth_real - disc_real))
            d_loss_fake = torch.mean(F.relu(disc_fake_for_disc - self.disc_label_smooth_fake))
            gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

            total_disc_loss = gan_loss + self.cls_loss_weight * cls_loss if self.classifier_enabled else gan_loss

            # R1 gradient penalty on real images (when enabled and disc is active)
            if self.r1_gamma > 0 and disc_factor > 0:
                imgs_r1 = imgs.detach().requires_grad_(True)
                if self.classifier_enabled:
                    disc_real_r1, _ = self.discriminator(imgs_r1, return_cls=True)
                else:
                    disc_real_r1 = self.discriminator(imgs_r1, return_cls=False)
                grad_real = torch.autograd.grad(
                    outputs=disc_real_r1,
                    inputs=imgs_r1,
                    grad_outputs=torch.ones_like(disc_real_r1),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                r1_penalty = grad_real.pow(2).sum()
                total_disc_loss = total_disc_loss + self.r1_gamma * r1_penalty

        # W&B Logging
        if self.run is not None:
            log_dict = {
                "generator/epoch": self.current_epoch,
                "generator/loss_reconstruction": rec_loss.mean().item(),
                "generator/loss_perceptual": perceptual_loss.mean().item(),
                "generator/loss_generator": g_loss.item(),
                "discriminator/loss_gan": gan_loss.item(),
                "generator/loss_total": total_loss.item(),
                "discriminator/loss_real": d_loss_real.item(),
                "discriminator/loss_fake": d_loss_fake.item(),
                "discriminator/factor": disc_factor,
                "generator/lambda": lmda.item() if torch.is_tensor(lmda) else lmda,
                "generator/learning_rate": self.opt_model.param_groups[0]["lr"],
                "discriminator/learning_rate": self.opt_disc.param_groups[0]["lr"],
            }

            # Mode-specific latent loss + codebook metrics (logged every step)
            # Note: commitment_loss = beta * quantization_error (same underlying quantity; only one is used in the loss when EMA).
            if self.vq_mode:
                log_dict["vqgan/quantization_error"] = utilization_stats.get("embedding_loss", 0)
                log_dict["vqgan/commitment_loss"] = utilization_stats.get("commitment_loss", codebook_loss.item())
                log_dict["vqgan/codebook_utilization"] = utilization_stats.get("utilization_pct", 0)
                log_dict["vqgan/codebook_perplexity"] = utilization_stats.get("perplexity", 0)
                log_dict["vqgan/active_codes"] = utilization_stats.get("active_codes", 0)
            else:
                log_dict["vaegan/kl_loss"] = kl_loss.item()
                log_dict["vaegan/kl_weight"] = current_kl_weight

            # Classifier metrics
            if self.classifier_enabled:
                with torch.no_grad():
                    cls_preds = cls_real.argmax(dim=1)
                    overall_acc = (cls_preds == labels).float().mean().item()

                log_dict.update({
                    "classifier/loss": cls_loss.item(),
                    "classifier/acc_overall": overall_acc,
                    "discriminator/loss_total": total_disc_loss.item(),
                })
                if self.normal_class_idx is not None:
                    normal_mask = labels == self.normal_class_idx
                    faulty_mask = labels != self.normal_class_idx
                    normal_acc = (cls_preds[normal_mask] == labels[normal_mask]).float().mean().item() if normal_mask.sum() > 0 else 0.0
                    faulty_acc = (cls_preds[faulty_mask] == labels[faulty_mask]).float().mean().item() if faulty_mask.sum() > 0 else 0.0
                    log_dict["classifier/acc_normal"] = normal_acc
                    log_dict["classifier/acc_faulty"] = faulty_acc
                    log_dict["classifier/acc_gap_nf"] = normal_acc - faulty_acc

            wandb.log(log_dict, step=self.global_step)

            # Image logging removed from steps - only at epoch end

        # Backpropagation with optional mixed precision + gradient clipping.
        # No retain_graph needed: disc losses use detached inputs (independent graph).
        # calculate_lambda's autograd.grad calls use retain_graph=True internally,
        # which keeps the graph alive for this backward() call.
        self.opt_model.zero_grad()
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.opt_model)
            torch.nn.utils.clip_grad_norm_(self._get_model_params(), self.grad_clip_norm)
            self.scaler.step(self.opt_model)
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._get_model_params(), self.grad_clip_norm)
            self.opt_model.step()

        self.opt_disc.zero_grad()
        if self.use_amp:
            self.scaler.scale(total_disc_loss).backward()
            self.scaler.unscale_(self.opt_disc)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip_norm)
            self.scaler.step(self.opt_disc)
            self.scaler.update()
        else:
            total_disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip_norm)
            self.opt_disc.step()

        # Detach outputs: backward() consumed the graph but grad_fn chains
        # still pin GPU memory until the Python objects are collected.
        return decoded_images.detach(), total_loss.item(), total_disc_loss.item()


    def _log_epoch_images(self, epoch: int, imgs: torch.Tensor, decoded_images: torch.Tensor, labels: torch.Tensor = None):
        """Log three image grids: original_samples, reconstructed, and generated (all in class order)."""
        if self.run is None:
            return

        with torch.no_grad():
            log_dict = {}

            # Get class-ordered originals and reconstructions
            if labels is not None and self.class_display_order:
                orig_stack, recon_stack = self._get_class_ordered_samples(imgs, decoded_images, labels)
                if orig_stack is not None:
                    # Original samples grid
                    orig_grid = torchvision.utils.make_grid(
                        orig_stack, nrow=len(self.class_display_order), normalize=True, value_range=(-1, 1)
                    )
                    log_dict["generator/input"] = wandb.Image(
                        orig_grid.cpu(), caption=f"Epoch {epoch + 1}: Original samples (class order)"
                    )
                    
                    # Reconstructed grid
                    recon_grid = torchvision.utils.make_grid(
                        recon_stack, nrow=len(self.class_display_order), normalize=True, value_range=(-1, 1)
                    )
                    log_dict["generator/reconstructed"] = wandb.Image(
                        recon_grid.cpu(), caption=f"Epoch {epoch + 1}: Reconstructed (class order)"
                    )

            # Generated samples: one per class in class_display_order (same order as original/recon)
            n_generate = len(self.class_display_order) if self.class_display_order else self.n_display_images
            nrow_gen = len(self.class_display_order) if self.class_display_order else n_generate
            gen_labels = torch.tensor(self.class_display_order, device=self.device, dtype=torch.long) if self.class_display_order else None
            if self.vq_mode:
                generated = self._generate_vqgan_samples(n_generate, gen_labels)
            else:
                generated = self._generate_vaegan_samples(n_generate, gen_labels)

            if generated is not None:
                gen_grid = torchvision.utils.make_grid(
                    generated, nrow=nrow_gen, normalize=True, value_range=(-1, 1)
                )
                caption_gen = f"Epoch {epoch + 1}: Generated samples (class order)" if self.class_display_order else f"Epoch {epoch + 1}: Generated samples"
                log_dict["generator/generated"] = wandb.Image(
                    gen_grid.cpu(), caption=caption_gen
                )

            wandb.log(log_dict, step=self.global_step)

            # Local GIF frame (using first available reconstruction)
            if len(log_dict) > 0 and "generator/reconstructed" in log_dict:
                # Create GIF from original and reconstructed
                if "generator/input" in log_dict:
                    combined = torch.cat([orig_grid, recon_grid], dim=1)
                    gif_frame = combined.cpu().permute(1, 2, 0).numpy()
                    gif_frame = (gif_frame - gif_frame.min()) / (gif_frame.max() - gif_frame.min() + 1e-8)
                    gif_frame = (gif_frame * 255).astype(np.uint8)
                    self.gif_images.append(gif_frame)
                    
                    # Limit to last gif_frame_limit frames to prevent memory leak
                    if len(self.gif_images) > self.gif_frame_limit:
                        self.gif_images = self.gif_images[-self.gif_frame_limit:]

                    imageio.mimsave(
                        os.path.join(self.experiment_dir, "reconstruction.gif"),
                        self.gif_images,
                        fps=5,
                    )

    def _get_class_ordered_samples(self, imgs: torch.Tensor, decoded: torch.Tensor, labels: torch.Tensor) -> tuple:
        """Get one sample per class in class_display_order. Missing classes filled with zeros.
        
        Returns:
            (orig_stack, recon_stack): Stacked tensors of shape [n_classes, C, H, W]
        """
        if not self.class_display_order or imgs.shape[0] == 0:
            return None, None
        C, H, W = imgs.shape[1], imgs.shape[2], imgs.shape[3]
        device = imgs.device
        placeholder = torch.zeros(1, C, H, W, device=device, dtype=imgs.dtype)

        orig_list, recon_list = [], []
        for class_idx in self.class_display_order:
            idx = (labels == class_idx).nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                i = idx[0].item()
                orig_list.append(imgs[i : i + 1])
                recon_list.append(decoded[i : i + 1])
            else:
                orig_list.append(placeholder)
                recon_list.append(placeholder)

        orig_stack = torch.cat(orig_list, dim=0)
        recon_stack = torch.cat(recon_list, dim=0)
        return orig_stack, recon_stack

    def _generate_vaegan_samples(self, n_samples: int, labels: torch.Tensor = None) -> torch.Tensor:
        """Generate class-conditioned samples from prior N(0,I) for VAEGAN."""
        z_samples = torch.randn(
            n_samples, self.model.latent_channels,
            self.model.latent_size, self.model.latent_size,
            device=self.device,
        )
        return self.model.decode(z_samples, labels=labels)

    def _generate_vqgan_samples(self, n_samples: int, labels: torch.Tensor = None) -> torch.Tensor:
        """Generate class-conditioned samples for VQGAN by sampling random codebook vectors."""
        num_tokens = self.model.latent_size * self.model.latent_size
        random_indices = torch.randint(
            0, self.model.codebook.num_codebook_vectors,
            (n_samples * num_tokens,),
            device=self.device
        )
        vectors = self.model.codebook.codebook(random_indices)
        quant_z = vectors.reshape(
            n_samples, self.model.latent_size, self.model.latent_size, self.model.latent_channels
        ).permute(0, 3, 1, 2)
        return self.model.decode(quant_z, labels=labels)

    def _save_checkpoint(self, epoch: int):
        """Save model and discriminator checkpoints for a given epoch."""
        ckpt_dir = os.path.join(self.experiment_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        name = "vqgan" if self.vq_mode else "vaegan"

        # Epoch-specific checkpoints
        self.model.save_checkpoint(os.path.join(ckpt_dir, f"{name}_epoch{epoch}.pt"))
        torch.save(
            self.discriminator.state_dict(),
            os.path.join(ckpt_dir, f"disc_epoch{epoch}.pt"),
        )

        # Latest checkpoints (overwritten each save)
        self.model.save_checkpoint(os.path.join(ckpt_dir, f"{name}.pt"))
        torch.save(
            self.discriminator.state_dict(),
            os.path.join(ckpt_dir, "disc.pt"),
        )

        print(f"[INFO] Saved checkpoint at epoch {epoch}")

    def train(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        """Train the model for the given number of epochs.

        Args:
            dataloader: Yields (images, labels) if classifier enabled, else images.
            epochs: Number of training epochs.
        """
        mode_name = "VQGAN" if self.vq_mode else "VAEGAN"
        if self.classifier_enabled:
            mode_name += "+AC"

        steps_per_epoch = len(dataloader)
        sched_model, sched_disc = self._create_schedulers(epochs, steps_per_epoch)

        for epoch in range(epochs):
            self.current_epoch = epoch
            last_imgs = None
            last_decoded = None
            last_labels = None
            last_total_loss = None
            last_disc_loss = None

            for index, batch in enumerate(dataloader):
                if self.classifier_enabled:
                    if isinstance(batch, (tuple, list)) and len(batch) == 2:
                        imgs, labels = batch
                        labels = labels.to(self.device)
                    else:
                        raise ValueError(
                            "Classifier enabled but dataloader does not return (imgs, labels). "
                            "Set return_labels=True in the dataloader."
                        )
                else:
                    if isinstance(batch, (tuple, list)):
                        imgs = batch[0]
                    else:
                        imgs = batch
                    labels = None

                imgs = imgs.to(self.device)
                decoded_images, total_loss, disc_loss = self.step(imgs, labels)
                # step() returns detached decoded_images and scalar losses (.item())

                last_imgs = imgs.detach()
                last_decoded = decoded_images
                last_labels = labels
                last_total_loss = total_loss
                last_disc_loss = disc_loss
                
                # Codebook reset (VQ mode only) - step-based
                if self.vq_mode and self.codebook_reset_steps > 0:
                    self.steps_since_codebook_reset += 1
                    if self.steps_since_codebook_reset >= self.codebook_reset_steps:
                        if imgs.numel() > 0:
                            with torch.no_grad():
                                encoded = self.model.encoder(imgs)
                                quant_x = self.model.quant_conv(encoded)
                                z_flattened = quant_x.permute(0, 2, 3, 1).contiguous().view(-1, self.model.latent_channels)
                                n_reset = self.model.codebook.reset_dead_codes(z_flattened, self.codebook_usage_threshold)
                            del encoded, quant_x, z_flattened
                            torch.cuda.empty_cache()
                            
                            if n_reset > 0:
                                print(f"[INFO] Step {self.global_step}: Reset {n_reset} dead codebook entries")
                                if self.run is not None:
                                    wandb.log({"vqgan/codebook_reset": n_reset}, step=self.global_step)
                            
                        self.steps_since_codebook_reset = 0
                
                self.global_step += 1

                if sched_model is not None:
                    sched_model.step()
                if sched_disc is not None:
                    sched_disc.step()

                # Clear CUDA cache periodically to reduce memory fragmentation
                if index % 10 == 0:
                    torch.cuda.empty_cache()

                if index % self.print_every_n_steps == 0:
                    print(
                        f"Epoch: {epoch + 1}/{epochs} | Batch: {index}/{len(dataloader)} | "
                        f"{mode_name} Loss: {total_loss:.4f} | Disc Loss: {disc_loss:.4f}"
                    )

                # Log images (original, reconstructed, generated) every N steps
                if self.global_step % self.image_log_every_n_steps == 0:
                    self._log_epoch_images(epoch, last_imgs, last_decoded, last_labels)

            # Free epoch logging tensors and clear CUDA cache
            del last_imgs, last_decoded, last_labels, last_total_loss, last_disc_loss
            torch.cuda.empty_cache()

            # Periodic checkpoint saving
            if (epoch + 1) % self.save_every_n_epochs == 0:
                self._save_checkpoint(epoch + 1)
