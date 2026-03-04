"""
Trainer for the autoregressive transformer prior (VQGAN mode only).

Trains a GPT model to predict codebook index sequences for autoregressive sampling.
Supports diversity auxiliary loss and label smoothing to avoid perplexity collapse.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb

from utils import get_optimizer, get_scheduler


# --- Trainer class ---


class TransformerTrainer:
    """Trainer for the VQGAN transformer prior.

    Args:
        model: VQGANTransformer model instance.
        run: wandb run object (or None to disable logging).
        experiment_dir: Directory for saving outputs.
        device: Device string.
        learning_rate: Learning rate.
        beta1: AdamW beta1.
        beta2: AdamW beta2.
        logging_config: Dict with logging interval settings.
    """

    def __init__(
        self,
        model: nn.Module,
        run,
        experiment_dir: str = "experiments",
        device: str = "cuda",
        optimizer: str = "adamw",
        learning_rate: float = 4.5e-06,
        beta1: float = 0.9,
        beta2: float = 0.95,
        weight_decay: float = 0.01,
        scheduler: str = "none",
        scheduler_kwargs: dict = None,
        logging_config: dict = None,
        class_display_order: list = None,
        initial_step: int = 0,
        transformer_diversity_weight: float = 0.0,
        transformer_label_smoothing: float = 0.0,
    ):
        self.run = run
        self.experiment_dir = experiment_dir
        self.model = model
        self.device = device
        self.global_step = initial_step
        self.class_display_order = class_display_order or []

        # Logging intervals from config
        log_cfg = logging_config or {}
        self.print_every_n_steps = log_cfg.get("print_every_n_steps", 10)
        self.image_log_every_n_steps = log_cfg.get("image_log_every_n_steps", 50)
        self.save_every_n_epochs = log_cfg.get("save_every_n_epochs", 20)

        self.optimizer_name = (optimizer or "adamw").lower()
        self.scheduler_name = (scheduler or "none").lower()
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.transformer_diversity_weight = transformer_diversity_weight
        self.transformer_label_smoothing = transformer_label_smoothing
        self.optim = self._configure_optimizers(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2, weight_decay=weight_decay
        )

    def _configure_optimizers(
        self, learning_rate: float, beta1: float, beta2: float, weight_decay: float = 0.01
    ) -> torch.optim.Optimizer:
        """Build optimizer with selective weight decay (transformer convention)."""
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")
        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return get_optimizer(
            self.optimizer_name,
            optim_groups,
            learning_rate=learning_rate,
            betas=(beta1, beta2),
            eps=1e-8,
        )

    def _save_checkpoint(self, epoch: int) -> None:
        """Save transformer checkpoint for a given epoch."""
        ckpt_dir = os.path.join(self.experiment_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Epoch-specific checkpoint
        self.model.save_checkpoint(os.path.join(ckpt_dir, f"transformer_epoch{epoch}.pt"))

        # Latest checkpoint (overwritten each save)
        self.model.save_checkpoint(os.path.join(ckpt_dir, "transformer.pt"))

        print(f"[INFO] Saved transformer checkpoint at epoch {epoch}")

    def train(self, dataloader: torch.utils.data.DataLoader, epochs: int) -> None:
        """Train the transformer prior.

        Args:
            dataloader: Yields (images, labels) or images.
            epochs: Number of training epochs.
        """
        # Keep VQGAN (encoder, codebook, decoder) frozen: no EMA updates or codebook reinit.
        self.model.vaevqgan.eval()
        if hasattr(self.model.vaevqgan, "codebook") and self.model.vaevqgan.codebook is not None:
            self.model.vaevqgan.codebook.eval()

        steps_per_epoch = len(dataloader)
        sk = self.scheduler_kwargs
        scheduler = get_scheduler(
            self.scheduler_name,
            self.optim,
            T_0=sk.get("T_0", 10),
            T_mult=sk.get("T_mult", 2),
            eta_min=sk.get("eta_min", 1.0e-6),
            steps_per_epoch=steps_per_epoch,
        )

        for epoch in range(epochs):
            for index, batch in enumerate(dataloader):
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    imgs, labels = batch
                    labels = labels.to(device=self.device)
                elif isinstance(batch, (tuple, list)):
                    imgs = batch[0]
                    labels = None
                else:
                    imgs = batch
                    labels = None

                self.optim.zero_grad()
                imgs = imgs.to(device=self.device)
                logits, targets = self.model(imgs, labels=labels)
                # Cross-entropy over full vocab (targets are code indices; SOS in context only)
                ce_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    label_smoothing=self.transformer_label_smoothing,
                )
                loss = ce_loss

                # Diversity auxiliary: encourage marginal over codebook indices to be uniform
                if self.transformer_diversity_weight > 0:
                    num_codes = self.model.vaevqgan.num_codebook_vectors
                    logits_codes = logits[..., :num_codes]
                    probs = F.softmax(logits_codes, dim=-1)
                    marginal = probs.mean(dim=(0, 1))
                    eps = 1e-8
                    entropy = -torch.sum(marginal * torch.log(marginal + eps))
                    log_num_codes = math.log(num_codes)
                    diversity_loss = log_num_codes - entropy
                    loss = loss + self.transformer_diversity_weight * diversity_loss

                loss.backward()
                self.optim.step()

                # W&B logging
                if self.run is not None:
                    perplexity = torch.exp(ce_loss.detach()).item()
                    log_dict = {
                        "transformer/loss_cross_entropy": ce_loss.item(),
                        "transformer/perplexity": perplexity,
                        "transformer/learning_rate": self.optim.param_groups[0]["lr"],
                    }
                    if self.transformer_diversity_weight > 0:
                        log_dict["transformer/diversity_loss"] = diversity_loss.item()
                    wandb.log(log_dict, step=self.global_step)

                self.global_step += 1

                if scheduler is not None:
                    scheduler.step()

                if index % self.print_every_n_steps == 0:
                    perplexity = torch.exp(ce_loss.detach()).item()
                    print(
                        f"Epoch: {epoch + 1}/{epochs} | Batch: {index}/{len(dataloader)} | "
                        f"CE Loss: {ce_loss.item():.4f} | Perplexity: {perplexity:.2f}"
                    )

                # Log class-ordered generated grid at image_log_every_n_steps
                if index % self.image_log_every_n_steps == 0 and self.run is not None:
                    log_dict = {}
                    if self.class_display_order:
                        try:
                            with torch.no_grad():
                                gen_labels = torch.tensor(
                                    self.class_display_order, device=self.device, dtype=torch.long
                                )
                                n_gen = len(gen_labels)
                                start_indices = torch.zeros((n_gen, 0), device=self.device).long()
                                sample_indices = self.model.sample(
                                    start_indices, c=None, steps=256, labels=gen_labels
                                )
                                generated = self.model.z_to_image(sample_indices, labels=gen_labels)
                                gen_grid = torchvision.utils.make_grid(
                                    generated, nrow=n_gen, normalize=True, value_range=(-1, 1)
                                )
                                log_dict["transformer/generated"] = wandb.Image(
                                    gen_grid.permute(1, 2, 0).cpu().numpy(),
                                    caption=f"Epoch {epoch + 1}, Batch {index}: Generated (class order)"
                                )
                        except Exception as e:
                            print(f"[WARNING] Failed to log transformer/generated: {e}")

                    if log_dict:
                        wandb.log(log_dict, step=self.global_step)

            # Periodic checkpoint saving
            if (epoch + 1) % self.save_every_n_epochs == 0:
                self._save_checkpoint(epoch + 1)
