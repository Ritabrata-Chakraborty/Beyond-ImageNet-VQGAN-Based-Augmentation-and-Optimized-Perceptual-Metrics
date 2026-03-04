from __future__ import annotations

import os

import torch
import torchvision
from utils import reproducibility

from trainer import VAEVQGANTrainer, TransformerTrainer


# --- Trainer ---


class Trainer:
    """Top-level orchestrator for VAE-VQGAN training and generation.

    Reads all sub-configs (trainer, classifier, logging, training) from the
    full config dict and delegates to VAEVQGANTrainer / TransformerTrainer.

    Args:
        model: VAEVQGAN model instance.
        transformer: VQGANTransformer instance (or None for VAE mode).
        run: wandb run object (or None).
        config: Full config dict (must contain trainer, classifier, logging, training keys).
        vq_mode: If True, VQGAN mode; if False, VAEGAN mode.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        transformer: torch.nn.Module,
        run,
        config: dict,
        vq_mode: bool = False,
    ) -> None:
        self.model = model
        self.transformer = transformer
        self.vq_mode = vq_mode
        self.run = run
        self.config = config

        training_cfg = config["training"]
        self.device = torch.device(training_cfg["device"]) if isinstance(training_cfg["device"], str) else training_cfg["device"]
        self.experiment_dir = training_cfg["experiment_dir"]

        seed = training_cfg["seed"]
        print(f"[INFO] Setting seed to {seed}")
        reproducibility(seed)

        print(f"[INFO] Results will be saved in {self.experiment_dir}")
        os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)

    @staticmethod
    def _extract_scheduler_kwargs(raw_cfg: dict) -> dict:
        """Extract scheduler_* keys into a scheduler_kwargs dict."""
        scheduler_kwargs = {
            "T_0": raw_cfg.get("scheduler_T_0", 10),
            "T_mult": raw_cfg.get("scheduler_T_mult", 2),
            "eta_min": raw_cfg.get("scheduler_eta_min", 1.0e-6),
        }
        trainer_cfg = {
            k: v for k, v in raw_cfg.items()
            if not (k.startswith("scheduler_") and k != "scheduler")
        }
        trainer_cfg["scheduler_kwargs"] = scheduler_kwargs
        return trainer_cfg

    def train_vae_vqgan(self, dataloader: torch.utils.data.DataLoader, epochs: int = 500) -> None:
        """Train VQGAN or VAEGAN based on vq_mode."""
        clf_cfg = self.config.get("classifier", {})
        logging_cfg = self.config.get("logging", {})
        raw_cfg = self.config["trainer"]["vae_vqgan"]
        trainer_cfg = self._extract_scheduler_kwargs(raw_cfg)

        mode_name = "VQGAN" if self.vq_mode else "VAEGAN"
        if clf_cfg.get("enabled", False):
            mode_name += "+AC"

        print(f"[INFO] Training {mode_name} on {self.device} for {epochs} epoch(s).")
        self.model.to(self.device)

        self.vae_vqgan_trainer = VAEVQGANTrainer(
            model=self.model,
            run=self.run,
            vq_mode=self.vq_mode,
            device=self.device,
            experiment_dir=self.experiment_dir,
            # Classifier config
            classifier_enabled=clf_cfg.get("enabled", False),
            num_classes=clf_cfg.get("num_classes", 9),
            cls_loss_weight=clf_cfg.get("loss_weight", 1.0),
            classifier_dropout=clf_cfg.get("dropout", 0.3),
            normal_class_idx=clf_cfg.get("normal_class_idx"),  # None for 9 faulty classes only
            class_display_order=clf_cfg.get("class_display_order"),
            # Logging config
            logging_config=logging_cfg,
            # Trainer hyperparameters
            **trainer_cfg,
        )

        self.vae_vqgan_trainer.train(dataloader=dataloader, epochs=epochs)

        # Final checkpoint (also saved periodically by VAEVQGANTrainer)
        checkpoint_name = "vqgan.pt" if self.vq_mode else "vaegan.pt"
        self.model.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", checkpoint_name)
        )
        torch.save(
            self.vae_vqgan_trainer.discriminator.state_dict(),
            os.path.join(self.experiment_dir, "checkpoints", "disc.pt"),
        )

    def train_transformers(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1) -> None:
        """Train transformer prior (VQ mode only)."""
        if not self.vq_mode:
            print("[WARNING] Transformer training skipped -- only applicable for VQ mode.")
            return

        if self.transformer is None:
            raise ValueError("Transformer model not provided.")

        logging_cfg = self.config.get("logging", {})
        raw_trans = self.config["trainer"]["transformer"]
        transformer_cfg = self._extract_scheduler_kwargs(raw_trans)

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")
        # Freeze VQGAN (no encoder/decoder/codebook updates). Ensures no codebook EMA or reinit.
        self.model.eval()
        if hasattr(self.model, "codebook") and self.model.codebook is not None:
            self.model.codebook.eval()
        self.transformer = self.transformer.to(self.device)

        clf_cfg = self.config.get("classifier", {})
        initial_step = self.vae_vqgan_trainer.global_step if hasattr(self, "vae_vqgan_trainer") and self.vae_vqgan_trainer is not None else 0
        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            logging_config=logging_cfg,
            class_display_order=clf_cfg.get("class_display_order"),
            initial_step=initial_step,
            **transformer_cfg,
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "transformer.pt")
        )

    def generate_images(self, n_images: int = 5) -> None:
        """Generate sample images (simple preview after training).

        VQ mode: autoregressive sampling via transformer.
        VAE mode: sample z ~ N(0, I).
        Saves n_images files, each with 4 images in a grid.
        For full 9-column class-ordered grids and GIFs, use generate.py.
        """
        print(f"[INFO] Generating {n_images} images...")
        self.model.to(self.device)
        self.model.eval()

        clf_cfg = self.config.get("classifier", {})
        num_classes = self.model.num_classes
        class_display_order = clf_cfg.get("class_display_order", list(range(num_classes))) if num_classes > 0 else None

        sos_token = 0
        if self.vq_mode and self.config.get("architecture", {}).get("transformer"):
            sos_token = self.config["architecture"]["transformer"].get("sos_token", 0)

        with torch.no_grad():
            for i in range(n_images):
                # Build labels for class-conditioned generation
                if class_display_order is not None:
                    gen_labels = torch.tensor(class_display_order, device=self.device)
                    n_gen = len(gen_labels)
                else:
                    gen_labels = None
                    n_gen = 9

                if self.vq_mode:
                    if self.transformer is None:
                        print("[WARNING] Cannot generate VQ samples without transformer.")
                        return
                    self.transformer = self.transformer.to(self.device)
                    start_indices = torch.zeros((n_gen, 0)).long().to(self.device)
                    sos_tokens = torch.ones(n_gen, 1, device=self.device).long() * sos_token
                    sample_indices = self.transformer.sample(
                        start_indices, sos_tokens, steps=256, labels=gen_labels,
                    )
                    sampled_imgs = self.transformer.z_to_image(sample_indices, labels=gen_labels)
                else:
                    sampled_imgs = self.model.sample(n_samples=n_gen, device=self.device, labels=gen_labels)

                torchvision.utils.save_image(
                    sampled_imgs,
                    os.path.join(self.experiment_dir, f"generated_{i}.jpg"),
                    nrow=n_gen,
                )
