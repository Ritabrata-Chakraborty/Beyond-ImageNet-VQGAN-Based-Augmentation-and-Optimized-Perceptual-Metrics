"""Train VAE-VQGAN or VQGAN+Transformer from a YAML config; supports W&B logging."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add external/perceptual_similarity for LPIPS (trainer imports lpips)
_project_root = Path(__file__).resolve().parents[2]
_perceptual_dir = _project_root / "external" / "perceptual_similarity"
if _perceptual_dir.is_dir() and str(_perceptual_dir) not in sys.path:
    sys.path.insert(0, str(_perceptual_dir))

import yaml
import wandb

from dataloader import load_dataloader
from trainer import Trainer
from transformer import VQGANTransformer
from vae_vqgan import VAEVQGAN


def main(config: dict, config_path: str) -> None:
    """Run training pipeline: resolve paths, build model/dataloader, train and save checkpoints."""
    vq_mode = config["mode"]["vq"]
    clf = config.get("classifier", {})
    data_cfg = config["data"]
    training_cfg = config["training"]
    logging_cfg = config["logging"]

    # Resolve dataset_path relative to project root when not absolute
    project_root = Path(__file__).resolve().parents[2]
    dp = data_cfg["dataset_path"]
    if not os.path.isabs(dp):
        data_cfg = dict(data_cfg)
        data_cfg["dataset_path"] = str(project_root / dp)

    # Resolve perceptual paths (LPIPS from external/perceptual_similarity) relative to project root
    trainer_cfg = config.get("trainer", {}).get("vae_vqgan", {})
    for key in ("perceptual_model_path", "perceptual_backbone_path"):
        p = trainer_cfg.get(key)
        if p and not os.path.isabs(p):
            config["trainer"]["vae_vqgan"] = dict(config["trainer"]["vae_vqgan"])
            config["trainer"]["vae_vqgan"][key] = str(project_root / p)

    # Name experiment dir and W&B run from config file name (e.g. vqgan-rmsprop-cosine_restarts)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    config["training"] = dict(training_cfg)
    config["training"]["experiment_dir"] = str(project_root / "experiments" / "generative" / config_name)

    model_name = ("VQGAN" if vq_mode else "VAEGAN") + ("+AC" if clf.get("enabled", False) else "")
    print(f"[INFO] Running in {model_name} mode")

    model = VAEVQGAN(**config["architecture"]["vae_vqgan"], vq_mode=vq_mode)

    transformer = None
    if vq_mode:
        transformer = VQGANTransformer(
            model,
            device=training_cfg["device"],
            **config["architecture"]["transformer"],
        )

    dataloader = load_dataloader(
        name=data_cfg["dataset_name"],
        dataset_path=data_cfg["dataset_path"],
        batch_size=data_cfg["batch_size"],
        image_size=data_cfg["image_size"],
        num_workers=data_cfg["num_workers"],
        return_labels=clf.get("enabled", False),
        balanced=clf.get("balanced", True),
    )

    run = wandb.init(
        project=logging_cfg["wandb_project"],
        name=config_name,
        config=config,
    )

    trainer = Trainer(
        model,
        transformer,
        run=run,
        config=config,
        vq_mode=vq_mode,
    )

    trainer.train_vae_vqgan(dataloader, epochs=training_cfg["epochs"])

    if vq_mode and training_cfg.get("train_transformer", False):
        trainer.train_transformers(dataloader, epochs=training_cfg["transformer_epochs"])

    trainer.generate_images()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE-VQGAN or VQGAN+Transformer from config.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/default.yml",
        help="Path to config YAML.",
    )
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    main(config, args.config_path)
