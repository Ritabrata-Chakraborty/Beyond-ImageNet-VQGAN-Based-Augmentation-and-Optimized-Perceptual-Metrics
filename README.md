# Beyond ImageNet

Research code for **VQGAN-based augmentation** and **optimized perceptual metrics** applied to CWRU CWT scalograms for bearing fault diagnosis. This repository accompanies the paper:

**Beyond ImageNet: VQGAN-Based Augmentation and Optimized Perceptual Metrics for Bearing Fault Diagnosis**  
Ritabrata Chakraborty, Pradeep Kundu  
*In preparation for IEEE Transactions on Industrial Informatics.*

## Citation

```bibtex
@article{chakraborty2025beyond,
  title   = {Beyond ImageNet: VQGAN-Based Augmentation and Optimized Perceptual Metrics for Bearing Fault Diagnosis},
  author  = {Chakraborty, Ritabrata and Kundu, Pradeep},
  journal = {IEEE Transactions on Industrial Informatics},
  year    = {2025},
  note    = {In preparation}
}
```

## Components

- **Data prep** — Build CWT scalograms from raw CWRU, perturb for 2AFC, convert to LPIPS 2AFC format. Scripts in `src/data_prep/`; outputs under `data/`.
- **Perceptual (LPIPS)** — Backbone training (VGG/AlexNet/SqueezeNet) on scalograms, layer selection, separability analysis. Code in `src/perceptual/`; LPIPS train/eval via **our modified** `external/perceptual_similarity`; 2AFC data in `data/2afc`.
- **Generative (VAE-VQGAN)** — VQGAN/VAEGAN for CWT scalograms; reads `data/cwru_cwt`, writes to `data/generated/`. Code in `src/generative/`; uses LPIPS from `external/perceptual_similarity` for perceptual loss. CMMD is run in-process via `external/cmmd`.
- **Classification (LiteFormer2D)** — 2D LiteFormer variants (A–E) on CWRU CWT. Code in `src/classification/`; dataset `data/cwru_cwt`. Augment-and-train (real + gen_vaegan) in `experiments/classification/augment_and_train/`.
- **Evaluation (CMMD)** — CMMD metric (CLIP-based). Invoked from `src/generative/generate.py` or run via `external/cmmd`; reference/generated dirs under `data/`.

## Quick start

```bash
# From project root; ensure data/ is populated (see docs/PIPELINE.md)
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)/src:$(pwd)/external/perceptual_similarity"

# Data prep: build CWT, perturb in-place, then convert to 2AFC
./scripts/data_prep/run_data_prep.sh build
./scripts/data_prep/run_data_prep.sh perturb
./scripts/data_prep/run_data_prep.sh 2afc

# Train LPIPS (from external/perceptual_similarity with dataroot=data/2afc)
# See experiments/perceptual/ for scripts.
```

## Layout

| Path | Description |
|------|-------------|
| `data/` | Datasets and generated outputs (cwru, cwru_cwt, 2afc, generated). |
| `src/` | Our code: data_prep, perceptual, generative, classification. |
| `external/` | Third-party code: Scenic, **modified** perceptual_similarity, **modified** cmmd (see [License](#license) and [Third-party code](#third-party-code)). |
| `experiments/` | Run configs and sweep scripts (perceptual, generative, classification, augment_and_train). |
| `configs/` | Project-wide path and env config (e.g. paths.yml). |
| `docs/` | Pipeline and technical documentation. |
| `COMPILED_READMES.md` | Consolidated README, SETUP, and CONTRIBUTING content from across the repo. |

See `docs/PIPELINE.md` for data flow and script-to-artifact mapping.

## Scripts (entry points)

All runnable bash entry points live under `scripts/`. Run from project root.

| Location | Script | Description |
|----------|--------|-------------|
| **data_prep/** | `run_data_prep.sh` | Modes: `build` (CWT from raw → data/cwru_cwt), `perturb`, `2afc` (→ data/2afc). |
| **perceptual/** | `train_scalograms.sh` | Train LPIPS on 2AFC scalograms. |
| | `train_test_vgg_layer_sweep.sh` | VGG layer sweep; appends CSV. |
| | `eval_valsets.sh` | Evaluate LPIPS on validation 2AFC sets. |
| | `run_layer_selection_all.sh` | Layer selection for all models/cases. |
| **classification/** | `run.sh` | Train or test LiteFormer 2D. |
| | `run_augment_and_train.sh` | Augmented datasets + train. |
| **generative/** | `generate.sh` | Generate images for VAEGAN/VQGAN epochs; optional `--n-per-class N`. |
| **evaluation/** | (CMMD) | CMMD is called from `generate.py`; or run `external/cmmd` directly. |

## Generative (VAE-VQGAN) summary

Unified PyTorch VQGAN/VAEGAN for CWT scalograms: class-balanced training, AC-GAN discriminator, optional transformer prior (VQ mode), W&B logging. Config: `src/generative/configs/`; checkpoints under `experiments/generative/`. See `docs/` (ARCHITECTURE.md, TRAINING.md, SAMPLING.md, METRICS.md, CONFIG.md).

## Dependencies

- **pip:** From project root run `pip install -r requirements.txt`.
- **conda:** From project root run `conda env create -f environment.yml` (creates env `beyond_imagenet`), then `conda activate beyond_imagenet`.

Optional CMMD (when running from `generate.py`): uncomment the JAX/Flax lines in `requirements.txt` if you use CMMD in-process; Scenic is loaded via `PYTHONPATH` from `external/scenic`. For a dedicated CMMD venv, see the "external/cmmd/SETUP.md" section in [COMPILED_READMES.md](COMPILED_READMES.md).

## License

This project (code in `src/`, `scripts/`, `configs/`, `docs/`, and root config files) is released under the **MIT License** — see [LICENSE](LICENSE).

### Third-party code

The `external/` directory contains code from the following projects. **Their respective licenses apply** to those parts; we do not claim to relicense them.

| Component | Source | License | Our modifications |
|-----------|--------|---------|-------------------|
| **Perceptual Similarity (LPIPS)** | [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) | BSD-2-Clause | **Heavily modified**; we use our version in `external/perceptual_similarity` for scalogram 2AFC training and evaluation. |
| **CMMD** | [google-research/cmmd](https://github.com/google-research/google-research/tree/master/cmmd) | Apache-2.0 | **Modified** `cmmd/io_util.py` for our I/O and environment. |
| **Scenic** | [google-research/scenic](https://github.com/google-research/scenic) | Apache-2.0 | Used as-is (CLIP embeddings for CMMD). |

See the README or license files inside each `external/<name>/` directory for full attribution and terms.
