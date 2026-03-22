# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Latent Diffusion Models (LDM) — a research framework for high-resolution image synthesis by applying diffusion models in a learned latent space rather than pixel space. Implements the paper "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022) from CompVis. Includes Retrieval-Augmented Diffusion Models (RDM).

## Environment Setup

```bash
conda env create -f environment.yaml
conda activate ldm
pip install -e .
```

For RDM features, additional packages are needed:
```bash
pip install transformers==4.19.2 scann kornia==0.6.4 torchmetrics==0.6.0
pip install git+https://github.com/arogozhnikov/einops.git
```

Key dependencies: PyTorch 1.7.0, pytorch-lightning 1.4.2, omegaconf 2.1.1, transformers, einops, taming-transformers (git), CLIP (git).

## Common Commands

### Training
```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/latent-diffusion/<config>.yaml -t --gpus 0,
```
Resume training: `python main.py --base <config>.yaml -t --gpus 0, --resume logs/<run_dir>`

### Inference
```bash
# Text-to-image
python scripts/txt2img.py --prompt "a painting of a virus monster" --ddim_eta 0.0 --n_samples 4 --n_iter 4 --scale 5.0 --ddim_steps 50

# Unconditional sampling
python scripts/sample_diffusion.py -r models/ldm/<model>/model.ckpt -l <logdir> -n <n_samples> --batch_size <bs> -c <ddim_steps> -e <ddim_eta>

# Inpainting
python scripts/inpaint.py --indir data/inpainting_examples/ --outdir outputs/inpainting_results

# Retrieval-augmented (kNN)
python scripts/knn2img.py --prompt "a happy bear reading a newspaper" --database_name artbench-art_nouveau --knn <k>
```

### Download pretrained models
```bash
bash scripts/download_models.sh        # Diffusion model checkpoints
bash scripts/download_first_stages.sh  # Autoencoder checkpoints
```

## Architecture

### Core Design Pattern

Everything is **config-driven** via OmegaConf YAML files. Components are instantiated dynamically using `ldm.util.instantiate_from_config()`, which reads a `target` key (dotted Python class path) and `params` dict from config. This is the central factory pattern used throughout the codebase.

### Three-Stage Pipeline

1. **First Stage Model** (frozen autoencoder) — compresses images to a latent space
   - `ldm.models.autoencoder.VQModel` — Vector-Quantized VAE
   - `ldm.models.autoencoder.AutoencoderKL` — KL-regularized VAE
   - Configured under `model.params.first_stage_config` in YAML

2. **Diffusion Model** — operates in the latent space
   - `ldm.models.diffusion.ddpm.DDPM` — base DDPM (pixel-space)
   - `ldm.models.diffusion.ddpm.LatentDiffusion(DDPM)` — main LDM class, inherits DDPM
   - UNet backbone: `ldm.modules.diffusionmodules.openaimodel.UNetModel`
   - Configured under `model` in YAML

3. **Conditioning** — text, class labels, or concatenation
   - `conditioning_key` in config: `crossattn` (text/CLIP), `concat`, or `hybrid`
   - `DiffusionWrapper` (inner class in ddpm.py) routes conditioning to the UNet
   - Encoders in `ldm/modules/encoders/modules.py`: `FrozenCLIPEmbedder`, `ClassEmbedder`, `FrozenCLIPTextEmbedder`
   - Configured under `model.params.cond_stage_config`

### Samplers

- `ldm.models.diffusion.ddim.DDIMSampler` — deterministic fast sampler (use `eta=0.0` for deterministic)
- `ldm.models.diffusion.plms.PLMSSampler` — pseudo-numerical methods, faster than DDIM

### Training (main.py)

`main.py` is the training entry point using PyTorch Lightning. It:
- Loads YAML configs via OmegaConf (supports merging multiple configs and CLI overrides)
- Sets up `DataModuleFromConfig` which wraps arbitrary dataset classes
- Registers callbacks: `SetupCallback`, `ImageLogger`, `CUDACallback`, `LearningRateMonitor`
- Training logs go to `logs/` directory

### Key Module Locations

- `ldm/models/diffusion/ddpm.py` — DDPM and LatentDiffusion (the main model classes, ~1400 lines)
- `ldm/modules/diffusionmodules/openaimodel.py` — UNet architecture
- `ldm/modules/attention.py` — spatial transformers and cross-attention
- `ldm/modules/diffusionmodules/model.py` — Encoder/Decoder blocks for autoencoders
- `ldm/modules/losses/` — VQ perceptual loss, contrastive perceptual loss
- `ldm/data/` — dataset classes (LSUN, ImageNet, base iterable dataset)

### Config Structure

```yaml
model:
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
    first_stage_config:
      target: ldm.models.autoencoder.VQModel  # or AutoencoderKL
    cond_stage_config:
      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder

data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 4
    train:
      target: <dataset_class>
    validation:
      target: <dataset_class>

lightning:
  trainer:
    max_epochs: ...
```

## Conventions

- All trainable models inherit from `pl.LightningModule`
- EMA weights toggled via `model.ema_scope()` context manager during inference
- Tensor operations use `einops` (`rearrange`, `repeat`) rather than manual reshaping
- The codebase has no test suite; validation is done through inference scripts and config-based evaluation
