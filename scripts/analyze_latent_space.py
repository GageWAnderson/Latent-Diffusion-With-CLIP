"""
t-SNE analysis of the AutoencoderKL latent space on LSUN Churches.

Usage:
    python scripts/analyze_latent_space.py \
        --config configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml \
        --ckpt models/first_stage_models/kl-f8/model.ckpt \
        --n_samples 2000 \
        --batch_size 32
"""

import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldm.util import instantiate_from_config


def load_autoencoder(config, ckpt_path, device):
    ae_cfg = config.model.params.first_stage_config
    model = instantiate_from_config(ae_cfg)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # checkpoint may be a plain state_dict or wrapped under "state_dict"
    if "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    model.to(device)
    model.eval()
    return model


def collect_embeddings(model, dataloader, n_samples, device):
    embeddings = []
    collected = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding"):
            x = batch["image"]  # (B, H, W, 3), float in [-1, 1] from LSUNBase
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
            x = x.to(device)

            posterior = model.encode(x)
            z = posterior.mode()  # (B, 4, 32, 32) — deterministic latent mean

            z_flat = z.view(z.size(0), -1).cpu().numpy()  # (B, 4096)
            embeddings.append(z_flat)
            collected += z_flat.shape[0]

            if collected >= n_samples:
                break

    embeddings = np.concatenate(embeddings, axis=0)[:n_samples]
    print(f"Collected {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")
    return embeddings


def run_pca_tsne(embeddings, pca_components, perplexity):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    print(f"Running PCA: {embeddings.shape[1]} → {pca_components} dims...")
    pca = PCA(n_components=pca_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA explains {explained:.1%} of variance")

    print(f"Running t-SNE: {pca_components} → 2 dims (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    coords = tsne.fit_transform(reduced)
    return coords


def plot_and_save(coords, outdir):
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "latent_tsne.png")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(coords[:, 0], coords[:, 1], s=4, alpha=0.5, linewidths=0)
    ax.set_title(f"t-SNE of AutoencoderKL latent space ({coords.shape[0]} samples)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {outpath}")

    # Also save raw coords for later reuse
    np_path = os.path.join(outdir, "latent_tsne_coords.npy")
    np.save(np_path, coords)
    print(f"Saved raw coords to {np_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml")
    parser.add_argument("--ckpt", default="models/first_stage_models/kl-f8/model.ckpt")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--outdir", default="outputs/")
    parser.add_argument("--pca_components", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--num_workers", type=int, default=4)
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = OmegaConf.load(opt.config)

    print("Loading autoencoder...")
    model = load_autoencoder(config, opt.ckpt, device)

    print("Loading dataset...")
    dataset = instantiate_from_config(config.data.params.validation)
    if opt.n_samples < len(dataset):
        indices = np.random.default_rng(0).choice(len(dataset), size=opt.n_samples, replace=False)
        dataset = Subset(dataset, indices.tolist())
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=(device.type == "cuda"))

    embeddings = collect_embeddings(model, dataloader, opt.n_samples, device)
    coords = run_pca_tsne(embeddings, opt.pca_components, opt.perplexity)
    plot_and_save(coords, opt.outdir)


if __name__ == "__main__":
    main()
