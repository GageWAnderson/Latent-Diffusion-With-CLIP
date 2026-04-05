"""
Latent space diagnostics for AutoencoderKL.

Runs six targeted analyses to determine whether the VAE is encoding meaningful
information or suffering from posterior collapse.

Diagnostics:
  1. Reconstruction sanity check
  2. Per-channel KL analysis
  3. Latent interpolation test
  4. t-SNE colored by image statistics
  5. Prior samples vs. encoder reconstructions
  6. Channel spatial-mean t-SNE

Usage:
    python scripts/diagnose_latent_space.py \
        --config configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml \
        --ckpt models/first_stage_models/kl-f8/model.ckpt \
        --n_samples 2000 \
        --batch_size 32 \
        --reuse_coords

    # Fast smoke test (no t-SNE):
    python scripts/diagnose_latent_space.py ... --n_samples 64 --skip 4,6
"""

import argparse
import os
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldm.util import instantiate_from_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_autoencoder(config, ckpt_path, device):
    ae_cfg = config.model.params.first_stage_config
    model = instantiate_from_config(ae_cfg)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    model.to(device)
    model.eval()
    return model


def tensor_to_pil(t):
    """(B, 3, H, W) tensor in approximately [-1, 1] → list of B PIL Images."""
    t = t.clamp(-1.0, 1.0)
    arr = ((t + 1.0) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]


def save_image_grid(pil_images, labels, ncols, outpath):
    """Save a grid of PIL images with optional text labels below each image."""
    n = len(pil_images)
    nrows = (n + ncols - 1) // ncols
    img_w, img_h = pil_images[0].size
    label_pad = 18 if labels is not None else 0

    canvas = Image.new("RGB", (ncols * img_w, nrows * (img_h + label_pad)), color=(240, 240, 240))
    draw = ImageDraw.Draw(canvas) if labels is not None else None

    for idx, img in enumerate(pil_images):
        row, col = divmod(idx, ncols)
        x = col * img_w
        y = row * (img_h + label_pad)
        canvas.paste(img, (x, y))
        if draw is not None and labels is not None and idx < len(labels):
            draw.text((x + 2, y + img_h + 2), str(labels[idx])[:20], fill=(50, 50, 50))

    canvas.save(outpath)
    print(f"Saved {outpath}")


def parse_skip(skip_str):
    if not skip_str.strip():
        return set()
    return {int(x.strip()) for x in skip_str.split(",") if x.strip()}


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

DataBundle = namedtuple("DataBundle", [
    "raw_images_np",   # (N, H, W, 3) float32 in [-1, 1]
    "means_np",        # (N, 4, 32, 32)
    "logvars_np",      # (N, 4, 32, 32)
    "vars_np",         # (N, 4, 32, 32)
    "channel_means",   # (N, 4)  — spatial average of z per channel
    "embeddings_flat", # (N, 4096)
])


def collect_all(model, dataloader, n_samples, device):
    lists = {k: [] for k in DataBundle._fields}
    collected = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting"):
            x_np = batch["image"].numpy()  # (B, H, W, 3)
            x_t = (
                torch.from_numpy(x_np)
                .permute(0, 3, 1, 2)
                .to(memory_format=torch.contiguous_format)
                .float()
                .to(device)
            )

            posterior = model.encode(x_t)
            z = posterior.mode()  # (B, 4, 32, 32)

            lists["raw_images_np"].append(x_np)
            lists["means_np"].append(posterior.mean.cpu().numpy())
            lists["logvars_np"].append(posterior.logvar.cpu().numpy())
            lists["vars_np"].append(posterior.var.cpu().numpy())
            lists["channel_means"].append(z.mean(dim=[2, 3]).cpu().numpy())  # (B, 4)
            lists["embeddings_flat"].append(z.view(z.size(0), -1).cpu().numpy())

            collected += x_np.shape[0]
            if collected >= n_samples:
                break

    return DataBundle(**{
        k: np.concatenate(v, axis=0)[:n_samples]
        for k, v in lists.items()
    })


# ---------------------------------------------------------------------------
# Diagnostic 1 — Reconstruction sanity check
# ---------------------------------------------------------------------------

def diag_1_reconstruction(model, bundle, outdir, device):
    x_np = bundle.raw_images_np[:8]
    x_t = (
        torch.from_numpy(x_np)
        .permute(0, 3, 1, 2)
        .to(memory_format=torch.contiguous_format)
        .float()
        .to(device)
    )

    with torch.no_grad():
        posterior = model.encode(x_t)
        z = posterior.mode()
        recon_t = model.decode(z)

    originals = tensor_to_pil(x_t.cpu())
    recons = tensor_to_pil(recon_t.cpu())

    pairs, labels = [], []
    for i in range(len(originals)):
        pairs.extend([originals[i], recons[i]])
        labels.extend([f"orig {i}", f"recon {i}"])

    save_image_grid(pairs, labels, ncols=2, outpath=os.path.join(outdir, "diag_1_reconstruction.png"))


# ---------------------------------------------------------------------------
# Diagnostic 2 — Per-channel KL analysis
# ---------------------------------------------------------------------------

def diag_2_kl_per_channel(bundle, outdir):
    mean = bundle.means_np
    logvar = bundle.logvars_np
    var = bundle.vars_np

    # Element-wise KL(q || N(0,I)): 0.5 * (μ² + σ² - log(σ²) - 1)
    kl_elem = 0.5 * (mean ** 2 + var - logvar - 1.0)  # (N, 4, 32, 32)

    mean_kl = kl_elem.mean(axis=(0, 2, 3))   # (4,)
    active_frac = (kl_elem > 0.1).astype(np.float32).mean(axis=(0, 2, 3))  # (4,)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(4)
    colors = ["C0", "C1", "C2", "C3"]
    bars = ax.bar(x, mean_kl, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Channel {i}" for i in range(4)])
    ax.set_ylabel("Mean KL divergence vs N(0,I)")
    ax.set_title(f"Per-channel KL ({bundle.means_np.shape[0]} samples)\n"
                 f"Total mean KL = {mean_kl.sum():.3f}")

    for bar, af in zip(bars, active_frac):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mean_kl) * 0.01,
            f"{af:.1%} active",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylim(0, max(mean_kl) * 1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "diag_2_kl_per_channel.png"), dpi=150)
    plt.close(fig)
    print(f"Mean KL per channel: {mean_kl}")
    print(f"Active fraction per channel (KL > 0.1): {active_frac}")
    print(f"Saved {os.path.join(outdir, 'diag_2_kl_per_channel.png')}")


# ---------------------------------------------------------------------------
# Diagnostic 3 — Latent interpolation
# ---------------------------------------------------------------------------

def diag_3_interpolation(model, bundle, outdir, device):
    x_np = bundle.raw_images_np[:2]
    x_t = (
        torch.from_numpy(x_np)
        .permute(0, 3, 1, 2)
        .to(memory_format=torch.contiguous_format)
        .float()
        .to(device)
    )

    with torch.no_grad():
        z_a = model.encode(x_t[[0]]).mode()  # (1, 4, 32, 32)
        z_b = model.encode(x_t[[1]]).mode()

        alphas = np.linspace(0.0, 1.0, 9)
        decoded = []
        for alpha in alphas:
            z_interp = (1 - alpha) * z_a + alpha * z_b
            decoded.append(model.decode(z_interp).cpu())

    decoded_cat = torch.cat(decoded, dim=0)  # (9, 3, H, W)
    pil_images = tensor_to_pil(decoded_cat)

    labels = [f"a={a:.3f}" for a in alphas]
    labels[0] = "img A"
    labels[-1] = "img B"

    save_image_grid(pil_images, labels, ncols=9, outpath=os.path.join(outdir, "diag_3_interpolation.png"))


# ---------------------------------------------------------------------------
# Diagnostic 4 — t-SNE colored by image statistics
# ---------------------------------------------------------------------------

def diag_4_tsne_colored(bundle, outdir, reuse_coords):
    imgs_01 = (bundle.raw_images_np + 1.0) / 2.0  # (N, H, W, 3) in [0, 1]
    N = imgs_01.shape[0]

    brightness = imgs_01.mean(axis=(1, 2, 3))
    texture_var = imgs_01.std(axis=(1, 2, 3))

    print("Computing per-image mean hue...")
    mean_hue = np.zeros(N, dtype=np.float32)
    for i in tqdm(range(N), desc="Hue"):
        pil_rgb = Image.fromarray((imgs_01[i] * 255).clip(0, 255).astype(np.uint8))
        pil_hsv = pil_rgb.convert("HSV")
        h_channel = np.array(pil_hsv)[:, :, 0].astype(np.float32) / 255.0
        mean_hue[i] = h_channel.mean()

    # --- t-SNE coords
    coords_path = os.path.join(outdir, "latent_tsne_coords.npy")
    coords = None
    if reuse_coords and os.path.exists(coords_path):
        loaded = np.load(coords_path)
        if loaded.shape[0] == N:
            coords = loaded
            print(f"Reusing t-SNE coords from {coords_path}")
        else:
            print(f"Coord shape mismatch ({loaded.shape[0]} vs {N}); recomputing.")

    if coords is None:
        print(f"Running PCA: {bundle.embeddings_flat.shape[1]} → 50 dims...")
        pca = PCA(n_components=50, random_state=42)
        reduced = pca.fit_transform(bundle.embeddings_flat)
        print(f"PCA explains {pca.explained_variance_ratio_.sum():.1%} of variance")
        print("Running t-SNE (this may take 1-2 minutes)...")
        coords = TSNE(n_components=2, perplexity=30.0, random_state=42, n_jobs=-1).fit_transform(reduced)
        np.save(coords_path, coords)
        print(f"Saved coords to {coords_path}")

    stats = [brightness, texture_var, mean_hue]
    stat_names = ["Mean Brightness", "Texture Variance (std)", "Mean Hue"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, values, name in zip(axes, stats, stat_names):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=3, alpha=0.5,
                        cmap="viridis", linewidths=0)
        plt.colorbar(sc, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_aspect("equal")

    fig.suptitle(f"t-SNE colored by image statistics ({N} samples)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "diag_4_tsne_colored.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {os.path.join(outdir, 'diag_4_tsne_colored.png')}")


# ---------------------------------------------------------------------------
# Diagnostic 5 — Prior samples vs. encoder reconstructions
# ---------------------------------------------------------------------------

def diag_5_prior_vs_reconstruction(model, bundle, outdir, device):
    x_np = bundle.raw_images_np[:8]
    x_t = (
        torch.from_numpy(x_np)
        .permute(0, 3, 1, 2)
        .to(memory_format=torch.contiguous_format)
        .float()
        .to(device)
    )

    with torch.no_grad():
        posterior = model.encode(x_t)
        z_real = posterior.mode()
        z_random = torch.randn_like(z_real)  # pure N(0, I) — on same device as z_real
        recon_real = model.decode(z_real)
        recon_rand = model.decode(z_random)

    originals = tensor_to_pil(x_t.cpu())
    from_enc = tensor_to_pil(recon_real.cpu())
    from_prior = tensor_to_pil(recon_rand.cpu())

    triplets, labels = [], []
    for i in range(len(originals)):
        triplets.extend([originals[i], from_enc[i], from_prior[i]])
        labels.extend([f"orig {i}", f"encoder {i}", f"prior {i}"])

    save_image_grid(
        triplets, labels, ncols=3,
        outpath=os.path.join(outdir, "diag_5_prior_vs_reconstruction.png"),
    )


# ---------------------------------------------------------------------------
# Diagnostic 6 — Channel spatial-mean t-SNE
# ---------------------------------------------------------------------------

def diag_6_channel_mean_tsne(bundle, outdir):
    X = bundle.channel_means   # (N, 4)
    N = X.shape[0]

    perplexity = min(30.0, N - 1)
    print(f"Running t-SNE on {N} × 4 channel means (perplexity={perplexity})...")
    coords = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1).fit_transform(X)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ch in range(4):
        ax = axes[ch]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=X[:, ch], s=3, alpha=0.5,
                        cmap="coolwarm", linewidths=0)
        plt.colorbar(sc, ax=ax)
        ax.set_title(f"Channel {ch} spatial mean")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_aspect("equal")

    fig.suptitle(f"t-SNE of 4-dim channel spatial means ({N} samples)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "diag_6_channel_mean_tsne.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {os.path.join(outdir, 'diag_6_channel_mean_tsne.png')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Latent space diagnostics for AutoencoderKL")
    parser.add_argument("--config", default="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml")
    parser.add_argument("--ckpt", default="models/first_stage_models/kl-f8/model.ckpt")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--outdir", default="outputs/")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--reuse_coords", action="store_true",
        help="Load pre-computed t-SNE coords from outputs/latent_tsne_coords.npy for diag 4",
    )
    parser.add_argument(
        "--skip", type=str, default="",
        help="Comma-separated diagnostic numbers to skip, e.g. '3,5'",
    )
    opt = parser.parse_args()

    skip = parse_skip(opt.skip)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(opt.outdir, exist_ok=True)

    config = OmegaConf.load(opt.config)

    print("Loading autoencoder...")
    model = load_autoencoder(config, opt.ckpt, device)

    print("Loading dataset...")
    dataset = instantiate_from_config(config.data.params.validation)
    n_samples = min(opt.n_samples, len(dataset))
    if n_samples < len(dataset):
        indices = np.random.default_rng(0).choice(len(dataset), size=n_samples, replace=False)
        dataset = Subset(dataset, indices.tolist())
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Collecting embeddings for {n_samples} samples...")
    bundle = collect_all(model, dataloader, n_samples, device)

    diags = {
        1: lambda: diag_1_reconstruction(model, bundle, opt.outdir, device),
        2: lambda: diag_2_kl_per_channel(bundle, opt.outdir),
        3: lambda: diag_3_interpolation(model, bundle, opt.outdir, device),
        4: lambda: diag_4_tsne_colored(bundle, opt.outdir, opt.reuse_coords),
        5: lambda: diag_5_prior_vs_reconstruction(model, bundle, opt.outdir, device),
        6: lambda: diag_6_channel_mean_tsne(bundle, opt.outdir),
    }

    for num, fn in diags.items():
        if num in skip:
            print(f"\nSkipping diagnostic {num}")
            continue
        print(f"\n{'='*50}")
        print(f"Diagnostic {num}")
        print(f"{'='*50}")
        fn()

    print("\nAll diagnostics complete.")


if __name__ == "__main__":
    main()
