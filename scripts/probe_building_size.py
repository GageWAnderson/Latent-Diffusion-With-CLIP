"""
Linear probe for "building size" in the AutoencoderKL latent space.

Two-pass usage:
  1. Generate a labeling grid:
     python scripts/probe_building_size.py --label_only

  2. Fit the probe and visualize:
     python scripts/probe_building_size.py \
         --big_indices "12,45,78,102,156" \
         --small_indices "3,27,89,134,201"
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldm.util import instantiate_from_config

# python scripts/probe_building_size.py \
#       --big_indices "676,641,656,444,3" \
#       --small_indices "534,530,627,634,779" \
#       --reuse_coords


# ---------------------------------------------------------------------------
# Helpers (adapted from analyze_latent_space.py / diagnose_latent_space.py)
# ---------------------------------------------------------------------------


def load_autoencoder(config, ckpt_path, device):
    ae_cfg = config.model.params.first_stage_config
    model = instantiate_from_config(ae_cfg)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(
            f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    model.to(device)
    model.eval()
    return model


def collect_embeddings(model, dataloader, n_samples, device):
    embeddings = []
    thumbnails = []
    collected = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding"):
            x = batch["image"]  # (B, H, W, 3) float in [-1, 1]
            thumbnails.append(x.numpy())

            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
            x = x.to(device)

            posterior = model.encode(x)
            z = posterior.mode()

            z_flat = z.view(z.size(0), -1).cpu().numpy()
            embeddings.append(z_flat)
            collected += z_flat.shape[0]

            if collected >= n_samples:
                break

    embeddings = np.concatenate(embeddings, axis=0)[:n_samples]
    thumbnails = np.concatenate(thumbnails, axis=0)[:n_samples]
    print(f"Collected {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")
    return embeddings, thumbnails


def tensor_to_pil(t):
    """(B, 3, H, W) tensor in [-1, 1] -> list of PIL Images."""
    t = t.clamp(-1.0, 1.0)
    arr = ((t + 1.0) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]


def np_to_pil(img_np, thumb_size=128):
    """(H, W, 3) float in [-1, 1] -> PIL Image resized to thumb_size."""
    img_uint8 = ((img_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_uint8)
    img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
    return img


def save_image_grid(pil_images, labels, ncols, outpath):
    n = len(pil_images)
    nrows = (n + ncols - 1) // ncols
    img_w, img_h = pil_images[0].size
    label_pad = 18 if labels is not None else 0

    canvas = Image.new(
        "RGB", (ncols * img_w, nrows * (img_h + label_pad)), color=(240, 240, 240)
    )
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


def parse_indices(s):
    if not s or not s.strip():
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Phase 1: Labeling grid
# ---------------------------------------------------------------------------


def make_labeling_grid(thumbnails, outdir, thumb_size=128, ncols=20):
    n = len(thumbnails)
    pil_images = [
        np_to_pil(thumbnails[i], thumb_size) for i in tqdm(range(n), desc="Grid")
    ]
    labels = [str(i) for i in range(n)]
    outpath = os.path.join(outdir, "labeling_grid.png")
    save_image_grid(pil_images, labels, ncols, outpath)


# ---------------------------------------------------------------------------
# Phase 2: Fit linear probe
# ---------------------------------------------------------------------------


def fit_probe(embeddings, big_indices, small_indices):
    labeled_indices = big_indices + small_indices
    labels = np.array([1] * len(big_indices) + [0] * len(small_indices))

    X = embeddings[labeled_indices]
    y = labels

    print(
        f"Fitting logistic regression on {len(big_indices)} big + {len(small_indices)} small = {len(y)} labels"
    )

    # Use fewer CV folds if we have very few samples
    n_folds = min(5, len(y))
    if n_folds < 2:
        print("Warning: too few labels for cross-validation, using all data without CV")
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
        clf.fit(X, y)
        print(f"Training accuracy: {clf.score(X, y):.1%}")
    else:
        clf = LogisticRegressionCV(
            penalty="l2",
            Cs=10,
            cv=n_folds,
            max_iter=1000,
            random_state=42,
        )
        clf.fit(X, y)
        print(f"Best C: {clf.C_[0]:.4f}")
        print(f"Training accuracy: {clf.score(X, y):.1%}")

    return clf


# ---------------------------------------------------------------------------
# Phase 3: Score & visualize
# ---------------------------------------------------------------------------


def score_all(clf, embeddings):
    probs = clf.predict_proba(embeddings)[:, 1]  # P(big)
    print(
        f"P(big) — min: {probs.min():.3f}, max: {probs.max():.3f}, "
        f"mean: {probs.mean():.3f}, std: {probs.std():.3f}"
    )
    return probs


def save_top_bottom_grids(thumbnails, probs, top_k, outdir, thumb_size=128):
    order = np.argsort(probs)

    # Top-K biggest
    top_big = order[-top_k:][::-1]
    pil_big = [np_to_pil(thumbnails[i], thumb_size) for i in top_big]
    labels_big = [f"#{i} P={probs[i]:.2f}" for i in top_big]
    save_image_grid(
        pil_big, labels_big, ncols=8, outpath=os.path.join(outdir, "probe_top_big.png")
    )

    # Top-K smallest
    top_small = order[:top_k]
    pil_small = [np_to_pil(thumbnails[i], thumb_size) for i in top_small]
    labels_small = [f"#{i} P={probs[i]:.2f}" for i in top_small]
    save_image_grid(
        pil_small,
        labels_small,
        ncols=8,
        outpath=os.path.join(outdir, "probe_top_small.png"),
    )


def plot_tsne_by_score(embeddings, probs, outdir, reuse_coords):
    N = embeddings.shape[0]
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
        print(f"Running PCA: {embeddings.shape[1]} -> 50 dims...")
        pca = PCA(n_components=50, random_state=42)
        reduced = pca.fit_transform(embeddings)
        print(f"PCA explains {pca.explained_variance_ratio_.sum():.1%} of variance")
        print("Running t-SNE...")
        coords = TSNE(
            n_components=2, perplexity=30.0, random_state=42, n_jobs=-1
        ).fit_transform(reduced)
        np.save(coords_path, coords)
        print(f"Saved coords to {coords_path}")

    fig, ax = plt.subplots(figsize=(10, 10))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=probs,
        s=4,
        alpha=0.5,
        cmap="RdYlBu_r",
        linewidths=0,
        vmin=0,
        vmax=1,
    )
    plt.colorbar(sc, ax=ax, label="P(big building)")
    ax.set_title(f"t-SNE colored by P(big building) ({N} samples)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_aspect("equal")
    fig.tight_layout()
    outpath = os.path.join(outdir, "probe_tsne_building_size.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {outpath}")


def plot_histogram(probs, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(probs, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("P(big building)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of P(big building) ({len(probs)} samples)")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="decision boundary")
    ax.legend()
    fig.tight_layout()
    outpath = os.path.join(outdir, "probe_score_histogram.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Linear probe for building size in latent space"
    )
    parser.add_argument(
        "--config", default="configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
    )
    parser.add_argument("--ckpt", default="models/first_stage_models/kl-f8/model.ckpt")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--outdir", default="outputs/")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--label_only",
        action="store_true",
        help="Only generate the labeling grid, then exit",
    )
    parser.add_argument(
        "--big_indices",
        type=str,
        default="",
        help="Comma-separated indices of 'big building' images",
    )
    parser.add_argument(
        "--small_indices",
        type=str,
        default="",
        help="Comma-separated indices of 'small building' images",
    )
    parser.add_argument(
        "--top_k", type=int, default=32, help="Number of top/bottom images to display"
    )
    parser.add_argument(
        "--reuse_coords",
        action="store_true",
        help="Reuse cached t-SNE coords from outputs/latent_tsne_coords.npy",
    )
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(opt.outdir, exist_ok=True)

    config = OmegaConf.load(opt.config)

    print("Loading autoencoder...")
    model = load_autoencoder(config, opt.ckpt, device)

    print("Loading dataset...")
    dataset = instantiate_from_config(config.data.params.validation)
    if opt.n_samples < len(dataset):
        indices = np.random.default_rng(0).choice(
            len(dataset), size=opt.n_samples, replace=False
        )
        dataset = Subset(dataset, indices.tolist())
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    embeddings, thumbnails = collect_embeddings(
        model, dataloader, opt.n_samples, device
    )

    # Phase 1: labeling grid
    make_labeling_grid(thumbnails, opt.outdir)

    if opt.label_only:
        print(
            "\n--label_only set. Examine outputs/labeling_grid.png and re-run with "
            "--big_indices and --small_indices."
        )
        return

    # Parse labels
    big_indices = parse_indices(opt.big_indices)
    small_indices = parse_indices(opt.small_indices)

    if not big_indices or not small_indices:
        print("Error: provide both --big_indices and --small_indices to fit the probe.")
        sys.exit(1)

    max_idx = embeddings.shape[0] - 1
    for idx in big_indices + small_indices:
        if idx < 0 or idx > max_idx:
            print(f"Error: index {idx} out of range [0, {max_idx}]")
            sys.exit(1)

    # Phase 2: fit probe
    clf = fit_probe(embeddings, big_indices, small_indices)

    # Phase 3: score & visualize
    probs = score_all(clf, embeddings)

    np.save(os.path.join(opt.outdir, "probe_scores.npy"), probs)
    print(f"Saved raw scores to {os.path.join(opt.outdir, 'probe_scores.npy')}")

    save_top_bottom_grids(thumbnails, probs, opt.top_k, opt.outdir)
    plot_tsne_by_score(embeddings, probs, opt.outdir, opt.reuse_coords)
    plot_histogram(probs, opt.outdir)

    print("\nDone. Check outputs/ for results.")


if __name__ == "__main__":
    main()
