# Linear Probing: Building Size in the Latent Space

## Goal

Test whether the AutoencoderKL latent space (trained on LSUN Churches) encodes "building size" as a linearly separable factor. The approach: hand-label a small set of images as big vs. small buildings, fit a regularized linear classifier on their latent vectors, then score all unlabeled images with P(big).

## Method

1. Encoded 2000 LSUN Churches validation images through the KL-f8 autoencoder to get 4096-dim latent vectors (flattened from 4x32x32).
2. Manually labeled 5 "big" and 5 "small" building examples from a thumbnail grid:
   - Big: indices 676, 641, 656, 444, 3
   - Small: indices 534, 530, 627, 634, 779
3. Fit an L2-regularized logistic regression (with cross-validated regularization strength) on the 10 labeled embeddings.
4. Scored all 2000 images with P(big building) and visualized the results.

## Results

The probe successfully separates building size from just 10 labels:

- **`probe_top_big.png`** — Top-32 highest P(big) images. These are dominated by large cathedrals, grand Gothic/Baroque churches, and monumental facades photographed from close range. Scores range ~0.70-0.84.
- **`probe_top_small.png`** — Top-32 lowest P(big) images. These show small rural chapels, modest wooden churches, and distant shots where the building is small in the frame. Scores range ~0.15-0.34.
- **`probe_tsne_building_size.png`** — t-SNE of the latent space colored by P(big). The scores are fairly spread (most images land in the 0.4-0.7 range), suggesting building size is a continuous factor rather than a sharp binary split. The t-SNE shows mild spatial clustering by score but not a clean separation, which is expected: building size is one of many entangled factors (viewpoint, lighting, style).

## Takeaway

The latent space does encode building size in a direction that a linear classifier can recover, even with very few labels. The probe picks up on a mix of actual building scale, architectural grandeur, and how much of the frame the building occupies. A larger labeled set (~50+ per class) and clearer labeling criteria (e.g., physical size vs. visual dominance) would sharpen the separation.

## Script

See `scripts/probe_building_size.py` for the full pipeline.
