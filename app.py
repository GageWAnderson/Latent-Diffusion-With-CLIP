import os
import spaces
import gradio as gr
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# --- Configuration ---
CONFIG_PATH = "configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml"
MODEL_REPO_ID = "Harbinger67/Churches-Latent-Diffusion"
CKPT_FILENAME = "model.ckpt"

model = None


def download_checkpoint(repo_id, filename):
    print(f"Downloading checkpoint '{filename}' from {repo_id}...")
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"Checkpoint downloaded to {path}")
    return path


def load_model(config_path, ckpt_path):
    config = OmegaConf.load(config_path)
    # Remove first-stage ckpt_path — weights are already in the main checkpoint
    if "ckpt_path" in config.model.params.first_stage_config.get("params", {}):
        del config.model.params.first_stage_config.params.ckpt_path
    print(f"Loading model from {ckpt_path}")
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    # Load on CPU at startup; ZeroGPU provides CUDA on-demand via @spaces.GPU
    model = model.to("cpu")
    model.eval()
    return model


try:
    ckpt_path = download_checkpoint(MODEL_REPO_ID, CKPT_FILENAME)
    print("Loading model... this may take a while.")
    model = load_model(CONFIG_PATH, ckpt_path)
    print("Model loaded.")
except Exception as e:
    print(f"WARNING: Could not load model: {e}")
    print(f"Ensure '{CKPT_FILENAME}' exists in the model repo '{MODEL_REPO_ID}'.")


@spaces.GPU
def generate(
    num_samples,
    ddim_steps,
    ddim_eta,
    use_plms,
    seed,
):
    if model is None:
        raise gr.Error(
            f"Model not loaded. Ensure '{CKPT_FILENAME}' exists in "
            f"'{MODEL_REPO_ID}' and restart the Space."
        )

    # Move model to GPU (ZeroGPU provides CUDA inside @spaces.GPU)
    model.to("cuda")

    if seed != -1:
        torch.manual_seed(seed)

    if use_plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    num_samples = int(num_samples)
    shape = [4, 32, 32]  # latent shape for 256x256 images with kl-f8

    with torch.no_grad():
        with model.ema_scope():
            samples, _ = sampler.sample(
                S=ddim_steps,
                conditioning=None,
                batch_size=num_samples,
                shape=shape,
                verbose=False,
                eta=ddim_eta,
            )

            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

    images = []
    for x in x_samples:
        img = 255.0 * rearrange(x.cpu().numpy(), "c h w -> h w c")
        images.append(Image.fromarray(img.astype(np.uint8)))

    return images


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Slider(1, 4, value=1, step=1, label="Number of Samples"),
        gr.Slider(10, 200, value=50, step=10, label="DDIM Steps"),
        gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="DDIM Eta (0 = deterministic)"),
        gr.Checkbox(label="Use PLMS Sampler", value=False),
        gr.Number(label="Seed (-1 = random)", value=-1, precision=0),
    ],
    outputs=gr.Gallery(label="Generated Images", columns=2),
    title="Churches Latent Diffusion (Unconditional)",
    description="Unconditional image generation using a Latent Diffusion Model with KL-f8 autoencoder, trained on LSUN Churches.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
