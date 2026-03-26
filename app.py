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
CONFIG_PATH = "configs/latent-diffusion/txt2img-clip-ldm-kl-8.yaml"
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
    # Override cache_dir for HF Spaces (no /mnt/data available)
    if hasattr(config.model.params, "cond_stage_config"):
        cond_params = config.model.params.cond_stage_config.get("params", {})
        if "cache_dir" in cond_params:
            config.model.params.cond_stage_config.params.cache_dir = None
    print(f"Loading model from {ckpt_path}")
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"]
    # Fix shape mismatch: checkpoint has 3D proj_out weights (C, C, 1) but
    # nn.Conv2d expects 4D (C, C, 1, 1). Unsqueeze the trailing dimension.
    # EMA keys omit dots (e.g. "model_ema.diffusion_modelinput_blocks11proj_outweight")
    # so match both "proj_out.weight" and "proj_outweight".
    for k in list(sd.keys()):
        if ("proj_out.weight" in k or "proj_outweight" in k) and sd[k].ndim == 3:
            sd[k] = sd[k].unsqueeze(-1)
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
    prompt,
    num_samples,
    ddim_steps,
    guidance_scale,
    ddim_eta,
    use_plms,
    height,
    width,
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
    shape = [4, height // 8, width // 8]

    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if guidance_scale != 1.0:
                uc = model.get_learned_conditioning(num_samples * [""])
            c = model.get_learned_conditioning(num_samples * [prompt])

            samples, _ = sampler.sample(
                S=ddim_steps,
                conditioning=c,
                batch_size=num_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=guidance_scale,
                unconditional_conditioning=uc,
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
        gr.Textbox(
            label="Prompt",
            placeholder="a painting of a virus monster playing guitar",
            value="a painting of a virus monster playing guitar",
        ),
        gr.Slider(1, 4, value=1, step=1, label="Number of Samples"),
        gr.Slider(10, 200, value=50, step=10, label="DDIM Steps"),
        gr.Slider(1.0, 15.0, value=5.0, step=0.5, label="Guidance Scale"),
        gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="DDIM Eta (0 = deterministic)"),
        gr.Checkbox(label="Use PLMS Sampler", value=False),
        gr.Slider(128, 512, value=256, step=64, label="Height"),
        gr.Slider(128, 512, value=256, step=64, label="Width"),
        gr.Number(label="Seed (-1 = random)", value=-1, precision=0),
    ],
    outputs=gr.Gallery(label="Generated Images", columns=2),
    title="Churches Latent Diffusion with CLIP",
    description="Text-to-image generation using a Latent Diffusion Model with CLIP text conditioning, trained on church images.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
