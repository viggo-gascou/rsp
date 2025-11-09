"""Loading functions for the project."""

import torch
from diffusers.models.autoencoders.vq_model import VQModel
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from .semanticdiffusion import SemanticDiffusion
from .unet import UNet


def load_model(
    model_id: str,
    device: str = "cuda",
    h_space: str = "after",
    num_inference_steps: int = 25,
) -> SemanticDiffusion:
    """Load a semantic diffusion model.

    Args:
        model_id: The model ID.
        device: The device to load the model on.
        h_space: The horizontal space.
        num_inference_steps: The number of inference steps.

    Returns:
        SemanticDiffusion: The loaded semantic diffusion model.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            f"Device '{device}' was specified but is not available, "
            "please check your CUDA installation, or try changing the device to 'cpu'."
        )

    if model_id not in ["pixel", "ldm"]:
        raise ValueError(f"Invalid model_id: {model_id}")

    if "pixel" == model_id:
        model_id = "google/ddpm-ema-celebahq-256"
        model = UNet2DModel.from_pretrained(
            model_id, use_safetensors=False, device_map=device
        )
        scheduler = DDIMScheduler.from_pretrained(model_id)
        vqvae = None
    elif "ldm" == model_id:
        model_id = "CompVis/ldm-celebahq-256"
        vqvae = VQModel.from_pretrained(model_id, subfolder="vqvae", device_map=device)
        model = UNet2DModel.from_pretrained(
            model_id, subfolder="unet", device_map=device
        )
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    else:
        raise ValueError(f"Invalid model_id: {model_id}")

    unet = UNet(model, h_space=h_space)
    sd = SemanticDiffusion(
        unet,
        scheduler,
        vqvae=vqvae,
        model_id=model_id,
        num_inference_steps=num_inference_steps,
    )
    return sd
