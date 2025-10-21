"""Semantic Diffusion class for generating images using a semantic diffusion model."""

import torch
from diffusers.models.autoencoders.vq_model import VQModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from tqdm import tqdm

from .diffusion import Diffusion
from .interpolation import Interpolations
from .stateclass import Q
from .unet import UNet
from .utils import image_grid


class SemanticDiffusion(Interpolations):
    """Semantic Diffusion class."""

    def __init__(
        self,
        unet: UNet,
        scheduler: DDIMScheduler,
        model_id: str,
        vqvae: VQModel | None,
        num_inference_steps=25,
        diffusers_default_scheduler=False,
        resize_to=256,
    ):
        """Initialize the SemanticDiffusion class."""
        self.vqvae = vqvae
        self.diff = Diffusion(
            scheduler=scheduler,
            unet=unet,
            diffusers_default_scheduler=diffusers_default_scheduler,
        )
        self.device = unet.device
        self.model_id = model_id
        self.is_conditional = False
        self.h_space = self.diff.unet.h_space
        self.resize_to = resize_to
        if self.vqvae is not None:
            self.img_size = 256
        else:
            self.img_size = self.diff.unet.model.sample_size
        self.set_inference_steps(num_inference_steps=num_inference_steps)

    # def get_eta_schedule(self, fraction=0.2, eta_scale=1, where="end"):
    #     """Get the eta schedule for the diffusion process."""
    #     T = self.num_inference_steps
    #     nr_on = int(T * fraction)
    #     nr_off = T - nr_on
    #     if where == "end":
    #         etas = torch.tensor([0] * nr_off + [1] * nr_on) * eta_scale
    #     elif where == "end":
    #         etas = torch.tensor([1] * nr_on + [0] * nr_off) * eta_scale
    #     else:
    #         raise NotImplementedError
    #     return etas

    def set_inference_steps(self, num_inference_steps=50):
        """Set the number of inference steps for the diffusion process."""
        self.num_inference_steps = num_inference_steps
        self.diff.update_inference_steps(num_inference_steps)
        self.model_label = (
            self.model_id.replace("/", "-")
            + f"steps{self.num_inference_steps}-hspace-{self.h_space}"
        )

    def encode(self, q, method_from="x0", **kwargs):
        """Encode the input data using the VQVAE model or the diffusion model."""
        with torch.autocast("cuda"), torch.inference_mode():
            if self.vqvae is not None:
                q.w0 = self.vqvae.encode(q.x0).latents
                q.wT, q.hs, q.zs = self.diff.forward(
                    q.w0, etas=q.etas, method_from=method_from, **kwargs
                )
            else:
                q.xT, q.hs, q.zs = self.diff.forward(
                    q.x0, etas=q.etas, method_from=method_from, **kwargs
                )
        return q

    def decode(self, q, **kwargs):
        """Decode the input data using the VQVAE model or the diffusion model."""
        if self.vqvae is not None:
            q.w0, q.hs, q.zs = self.diff.reverse_process(
                q.wT, zs=q.zs, etas=q.etas, delta_hs=q.delta_hs, asyrp=q.asyrp, **kwargs
            )
            with torch.autocast("cuda"), torch.inference_mode():
                q.x0 = self.vqvae.decode(q.w0).sample
        else:
            q.x0, q.hs, q.zs = self.diff.reverse_process(
                q.xT, zs=q.zs, etas=q.etas, delta_hs=q.delta_hs, asyrp=q.asyrp, **kwargs
            )
        return q

    def sample(
        self,
        decode=True,
        seed: int | None | torch.Tensor = None,
        variance_seed=None,
        etas=None,
        **kwargs,
    ):
        """Samples random noise in the dimensions of the Unet."""
        if seed is None:
            seed = torch.randint(int(1e6), (1,))
        if etas is not None:
            q = Q(seed=seed, etas=etas)
        else:
            q = Q(seed=seed)
        sample = self.diff.unet.sample(seed=seed)
        if self.vqvae is not None:
            q.wT = sample
        else:
            q.xT = sample
        if etas is not None:
            if variance_seed is None:
                variance_seed = seed + 1
            ## Very important the the first zt is not equal to xT
            ## if seed xT and seed zs is equal horrible stuff happens
            q.zs = self.sample_variance_noise(seed=variance_seed)

        if decode:
            q = self.decode(q, **kwargs)
        return q

    def sample_seeds(
        self, etas=None, num_imgs=25, imsize=None, plot_seed_nr=True, rows=5, cols=5
    ):
        """Samples random seeds for the diffusion model."""
        qs = [self.sample(etas=etas) for _ in tqdm(range(num_imgs))]
        imgs = [q.x0 for q in qs]
        labels = [str(int(q.seed)) for q in qs] if plot_seed_nr else None
        return image_grid(imgs, titles=labels, size=imsize, rows=rows, cols=cols)

    def sample_variance_noise(self, seed=None):
        """Samples random noise for the diffusion model."""
        return self.diff.sample_variance_noise(seed=seed)

    def apply_direction(self, q, n, scale: float | torch.Tensor = 1.0, space="hspace"):
        """Applies a direction to a diffusion model."""
        q_edit = q.copy()
        if space == "noise":
            if self.vqvae is not None:
                q_edit.wT = q_edit.wT + scale * n.wT.to(self.device)
            else:
                q_edit.xT = q_edit.xT + scale * n.xT.to(self.device)
            if q_edit.etas is not None:
                q_edit.zs = (
                    q_edit.zs + scale * n.zs.to(self.device)
                    if n.zs is not None
                    else q_edit.zs
                )
        elif space == "hspace":
            q_edit.delta_hs = scale * n.delta_hs.to(self.device)
        q_edit = self.decode(q_edit)
        return q_edit

    def interpolate_direction(
        self,
        q,
        n,
        space="hspace",
        t1=0,
        t2=1,
        numsteps=5,
        vertical=False,
        plot_strength=True,
    ):
        """Interpolates between two directions in the and returns the output image."""
        qs = []
        interval = torch.linspace(t1, t2, numsteps)
        for t in interval:
            q_edit = self.apply_direction(q, n, scale=t, space=space)
            qs.append(q_edit)

        imgs = torch.cat([q.x0 for q in qs])
        interval = [str(i.item()) for i in interval]
        if plot_strength is False:
            interval = None

        if vertical:
            plot = image_grid(
                imgs, titles=interval, cols=1, rows=len(imgs), size=self.resize_to
            )
        else:
            plot = image_grid(imgs, titles=interval, size=self.resize_to)

        return plot

    def interpolate(
        self, q1, q2, space="pixel", method="lerp", t1=0, t2=1, numsteps=5, to_img=False
    ):
        """Interpolates between two directions in the diffusion model."""
        qs = []

        for t in tqdm(torch.linspace(t1, t2, numsteps)):
            q = q1.copy()
            if space == "pixel":
                q.x0 = self.interp(q1.x0, q2.x0, t, method=method)

            elif space == "noise":
                if self.vqvae is not None:
                    q.wT = self.interp(q1.wT, q2.wT, t, method=method)
                else:
                    q.xT = self.interp(q1.xT, q2.xT, t, method=method)
                if q.etas is not None:
                    q.zs = self.interp(q1.zs, q2.zs, t, method=method)
                q = self.decode(q)

            elif space == "hspace":
                if self.vqvae is not None:
                    q.wT = q1.wT
                else:
                    q.xT = q1.xT
                q.delta_hs = self.interp(
                    torch.zeros_like(q2.delta_hs), q2.delta_hs, t, method=method
                )
                q = self.decode(q)
            elif space == "vq-denoisedspace":
                assert self.vqvae is not None
                raise NotImplementedError
                # q.wT = self.interp(q1.wT, q2.wT, t, method=method
            else:
                raise NotImplementedError
            qs.append(q)
        if to_img:
            qs = torch.cat([q.x0 for q in qs])
        return qs

    def show(self, q: Q | list[Q]):
        """Show the output image the diffusion model."""
        if isinstance(q, Q):
            q = [q]
        imgs = [q_.x0 for q_ in q]
        imgs = torch.cat(imgs)

        return image_grid(imgs, size=self.resize_to)
