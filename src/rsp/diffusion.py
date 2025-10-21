"""Custom Diffusion model used for the project."""

import logging

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from tqdm import tqdm

from .unet import UNet

logger = logging.getLogger("rsp")


class Diffusion:
    """Diffusion model used for the project."""

    def __init__(
        self,
        unet: UNet,
        scheduler: DDIMScheduler,
        diffusers_default_scheduler=False,
    ):
        """Initialize the diffusion model."""
        self.unet = unet
        self.device = self.unet.device

        self.scheduler = scheduler
        self.diffusers_default_scheduler = diffusers_default_scheduler
        self.sanity_check()

        self.update_inference_steps(num_inference_steps=50)

    def update_inference_steps(self, num_inference_steps=50):
        """Update the inference steps."""
        self.num_inference_steps = num_inference_steps
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        self.t_to_idx = {int(v): k for k, v in enumerate(self.timesteps)}

        self.h_shape = self.get_h_shape()
        self.variance_noise_shape = (
            self.num_inference_steps,
            self.unet.model.config["in_channels"],
            self.unet.model.config["sample_size"],
            self.unet.model.config["sample_size"],
        )

    def get_h_shape(self):
        """Return the shape of the h tensors."""
        xT = self.unet.sample()
        with torch.no_grad():
            out = self.unet.forward(xT, timestep=self.timesteps[-1])
        return (self.num_inference_steps,) + tuple(out.h.shape[1:])

    def sanity_check(self):
        """Perform sanity checks."""
        if self.scheduler.config["clip_sample"]:
            logger.warning("Scheduler assumes clipping, setting to false")
            self.scheduler.config["clip_sample"] = False

    def get_variance(self, timestep):
        """Calculate the variance for a given timestep."""
        prev_timestep = (
            timestep
            - self.scheduler.config["num_train_timesteps"]
            // self.scheduler.num_inference_steps
        )
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )
        return variance

    def sample_variance_noise(self, seed=None):
        """Samples variance noise."""
        if seed is None:
            seed = torch.randint(int(1e6), (1,))
        return torch.randn(
            size=self.variance_noise_shape, generator=torch.manual_seed(seed)
        ).to(self.device)

    def reverse_step(
        self, model_output, timestep, sample, eta=0, asyrp=None, variance_noise=None
    ):
        """Reverse step for diffusion process."""
        # 1. get previous step value (=t-1)
        prev_timestep = (
            timestep
            - self.scheduler.config["num_train_timesteps"]
            // self.scheduler.num_inference_steps
        )
        # 2. compute alphas, betas

        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        # variance = self.scheduler._get_variance(timestep, prev_timestep)
        variance = self.get_variance(timestep)  # , prev_timestep)
        std_dev_t = eta * variance ** (0.5)
        # Take care of asymetric reverse process (asyrp)
        model_output_direction = model_output if asyrp is None else asyrp
        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
            0.5
        ) * model_output_direction
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        )
        # 8. Add noice if eta > 0
        if eta > 0:
            if variance_noise is None:
                variance_noise = torch.randn(model_output.shape, device=self.device)
            sigma_z = eta * variance ** (0.5) * variance_noise
            prev_sample = prev_sample + sigma_z

        return prev_sample

    def reverse_process(
        self, xT, etas=0, prog_bar=False, zs=None, delta_hs=None, asyrp=False
    ):
        """Reverse process for diffusion."""
        if etas is None:
            etas = 0
        if isinstance(etas, (int, float)):
            etas = [etas] * self.num_inference_steps
        assert len(etas) == self.num_inference_steps

        xt = xT
        hs = torch.zeros(self.h_shape).to(self.device)
        op = tqdm(self.timesteps) if prog_bar else self.timesteps
        for t in op:
            idx = self.t_to_idx[int(t)]
            delta_h = None if delta_hs is None else delta_hs[idx][None]

            with torch.no_grad():
                out = self.unet.forward(xt, timestep=t, delta_h=delta_h)
            hs[idx] = out.h.squeeze()

            # Support for asyrp
            # ++++++++++++++++++++++++++++++++++++++++
            if asyrp and delta_hs is not None:
                with torch.no_grad():
                    out_asyrp = self.unet.forward(xt, timestep=t)
                residual_d = out_asyrp.out
            else:
                residual_d = None
            # ----------------------------------------------------
            z = zs[idx] if zs is not None else None

            # 2. compute less noisy image and set x_t -> x_t-1
            xt = self.reverse_step(
                out.out, t, xt, asyrp=residual_d, eta=etas[idx], variance_noise=z
            )
        return xt, hs, zs

    def add_noise(self, original_samples, noise, timesteps):
        """Adds noise to original samples."""
        # Make sure alphas_cumprod and timestep have same device and dtype as
        # original_samples
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        # timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = self.scheduler.alphas_cumprod[timesteps] ** 0.5
        # sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        #     sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (
            1 - self.scheduler.alphas_cumprod[timesteps]
        ) ** 0.5
        # sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        #     sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def forward_step(self, model_output, timestep, sample):
        """Forward step for diffusion process."""
        next_timestep = min(
            self.scheduler.config["num_train_timesteps"] - 2,
            timestep
            + self.scheduler.config["num_train_timesteps"]
            // self.scheduler.num_inference_steps,
        )

        # 2. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)

        next_sample = self.scheduler.add_noise(
            pred_original_sample, model_output, torch.IntTensor([next_timestep])
        )
        return next_sample

    def forward(self, x0, etas=None, method_from="x0", prog_bar=False):
        """Forward process for diffusion."""
        if etas is None or (isinstance(etas, (int, float)) and etas == 0):
            eta_is_zero = True
            etas = torch.empty(0)
        else:
            eta_is_zero = False
            if isinstance(etas, (int, float)):
                etas = [etas] * self.num_inference_steps
        xts = self.sample_xts_from_x0(x0, method_from=method_from)
        alpha_bar = self.scheduler.alphas_cumprod
        zs = torch.zeros_like(self.sample_variance_noise())

        xt = x0
        hs = torch.zeros(self.h_shape).to(self.device)
        op = tqdm(reversed(self.timesteps)) if prog_bar else reversed(self.timesteps)

        for t in op:
            idx = self.t_to_idx[int(t)]
            # 1. predict noise residual
            if not eta_is_zero:
                xt = xts[idx][None]

            with torch.no_grad():
                out = self.unet.forward(xt, timestep=t)
            hs[idx] = out.h.squeeze()

            if eta_is_zero:
                # 2. compute more noisy image and set x_t -> x_t+1
                xt = self.forward_step(out.out, t, xt)

            else:
                xtm1 = xts[idx + 1][None]
                # pred of x0
                pred_original_sample = (
                    xt - (1 - alpha_bar[t]) ** 0.5 * out.out
                ) / alpha_bar[t] ** 0.5

                # direction to xt
                prev_timestep = (
                    t
                    - self.scheduler.config["num_train_timesteps"]
                    // self.scheduler.num_inference_steps
                )
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[prev_timestep]
                    if prev_timestep >= 0
                    else self.scheduler.final_alpha_cumprod
                )

                variance = self.get_variance(t)
                pred_sample_direction = (
                    1 - alpha_prod_t_prev - etas[idx] * variance
                ) ** (0.5) * out.out

                mu_xt = (
                    alpha_prod_t_prev ** (0.5) * pred_original_sample
                    + pred_sample_direction
                )

                z = (xtm1 - mu_xt) / (etas[idx] * variance**0.5)
                zs[idx] = z
        if zs is not None:
            zs[-1] = torch.zeros_like(zs[-1])

        return xt, hs, zs

    def sample_xts_from_x0(self, x0, method_from="x0"):
        """Samples from P(x_1:T|x_0)."""
        assert method_from in ["x0", "x_prev", "dpm"]
        # torch.manual_seed(43256465436)

        alpha_bar = self.scheduler.alphas_cumprod
        sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
        alphas = self.scheduler.alphas
        betas = 1 - alphas

        if method_from == "x0":
            xts = torch.zeros(size=self.variance_noise_shape).to(x0.device)
            for t in reversed(self.timesteps):
                idx = self.t_to_idx[int(t)]
                xts[idx] = (
                    x0 * (alpha_bar[t] ** 0.5)
                    + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
                )
            xts = torch.cat([xts, x0], dim=0)

        elif method_from == "x_prev":
            xts = torch.zeros(size=self.variance_noise_shape).to(x0.device)
            x_next = x0.clone()

            for t in reversed(self.timesteps):
                noise = torch.randn_like(x0)
                idx = self.t_to_idx[int(t)]
                xt = ((1 - betas[t]) ** 0.5) * x_next + noise * (betas[t] ** 0.5)
                x_next = xt
                xts[idx] = xt

            xts = torch.cat([xts, x0], dim=0)

        elif method_from == "dpm":
            xts = torch.zeros(size=self.variance_noise_shape).to(x0.device)
            x0.clone()
            t_final = self.timesteps[0]
            xT = (
                x0 * (alpha_bar[t_final] ** 0.5)
                + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t_final]
            )
            xt = xT.clone()
            for t in self.timesteps:
                idx = self.t_to_idx[int(t)]
                xtm1 = self.sample_xtm1_from_xt_x0(xt, x0, t)
                xt = xtm1
                xts[idx] = xt
            xts = torch.cat([xts, x0], dim=0)
        else:
            raise ValueError("Invalid method_from, choose from 'x0', 'x_prev', 'dpm'")

        return xts

    def mu_tilde(self, xt, x0, timestep):
        """mu_tilde (x_t, x_0) DDPM paper eq. 7."""
        prev_timestep = (
            timestep
            - self.scheduler.config["num_train_timesteps"]
            // self.scheduler.num_inference_steps
        )
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        alpha_t = self.scheduler.alphas[timestep]
        beta_t = 1 - alpha_t
        alpha_bar = self.scheduler.alphas_cumprod[timestep]
        return ((alpha_prod_t_prev**0.5 * beta_t) / (1 - alpha_bar)) * x0 + (
            (alpha_t**0.5 * (1 - alpha_prod_t_prev)) / (1 - alpha_bar)
        ) * xt

    def sample_xtm1_from_xt_x0(self, xt, x0, t):
        """DDPM paper equation 6."""
        prev_timestep = (
            t
            - self.scheduler.config["num_train_timesteps"]
            // self.scheduler.num_inference_steps
        )
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        alpha_t = self.scheduler.alphas[t]
        beta_t = 1 - alpha_t
        alpha_bar = self.scheduler.alphas_cumprod[t]
        beta_tilde_t = ((1 - alpha_prod_t_prev) / (1 - alpha_bar)) * beta_t
        return self.mu_tilde(xt, x0, t) + beta_tilde_t**0.5 * torch.randn_like(x0)
