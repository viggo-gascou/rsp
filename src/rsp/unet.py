"""Custom UNet model used for the project."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from diffusers.models.resnet import ResnetBlock2D, ResnetBlockCondNorm2D
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D


def forward_ResnetBlock2D(
    resnet_block: ResnetBlockCondNorm2D | ResnetBlock2D,
    input_tensor: torch.Tensor,
    temb: Optional[torch.Tensor],
    inject_into_botleneck: Optional[torch.Tensor] = None,
):
    """Forward pass for ResnetBlock2D.

    Args:
        resnet_block: The ResnetBlock of the UNet model.
        input_tensor: Input tensor.
        temb: Temporal embedding.
        inject_into_botleneck (Optional): Injection tensor. Defaults to None.
    """
    ## From https://github.com/huggingface/diffusers/blob/fc94c60c8373862c509e388f3f4065d98cedf589/src/diffusers/models/resnet.py#L367
    hidden_states = input_tensor

    hidden_states = resnet_block.norm1(hidden_states)
    hidden_states = resnet_block.nonlinearity(hidden_states)

    if resnet_block.upsample is not None:
        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            input_tensor = input_tensor.contiguous()
            hidden_states = hidden_states.contiguous()
        input_tensor = resnet_block.upsample(input_tensor)
        hidden_states = resnet_block.upsample(hidden_states)
    elif resnet_block.downsample is not None:
        input_tensor = resnet_block.downsample(input_tensor)
        hidden_states = resnet_block.downsample(hidden_states)

    hidden_states = resnet_block.conv1(hidden_states)

    if temb is not None:
        new_temb = resnet_block.time_emb_proj(resnet_block.nonlinearity(temb))[
            :, :, None, None
        ]
        hidden_states = hidden_states + new_temb

    hidden_states = resnet_block.norm2(hidden_states)

    ### Middle of block injection
    bottleneck = hidden_states.clone()
    if inject_into_botleneck is not None:
        hidden_states = bottleneck + inject_into_botleneck

    hidden_states = resnet_block.nonlinearity(hidden_states)

    hidden_states = resnet_block.dropout(hidden_states)
    hidden_states = resnet_block.conv2(hidden_states)

    # if self.conv_shortcut is not None:
    #     input_tensor = self.conv_shortcut(input_tensor)

    output_tensor = input_tensor + hidden_states

    return output_tensor, bottleneck


def mid_block_forward(
    mid_block: UNetMidBlock2D,
    hidden_states: torch.Tensor,
    temb=None,
    encoder_states=None,
    inject_into_botleneck=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for middle block of UNet.

    Args:
        mid_block: The middle block of UNet.
        hidden_states: Hidden states.
        temb: Time embedding.
        encoder_states: Encoder states.
        inject_into_botleneck: Inject into bottleneck.

    Returns:
        A pair, containing the hidden states and the bottleneck.
    """
    ## https://github.com/huggingface/diffusers/blob/fc94c60c8373862c509e388f3f4065d98cedf589/src/diffusers/models/unet_2d_blocks.py#L246
    hidden_states = mid_block.resnets[0](hidden_states, temb)
    bottleneck = None

    for attn, resnet in zip(mid_block.attentions, mid_block.resnets[1:]):
        hidden_states = attn(hidden_states)
        hidden_states, bottleneck = forward_ResnetBlock2D(
            resnet, hidden_states, temb, inject_into_botleneck=inject_into_botleneck
        )

    if bottleneck is None:
        raise ValueError("bottleneck is None, something bad happened!")

    return hidden_states, bottleneck


@dataclass
class UNetOutput:
    """Output of a UNet model.

    Args:
        out: Output of last layer of model.
        h: Hidden states output.
    """

    out: torch.Tensor
    h: Optional[torch.Tensor]


class UNet:
    """UNet model."""

    def __init__(self, model: UNet2DModel, h_space: Optional[str] = None):
        """Initialize the UNet model."""
        if h_space not in [None, "before", "after", "middle"]:
            raise ValueError(
                "h_space must be one of [None, 'before', 'after', 'middle']"
            )
        self.h_space = h_space
        self.model = model
        self.device = model.device

    def time_embedding(
        self, timestep: int | torch.Tensor, batch_dim: int
    ) -> torch.Tensor:
        """Compute the time embedding for the UNet model.

        Args:
            timestep: The timestep to compute the embedding for.
            batch_dim: The batch dimension.

        Returns:
            The time embedding.
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=self.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(self.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(
            batch_dim, dtype=timesteps.dtype, device=timesteps.device
        )

        t_emb = self.model.time_proj(timesteps)
        emb = self.model.time_embedding(t_emb)
        return emb

    def forward(
        self,
        sample: torch.Tensor,
        timestep: int | torch.Tensor,
        delta_h: Optional[torch.Tensor] = None,
    ):
        """Compute the forward pass of the UNet model.

        Args:
            sample: The input sample.
            timestep: The timestep to compute the embedding for.
            delta_h: The delta height.

        Returns:
            The output of the forward pass.
        """
        # Modified from From: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d.py
        bottleneck = None

        # 1. Positional embedding
        emb = self.time_embedding(timestep, batch_dim=sample.shape[0])

        # 2. pre-process
        skip_sample = sample
        sample = self.model.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.model.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # 4. mid
        if self.h_space == "before":
            bottleneck = sample.clone()
            if delta_h is not None:
                sample = bottleneck + delta_h

        if self.h_space == "middle":
            sample, bottleneck = mid_block_forward(
                self.model.mid_block,
                sample,
                temb=emb,
                encoder_states=None,
                inject_into_botleneck=delta_h,
            )

        else:
            sample = self.model.mid_block(sample, emb)

        if self.h_space == "after":
            bottleneck = sample.clone()
            if delta_h is not None:
                sample = bottleneck + delta_h

        # 5. up
        skip_sample = None
        for upsample_block in self.model.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]
            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(
                    sample, res_samples, emb, skip_sample
                )
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process make sure hidden states is in float32 when running
        # in half-precision
        sample = self.model.conv_norm_out(sample.float()).type(sample.dtype)
        sample = self.model.conv_act(sample)
        sample = self.model.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        return UNetOutput(out=sample, h=None if self.h_space is None else bottleneck)

    def sample(self, num_samples: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        """Samples random noise in the dimensions of the Unet.

        Args:
            num_samples: Number of samples to generate.
            seed (Optional): Seed for the random number generator.

        Returns:
            Random noise tensor.
        """
        return torch.randn(
            num_samples,
            self.model.config.in_channels,
            self.model.sample_size,
            self.model.sample_size,
            generator=torch.manual_seed(seed) if seed is not None else None,
        ).to(self.device)
