"""Representation state class."""

import copy
from typing import Optional

import torch


class Q:
    """Representation state class."""

    def __init__(
        self,
        x0: Optional[torch.Tensor] = None,
        xT: Optional[torch.Tensor] = None,
        w0: Optional[torch.Tensor] = None,
        wT: Optional[torch.Tensor] = None,
        hs: Optional[torch.Tensor] = None,
        zs: Optional[torch.Tensor] = None,
        etas: Optional[torch.Tensor] = None,
        delta_hs: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        asyrp: Optional[bool] = None,
        cfg_scale: Optional[float] = None,
        label: Optional[str] = None,
    ):
        """Initialize the state class.

        Args:
            x0: Clean image tensor.
            xT: Noisy image tensor.
            w0: Clean latent tensor (in case is_vq=True).
            wT: Noisy latent tensor.
            hs: h-space representation tensor.
            zs: Variance noise added tensor (pre-sampled).
            etas: eta schedule tensor.
            delta_hs: delta hs to be injected during decoding tensor.
            seed: Seed for random number generators.
            asyrp: If delta hs injection is asymetrical.
            cfg_scale: Classifier free guidance scale.
            label (Optional): Label.
        """
        self.x0 = x0
        self.xT = xT
        self.w0 = w0
        self.wT = wT

        self.zs = zs
        self.hs = hs

        # Modifiers
        self.etas = etas
        self.delta_hs = delta_hs
        self.seed = seed
        self.asyrp = asyrp
        self.cfg_scale = cfg_scale
        self.label = label

    def copy(self):
        """Create a deepcopy of the state class."""
        # return  Q(**self.__dict__.copy())
        return Q(**copy.deepcopy(self.__dict__))

    def set_delta_hs(self, delta_hs: torch.Tensor) -> "Q":
        """Set the delta hs to be injected during decoding.

        Args:
            delta_hs: Delta hs to be injected during decoding.
        """
        self.delta_hs = delta_hs
        return self

    def to_string(self) -> str:
        """Return a string representation of the state class."""
        string = f"seed={self.seed}-etas={self.etas}"
        if self.label is not None:
            string += f"label={self.label}"
        return string

    def __add__(self, other):
        """Add two state classes."""
        return Q(delta_hs=self.delta_hs + other.delta_hs)

    def __sub__(self, other):
        """Subtract two state classes."""
        return Q(delta_hs=self.delta_hs - other.delta_hs)
