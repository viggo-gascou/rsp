"""Representation state class."""

import copy
from dataclasses import dataclass, field

import torch


def _default_empty_tensor() -> torch.Tensor:
    """Create an empty tensor, used only in the stateclass for initialization."""
    return torch.empty(0)


@dataclass
class Q:
    """Representation state class.

    Attributes:
        x0:
            Clean image tensor. Defaults to an empty tensor.
        xT:
            Noisy image tensor. Defaults to an empty tensor.
        w0:
            Clean latent tensor (in case is_vq=True). Defaults to an empty tensor.
        wT:
            Noisy latent tensor. Defaults to an empty tensor.
        hs:
            h-space representation tensor. Defaults to an empty tensor.
        zs:
            Variance noise added tensor (pre-sampled). Default to an empty tensor.
        etas:
            eta schedule tensor. Default to an empty tensor.
        delta_hs:
            delta hs to be injected during decoding tensor. Default to an empty tensor.
        seed:
            Seed for random number generators. Defaults to 0.
        asyrp:
            If delta hs injection is asymetrical. Defaults to False.
        cfg_scale:
            Classifier free guidance scale. Defaults to 1.0.
        label:
            Label. Defaults to an empty string.
    """

    x0: torch.Tensor = field(default_factory=_default_empty_tensor)
    xT: torch.Tensor = field(default_factory=_default_empty_tensor)
    w0: torch.Tensor = field(default_factory=_default_empty_tensor)
    wT: torch.Tensor = field(default_factory=_default_empty_tensor)
    hs: torch.Tensor = field(default_factory=_default_empty_tensor)
    zs: torch.Tensor = field(default_factory=_default_empty_tensor)
    etas: torch.Tensor | None = field(default=None)
    delta_hs: torch.Tensor = field(default_factory=_default_empty_tensor)
    seed: int | torch.Tensor = field(default=0)
    asyrp: bool = field(default=False)
    cfg_scale: float = field(default=1.0)
    label: str = field(default="")

    def copy(self):
        """Create a deepcopy of the state class."""
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
        string += f"label={self.label}"
        return string

    def __add__(self, other):
        """Add two state classes."""
        if self.delta_hs.numel() <= 0 or other.delta_hs.numel() <= 0:
            raise ValueError("Cannot add two Q instances with empty delta_hs")
        return Q(delta_hs=self.delta_hs + other.delta_hs)

    def __sub__(self, other):
        """Subtract two state classes."""
        if self.delta_hs.numel() <= 0 or other.delta_hs.numel() <= 0:
            raise ValueError("Cannot subtract two Q instances with empty delta_hs")
        return Q(delta_hs=self.delta_hs - other.delta_hs)

    def to_state_dict(self) -> dict:
        """Convert the state class to a dictionary."""
        return {
            "x0": self.x0,
            "xT": self.xT,
            "w0": self.w0,
            "wT": self.wT,
            "zs": self.zs,
            "hs": self.hs,
            "etas": self.etas,
            "delta_hs": self.delta_hs,
            "seed": self.seed,
            "asyrp": self.asyrp,
            "cfg_scale": self.cfg_scale,
            "label": self.label,
        }

    def from_state_dict(self, state_dict: dict) -> "Q":
        """Create a state class from a dictionary."""
        return Q(**state_dict)
