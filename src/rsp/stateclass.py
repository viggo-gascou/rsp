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
    zs: torch.Tensor | None = field(default=None)
    etas: torch.Tensor | None = field(default=None)
    delta_hs: torch.Tensor | None = field(default=None)
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

    def _is_initialized(self, tensor: torch.Tensor | None) -> bool:
        if tensor is None:
            return False
        return tensor.numel() <= 0

    def __add__(self, other):
        """Add two state classes."""
        if self._is_initialized(self.delta_hs) and self._is_initialized(other.delta_hs):
            raise ValueError("Cannot add two Q instances with empty delta_hs")
        return Q(delta_hs=self.delta_hs + other.delta_hs)

    def __sub__(self, other):
        """Subtract two state classes."""
        if self._is_initialized(self.delta_hs) and self._is_initialized(other.delta_hs):
            raise ValueError("Cannot subtract two Q instances with empty delta_hs")
        return Q(delta_hs=self.delta_hs - other.delta_hs)

    def to_state_dict(self) -> dict:
        """Convert the state class to a dictionary."""
        label_bytes = self.label.encode("utf-8") if self.label else b" "
        label_tensor = torch.frombuffer(bytearray(label_bytes), dtype=torch.uint8)
        seed = torch.Tensor([self.seed]) if isinstance(self.seed, int) else self.seed

        return {
            "x0": self.x0,
            "xT": self.xT,
            "w0": self.w0,
            "wT": self.wT,
            "hs": self.hs,
            "zs": self.zs if self.zs is not None else torch.empty(0),
            "etas": self.etas if self.etas is not None else torch.empty(0),
            "delta_hs": self.delta_hs if self.delta_hs is not None else torch.empty(0),
            "seed": seed,
            "asyrp": torch.Tensor([self.asyrp]),
            "cfg_scale": torch.Tensor([self.cfg_scale]),
            "label": label_tensor,
        }

    def from_state_dict(self, state_dict: dict) -> "Q":
        """Create a state class from a dictionary."""
        # Convert label tensor back to string
        label_tensor = state_dict.get("label", torch.empty(0, dtype=torch.uint8))
        if label_tensor.numel() > 0:
            label = bytes(label_tensor.numpy()).decode("utf-8")
        else:
            label = ""

        return Q(
            x0=state_dict["x0"],
            xT=state_dict["xT"],
            w0=state_dict["w0"],
            wT=state_dict["wT"],
            zs=state_dict["zs"],
            hs=state_dict["hs"],
            etas=state_dict["etas"],
            delta_hs=state_dict["delta_hs"],
            seed=state_dict["seed"],
            asyrp=state_dict["asyrp"],
            cfg_scale=state_dict["cfg_scale"],
            label=label,
        )
