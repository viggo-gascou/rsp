"""Module for interpolation."""

import numpy as np
import torch


class Interpolations:
    """Class for interpolation."""

    def interpolation(self, z1, z2, numsteps=5, t_min=0, t_max=1, method="lerp"):
        """Interpolate between two tensors z1 and z2 using the specified method."""
        return torch.cat(
            [
                self.interp(z1, z2, t, method=method)
                for t in torch.linspace(t_min, t_max, numsteps)
            ]
        )

    def interp(self, z1, z2, t, method="lerp"):
        """Interpolate between two tensors z1 and z2 using the specified method."""
        if method == "lerp":
            return self.lerp(z1, z2, t)
        elif method == "slerp":
            return self.slerp(z1, z2, t)
        elif method == "sqlerp":
            return self.sqlerp(z1, z2, t)
        else:
            raise NotImplementedError("only lerp and slerp implemented")

    def slerp(self, v0, v1, t, DOT_THRESHOLD=0.9995):
        """Helper function to spherically interpolate two arrays v1 v2."""
        inputs_are_torch = False
        if not isinstance(v0, np.ndarray):
            inputs_are_torch = True
            input_device = v0.device
            v0 = v0.cpu().numpy()
            v1 = v1.cpu().numpy()
            t = t.cpu().numpy()
        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > DOT_THRESHOLD:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1

        if inputs_are_torch:
            v2 = torch.from_numpy(v2).to(input_device)
        return v2

    def lerp(self, z1, z2, t):
        """Interpolate between two tensors using linear interpolation."""
        return z1 * (1 - t) + z2 * t

    def sqlerp(self, z1, z2, t):
        """Interpolate between two tensors using squared linear interpolation."""
        return z1 * (1 - t) ** 0.5 + z2 * t**0.5
