"""
Time distortion for variable step sizes in training and sampling.

Time distortion enables non-uniform sampling of time steps, which is critical
for discrete graph generation where steps near t=1 are more important for
preserving global graph properties.
"""

import torch
from typing import Literal

TimeDistortionType = Literal["identity", "polydec", "polyinc", "cos", "revcos"]


class TimeDistorter:
    """
    Time distortion for variable step sizes in training and sampling.

    Available distortions:
    - "identity": f(t) = t (uniform steps)
    - "polydec": f(t) = 2t - t^2 (smaller steps near t=1, critical for planarity)
    - "polyinc": f(t) = t^2 (smaller steps near t=0)
    - "cos": f(t) = (1 - cos(pi*t))/2 (emphasize boundaries)
    - "revcos": f(t) = 2t - (1 - cos(pi*t))/2

    The "polydec" distortion is particularly important for graph generation
    as it ensures smaller step sizes near t=1 where global properties
    (like planarity) are determined. Using polydec can improve planar
    validity from ~77% to ~99%.

    Args:
        train_distortion: Time distortion type to use during training
        sample_distortion: Default time distortion type for sampling

    Example:
        >>> distorter = TimeDistorter(train_distortion="identity", sample_distortion="polydec")
        >>> t = distorter.train_ft(batch_size=32, device="cuda")  # Sample training times
        >>> t_distorted = distorter.sample_ft(t, "polydec")  # Apply distortion for sampling
    """

    def __init__(
        self,
        train_distortion: TimeDistortionType = "identity",
        sample_distortion: TimeDistortionType = "identity",
    ):
        self.train_distortion = train_distortion
        self.sample_distortion = sample_distortion

    def train_ft(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample distorted times for training.

        Samples uniform times and applies the training distortion.

        Args:
            batch_size: Number of time samples
            device: Device to place the tensor on

        Returns:
            Tensor of shape (batch_size, 1) with distorted times in [0, 1]
        """
        t_uniform = torch.rand((batch_size, 1), device=device)
        return self._apply_distortion(t_uniform, self.train_distortion)

    def sample_ft(
        self,
        t: torch.Tensor,
        distortion: TimeDistortionType = None
    ) -> torch.Tensor:
        """
        Apply distortion for sampling.

        Args:
            t: Input times in [0, 1]
            distortion: Distortion type to use. If None, uses default sample_distortion.

        Returns:
            Distorted times
        """
        if distortion is None:
            distortion = self.sample_distortion
        return self._apply_distortion(t, distortion)

    def _apply_distortion(self, t: torch.Tensor, distortion: str) -> torch.Tensor:
        """
        Apply time distortion function.

        Args:
            t: Input times in [0, 1]
            distortion: Type of distortion to apply

        Returns:
            Distorted times in [0, 1]
        """
        # Clamp to valid range
        t = t.clamp(0, 1)

        if distortion == "identity":
            return t
        elif distortion == "polydec":
            # f(t) = 2t - t^2: derivative is 2 - 2t, smaller steps near t=1
            return 2 * t - t ** 2
        elif distortion == "polyinc":
            # f(t) = t^2: derivative is 2t, smaller steps near t=0
            return t ** 2
        elif distortion == "cos":
            # f(t) = (1 - cos(pi*t))/2: emphasizes boundaries
            return (1 - torch.cos(t * torch.pi)) / 2
        elif distortion == "revcos":
            # f(t) = 2t - (1 - cos(pi*t))/2: reverse cosine
            return 2 * t - (1 - torch.cos(t * torch.pi)) / 2
        else:
            raise ValueError(f"Unknown distortion type: {distortion}")

    def __repr__(self) -> str:
        return (
            f"TimeDistorter(train={self.train_distortion}, "
            f"sample={self.sample_distortion})"
        )
