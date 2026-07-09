"""
Size distributions for graph generation.

Decouples graph-size sampling from the CTMC denoising process. During
``DeFoGModel.sample``, the first step draws the number of nodes for each graph
from a :class:`SizeDistribution`. This can be:

- :class:`EmpiricalSizeDistribution` -- the marginal ``P(n)`` histogram over the
  training set (the historical default).
- :class:`FixedSizeDistribution` -- a single fixed size for every graph.
- :class:`ExplicitSizeDistribution` -- an explicit per-sample list of sizes.
- :class:`UniformSizeDistribution` -- uniform over a user-specified range.
- :class:`CategoricalSizeDistribution` -- an arbitrary user-specified pmf.
- :class:`ConditionalSizeDistribution` -- ``P(n | c)`` estimated from the
  training ``(condition, size)`` pairs, so that size-correlated properties
  (molecular weight, edge count, diameter, ...) draw a *consistent* size.

The condition passed to ``sample`` must live in the **same (normalized) space**
the model was trained on; conditioning-unaware distributions simply ignore it.
"""

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

import torch


class SizeDistribution(ABC):
    """Abstract base class for graph-size samplers."""

    @abstractmethod
    def sample(
        self,
        num_samples: int,
        condition: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Sample per-graph node counts.

        Args:
            num_samples: Number of graph sizes to draw.
            condition: Optional (num_samples, cond_dim) normalized condition;
                used only by condition-aware distributions.
            device: Device for the returned tensor.
            generator: Optional torch.Generator for reproducibility.

        Returns:
            Long tensor of shape (num_samples,), each entry >= 1.
        """

    @property
    @abstractmethod
    def max_size(self) -> int:
        """Largest size this distribution can produce (for allocation/validation)."""

    def log_prob(
        self, sizes: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Log-probability of ``sizes`` (optional; not all distributions implement it)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement log_prob"
        )

    @staticmethod
    def _to(tensor: torch.Tensor, device: Optional[torch.device]) -> torch.Tensor:
        return tensor.to(device) if device is not None else tensor


class EmpiricalSizeDistribution(SizeDistribution):
    """Marginal ``P(n)`` from a training-set size histogram (condition-agnostic)."""

    def __init__(self, histogram: Union[torch.Tensor, dict]):
        if isinstance(histogram, dict):
            max_n = max(histogram)
            prob = torch.zeros(max_n + 1)
            for size, count in histogram.items():
                prob[size] = count
        else:
            prob = torch.as_tensor(histogram).float()
        total = prob.sum()
        assert total > 0, "Size histogram must have positive mass"
        self.prob = prob / total

    def sample(self, num_samples, condition=None, device=None, generator=None):
        idx = torch.multinomial(
            self.prob, num_samples, replacement=True, generator=generator
        )
        return self._to(idx.long(), device)

    @property
    def max_size(self) -> int:
        return len(self.prob) - 1

    def log_prob(self, sizes, condition=None):
        p = self.prob.to(sizes.device)
        return torch.log(p[sizes] + 1e-30)


class FixedSizeDistribution(SizeDistribution):
    """Every graph has exactly ``size`` nodes."""

    def __init__(self, size: int):
        assert size >= 1, "size must be >= 1"
        self.size = int(size)

    def sample(self, num_samples, condition=None, device=None, generator=None):
        return torch.full((num_samples,), self.size, dtype=torch.long, device=device)

    @property
    def max_size(self) -> int:
        return self.size


class ExplicitSizeDistribution(SizeDistribution):
    """An explicit, per-sample list of sizes (one entry per requested sample)."""

    def __init__(self, sizes: Union[torch.Tensor, Sequence[int]]):
        self.sizes = torch.as_tensor(sizes, dtype=torch.long).view(-1)
        assert (self.sizes >= 1).all(), "all sizes must be >= 1"

    def sample(self, num_samples, condition=None, device=None, generator=None):
        assert len(self.sizes) == num_samples, (
            f"ExplicitSizeDistribution has {len(self.sizes)} sizes but "
            f"{num_samples} samples were requested"
        )
        return self._to(self.sizes, device)

    @property
    def max_size(self) -> int:
        return int(self.sizes.max())


class UniformSizeDistribution(SizeDistribution):
    """Uniform over the integer range ``[min_size, max_size]`` (inclusive)."""

    def __init__(self, min_size: int, max_size: int):
        assert 1 <= min_size <= max_size, "require 1 <= min_size <= max_size"
        self.min_size = int(min_size)
        self._max_size = int(max_size)

    def sample(self, num_samples, condition=None, device=None, generator=None):
        return torch.randint(
            self.min_size,
            self._max_size + 1,
            (num_samples,),
            dtype=torch.long,
            device=device,
            generator=generator,
        )

    @property
    def max_size(self) -> int:
        return self._max_size


class CategoricalSizeDistribution(SizeDistribution):
    """Arbitrary user-specified pmf over a set of sizes."""

    def __init__(
        self,
        sizes: Union[torch.Tensor, Sequence[int]],
        probs: Optional[Union[torch.Tensor, Sequence[float]]] = None,
    ):
        self.sizes = torch.as_tensor(sizes, dtype=torch.long).view(-1)
        assert (self.sizes >= 1).all(), "all sizes must be >= 1"
        if probs is None:
            probs = torch.ones(len(self.sizes))
        else:
            probs = torch.as_tensor(probs, dtype=torch.float).view(-1)
        assert len(probs) == len(self.sizes), "sizes and probs must align"
        assert (probs >= 0).all() and probs.sum() > 0, "probs must be non-negative"
        self.probs = probs / probs.sum()

    def sample(self, num_samples, condition=None, device=None, generator=None):
        idx = torch.multinomial(
            self.probs, num_samples, replacement=True, generator=generator
        )
        return self._to(self.sizes[idx], device)

    @property
    def max_size(self) -> int:
        return int(self.sizes.max())

    def log_prob(self, sizes, condition=None):
        lookup = {int(s): float(p) for s, p in zip(self.sizes, self.probs)}
        p = torch.tensor(
            [lookup.get(int(s), 0.0) for s in sizes], device=sizes.device
        )
        return torch.log(p + 1e-30)


class ConditionalSizeDistribution(SizeDistribution):
    """
    ``P(n | c)`` estimated from training ``(condition, size)`` pairs.

    Two estimators are available:

    - ``method="kernel"`` (default): non-parametric Nadaraya-Watson resampling.
      For a query ``c`` each training size ``n_j`` is weighted by a Gaussian
      kernel of the property distance ``||c - c_j||``, and a size is drawn from
      that weighted set. Captures multi-modal / nonlinear size dependence but
      does not extrapolate beyond the training support (an extreme target
      collapses onto the nearest training region).
    - ``method="regression"``: fit ``n ~ Normal(a . c + b, sigma^2)`` by least
      squares and sample. Extrapolates smoothly to novel targets but assumes a
      (near-)linear, unimodal relationship.

    When ``condition`` is ``None`` at sampling time, both methods fall back to
    the marginal ``P(n)`` over the stored sizes.
    """

    def __init__(
        self,
        conditions: torch.Tensor,
        sizes: torch.Tensor,
        method: str = "kernel",
        bandwidth: Union[str, float] = "median",
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ):
        assert method in ("kernel", "regression"), f"unknown method {method!r}"
        conditions = torch.as_tensor(conditions, dtype=torch.float)
        if conditions.dim() == 1:
            conditions = conditions.unsqueeze(-1)
        sizes = torch.as_tensor(sizes, dtype=torch.long).view(-1)
        assert conditions.size(0) == sizes.size(0), "conditions and sizes must align"
        assert conditions.size(0) > 0, "need at least one (condition, size) pair"

        self.conditions = conditions
        self.sizes = sizes
        self.method = method
        self._min_size = int(min_size) if min_size is not None else int(sizes.min())
        self._max_size = int(max_size) if max_size is not None else int(sizes.max())

        # Marginal fallback for condition=None.
        self._marginal = CategoricalSizeDistribution(
            *torch.unique(sizes, return_counts=True)
        )

        if method == "kernel":
            if bandwidth == "median":
                self.bandwidth = self._median_bandwidth(conditions)
            else:
                self.bandwidth = float(bandwidth)
        else:  # regression
            self._fit_regression(conditions, sizes)

    # -- estimators ----------------------------------------------------------

    @staticmethod
    def _median_bandwidth(conditions: torch.Tensor) -> float:
        n = conditions.size(0)
        if n < 2:
            return 1.0
        m = min(n, 1000)
        idx = torch.randperm(n)[:m]
        d = torch.pdist(conditions[idx])
        if d.numel() == 0:
            return 1.0
        med = d.median()
        return float(med.clamp(min=1e-3))

    def _fit_regression(self, conditions: torch.Tensor, sizes: torch.Tensor):
        # n ~ [c, 1] @ beta ; homoscedastic residual std.
        X = torch.cat([conditions, torch.ones(conditions.size(0), 1)], dim=1)
        y = sizes.float().unsqueeze(1)
        self._beta = torch.linalg.lstsq(X, y).solution  # (D+1, 1)
        resid = (y - X @ self._beta).squeeze(1)
        sigma = resid.std() if resid.numel() > 1 else torch.tensor(0.0)
        self._sigma = float(sigma) if torch.isfinite(sigma) else 0.0

    # -- sampling ------------------------------------------------------------

    def sample(self, num_samples, condition=None, device=None, generator=None):
        if condition is None:
            return self._marginal.sample(
                num_samples, device=device, generator=generator
            )

        c = torch.as_tensor(condition, dtype=torch.float)
        if c.dim() == 1:
            c = c.unsqueeze(0)
        c = c.to(self.conditions.device)
        assert c.size(0) == num_samples, (
            f"condition has {c.size(0)} rows but {num_samples} samples requested"
        )

        if self.method == "kernel":
            n = self._sample_kernel(c, generator)
        else:
            n = self._sample_regression(c, generator)

        n = n.clamp(self._min_size, self._max_size).long()
        return self._to(n, device)

    def _sample_kernel(self, c: torch.Tensor, generator) -> torch.Tensor:
        # (num_samples, N) squared distances in property space.
        d2 = torch.cdist(c, self.conditions) ** 2
        # Subtract per-row min so the nearest neighbour always keeps weight 1;
        # this avoids all-zero rows for far-away (extrapolated) queries and is
        # numerically stable.
        d2 = d2 - d2.min(dim=1, keepdim=True).values
        w = torch.exp(-d2 / (2 * self.bandwidth ** 2))
        idx = torch.multinomial(w, 1, generator=generator).squeeze(1)
        return self.sizes[idx]

    def _sample_regression(self, c: torch.Tensor, generator) -> torch.Tensor:
        X = torch.cat([c, torch.ones(c.size(0), 1, device=c.device)], dim=1)
        mean = (X @ self._beta.to(c.device)).squeeze(1)
        noise = torch.randn(c.size(0), generator=generator, device=c.device)
        return torch.round(mean + self._sigma * noise)

    @property
    def max_size(self) -> int:
        return self._max_size

    # -- construction helpers ------------------------------------------------

    @classmethod
    def from_dataloader(
        cls, dataloader, method: str = "kernel", **kwargs
    ) -> "ConditionalSizeDistribution":
        """
        Build from a PyG dataloader whose graphs carry a condition in ``batch.y``.

        Per-graph size is the number of nodes; the condition is ``batch.y``
        (expected already normalized, one row per graph).
        """
        conditions, sizes = [], []
        for batch in dataloader:
            if getattr(batch, "y", None) is None:
                raise ValueError(
                    "ConditionalSizeDistribution.from_dataloader requires graphs "
                    "with a condition in `.y`"
                )
            counts = torch.bincount(batch.batch)  # (num_graphs,)
            conditions.append(batch.y.view(counts.size(0), -1).float())
            sizes.append(counts)
        conditions = torch.cat(conditions, dim=0)
        sizes = torch.cat(sizes, dim=0)
        return cls(conditions, sizes, method=method, **kwargs)
