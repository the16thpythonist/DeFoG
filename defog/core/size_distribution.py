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
- :class:`ConditionalSizeDistribution` -- ``P(n | c)`` estimated non-parametrically
  from training ``(condition, size)`` pairs, so that size-correlated properties
  (molecular weight, edge count, diameter, ...) draw a *consistent* size.
- :class:`LearnedSizeDistribution` -- ``P(n | c)`` from a small trained MLP with a
  categorical output, the parametric counterpart of the above.
- :class:`ComposedSizeDistribution` -- product-of-experts over several
  :class:`LearnedSizeDistribution` branches, for multi-adapter steering.

**Which space is the condition in?** The older, non-parametric distributions take a
condition in whatever space their stored training conditions were in, and ignore it
entirely if they are conditioning-unaware. :class:`LearnedSizeDistribution` instead
takes the **RAW** condition and normalizes internally from its own buffers, matching
the convention :class:`~defog.core.adapter.AdaLNAdapter` and ``ConditionBranch``
already use. Prefer that convention for anything new: "the caller normalizes" is how
a target ends up normalized twice, or by the wrong statistics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ===========================================================================
# Learned P(n | c)
# ===========================================================================
_NEG_INF = float("-inf")


class LearnedSizeDistribution(nn.Module, SizeDistribution):
    """``P(n | c)`` from a small MLP with a CATEGORICAL output over node counts.

    The parametric counterpart to :class:`ConditionalSizeDistribution`, and the
    mechanism FreeGress (arXiv 2312.17397 §3.1) calls ``p_xi(n | y)``: a two-hidden-layer
    ReLU network with a softmax over sizes, replacing the dataset's marginal ``P(n)``.
    Its argument is that a target property may only be attainable at particular atom
    counts, so drawing ``n`` from the marginal and then asking the denoiser to hit the
    target means fighting the size draw for the whole trajectory.

    Measured on the ZINC reference train split, that argument holds here too:
    ``E[n | logP]`` moves from 20.2 to 26.7 heavy atoms across logP deciles (a 1.4-sigma
    swing) while the marginal offers 23.2 regardless, and the mismatch is worst in the
    top and bottom deciles -- exactly where targeting is hardest. Note what it does NOT
    do: the conditional spread barely narrows (std 4.51 -> 4.11). This is a bias
    correction on the size draw, not a variance reduction, and should be described that
    way.

    Three design choices worth stating, because each has an alternative that looks
    equivalent and is not:

    * **Categorical output, not a regressed mean.** A Gaussian head cannot express the
      multimodality of real size distributions, and -- more importantly here -- gives no
      exact ``log_pmf`` for :class:`ComposedSizeDistribution` to combine. (Note that
      :class:`ConditionalSizeDistribution` implements no ``log_prob`` at all, which is
      why it cannot be composed.)
    * **The condition arrives RAW and is normalized inside**, from the ``cond_mean`` /
      ``cond_std`` buffers, exactly as :meth:`AdaLNAdapter.normalize` does. Pair this
      model with an adapter via :meth:`check_compatible` rather than trusting a caller
      to have applied the right statistics.
    * **Sizes with zero marginal mass stay at probability zero**, at fit time and at
      sampling time. This mirrors the invariant in ``DeFoGModel._blend_rates`` that
      structurally-forbidden transitions stay forbidden: the model may reweight the
      support, never extend it.

    Args:
        cond_dim: Width of the raw condition (1 for a scalar property).
        min_size, max_size: Inclusive node-count grid. Bin ``i`` is size ``min_size + i``.
        hidden, layers: MLP width / number of hidden layers.
        cond_mean, cond_std: Normalization statistics for the raw condition.
        marginal: ``P(n)`` over the grid -- the ``condition=None`` fallback, the
            product-of-experts anchor, and the baseline the fitter scores against.
            Defaults to uniform, which makes every size "supported"; pass the real
            training histogram in anything but a unit test.
        cond_encoder: Reserved for wide conditions (fingerprints, spectra). Declared in
            the config format now so adding one later needs no package migration; no
            encoder is wired in this version.
        property_name, property_from: Provenance. ``property_from`` records the label
            convention ("decoded" / "source") the model was fit under -- a size model fit
            on source labels and paired with an adapter conditioned on decoded ones will
            disagree precisely at the extremes.
    """

    def __init__(
        self,
        cond_dim: int,
        min_size: int,
        max_size: int,
        hidden: int = 512,
        layers: int = 2,
        cond_mean=None,
        cond_std=None,
        marginal=None,
        cond_encoder: Optional[nn.Module] = None,
        property_name: str = "",
        property_from: str = "",
    ):
        nn.Module.__init__(self)
        assert min_size >= 1, "min_size must be >= 1"
        assert max_size >= min_size, "max_size must be >= min_size"
        assert layers >= 1, "need at least one hidden layer"
        self.cond_dim = int(cond_dim)
        self._min_size = int(min_size)
        self._max_size = int(max_size)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.property_name = property_name
        self.property_from = property_from

        if cond_encoder is not None:
            raise NotImplementedError(
                "cond_encoder is a reserved config slot; no encoder is wired in this "
                "version. Scalar-property conditions only."
            )
        self.cond_encoder = None

        n_bins = self._max_size - self._min_size + 1
        net: List[nn.Module] = []
        in_dim = self.cond_dim
        for _ in range(self.layers):
            net += [nn.Linear(in_dim, self.hidden), nn.ReLU()]
            in_dim = self.hidden
        net += [nn.Linear(in_dim, n_bins)]
        self.net = nn.Sequential(*net)

        m = torch.zeros(self.cond_dim) if cond_mean is None else \
            torch.as_tensor(cond_mean, dtype=torch.float32).reshape(-1)
        s = torch.ones(self.cond_dim) if cond_std is None else \
            torch.as_tensor(cond_std, dtype=torch.float32).reshape(-1).clamp_min(1e-6)
        assert m.numel() == self.cond_dim and s.numel() == self.cond_dim, \
            f"cond_mean/cond_std must have {self.cond_dim} entries"

        if marginal is None:
            p = torch.ones(n_bins)
        else:
            p = torch.as_tensor(marginal, dtype=torch.float32).reshape(-1)
            assert p.numel() == n_bins, (
                f"marginal has {p.numel()} bins but the grid "
                f"{self._min_size}..{self._max_size} has {n_bins}"
            )
            assert (p >= 0).all() and p.sum() > 0, "marginal must be non-negative with mass"
        self.register_buffer("cond_mean", m)
        self.register_buffer("cond_std", s)
        self.register_buffer("marginal", p / p.sum())

    # -- grid ----------------------------------------------------------------

    @property
    def min_size(self) -> int:
        return self._min_size

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def n_bins(self) -> int:
        return self._max_size - self._min_size + 1

    @property
    def support(self) -> torch.Tensor:
        """Boolean mask of sizes the training marginal actually gave mass to."""
        return self.marginal > 0

    @property
    def log_marginal(self) -> torch.Tensor:
        """``log P(n)`` over the grid; ``-inf`` off-support."""
        return torch.log(self.marginal)

    def sizes(self) -> torch.Tensor:
        """The node counts each bin stands for, ``(n_bins,)``."""
        return torch.arange(self._min_size, self._max_size + 1, device=self.marginal.device)

    # -- forward -------------------------------------------------------------

    def normalize(self, c: torch.Tensor) -> torch.Tensor:
        return (c - self.cond_mean) / self.cond_std

    def _prepare(self, condition, num_samples: Optional[int]) -> torch.Tensor:
        c = torch.as_tensor(condition, dtype=torch.float32, device=self.marginal.device)
        if c.dim() == 1:
            c = c.unsqueeze(0) if self.cond_dim > 1 else c.reshape(-1, 1)
        assert c.size(-1) == self.cond_dim, \
            f"condition has width {c.size(-1)}, expected {self.cond_dim}"
        if num_samples is not None and c.size(0) == 1 and num_samples > 1:
            c = c.expand(num_samples, -1)
        if num_samples is not None:
            assert c.size(0) == num_samples, \
                f"condition has {c.size(0)} rows but {num_samples} samples were requested"
        return c

    def log_pmf(self, condition=None, num_samples: Optional[int] = None) -> torch.Tensor:
        """Normalized ``log P(n | c)``, shape ``(bs, n_bins)``.

        ``condition=None`` returns the marginal, broadcast to ``num_samples`` rows --
        the same fallback :class:`ConditionalSizeDistribution` makes, so swapping the two
        cannot silently change unconditional behaviour.
        """
        support = self.support
        if condition is None:
            lp = self.log_marginal.unsqueeze(0)
            return lp.expand(num_samples or 1, -1)
        c = self._prepare(condition, num_samples)
        logits = self.net(self.normalize(c))
        return torch.log_softmax(logits.masked_fill(~support, _NEG_INF), dim=-1)

    def forward(self, condition=None, num_samples: Optional[int] = None) -> torch.Tensor:
        return self.log_pmf(condition, num_samples)

    # -- SizeDistribution ----------------------------------------------------

    def sample(self, num_samples, condition=None, device=None, generator=None):
        with torch.no_grad():
            probs = self.log_pmf(condition, num_samples=num_samples).exp()
        if probs.size(0) == 1 and num_samples > 1:
            probs = probs.expand(num_samples, -1)
        # multinomial on CPU: `generator` is a CPU generator by convention here (see
        # ConditionalSizeDistribution._sample_kernel), and the grid is tiny.
        idx = torch.multinomial(probs.cpu(), 1, generator=generator).squeeze(1)
        return self._to((idx + self._min_size).long(), device)

    def log_prob(self, sizes, condition=None):
        sizes = torch.as_tensor(sizes, dtype=torch.long).reshape(-1)
        lp = self.log_pmf(condition, num_samples=sizes.numel())
        if lp.size(0) == 1 and sizes.numel() > 1:
            lp = lp.expand(sizes.numel(), -1)
        bins = sizes.to(lp.device) - self._min_size
        inside = (bins >= 0) & (bins < self.n_bins)
        out = torch.full((sizes.numel(),), _NEG_INF, device=lp.device, dtype=lp.dtype)
        if inside.any():
            out[inside] = lp[inside].gather(1, bins[inside].unsqueeze(1)).squeeze(1)
        return out.to(sizes.device)

    # -- pairing -------------------------------------------------------------

    def check_compatible(self, adapter, tol: float = 1e-4):
        """Assert this model and ``adapter`` speak the same conditioning language.

        Catches the failure the ZINC heads already hit once: a model fit against one
        label convention paired with an adapter trained on another. Both parts are
        individually correct and the pair is wrong only at the extremes, which is where
        nobody is looking.
        """
        assert self.cond_dim == adapter.cond_dim, (
            f"size model expects a width-{self.cond_dim} condition but the adapter's is "
            f"{adapter.cond_dim}"
        )
        for name in ("cond_mean", "cond_std"):
            mine, theirs = getattr(self, name), getattr(adapter, name).to(self.marginal.device)
            assert torch.allclose(mine, theirs, atol=tol), (
                f"{name} differs between the size model ({mine.tolist()}) and the adapter "
                f"({theirs.tolist()}). They were fit on differently-scaled targets, so the "
                f"size draw will not match the steering."
            )
        return True

    # -- io ------------------------------------------------------------------

    def config(self) -> dict:
        """Architecture config needed to rebuild this model (buffers live in the
        state dict, and their shapes are all derivable from these fields)."""
        return {
            "cond_dim": self.cond_dim, "min_size": self._min_size,
            "max_size": self._max_size, "hidden": self.hidden, "layers": self.layers,
            "cond_encoder": None,
            "property_name": self.property_name, "property_from": self.property_from,
        }

    @classmethod
    def from_config(cls, config: dict, state_dict: dict, device="cpu"):
        known = {
            "cond_dim", "min_size", "max_size", "hidden", "layers",
            "property_name", "property_from",
        }
        cfg = {k: v for k, v in config.items() if k in known}
        model = cls(**cfg)
        model.load_state_dict(state_dict)      # includes cond_mean/cond_std/marginal
        return model.to(device).eval()

    def save(self, path):
        path = str(path)
        if not path.endswith(".ckpt"):
            path += ".ckpt"
        torch.save({"state_dict": self.state_dict(), "config": self.config()}, path)
        return path

    @classmethod
    def load(cls, path, device="cpu"):
        path = str(path)
        if not path.endswith(".ckpt"):
            path += ".ckpt"
        ck = torch.load(path, map_location=device, weights_only=False)
        return cls.from_config(ck["config"], ck["state_dict"], device=device)


# ===========================================================================
# Composition
# ===========================================================================
@dataclass
class SizeBranch:
    """One conditioned size model in a composition: the model, a RAW condition
    (the model normalizes internally), and its weight. Mirrors
    :class:`~defog.core.adapter.ConditionBranch`."""

    dist: LearnedSizeDistribution
    condition: Any
    weight: float = 1.0


class ComposedSizeDistribution(SizeDistribution):
    r"""Product-of-experts over several :class:`LearnedSizeDistribution` branches.

    When two adapters steer two properties, each has an opinion about how big the
    molecule should be, and the sampler needs one number. Assuming the targets are
    conditionally independent given the size,

    .. math::
        p(n \mid y_1 \ldots y_N) \;\propto\; P(n) \prod_i \frac{p_i(n \mid y_i)}{P(n)}

    which in log space is ``log P(n) + sum_i w_i [log p_i(n|y_i) - log P(n)]``,
    renormalized. That is exactly the algebra ``DeFoGModel._blend_rates`` already applies
    to rate matrices, with the dataset marginal in the role the frozen base plays there,
    and the same ``product`` / ``mean`` modes as ``AdapterComposition``.

    Two things make this better behaved than the rate-matrix blend: the grid is ~30
    categories, so the normalization is exact and needs none of the clamping that blend
    requires; and the resulting entropy is available in closed form, so collapse is
    observable rather than inferred (see :meth:`diagnostics`).

    **On the default mode.** ``product`` is correct only to the extent the branches carry
    independent information about ``n``, and the intuition that two size-correlated
    properties must be largely redundant is wrong here. Measured on 219,568 ZINC train
    molecules at matched 49-bucket resolution, the logP+QED joint recovers 0.236 nats
    against a sum-of-singles of 0.266 and a best-single of 0.169 -- 89% additive. The
    reason is that the two pull size in opposite directions (corr +0.398 and -0.321), so
    they act as near-orthogonal constraints rather than duplicate votes. ``mean`` remains
    available for the residual sub-additivity, and halves the deviation.
    """

    def __init__(self, branches: Sequence[SizeBranch], mode: str = "product",
                 marginal=None, max_divergence: float = 0.05):
        """
        Args:
            marginal: The single ``P(n)`` anchor every log-ratio is taken against.
                Defaults to the first branch's.
            max_divergence: Reject branches whose own marginal sits further than this in
                total-variation distance from the anchor.

        On why the anchor is a parameter and not a consensus: the formula contains ONE
        ``log P(n)``, so the composition picks an anchor -- it does not require the
        branches to have memorised identical copies of one. Demanding exact agreement
        conflates "these were fit on the same data" with "we use one anchor", and fails in
        normal use, since two size models fit from the same dataset still differ in which
        molecules RDKit dropped and how the split fell. What genuinely breaks the
        composition is an anchor that is *materially* wrong for a branch, so the guard is a
        distance with a threshold: TV = 0.05 means at most 5% of the probability mass sits
        somewhere different, which comfortably admits sampling noise and still catches a
        model fit on another dataset or vocabulary.
        """
        assert mode in ("product", "mean"), f"unknown mode {mode!r}"
        assert len(branches) >= 1, "need at least one branch"
        self.branches = list(branches)
        self.mode = mode

        first = self.branches[0].dist
        self._min_size, self._max_size = first.min_size, first.max_size
        for b in self.branches[1:]:
            assert (b.dist.min_size, b.dist.max_size) == (self._min_size, self._max_size), (
                f"branches disagree on the size grid: {self._min_size}..{self._max_size} "
                f"vs {b.dist.min_size}..{b.dist.max_size}. A product of experts needs one "
                f"shared support."
            )
        anchor = first.marginal if marginal is None else \
            torch.as_tensor(marginal, dtype=torch.float32).reshape(-1)
        assert anchor.numel() == self._max_size - self._min_size + 1, \
            "the anchor marginal does not match the branches' size grid"
        anchor = anchor / anchor.sum()
        for i, b in enumerate(self.branches):
            tv = float(0.5 * (b.dist.marginal.to(anchor.device) - anchor).abs().sum())
            if tv > max_divergence:
                raise AssertionError(
                    f"branch {i} ({b.dist.property_name or 'unnamed'}) carries a marginal "
                    f"P(n) that is {tv:.3f} in total variation from the anchor, above the "
                    f"{max_divergence} limit. The anchor is what every log-ratio is taken "
                    f"against, so a branch this far from it was fit on different data and "
                    f"its ratios are not comparable with the others'."
                )
        self._marginal = anchor

    # -- core ----------------------------------------------------------------

    def log_pmf(self, num_samples: Optional[int] = None) -> torch.Tensor:
        """Blended ``log q(n)``, shape ``(bs, n_bins)``."""
        support = self._marginal > 0
        keep = support.unsqueeze(0)
        # Off-support bins are -inf in every term; zero them before differencing so the
        # arithmetic never sees (-inf) - (-inf) = nan, then mask the result back out.
        lu = torch.where(keep, torch.log(self._marginal).unsqueeze(0),
                         torch.zeros(1, self._marginal.numel(), device=self._marginal.device))
        dev = torch.zeros_like(lu)
        for b in self.branches:
            lc = b.dist.log_pmf(b.condition, num_samples=num_samples)
            lc = torch.where(keep, lc, torch.zeros_like(lc))
            dev = dev + float(b.weight) * (lc - lu)
        if self.mode == "mean":
            dev = dev / len(self.branches)
        lq = (lu + dev).masked_fill(~keep, _NEG_INF)
        return torch.log_softmax(lq, dim=-1)

    def sample(self, num_samples, condition=None, device=None, generator=None):
        """``condition`` is ignored: each branch carries its own, as in
        ``AdapterComposition``."""
        with torch.no_grad():
            probs = self.log_pmf(num_samples=num_samples).exp()
        if probs.size(0) == 1 and num_samples > 1:
            probs = probs.expand(num_samples, -1)
        idx = torch.multinomial(probs.cpu(), 1, generator=generator).squeeze(1)
        return self._to((idx + self._min_size).long(), device)

    def log_prob(self, sizes, condition=None):
        sizes = torch.as_tensor(sizes, dtype=torch.long).reshape(-1)
        lp = self.log_pmf(num_samples=sizes.numel())
        if lp.size(0) == 1 and sizes.numel() > 1:
            lp = lp.expand(sizes.numel(), -1)
        bins = sizes.to(lp.device) - self._min_size
        inside = (bins >= 0) & (bins < lp.size(-1))
        out = torch.full((sizes.numel(),), _NEG_INF, device=lp.device, dtype=lp.dtype)
        if inside.any():
            out[inside] = lp[inside].gather(1, bins[inside].unsqueeze(1)).squeeze(1)
        return out.to(sizes.device)

    @property
    def max_size(self) -> int:
        return self._max_size

    # -- diagnostics ---------------------------------------------------------

    def diagnostics(self, num_samples: Optional[int] = None) -> Dict[str, float]:
        """Is the blend collapsing, and do the branches agree?

        * ``entropy`` / ``marginal_entropy`` -- nats. A blend far below the marginal has
          concentrated the size draw, which buys MAE with diversity; report it beside any
          MAE improvement rather than after someone asks.
        * ``agreement`` -- the smallest pairwise Bhattacharyya coefficient
          ``sum_n sqrt(p_i p_j)``, in [0, 1]. Near zero means the branches want disjoint
          size ranges, so the product is supported only on their thin overlap, which is
          where both models are least reliable.
        """
        with torch.no_grad():
            q = self.log_pmf(num_samples=num_samples).exp().mean(0)
            ent = float(-(q * torch.log(q.clamp_min(1e-30))).sum())
            m = self._marginal
            m_ent = float(-(m * torch.log(m.clamp_min(1e-30))).sum())
            per = [b.dist.log_pmf(b.condition, num_samples=num_samples).exp().mean(0)
                   for b in self.branches]
            agreement = 1.0
            for i in range(len(per)):
                for j in range(i + 1, len(per)):
                    agreement = min(agreement, float((per[i] * per[j]).sqrt().sum()))
            return {
                "entropy": ent,
                "marginal_entropy": m_ent,
                "entropy_ratio": ent / m_ent if m_ent > 0 else float("nan"),
                "agreement": agreement,
                "modal_size": int(q.argmax()) + self._min_size,
                "mean_size": float((q * torch.arange(
                    self._min_size, self._max_size + 1, dtype=q.dtype, device=q.device)).sum()),
            }
