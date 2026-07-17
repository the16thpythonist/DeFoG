"""
Exact posterior-based discrete guidance for DeFoG.

Implements the guidance framework of *"Discrete Guidance Matching: Exact Guidance
for Discrete Flow Matching"* (ICLR 2026, arXiv:2509.21912). The key identity
(their Theorem 1) is that the guided clean-graph posterior is just the pretrained
posterior reweighted coordinate-wise by a learned density-ratio expectation
``h_t`` and renormalized::

    q(x1 | x_t)  ∝  h_t(x1, x_t) · p(x1 | x_t)          (Eq. 6)

DeFoG already builds its whole CTMC *from* the predicted clean-graph marginals
``pred_X`` / ``pred_E`` (the softmax inside ``DeFoGModel.denoise_step``). So the
entire method is a pure transform of those two tensors, plugged into the model's
generic ``posterior_transform`` hook -- the base model stays guidance-agnostic,
exactly as it stays constraint-agnostic.

Design notes
------------
* The guidance network ``h`` is itself a :class:`DeFoGModel`. Parameterizing
  ``h = exp(g)`` (``g`` = the network's raw logits) makes the reweight the
  numerically clean ``softmax(g + log p)`` and the training objective the Bregman
  divergence ``exp(g) - r·g`` (with ``F(x)=<x, log x>``) -- no log of a network
  output, no ell-2 loss (which fails for positive density ratios).
* Two flavours of ``h``:
    - **fixed-target** (unconditional, ``cond_dim=0``): one network per target,
      trained by :class:`GuidanceModule`. Faithful to the paper.
    - **amortized** (conditional on the target property, ``cond_dim=1``): a single
      network that steers to *any* target value, trained by
      :class:`AmortizedPropertyGuidanceModule`. A target-conditioned extension
      beyond the paper's fixed-``r`` formulation; it reuses DeFoGModel's own
      conditioning machinery to inject the (normalized) target.
* :class:`ExactGuidance` is the sample-time reweighter driven by
  :class:`~defog.core.sampler.GuidedSampler`; it handles both flavours.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .model import DeFoGModel
from .data import to_dense, dense_to_pyg


# ===========================================================================
# Density ratio  r(x1) = q1(x1) / p1(x1)
# ===========================================================================
class DensityRatio(ABC):
    """A nonnegative per-graph ratio ``r(x1) = q1(x1)/p1(x1)`` over a batch of
    CLEAN dense one-hot graphs, returned as shape ``(bs,)``. Known only up to a
    positive constant (the sample-time softmax and the training ``r_scale`` both
    cancel any global factor). Off-support / invalid graphs should return ~0."""

    @abstractmethod
    def __call__(self, X1: torch.Tensor, E1: torch.Tensor,
                 node_mask: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        ...


class EnergyRatio(DensityRatio):
    """``r = exp(-gamma * E(x1))`` -- energy-guided sampling (paper Sec 4.1)."""

    def __init__(self, energy_fn: Callable, gamma: float = 1.0):
        self.energy_fn, self.gamma = energy_fn, gamma

    def __call__(self, X1, E1, node_mask):
        return torch.exp(-self.gamma * self.energy_fn(X1, E1, node_mask))


class ClassifierRatio(DensityRatio):
    """``r = p(y_target | x1)`` -- classifier / class-conditional guidance (Eq. 7)."""

    def __init__(self, prob_fn: Callable):
        self.prob_fn = prob_fn

    def __call__(self, X1, E1, node_mask):
        return self.prob_fn(X1, E1, node_mask).clamp_min(1e-12)


class RewardRatio(DensityRatio):
    """``r = exp(R(x1)/tau)`` -- RLHF preference alignment (paper Sec 3.5)."""

    def __init__(self, reward_fn: Callable, tau: float = 1.0):
        self.reward_fn, self.tau = reward_fn, tau

    def __call__(self, X1, E1, node_mask):
        return torch.exp(self.reward_fn(X1, E1, node_mask) / self.tau)


class MoleculePropertyEnergy:
    """Energy for molecular property steering: ``E(x1) = (prop(mol) - target)^2``.

    Decodes each dense one-hot graph to an RDKit molecule via a duck-typed
    ``domain`` (anything with ``.decode(pyg_data) -> Optional[Mol]``, e.g.
    :class:`~defog.domains.molecule.MoleculeDomain`) and evaluates ``prop_callback``
    (e.g. ``rdkit.Chem.Crippen.MolLogP``). Molecules that fail to decode get
    ``invalid_energy`` (so ``r ~ 0`` -- they lie off the source support). Depends
    only on ``x1``, so results can be cached per molecule.
    """

    def __init__(self, domain, prop_callback: Callable, target: float,
                 invalid_energy: float = 1e3):
        self.domain, self.prop = domain, prop_callback
        self.target, self.invalid = target, invalid_energy

    def __call__(self, X1, E1, node_mask):
        n = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, n)
        out = X1.new_full((len(datas),), float(self.invalid))
        for i, d in enumerate(datas):
            mol = self.domain.decode(d)
            if mol is not None:
                try:
                    out[i] = (float(self.prop(mol)) - self.target) ** 2
                except Exception:
                    pass
        return out


class MultiPropertyEnergy:
    """Joint energy over several molecular properties, for the FK reward.

    Decodes each dense one-hot graph ONCE and sums per-property normalized squared
    errors:  E(x1) = sum_i weight_i * ((prop_i(mol) - target_i) / scale_i)^2.
    Normalizing by ``scale_i`` (e.g. each property's dataset std) lets properties on
    very different ranges (logP ~ O(1), TPSA ~ O(100)) combine fairly. Invalid /
    undecodable graphs get ``invalid_energy`` (so their FK weight -> 0).

    Decoupled from the guidance networks: it is just a reward, passed to
    :class:`~defog.core.feynman_kac.JointGuidanceSampler` as ``energy_fn``.

    Args:
        domain: object with ``.decode(pyg_data) -> Optional[Mol]``.
        specs: list of ``(callback, target, scale, weight)`` tuples, one per property.
    """

    def __init__(self, domain, specs, invalid_energy: float = 1e3):
        self.domain = domain
        self.specs = [tuple(s) for s in specs]
        self.invalid = invalid_energy

    def __call__(self, X1, E1, node_mask):
        n = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, n)
        out = X1.new_full((len(datas),), float(self.invalid))
        for i, d in enumerate(datas):
            mol = self.domain.decode(d)
            if mol is None:
                continue
            try:
                e = 0.0
                for cb, tgt, scale, w in self.specs:
                    e += float(w) * ((float(cb(mol)) - float(tgt)) / float(scale)) ** 2
                out[i] = e
            except Exception:
                pass
        return out


# ===========================================================================
# Helpers
# ===========================================================================
def _edge_upper_mask(node_mask: torch.Tensor) -> torch.Tensor:
    """Undirected off-diagonal edges over real nodes: each edge once, no padding,
    no self-loops. Shape ``(bs, n, n)`` boolean."""
    m = node_mask[:, :, None] & node_mask[:, None, :]
    return torch.triu(m, diagonal=1)


def bregman_loss(g, X1, E1, r, node_mask, lambda_edge: float = 1.0,
                 g_clamp: float = 20.0):
    """Bregman objective for ``F(x) = <x, log x>`` with ``h = exp(g)`` (paper Eq. 11).

    Per coordinate, at the sample's realized clean class ``z``, the term is
    ``exp(g_z) - r · g_z``; its minimizer over ``g_z`` is ``E[r | x1=z, x_t]`` --
    exactly the conditional expectation ``h_t`` the guidance needs. ``g`` is the
    guidance network's :class:`PlaceHolder` output (raw logits). Nodes are masked
    by ``node_mask``; edges summed over the upper triangle only.
    """
    gX = g.X.gather(-1, X1.argmax(-1, keepdim=True)).squeeze(-1).clamp(max=g_clamp)
    gE = g.E.gather(-1, E1.argmax(-1, keepdim=True)).squeeze(-1).clamp(max=g_clamp)
    emask = _edge_upper_mask(node_mask)
    loss_X = (gX.exp() - r[:, None] * gX)[node_mask].mean()
    loss_E = (gE.exp() - r[:, None, None] * gE)[emask].mean()
    return loss_X + lambda_edge * loss_E


def build_guidance_network(base: DeFoGModel, cond_dim: int = 0,
                           **overrides) -> DeFoGModel:
    """A :class:`DeFoGModel` mirroring ``base``'s categorical dims / features /
    limit distribution, to be trained as the guidance network ``h``.

    ``cond_dim`` is the guidance net's OWN conditioning width -- decoupled from the
    base's conditioning (``0`` = fixed-target, ``1`` = amortized on a scalar target
    property). CFG is disabled on ``h`` (``cond_drop_prob=0``, ``guidance_scale=1``).
    ``overrides`` may shrink it (e.g. ``n_layers=5, hidden_dim=128``) -- it estimates
    a scalar field, not the full posterior, so it can be smaller than the base.
    """
    hp = dict(base.hparams)
    hp["cond_dim"] = cond_dim
    hp["cond_drop_prob"] = 0.0
    hp["guidance_scale"] = 1.0
    hp.update(overrides)
    return DeFoGModel(**hp)


# ===========================================================================
# Sample-time reweighter (Theorem 1)
# ===========================================================================
class ExactGuidance:
    """Exact posterior-based discrete guidance at sampling time.

    Holds a trained guidance network ``h`` whose logits ``g`` are read as ``log h``.
    :meth:`reweight` is the ``posterior_transform`` a
    :class:`~defog.core.sampler.GuidedSampler` runs each step::

        q ∝ h · p = exp(g) · p   ->   softmax(g + log p)   (per class dimension)

    Works for both flavours of ``h``:

    * ``h.cond_dim == 0`` (fixed target): ``h`` is fed a 0-width ``y``.
    * ``h.cond_dim == 1`` (amortized): the current :attr:`target` property value is
      z-scored with ``prop_mean``/``prop_std`` and injected through ``h``'s own
      conditioning path. Use :meth:`set_target` to switch targets between runs.

    ``g`` is memoized within a step (keyed on the current ``X_t``/``E_t``/``t``) so
    composing with CFG costs one ``h`` forward pass, not two.

    Guidance strength
    -----------------
    ``weight`` (``w``) scales the log-density-ratio before the softmax::

        q ∝ h^w · p = softmax(w · g + log p)

    ``w = 1`` is the exact Theorem-1 guidance. ``w > 1`` sharpens the tilt toward
    the target (stronger steering at the cost of diversity/validity) -- the discrete
    analogue of the classifier-guidance scale. It is an inference-time knob: because
    the raw ``g`` is memoized, ``w`` can be swept with no retraining. (``h^w`` is a
    heuristic amplification of the exact guidance, not itself exact.)
    """

    def __init__(self, h_model: DeFoGModel, density_ratio: Optional[DensityRatio] = None,
                 prop_mean: Optional[float] = None, prop_std: Optional[float] = None,
                 target: Optional[float] = None, weight: float = 1.0):
        self.h = h_model.eval()
        self.density_ratio = density_ratio
        # prop_mean / prop_std / target are scalars for a scalar-property (amortized)
        # net and per-dim vectors (len == cond_dim) for a high-dimensional / latent
        # condition (e.g. a fingerprint). Stored raw; normalized on demand in _g.
        self.prop_mean = prop_mean
        self.prop_std = prop_std
        self.target = target
        self.weight = float(weight)
        self._memo = None
        if h_model.cond_dim > 0:
            assert prop_mean is not None and prop_std is not None, (
                "a conditional (amortized) guidance network requires "
                "prop_mean/prop_std to normalize the target."
            )

    def set_target(self, target) -> "ExactGuidance":
        """Set the target for the next guided run -- a scalar (scalar-property net)
        or a length-``cond_dim`` vector (latent/fingerprint net)."""
        self.target = target
        self._memo = None
        return self

    def _normalized_target(self, device) -> torch.Tensor:
        """z-scored target as a ``(cond_dim,)`` tensor. Handles both a scalar
        target/mean/std (broadcast to cond_dim) and per-dim vectors."""
        t = torch.as_tensor(self.target, dtype=torch.float32, device=device).reshape(-1)
        m = torch.as_tensor(self.prop_mean, dtype=torch.float32, device=device).reshape(-1)
        s = torch.as_tensor(self.prop_std, dtype=torch.float32, device=device).reshape(-1)
        c = (t - m) / s
        if c.numel() == 1 and self.h.cond_dim > 1:
            c = c.expand(self.h.cond_dim)
        return c.reshape(self.h.cond_dim)

    def set_weight(self, weight: float) -> "ExactGuidance":
        """Set the guidance strength ``w`` (1.0 = exact; >1 = stronger steering)."""
        self.weight = float(weight)
        return self

    @torch.no_grad()
    def _g(self, noisy_data, node_mask):
        nd = noisy_data
        key = (id(nd["X_t"]), id(nd["E_t"]), float(nd["t"].reshape(-1)[0]))
        if self._memo is not None and self._memo[0] == key:
            return self._memo[1], self._memo[2]

        bs = nd["X_t"].size(0)
        device = nd["X_t"].device
        if self.h.cond_dim == 0:
            y_t = torch.zeros(bs, 0, device=device)
        else:
            assert self.target is not None, (
                "call set_target(value) before guided sampling with an amortized "
                "guidance network."
            )
            c_norm = self._normalized_target(device).unsqueeze(0).expand(bs, -1)
            y_t = self.h._embed_condition(c_norm)

        h_noisy = {"X_t": nd["X_t"], "E_t": nd["E_t"], "t": nd["t"],
                   "node_mask": node_mask, "y_t": y_t}
        g = self.h.forward(h_noisy, self.h._compute_extra_data(h_noisy), node_mask)
        gX = g.X
        gE = 0.5 * (g.E + g.E.transpose(1, 2))  # keep edge logits symmetric
        self._memo = (key, gX, gE)
        return gX, gE

    @torch.no_grad()
    def reweight(self, pred_X, pred_E, noisy_data, node_mask, eps: float = 1e-8):
        """``posterior_transform`` hook: return the guided marginals ``q``."""
        gX, gE = self._g(noisy_data, node_mask)
        w = self.weight
        q_X = F.softmax(w * gX + torch.log(pred_X.clamp_min(eps)), dim=-1)
        q_E = F.softmax(w * gE + torch.log(pred_E.clamp_min(eps)), dim=-1)
        return q_X, q_E

    def save(self, path):
        return self.h.save(path)

    @classmethod
    def load(cls, path, density_ratio=None, prop_mean=None, prop_std=None,
             device="cpu"):
        return cls(DeFoGModel.load(path, device=device), density_ratio,
                   prop_mean=prop_mean, prop_std=prop_std)


class CompositeGuidance:
    """Combine several :class:`ExactGuidance` objects into one posterior_transform
    (product-of-experts stacking).

    Each guidance's log-ratio ``g_i`` (from its own network, at its own target set
    via ``set_target``) is summed (weighted) and the pretrained posterior is
    reweighted once::

        q  ∝  softmax( Σ_i w_i * g_i  +  log p )        # mode="product"
        q  ∝  softmax( mean_i(w_i * g_i)  +  log p )     # mode="mean"

    ``mode="none"`` is a sentinel meaning "do not stack in the proposal" (callers
    should then pass ``proposal_transform=None`` / let the joint energy steer alone).

    Args:
        guidances: list of ExactGuidance (each with its target already set).
        mode: "product" (sum) | "mean" (average) | "none".
        weights: per-guidance strengths; default = each guidance's own ``.weight``.
    """

    def __init__(self, guidances, mode: str = "product", weights=None):
        assert mode in ("product", "mean", "none"), f"unknown mode {mode!r}"
        self.guidances = list(guidances)
        self.mode = mode
        self.weights = list(weights) if weights is not None else [g.weight for g in self.guidances]
        assert len(self.weights) == len(self.guidances)

    @torch.no_grad()
    def reweight(self, pred_X, pred_E, noisy_data, node_mask, eps: float = 1e-8):
        gX_tot = gE_tot = None
        for w, g in zip(self.weights, self.guidances):
            gX, gE = g._g(noisy_data, node_mask)      # raw log-ratio from each expert
            gX_tot = w * gX if gX_tot is None else gX_tot + w * gX
            gE_tot = w * gE if gE_tot is None else gE_tot + w * gE
        if self.mode == "mean":
            k = max(1, len(self.guidances))
            gX_tot, gE_tot = gX_tot / k, gE_tot / k
        q_X = F.softmax(gX_tot + torch.log(pred_X.clamp_min(eps)), dim=-1)
        q_E = F.softmax(gE_tot + torch.log(pred_E.clamp_min(eps)), dim=-1)
        return q_X, q_E

    def set_targets(self, targets):
        """Convenience: set each guidance's target from a list (same order)."""
        for g, t in zip(self.guidances, targets):
            g.set_target(t)
        return self


# ===========================================================================
# Training modules (Bregman objective)
# ===========================================================================
class _GuidanceModuleBase(pl.LightningModule):
    """Shared plumbing: freeze the base, optimize only ``h``, and keep the frozen
    base out of guidance checkpoints."""

    def _freeze_base(self, base: DeFoGModel):
        self.base = base.requires_grad_(False)

    def on_save_checkpoint(self, checkpoint):
        # Don't serialize the frozen base inside every guidance checkpoint.
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items()
            if not k.startswith("base.")
        }

    def configure_optimizers(self):
        return torch.optim.AdamW(self.h.parameters(), lr=self.lr, weight_decay=1e-5)

    @staticmethod
    def _dense(batch):
        d, node_mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        d = d.mask(node_mask)
        return d.X, d.E, node_mask


class GuidanceModule(_GuidanceModuleBase):
    """Train a fixed-target (unconditional) guidance network via the Bregman
    objective, against a FROZEN base. Primary usage::

        pl.Trainer(max_epochs=..., enable_checkpointing=False).fit(
            GuidanceModule(base, density_ratio), source_loader)

    ``density_ratio`` returns ``r(x1)`` of shape ``(bs,)`` (a :class:`DensityRatio`
    or any callable). It can be precomputed and cached per molecule as
    ``batch.r``.
    """

    def __init__(self, base: DeFoGModel, density_ratio, h_model: Optional[DeFoGModel] = None,
                 lr: float = 2e-4, lambda_edge: Optional[float] = None,
                 g_clamp: float = 20.0, r_scale: float = 1.0, **h_overrides):
        super().__init__()
        self._freeze_base(base)
        self.h = h_model if h_model is not None else build_guidance_network(base, cond_dim=0, **h_overrides)
        self.density_ratio = density_ratio
        self.lr, self.g_clamp, self.r_scale = lr, g_clamp, r_scale
        self.lambda_edge = lambda_edge if lambda_edge is not None else base.train_loss.lambda_edge

    def training_step(self, batch, batch_idx):
        self.base.eval()
        X1, E1, node_mask = self._dense(batch)
        bs = X1.size(0)

        r = getattr(batch, "r", None)
        if r is None:
            r = self.density_ratio(X1, E1, node_mask)
        r = (r.to(X1.device).reshape(bs) / self.r_scale).detach()

        y0 = torch.zeros(bs, 0, device=X1.device)
        t = torch.rand(bs, 1, device=X1.device)  # Eq. 11: t ~ U[0, 1]
        with torch.no_grad():
            noisy = self.base._apply_noise(X1, E1, y0, node_mask, t=t)
        g = self.h.forward(noisy, self.h._compute_extra_data(noisy), node_mask)

        loss = bregman_loss(g, X1, E1, r, node_mask, self.lambda_edge, self.g_clamp)
        self.log("guid/loss", loss, prog_bar=True, on_epoch=True, batch_size=bs)
        return loss

    def guidance(self) -> ExactGuidance:
        """A ready-to-sample :class:`ExactGuidance` around the trained ``h``."""
        return ExactGuidance(self.h, self.density_ratio)


class AmortizedPropertyGuidanceModule(_GuidanceModuleBase):
    """Train a single AMORTIZED guidance network that steers toward *any* target
    value of a scalar molecular property.

    ``h`` is a conditional :class:`DeFoGModel` (``cond_dim=1``) whose condition is
    the z-scored target value. Each step draws an independent target ``c`` from the
    empirical property distribution (``prop_values``) and forms the density ratio

        r = exp(-gamma * (prop(x1) - c)^2)

    using the PRECOMPUTED true property of the data molecule ``x1`` (attached to
    each graph as ``batch.<prop_attr>``) -- so no RDKit call happens in the training
    loop. The minimizer of the Bregman loss is ``E[r_c(x1) | x_t]`` as a function of
    the fed target ``c``, i.e. one network valid across the whole target range.

    Correctness rests on ``prop_values`` covering the eval targets (it is the
    empirical distribution, so any percentile target is in-support).
    """

    def __init__(self, base: DeFoGModel, prop_values, prop_mean: float, prop_std: float,
                 gamma: float = 4.0, prop_attr: str = "prop_val", prop_scale: float = 1.0,
                 h_model: Optional[DeFoGModel] = None, lr: float = 2e-4,
                 lambda_edge: Optional[float] = None, g_clamp: float = 20.0,
                 **h_overrides):
        super().__init__()
        self._freeze_base(base)
        self.h = h_model if h_model is not None else build_guidance_network(base, cond_dim=1, **h_overrides)
        self.register_buffer("prop_values", torch.as_tensor(prop_values, dtype=torch.float32))
        self.prop_mean, self.prop_std = float(prop_mean), float(prop_std)
        # energy r = exp(-gamma * ((prop - c)/prop_scale)^2); prop_scale (e.g. the
        # property's std) makes gamma property-agnostic across very different ranges.
        self.gamma, self.prop_attr, self.prop_scale = float(gamma), prop_attr, float(prop_scale)
        self.lr, self.g_clamp = lr, g_clamp
        self.lambda_edge = lambda_edge if lambda_edge is not None else base.train_loss.lambda_edge

    def training_step(self, batch, batch_idx):
        self.base.eval()
        X1, E1, node_mask = self._dense(batch)
        bs = X1.size(0)
        device = X1.device

        prop_x1 = getattr(batch, self.prop_attr).to(device).reshape(bs).float()
        idx = torch.randint(0, self.prop_values.numel(), (bs,), device=device)
        c = self.prop_values[idx]
        r = torch.exp(-self.gamma * ((prop_x1 - c) / self.prop_scale) ** 2).detach()

        t = torch.rand(bs, 1, device=device)  # Eq. 11: t ~ U[0, 1]
        y0 = torch.zeros(bs, 0, device=device)
        with torch.no_grad():
            noisy = self.base._apply_noise(X1, E1, y0, node_mask, t=t)

        c_norm = ((c - self.prop_mean) / self.prop_std).reshape(bs, 1)
        y_h = self.h._embed_condition(c_norm)
        h_noisy = {"X_t": noisy["X_t"], "E_t": noisy["E_t"], "t": noisy["t"],
                   "node_mask": node_mask, "y_t": y_h}
        g = self.h.forward(h_noisy, self.h._compute_extra_data(h_noisy), node_mask)

        loss = bregman_loss(g, X1, E1, r, node_mask, self.lambda_edge, self.g_clamp)
        self.log("guid/loss", loss, prog_bar=True, on_epoch=True, batch_size=bs)
        return loss

    def guidance(self) -> ExactGuidance:
        """A ready-to-sample :class:`ExactGuidance` around the trained conditional
        ``h``; call ``.set_target(value)`` before each guided run."""
        return ExactGuidance(self.h, prop_mean=self.prop_mean, prop_std=self.prop_std)


# ===========================================================================
# Generic high-dimensional / latent conditioning
# ===========================================================================
def tanimoto_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-wise Tanimoto (Jaccard) between two batches of ``{0,1}`` vectors.

    ``a``, ``b`` are ``(bs, d)``; returns ``(bs,)`` in ``[0, 1]``. Computed as
    ``|A∩B| / |A∪B|`` in pure tensor ops (no RDKit), so it is cheap to call in the
    training loop and differentiable-free (used only as a detached reward)."""
    inter = (a * b).sum(-1)
    union = a.sum(-1) + b.sum(-1) - inter
    return inter / union.clamp_min(eps)


class LatentGuidanceModule(_GuidanceModuleBase):
    """Amortized guidance on an ARBITRARY d-dim precomputed condition ``φ(x1)``
    (a latent vector / fingerprint / soft cluster-membership) -- the vector-valued
    generalization of :class:`AmortizedPropertyGuidanceModule`.

    The guidance net ``h`` (``cond_dim = d``) is fed a TARGET condition ``c`` and
    trained, via the same Bregman objective, so that ::

        h_t(x1=z, x_t ; c)  =  E[ r_c(x1) | x1=z, x_t ]

    with a target-conditioned reward ``r_c(x1) = reward_fn(φ(x1), c)`` (default:
    Tanimoto). At sample time ``ExactGuidance.set_target(c)`` steers the frozen base
    toward molecules whose ``φ`` is close to ``c``.

    Positive-biased pairing
    -----------------------
    A randomly paired ``(x1, c)`` almost never matches in high dimensions -- the
    reward collapses to ~0 and gradients vanish (concentration of distances). So the
    target is drawn with a controlled positive fraction ``pos_frac``:

    * with prob ``pos_frac``: ``c`` is the condition of a precomputed near-neighbor
      of ``x1`` (a real, high-similarity match -> genuine up-signal), and
    * otherwise: a random in-batch condition (a negative -> teaches ``h`` to
      *suppress* off-target).

    Data plumbing (attached upstream, per graph)
    --------------------------------------------
    * ``batch.<cond_attr>``: ``(d,)`` own condition ``φ(x1)``.
    * ``batch.<nbr_attr>``: ``(K,)`` LongTensor of near-neighbor row indices into
      ``cond_bank`` (e.g. top-K by Tanimoto).

    ``cond_bank`` is the ``(N, d)`` matrix of ALL conditions, used to look up the
    neighbor targets. It is kept OFF the checkpoint (a plain attribute, moved to the
    batch device lazily) so guidance checkpoints stay small.

    Args:
        base: frozen pretrained :class:`DeFoGModel`.
        cond_dim: condition width ``d`` (e.g. 128 for a 128-bit fingerprint).
        cond_bank: ``(N, d)`` array/tensor of all per-molecule conditions.
        cond_mean, cond_std: ``(d,)`` per-dim normalization stats for the condition
            fed to ``h`` (also persisted for sample time).
        reward_fn: ``(φ, c) -> (bs,)`` similarity in ``[0, 1]`` (default Tanimoto).
        reward_sharpen: exponent ``β`` applied as ``r <- r**β`` -- sharpens the tilt
            when the raw similarity is diffuse (128-bit folded FPs collide a lot).
        pos_frac: fraction of each batch drawn as positives (near-neighbor targets).
    """

    def __init__(self, base: DeFoGModel, cond_dim: int, cond_bank,
                 cond_mean, cond_std, reward_fn: Callable = tanimoto_similarity,
                 reward_sharpen: float = 1.0, pos_frac: float = 0.5,
                 cond_attr: str = "cond", nbr_attr: str = "nbr",
                 h_model: Optional[DeFoGModel] = None, lr: float = 2e-4,
                 lambda_edge: Optional[float] = None, g_clamp: float = 20.0,
                 **h_overrides):
        super().__init__()
        self._freeze_base(base)
        self.h = h_model if h_model is not None else build_guidance_network(
            base, cond_dim=cond_dim, **h_overrides)
        self.register_buffer("cond_mean", torch.as_tensor(cond_mean, dtype=torch.float32).reshape(-1))
        self.register_buffer("cond_std", torch.as_tensor(cond_std, dtype=torch.float32).reshape(-1).clamp_min(1e-6))
        # cond_bank: plain attribute (NOT a buffer/param) so it never enters the
        # checkpoint; moved to the batch device lazily in _bank().
        self._cond_bank = torch.as_tensor(cond_bank, dtype=torch.float32)
        self.reward_fn = reward_fn
        self.reward_sharpen = float(reward_sharpen)
        self.pos_frac = float(pos_frac)
        self.cond_attr, self.nbr_attr = cond_attr, nbr_attr
        self.lr, self.g_clamp = lr, g_clamp
        self.lambda_edge = lambda_edge if lambda_edge is not None else base.train_loss.lambda_edge

    def _bank(self, device) -> torch.Tensor:
        if self._cond_bank.device != device:
            self._cond_bank = self._cond_bank.to(device)
        return self._cond_bank

    def training_step(self, batch, batch_idx):
        self.base.eval()
        X1, E1, node_mask = self._dense(batch)
        bs = X1.size(0)
        device = X1.device

        phi = getattr(batch, self.cond_attr).to(device).view(bs, -1).float()   # (bs, d) own φ(x1)
        nbr = getattr(batch, self.nbr_attr).to(device).view(bs, -1).long()     # (bs, K) neighbor idx
        bank = self._bank(device)

        # positive-biased target construction
        is_pos = torch.rand(bs, device=device) < self.pos_frac
        col = torch.randint(0, nbr.size(1), (bs, 1), device=device)
        c_pos = bank[nbr.gather(1, col).squeeze(1)]            # a random near-neighbor's condition
        c_neg = phi[torch.randperm(bs, device=device)]        # a random in-batch condition
        c = torch.where(is_pos[:, None], c_pos, c_neg)

        r = self.reward_fn(phi, c)
        if self.reward_sharpen != 1.0:
            r = r.clamp(0.0, 1.0) ** self.reward_sharpen
        r = r.detach()

        t = torch.rand(bs, 1, device=device)  # Eq. 11: t ~ U[0, 1]
        y0 = torch.zeros(bs, 0, device=device)
        with torch.no_grad():
            noisy = self.base._apply_noise(X1, E1, y0, node_mask, t=t)

        c_norm = (c - self.cond_mean) / self.cond_std
        y_h = self.h._embed_condition(c_norm)
        h_noisy = {"X_t": noisy["X_t"], "E_t": noisy["E_t"], "t": noisy["t"],
                   "node_mask": node_mask, "y_t": y_h}
        g = self.h.forward(h_noisy, self.h._compute_extra_data(h_noisy), node_mask)

        loss = bregman_loss(g, X1, E1, r, node_mask, self.lambda_edge, self.g_clamp)
        self.log("guid/loss", loss, prog_bar=True, on_epoch=True, batch_size=bs)
        self.log("guid/r_mean", r.mean(), prog_bar=True, on_epoch=True, batch_size=bs)
        self.log("guid/pos_frac", is_pos.float().mean(), on_epoch=True, batch_size=bs)
        return loss

    def guidance(self) -> ExactGuidance:
        """A ready-to-sample :class:`ExactGuidance` around the trained latent ``h``;
        call ``.set_target(vector)`` (length ``cond_dim``) before each guided run."""
        return ExactGuidance(
            self.h,
            prop_mean=self.cond_mean.detach().cpu().numpy(),
            prop_std=self.cond_std.detach().cpu().numpy(),
        )
