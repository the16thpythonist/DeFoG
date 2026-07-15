"""
GDPO-style reinforcement-learning fine-tuning for DeFoG (training-time).

Aligns a pretrained :class:`~defog.core.model.DeFoGModel` to an arbitrary, possibly
non-differentiable reward on the *generated* molecule (e.g. an RDKit property, a
connectivity check, or a learned preference model) by GDPO's **eager policy
gradient** (Graph Diffusion Policy Optimization, arXiv:2402.16302).

Why DeFoG fits GDPO almost for free
-----------------------------------
GDPO frames denoising as an MDP (state = noisy graph ``G_t``, action = next graph,
reward paid only at the terminal clean graph ``G1``) and maximizes ``E[r(G1)]``.
Naive REINFORCE pushes ``grad log p_theta(G_{t-1}|G_t)`` per transition, which for
the clean-graph (x1) parameterization expands into a reward-agnostic sum over all
clean graphs -> high variance. GDPO's *eager* estimator replaces the transition
term with the log-prob of the trajectory's own realized endpoint::

    g(theta) = (1/K) sum_k  A_k * sum_{t in T_k}  grad_theta log p_theta(G1_k | G_{t,k})

DeFoG's network already emits the clean-graph marginals ``p_theta(G1|G_t) =
softmax(pred.X), softmax(pred.E)``. So ``log p_theta(G1|G_t)`` is exactly the
masking- and symmetry-aware cross-entropy the training loss already computes, with
the rollout's sampled endpoint as the one-hot target. The whole method reduces to
an **advantage-weighted cross-entropy of each rollout's endpoint** against the
network's clean prediction at subsampled noisy states along that rollout.

The feature is additive and optional: importing this module changes nothing until
you construct a :class:`GDPOTrainer`. It reuses the existing :class:`Sampler` for
rollouts (via a ``_post_loop`` endpoint stash, the same pattern
``InpaintingSampler`` uses) and the existing energy/reward classes in
:mod:`defog.core.guidance`.
"""

from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .data import dense_to_pyg
from .guidance import _edge_upper_mask
from .sampler import Sampler


# ===========================================================================
# Reward
# ===========================================================================
class Reward:
    """Reward contract: ``__call__(X1, E1, node_mask) -> (K,)`` tensor, higher is
    better. Mirrors the energy contract in :mod:`defog.core.guidance` (which is
    *lower*-is-better) so any ``MoleculePropertyEnergy`` / ``MultiPropertyEnergy``
    composes via :func:`reward_from_energy`. This base class is documentation; any
    callable with the same signature works."""

    def __call__(self, X1, E1, node_mask) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


def reward_from_energy(energy_fn, *, transform: str = "neg", beta: float = 1.0,
                       invalid_reward: float = -5.0, validity_bonus: float = 0.0):
    """Wrap a DeFoG energy (lower = better; invalid graphs get a large energy) as a
    bounded reward (higher = better).

    ``transform``: ``"neg"`` -> ``-energy`` ; ``"negexp"`` -> ``exp(-beta*energy)``.
    Undecodable / invalid graphs (energy >= ``0.9 * energy_fn.invalid``) are floored
    to a finite ``invalid_reward`` -- applied *before* any advantage whitening so a
    ``1e3`` invalid energy cannot wreck the batch mean/std. Valid graphs optionally
    get ``+validity_bonus``.
    """
    invalid_energy = float(getattr(energy_fn, "invalid", 1e3))
    thresh = 0.9 * invalid_energy

    def _reward(X1, E1, node_mask):
        e = energy_fn(X1, E1, node_mask).float().reshape(-1)
        invalid = e >= thresh
        if transform == "negexp":
            r = torch.exp(-beta * e)
        else:
            r = -e
        r = r + validity_bonus
        r = torch.where(invalid, torch.full_like(r, float(invalid_reward)), r)
        return r

    return _reward


# ===========================================================================
# Eager policy gradient
# ===========================================================================
def eager_logprob(model, X_t, E_t, y_t, t, X1, E1, node_mask,
                  lambda_edge: float = 1.0, reduction: str = "mean") -> torch.Tensor:
    """Per-graph ``log p_theta(G1 | G_t)`` = masking- and symmetry-aware log-
    likelihood of the endpoint ``(X1, E1)`` under the network's clean-graph
    prediction at the noisy state ``(X_t, E_t, t)``. Returns shape ``(bs,)``.

    ``X1`` / ``E1`` are one-hot in the network's OUTPUT class space (never stripped
    of any virtual/absorbing class). Node CE runs over real nodes; edge CE runs over
    the UPPER TRIANGLE of real-node pairs only (``pred.E`` is already symmetric --
    ``transformer.py`` symmetrizes the edge output -- so the triangle is to count
    each undirected edge once, not for symmetry safety).

    ``reduction``: ``"mean"`` divides node/edge sums by their real counts (size-
    invariant -- the default, so large molecules don't dominate the gradient purely
    by node/edge count on variable-size datasets); ``"sum"`` returns the true joint
    log-likelihood.
    """
    noisy = {"X_t": X_t, "E_t": E_t, "y_t": y_t, "t": t, "node_mask": node_mask}
    pred = model.forward(noisy, model._compute_extra_data(noisy), node_mask)
    logpX = F.log_softmax(pred.X, dim=-1)                    # (bs, n, dx)
    logpE = F.log_softmax(pred.E, dim=-1)                    # (bs, n, n, de)
    lpX = (X1 * logpX).sum(-1)                               # (bs, n)
    lpE = (E1 * logpE).sum(-1)                               # (bs, n, n)
    emask = _edge_upper_mask(node_mask)                      # (bs, n, n) bool
    node_sum = (lpX * node_mask).sum(-1)                     # (bs,)
    edge_sum = (lpE * emask).sum(dim=(1, 2))                 # (bs,)
    if reduction == "mean":
        n_nodes = node_mask.sum(-1).clamp_min(1)
        n_edges = emask.sum(dim=(1, 2)).clamp_min(1)
        return node_sum / n_nodes + lambda_edge * edge_sum / n_edges
    return node_sum + lambda_edge * edge_sum


def kl_clean(policy, ref, X_t, E_t, y_t, t, node_mask, lambda_edge: float = 1.0,
             reduction: str = "mean") -> torch.Tensor:
    """Forward KL ``KL(p_policy || p_ref)`` on the clean-graph marginals at a noisy
    state, averaged over the batch. The reference forward is under ``no_grad``.
    Keeps the fine-tuned policy from drifting far off the pretrained distribution
    (reward-hacking / diversity-collapse guard). Returns a scalar.

    ``reduction`` MUST match the policy-gradient term's reduction so ``kl_coef``
    multiplies a like-scaled, size-invariant quantity: ``"mean"`` divides the per-
    graph node/edge KL by their real counts (O(1)); ``"sum"`` keeps the joint KL
    (O(n), O(n^2)). If it did not match, a per-graph-SUM KL against a per-element-
    MEAN PG term makes ``kl_coef`` scale with n^2 and become untunable across sizes."""
    noisy = {"X_t": X_t, "E_t": E_t, "y_t": y_t, "t": t, "node_mask": node_mask}
    p = policy.forward(noisy, policy._compute_extra_data(noisy), node_mask)
    with torch.no_grad():
        q = ref.forward(noisy, ref._compute_extra_data(noisy), node_mask)
    pX, pE = F.log_softmax(p.X, -1), F.log_softmax(p.E, -1)
    qX, qE = F.log_softmax(q.X, -1), F.log_softmax(q.E, -1)
    klX = (pX.exp() * (pX - qX)).sum(-1)                     # (bs, n)
    klE = (pE.exp() * (pE - qE)).sum(-1)                     # (bs, n, n)
    emask = _edge_upper_mask(node_mask)
    node_kl = (klX * node_mask).sum(-1)                      # (bs,)
    edge_kl = (klE * emask).sum(dim=(1, 2))                  # (bs,)
    if reduction == "mean":
        node_kl = node_kl / node_mask.sum(-1).clamp_min(1)
        edge_kl = edge_kl / emask.sum(dim=(1, 2)).clamp_min(1)
    return (node_kl + lambda_edge * edge_kl).mean()


def group_advantage(r, groups=None, mode: str = "grpo", eps: float = 1e-4,
                    clip: Optional[float] = 5.0) -> torch.Tensor:
    """Turn per-trajectory rewards ``(K,)`` into advantages.

    ``mode``: ``"grpo"`` -> ``(r - mu)/(sigma + eps)`` (removes reward scale AND
    offset, so one ``lr`` transfers across arbitrary rewards); ``"mean"`` -> ``r -
    mu`` (offset-only control variate); ``"none"`` -> ``r``. With ``groups`` (int
    ids, e.g. per-target for multi-condition runs) the statistics are computed
    within each group. Optionally clamped to ``+-clip``.
    """
    r = r.float()

    def _white(rg):
        mu = rg.mean()
        if mode != "grpo":
            return rg - mu
        # population std (unbiased=False) so a SINGLETON group is 0, not NaN; and
        # fall back to zero advantage on a ~constant batch instead of dividing by
        # ~0 (no learnable signal that iteration anyway).
        sd = rg.std(unbiased=False) if rg.numel() > 1 else rg.new_tensor(0.0)
        return (rg - mu) / (sd + eps) if float(sd) > eps else torch.zeros_like(rg)

    if mode == "none":
        A = r.clone()
    elif groups is None:
        A = _white(r)
    else:
        A = torch.zeros_like(r)
        gt = torch.as_tensor(groups, device=r.device)
        for g in gt.unique():
            m = gt == g
            A[m] = _white(r[m])
    if clip is not None:
        A = A.clamp(-clip, clip)
    return A


# ===========================================================================
# EMA (standalone: the repo's EMACallback is Lightning-hook-driven, unusable here)
# ===========================================================================
class EMA:
    """Exponential moving average of model weights, driven manually from the plain
    training loop. Evaluation/checkpointing can sample from the smoothed weights."""

    def __init__(self, model, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)

    @torch.no_grad()
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self):
        return self.shadow


# ===========================================================================
# Rollout
# ===========================================================================
class RolloutSampler(Sampler):
    """A :class:`Sampler` that records the (subsampled) noisy states it visits and
    the terminal clean graph, for the eager-gradient recompute.

    Recording is passive and rides on the base ``Sampler.sample()`` orchestration
    (which owns the ``eval()`` toggle, ``no_grad``, the ``t=0`` nudge, and CFG
    resolution) -- so there is no duplicated loop logic:

    - ``_advance`` stashes the conditioning ``y`` once (it is returned unchanged by
      ``denoise_step`` every step, so it is constant across a rollout).
    - ``_pre_step`` records ``(X_t, E_t, t_norm)`` at the pre-selected subsample
      indices, using the exact distorted time fed to ``denoise_step``.
    - ``_post_loop`` stashes the terminal one-hot ``(X1, E1)`` in the network's
      output class space *before* ``ignore_virtual_classes`` strips it.

    All stashed tensors are detached (``sample()`` runs under ``no_grad`` anyway);
    the gradient is computed later by :func:`eager_logprob` in a fresh pass.
    """

    def __init__(self, model, *, subsample_idx: Optional[Sequence[int]] = None, **kwargs):
        super().__init__(model, **kwargs)
        self.subsample_idx = set(subsample_idx) if subsample_idx is not None else None
        self._step = 0
        self.trace_X: List[torch.Tensor] = []
        self.trace_E: List[torch.Tensor] = []
        self.trace_t: List[torch.Tensor] = []
        self.trace_y: Optional[torch.Tensor] = None
        self.endpoint: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.end_node_mask: Optional[torch.Tensor] = None

    def _desc(self) -> str:
        return "RL rollout"

    def _advance(self, t_int, X, E, y, node_mask, use_cfg):
        if self.trace_y is None:
            self.trace_y = y.detach()
        return super()._advance(t_int, X, E, y, node_mask, use_cfg)

    def _pre_step(self, X, E, t_norm, node_mask):
        if self.subsample_idx is None or self._step in self.subsample_idx:
            self.trace_X.append(X.detach())
            self.trace_E.append(E.detach())
            self.trace_t.append(t_norm.detach())
        self._step += 1
        return X, E

    def _post_loop(self, X, E, node_mask):
        self.endpoint = (X.detach(), E.detach())
        self.end_node_mask = node_mask.detach()
        return X, E


class RolloutBuffer:
    """One iteration of on-policy data: the subsampled noisy states, the shared
    endpoint / conditioning / node_mask, and the per-trajectory reward + advantage."""

    def __init__(self, states, X1, E1, y, node_mask, reward, advantage):
        self.states = states            # list of (X_t, E_t, t_norm), each (K, n, .)
        self.X1 = X1                    # (K, n, dx) one-hot endpoint (output space)
        self.E1 = E1                    # (K, n, n, de)
        self.y = y                      # (K, dy) constant conditioning
        self.node_mask = node_mask      # (K, n)
        self.reward = reward            # (K,)
        self.advantage = advantage      # (K,)

    def __len__(self):
        return len(self.states)


# ===========================================================================
# Trainer
# ===========================================================================
class GDPOTrainer:
    """GDPO eager-policy-gradient fine-tuner for a pretrained ``DeFoGModel``.

    Defaults reproduce faithful single-epoch eager REINFORCE (``kl_coef=0``,
    no PPO surrogate). The fine-tuned weights live in ``model`` (updated in place);
    :meth:`save` writes a plain DeFoG checkpoint that loads with ``DeFoGModel.load``
    and samples with the ordinary ``Sampler`` -- no reward at generation time.

    Args:
        model: the policy ``DeFoGModel``, fine-tuned IN PLACE.
        reward_fn: ``(X1, E1, node_mask) -> (K,)``, higher = better. Use the raw
            energy classes via :func:`reward_from_energy`, or any custom callable.
        rollout_size: K trajectories per iteration.
        sample_steps / eta / omega / time_distortion: rollout (exploration) policy.
            ``eta`` is the CTMC stochasticity = exploration temperature.
        size_dist / num_nodes / condition_sampler: how graph sizes / conditioning
            are drawn. ``condition_sampler() -> (cond (K,cond_dim), groups (K,))``
            for conditional / multi-target runs (else None).
        subsample_steps: how many noisy states per trajectory enter the gradient
            (None -> all). ``subsample``: "stratified" | "uniform" | "late".
        lambda_edge / reduction: passed to :func:`eager_logprob`.
        advantage_mode / advantage_clip / advantage_eps: variance reduction.
        kl_coef / ref_model: KL-to-frozen-reference strength (0 -> no ref built).
        lr / weight_decay / grad_clip / ema_decay: optimization.
    """

    def __init__(
        self,
        model,
        reward_fn: Callable,
        *,
        # rollout / exploration
        rollout_size: int = 64,
        sample_steps: int = 100,
        eta: float = 5.0,
        omega: float = 0.0,
        time_distortion: str = "polydec",
        size_dist=None,
        num_nodes=None,
        condition_sampler: Optional[Callable] = None,
        # eager gradient
        subsample_steps: Optional[int] = 16,
        subsample: str = "stratified",
        minibatch_size: Optional[int] = 16,
        lambda_edge: float = 1.0,
        reduction: str = "mean",
        # advantage
        advantage_mode: str = "grpo",
        advantage_clip: Optional[float] = 5.0,
        advantage_eps: float = 1e-4,
        # KL to reference
        kl_coef: float = 0.0,
        ref_model=None,
        # optim
        lr: float = 1e-5,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        ema_decay: Optional[float] = 0.999,
        device=None,
        seed: int = 0,
    ):
        self.model = model
        self.reward_fn = reward_fn
        self.rollout_size = int(rollout_size)
        self.sample_steps = int(sample_steps)
        self.eta = eta
        self.omega = omega
        self.time_distortion = time_distortion
        self.size_dist = size_dist
        self.num_nodes = num_nodes
        self.condition_sampler = condition_sampler
        self.subsample_steps = subsample_steps
        self.subsample = subsample
        self.minibatch_size = minibatch_size
        self.lambda_edge = lambda_edge
        self.reduction = reduction
        self.advantage_mode = advantage_mode
        self.advantage_clip = advantage_clip
        self.advantage_eps = advantage_eps
        self.kl_coef = float(kl_coef)
        self.grad_clip = grad_clip
        self.device = device if device is not None else model.device
        self.seed = seed

        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.ema = EMA(model, ema_decay) if ema_decay else None

        # Frozen reference for the KL term: rebuild from hyperparameters and copy
        # weights (NOT copy.deepcopy -- avoids duplicating a live LightningModule
        # with a stale trainer/logger). Only allocated when the KL is on.
        self.ref = None
        if self.kl_coef > 0:
            self.ref = ref_model if ref_model is not None else self._frozen_reference()

    def _frozen_reference(self):
        cls = type(self.model)
        ref = cls(**dict(self.model.hparams))
        ref.load_state_dict(self.model.state_dict())
        ref.to(self.device).eval().requires_grad_(False)
        return ref

    def _choose_subsample(self):
        """Indices (shared across the K trajectories) of the noisy states that enter
        the gradient this iteration, so every grad forward is a clean (K, .) batch."""
        S = self.sample_steps
        m = self.subsample_steps
        if m is None or m >= S:
            return None
        g = torch.Generator().manual_seed(self.seed + self._iter)
        if self.subsample == "uniform":
            idx = torch.randperm(S, generator=g)[:m]
        elif self.subsample == "late":
            # bias toward t->1, where the clean prediction most directly carries G1
            w = torch.linspace(0.2, 1.0, S) ** 2
            idx = torch.multinomial(w, m, replacement=False, generator=g)
        else:  # stratified: one index per equal-width bin over range(S)
            edges = torch.linspace(0, S, m + 1).long()
            idx = torch.tensor([
                int(torch.randint(int(edges[i]), max(int(edges[i]) + 1, int(edges[i + 1])),
                                  (1,), generator=g))
                for i in range(m)
            ])
        return sorted(int(i) for i in idx.tolist())

    @torch.no_grad()
    def rollout(self) -> RolloutBuffer:
        self._iter = getattr(self, "_iter", 0)
        idx = self._choose_subsample()
        cond, groups = (None, None)
        if self.condition_sampler is not None:
            cond, groups = self.condition_sampler()

        sampler = RolloutSampler(
            self.model, subsample_idx=idx, eta=self.eta, omega=self.omega,
            sample_steps=self.sample_steps, time_distortion=self.time_distortion,
            # CFG is a sampling heuristic, not a differentiable policy: force it off
            # so the behavior policy matches the plain-conditional gradient recompute
            # (else conditional rollouts are CFG-tilted but the gradient is not).
            guidance_scale=1.0,
        )
        # Sampler.sample owns eval()/no_grad/restore; we discard the pyg output and
        # read the stashed dense traces instead.
        sampler.sample(
            self.rollout_size, num_nodes=self.num_nodes, size_dist=self.size_dist,
            condition=cond, device=self.device, show_progress=False,
        )

        X1, E1 = sampler.endpoint
        node_mask = sampler.end_node_mask
        y = sampler.trace_y
        states = list(zip(sampler.trace_X, sampler.trace_E, sampler.trace_t))

        # Reward on the stripped endpoint (energy/reward classes decode via RDKit and
        # expect the original class space); no-op strip for marginal noise.
        Xr, Er, _ = self.model.limit_dist.ignore_virtual_classes(X1.clone(), E1.clone())
        r = self.reward_fn(Xr, Er, node_mask).to(self.device).float().reshape(-1)
        A = group_advantage(r, groups, self.advantage_mode, self.advantage_eps, self.advantage_clip)
        return RolloutBuffer(states, X1, E1, y, node_mask, r, A)

    def update(self, buf: RolloutBuffer) -> dict:
        model = self.model
        was_training = model.training
        # Score the gradient under eval() -- the SAME (dropout-free) network that
        # generated the rollout (Sampler.sample forces eval) and that is deployed
        # at sampling time. There is no BatchNorm, so eval() needs no train-mode
        # statistics; differentiating a randomly dropped-out subnet would make the
        # policy gradient biased (behavior != scored != deployed policy).
        model.eval()

        A = buf.advantage
        K = A.shape[0]
        n_states = max(1, len(buf.states))
        mb = self.minibatch_size or K
        self.opt.zero_grad()
        pg_loss = 0.0
        kl_val = 0.0
        # One backward per (noise level, trajectory chunk): only a single
        # (mb, .) autograd graph is resident at a time, so grad-forward memory is
        # decoupled from the rollout size K (essential on small GPUs). The chunk
        # contributions sum EXACTLY to the full-batch mean loss -(1/(K.n_states)) *
        # sum_k A_k log p_theta(G1_k|G_{t,k}).
        for (X_t, E_t, t) in buf.states:
            for j in range(0, K, mb):
                sl = slice(j, min(j + mb, K))
                nb = sl.stop - sl.start
                lp = eager_logprob(model, X_t[sl], E_t[sl], buf.y[sl], t[sl],
                                   buf.X1[sl], buf.E1[sl], buf.node_mask[sl],
                                   lambda_edge=self.lambda_edge, reduction=self.reduction)
                loss = -(A[sl] * lp).sum() / (K * n_states)
                if self.kl_coef > 0:
                    kl = kl_clean(model, self.ref, X_t[sl], E_t[sl], buf.y[sl], t[sl],
                                  buf.node_mask[sl], self.lambda_edge, reduction=self.reduction)
                    # kl is a mean over the chunk; weight nb/K so chunk sums = full mean
                    loss = loss + (self.kl_coef / n_states) * kl * (nb / K)
                    kl_val += float(kl.detach()) * (nb / K) / n_states
                loss.backward()
                pg_loss += float(loss.detach())

        gnorm = clip_grad_norm_(model.parameters(), self.grad_clip)
        self.opt.step()
        if self.ema:
            self.ema.update(model)
        if was_training:
            model.train()
        return {"loss": pg_loss, "kl": kl_val, "grad_norm": float(gnorm)}

    def step(self) -> dict:
        self._iter = getattr(self, "_iter", 0)
        buf = self.rollout()
        metrics = self.update(buf)
        r = buf.reward
        metrics.update({
            "reward_mean": float(r.mean()), "reward_std": float(r.std()),
            "reward_min": float(r.min()), "reward_max": float(r.max()),
            "adv_std": float(buf.advantage.std()),
        })
        self._iter += 1
        return metrics

    def fit(self, iterations: int, on_iter: Optional[Callable] = None) -> List[dict]:
        self._iter = getattr(self, "_iter", 0)
        history = []
        for _ in range(iterations):
            m = self.step()
            history.append(m)
            if on_iter is not None:
                on_iter(self._iter - 1, m)
        return history

    def save(self, path: str, use_ema: bool = True) -> str:
        """Save the fine-tuned policy as a plain DeFoG checkpoint. If ``use_ema`` and
        an EMA is tracked, the smoothed weights are written (originals restored)."""
        if use_ema and self.ema is not None:
            backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.ema.copy_to(self.model)
            out = self.model.save(path)
            self.model.load_state_dict(backup)
            return out
        return self.model.save(path)
