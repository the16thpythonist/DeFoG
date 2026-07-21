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
    mu`` (offset-only control variate; = Dr. GRPO, no std-norm); ``"none"`` -> ``r``. With ``groups`` (int
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

    def __init__(self, model, *, subsample_idx: Optional[Sequence[int]] = None,
                 group_ids=None, crn: bool = False, **kwargs):
        super().__init__(model, **kwargs)
        self.subsample_idx = set(subsample_idx) if subsample_idx is not None else None
        # Common random numbers: when `crn` and `group_ids` are given, every member of
        # an advantage group starts from the SAME initial noise + graph size (see
        # `_init_state`), so the group-relative advantage reflects the sampling
        # stochasticity (eta), not the luck of the initial draw. Needs eta>0 (or the
        # inherent multinomial step noise) for any within-group diversity.
        self.group_ids = group_ids
        self.crn = bool(crn)
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

    def _init_state(self, X, E, node_mask, n_nodes):
        """Common random numbers: overwrite every group member's initial state with
        its group representative's (member 0), so all members of an advantage group
        start from the same noised graph AND the same size. No-op unless `crn` and
        `group_ids` are set."""
        if not self.crn or self.group_ids is None:
            return X, E, node_mask, n_nodes
        g = torch.as_tensor(self.group_ids, device=X.device)
        for gid in torch.unique(g):
            idx = (g == gid).nonzero(as_tuple=True)[0]
            r = idx[0]
            X[idx] = X[r].clone()
            E[idx] = E[r].clone()
            node_mask[idx] = node_mask[r].clone()
            n_nodes[idx] = n_nodes[r].clone()
        return X, E, node_mask, n_nodes


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
        advantage_mode / advantage_clip / advantage_eps: variance reduction
            (advantage_mode default "mean" = Dr. GRPO; "grpo" adds std-normalization).
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
        # common random numbers: share the initial noise+size within each advantage
        # group so the group-relative baseline has lower variance (needs groups + eta>0).
        crn: bool = False,
        # eager gradient
        subsample_steps: Optional[int] = 16,
        subsample: str = "stratified",
        minibatch_size: Optional[int] = 16,
        lambda_edge: float = 1.0,
        reduction: str = "mean",
        # advantage. Default "mean" (Dr. GRPO): a mean baseline WITHOUT the per-group
        # std-normalization, which otherwise amplifies medium-variance groups and
        # biases learning toward mid-difficulty targets. "grpo" restores the std-
        # normalized form; "none" uses the raw reward.
        advantage_mode: str = "mean",
        advantage_clip: Optional[float] = 5.0,
        advantage_eps: float = 1e-4,
        # positive-only / RAFT: clamp the advantage to >=0 so the loss NEVER pushes
        # down low-reward endpoints (no unlikelihood term -> no atom-soup collapse).
        # For a binary reward this is exactly reward-ranked filtered fine-tuning; it
        # also makes the gradient fade as the reward saturates. Optional, off by default.
        positive_only: bool = False,
        # KL to reference
        kl_coef: float = 0.0,
        ref_model=None,
        # adaptive KL: if set, nudge kl_coef each step toward this target KL. Optional.
        kl_target: Optional[float] = None,
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
        self.crn = bool(crn)
        self.subsample_steps = subsample_steps
        self.subsample = subsample
        self.minibatch_size = minibatch_size
        self.lambda_edge = lambda_edge
        self.reduction = reduction
        self.advantage_mode = advantage_mode
        self.advantage_clip = advantage_clip
        self.advantage_eps = advantage_eps
        self.positive_only = bool(positive_only)
        self.kl_coef = float(kl_coef)
        self.kl_target = kl_target
        self.grad_clip = grad_clip
        self.device = device if device is not None else model.device
        self.seed = seed

        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.ema = EMA(model, ema_decay) if ema_decay else None

        # Reference for the KL term. Built from hyperparameters + weight copy (NOT
        # copy.deepcopy -- avoids duplicating a live LightningModule). Only when KL
        # is on. The reference is a fixed frozen copy of the pretrained weights.
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
            group_ids=groups, crn=self.crn,
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

        # positive-only / RAFT: drop the negative-advantage (unlikelihood) half of
        # the gradient, so we only ever push UP good endpoints, never push mass into
        # invalid space. No-op when off.
        A = buf.advantage.clamp_min(0.0) if self.positive_only else buf.advantage
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

        # adaptive KL controller: multiplicatively nudge kl_coef toward kl_target.
        if self.kl_target is not None and self.kl_coef > 0:
            err = metrics.get("kl", 0.0) / max(self.kl_target, 1e-8)
            self.kl_coef = float(min(max(self.kl_coef * (1.0 + 0.1 * (err - 1.0)), 1e-4), 1e3))

        r = buf.reward
        metrics.update({
            "reward_mean": float(r.mean()), "reward_std": float(r.std()),
            "reward_min": float(r.min()), "reward_max": float(r.max()),
            "adv_std": float(buf.advantage.std()),
            "pos_frac": float((buf.advantage > 0).float().mean()),
            "kl_coef": self.kl_coef,
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


# ===========================================================================
# Frozen-base ADAPTER RL fine-tuning (composability-safe: only the adapter moves)
# ===========================================================================
def _base_uncond_softmax(base, X_t, E_t, t, node_mask):
    """Frozen-base UNCONDITIONAL clean-graph softmax marginals -- group 0 of the
    rollout's composition (no adapter modulation). No grad (the base is frozen). Also
    returns the shared ``(noisy, extra)`` so the conditional branch reuses one extra-
    feature computation, exactly as ``_denoise_step_composed`` does."""
    y0 = torch.zeros(X_t.size(0), 0, device=X_t.device)
    noisy = {"X_t": X_t, "E_t": E_t, "y_t": y0, "t": t, "node_mask": node_mask}
    extra = base._compute_extra_data(noisy)
    with torch.no_grad():
        pu = base.forward(noisy, extra, node_mask)
    return F.softmax(pu.X, -1), F.softmax(pu.E, -1), noisy, extra


def _compose_logmarginals(base, adapter, noisy, extra, node_mask, cond, puX, puE,
                          weight: float = 1.0, mode: str = "product"):
    """Composed clean-graph log-marginals ``log softmax_C(q)`` for base+adapter, where
    ``q`` is the SAME product-of-experts blend the rollout applies at its terminal
    decode -- reusing the model's own ``_blend_logp`` so the scored policy is bit-
    identical to the behavior policy (GDPO's behavior==scored invariant, for ANY
    ``weight``, not only w=1). Grad flows only through the adapter's conditional
    branch; the frozen uncond marginals ``(puX, puE)`` enter as constants."""
    mod = adapter(cond, t=noisy["t"])
    pc = base.forward(noisy, extra, node_mask, cond_modulation=mod)
    pcX, pcE = F.softmax(pc.X, -1), F.softmax(pc.E, -1)
    w = puX.new_tensor([float(weight)])
    qX = base._blend_logp(torch.stack([puX, pcX]), w, mode)   # (bs, n, dx) unnormalized log q
    qE = base._blend_logp(torch.stack([puE, pcE]), w, mode)   # (bs, n, n, de)
    return F.log_softmax(qX, dim=-1), F.log_softmax(qE, dim=-1)


def _score_logprob(logpX, logpE, X1, E1, node_mask, lambda_edge, reduction):
    """Masking-/symmetry-aware ``log p_theta(G1|G_t)`` from clean-graph log-marginals
    (shared node/edge reduction for the adapter eager term)."""
    lpX = (X1 * logpX).sum(-1)
    lpE = (E1 * logpE).sum(-1)
    emask = _edge_upper_mask(node_mask)
    node_sum = (lpX * node_mask).sum(-1)
    edge_sum = (lpE * emask).sum(dim=(1, 2))
    if reduction == "mean":
        return node_sum / node_mask.sum(-1).clamp_min(1) + lambda_edge * edge_sum / emask.sum(dim=(1, 2)).clamp_min(1)
    return node_sum + lambda_edge * edge_sum


def _kl_from_logmarginals(pX, pE, qX, qE, node_mask, lambda_edge, reduction):
    """Forward KL ``KL(p||q)`` over clean-graph log-marginals, batch-averaged."""
    klX = (pX.exp() * (pX - qX)).sum(-1)
    klE = (pE.exp() * (pE - qE)).sum(-1)
    emask = _edge_upper_mask(node_mask)
    node_kl = (klX * node_mask).sum(-1)
    edge_kl = (klE * emask).sum(dim=(1, 2))
    if reduction == "mean":
        node_kl = node_kl / node_mask.sum(-1).clamp_min(1)
        edge_kl = edge_kl / emask.sum(dim=(1, 2)).clamp_min(1)
    return (node_kl + lambda_edge * edge_kl).mean()


def adapter_eager_logprob(base, adapter, X_t, E_t, cond, t, X1, E1, node_mask,
                          weight: float = 1.0, mode: str = "product",
                          lambda_edge: float = 1.0, reduction: str = "mean") -> torch.Tensor:
    """GDPO eager ``log p_theta(G1|G_t)`` through the COMPOSED base+adapter policy at
    CFG ``weight``/``mode`` (only the adapter has grad). At ``weight=1`` this reduces
    exactly to the plain conditional log-prob; at ``weight!=1`` it tracks the rollout's
    product-of-experts blend so behavior==scored and the gradient stays unbiased."""
    puX, puE, noisy, extra = _base_uncond_softmax(base, X_t, E_t, t, node_mask)
    logpX, logpE = _compose_logmarginals(base, adapter, noisy, extra, node_mask, cond,
                                         puX, puE, weight, mode)
    return _score_logprob(logpX, logpE, X1, E1, node_mask, lambda_edge, reduction)


def adapter_kl_clean(base, adapter, ref_adapter, X_t, E_t, cond, t, node_mask,
                     weight: float = 1.0, mode: str = "product",
                     lambda_edge: float = 1.0, reduction: str = "mean") -> torch.Tensor:
    """KL(composed policy || composed reference) on the clean marginals at the SAME
    CFG ``weight``/``mode`` as the eager term (reward-hacking guard vs the PRE-RL
    adapter). Both branches share the frozen uncond marginals."""
    puX, puE, noisy, extra = _base_uncond_softmax(base, X_t, E_t, t, node_mask)
    pX, pE = _compose_logmarginals(base, adapter, noisy, extra, node_mask, cond, puX, puE, weight, mode)
    with torch.no_grad():
        qX, qE = _compose_logmarginals(base, ref_adapter, noisy, extra, node_mask, cond, puX, puE, weight, mode)
    return _kl_from_logmarginals(pX, pE, qX, qE, node_mask, lambda_edge, reduction)


class AdapterGDPOTrainer(GDPOTrainer):
    """GDPO fine-tuning of a FROZEN-base AdaLN adapter -- only the adapter's params
    move, so the shared unconditional path (hence composability) is preserved.

    Differences from :class:`GDPOTrainer`:
      * Policy = frozen base + trainable adapter. Rollouts apply the adapter's
        modulation at a PER-ROW target (each trajectory carries its own condition)
        via a single-branch ``AdapterComposition`` at CFG ``rollout_weight`` on a
        ``RolloutSampler``. The eager-gradient recompute reproduces the SAME product-
        of-experts blend, so the scored policy matches the behavior policy at ANY
        weight (not only w=1) -- weight/mode are a single shared source of truth.
      * Reward is CONDITIONAL: ``cond_reward(X1, E1, node_mask, cond) -> (K,)`` (match
        to each rollout's own target); GRPO advantage grouped by target.
      * KL reference is a frozen copy of the PRE-RL adapter.

    ``condition_sampler() -> (cond (K, cond_dim) RAW targets, groups (K,))`` is required
    (pass it via the usual GDPOTrainer kwarg).
    """

    def __init__(self, base, adapter, cond_reward, *, ref_adapter=None, kl_coef: float = 0.0,
                 kl_target=None, rollout_weight: float = 1.0, rollout_mode: str = "product",
                 crn: bool = True,
                 lr: float = 1e-5, weight_decay: float = 1e-5, ema_decay=0.999, **gdpo_kw):
        # Bring up GDPO plumbing with model=base but suppress its base-ref/opt/ema
        # (kl_coef=0, ema_decay=None); we build adapter-scoped versions below.
        super().__init__(base, reward_fn=None, kl_coef=0.0, ema_decay=None,
                         lr=lr, weight_decay=weight_decay, **gdpo_kw)
        self.base = base.eval().requires_grad_(False)
        self.adapter = adapter
        self.cond_reward = cond_reward
        self.kl_coef = float(kl_coef)
        self.kl_target = kl_target
        # CFG weight/mode of the single adapter branch: the ONE source of truth for
        # BOTH the rollout composition and the eager-gradient recompute, so behavior
        # and scored policy can never diverge (GDPO requires them identical). weight=1
        # is the plain conditional; weight!=1 is a genuine product-of-experts that the
        # scoring now reproduces exactly. (For N=1, product and mean are identical.)
        self.rollout_weight = float(rollout_weight)
        self.rollout_mode = rollout_mode
        self.crn = bool(crn)   # CRN on by default: grouped-target adapter RL always has groups
        self.opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
        self.ema = EMA(adapter, ema_decay) if ema_decay else None
        self.ref_adapter = None
        if self.kl_coef > 0:
            self.ref_adapter = ref_adapter if ref_adapter is not None else self._frozen_adapter_ref()

    def _frozen_adapter_ref(self):
        ref = type(self.adapter)(**self.adapter._config())
        ref.load_state_dict(self.adapter.state_dict())
        return ref.to(self.device).eval().requires_grad_(False)

    @torch.no_grad()
    def rollout(self) -> RolloutBuffer:
        from .adapter import AdapterComposition, ConditionBranch
        self._iter = getattr(self, "_iter", 0)
        idx = self._choose_subsample()
        cond, groups = self.condition_sampler()
        cond = cond.to(self.device).float()
        comp = AdapterComposition([ConditionBranch(self.adapter, cond, self.rollout_weight)],
                                  base=self.base, mode=self.rollout_mode)
        sampler = RolloutSampler(self.base, subsample_idx=idx, eta=self.eta, omega=self.omega,
                                 sample_steps=self.sample_steps, time_distortion=self.time_distortion,
                                 group_ids=groups, crn=self.crn, guidance_scale=1.0)
        sampler.composition = comp
        sampler.sample(self.rollout_size, num_nodes=self.num_nodes, size_dist=self.size_dist,
                       device=self.device, show_progress=False)
        X1, E1 = sampler.endpoint
        node_mask = sampler.end_node_mask
        states = list(zip(sampler.trace_X, sampler.trace_E, sampler.trace_t))
        Xr, Er, _ = self.base.limit_dist.ignore_virtual_classes(X1.clone(), E1.clone())
        r = self.cond_reward(Xr, Er, node_mask, cond).to(self.device).float().reshape(-1)
        A = group_advantage(r, groups, self.advantage_mode, self.advantage_eps, self.advantage_clip)
        return RolloutBuffer(states, X1, E1, cond, node_mask, r, A)   # cond stored in the y slot

    def update(self, buf: RolloutBuffer) -> dict:
        was_training = self.adapter.training
        self.adapter.eval()
        A = buf.advantage.clamp_min(0.0) if self.positive_only else buf.advantage
        K = A.shape[0]
        n_states = max(1, len(buf.states))
        mb = self.minibatch_size or K
        cond = buf.y
        w, mode = self.rollout_weight, self.rollout_mode
        self.opt.zero_grad()
        pg_loss = kl_val = 0.0
        for (X_t, E_t, t) in buf.states:
            for j in range(0, K, mb):
                sl = slice(j, min(j + mb, K))
                nb = sl.stop - sl.start
                nm = buf.node_mask[sl]
                # Composed policy log-marginals (SAME PoE blend as the rollout). The
                # uncond marginals are computed once here and shared with the KL term.
                puX, puE, noisy, extra = _base_uncond_softmax(self.base, X_t[sl], E_t[sl], t[sl], nm)
                logpX, logpE = _compose_logmarginals(self.base, self.adapter, noisy, extra, nm,
                                                     cond[sl], puX, puE, w, mode)
                lp = _score_logprob(logpX, logpE, buf.X1[sl], buf.E1[sl], nm,
                                    self.lambda_edge, self.reduction)
                loss = -(A[sl] * lp).sum() / (K * n_states)
                if self.kl_coef > 0:
                    with torch.no_grad():
                        rX, rE = _compose_logmarginals(self.base, self.ref_adapter, noisy, extra, nm,
                                                       cond[sl], puX, puE, w, mode)
                    kl = _kl_from_logmarginals(logpX, logpE, rX, rE, nm, self.lambda_edge, self.reduction)
                    loss = loss + (self.kl_coef / n_states) * kl * (nb / K)
                    kl_val += float(kl.detach()) * (nb / K) / n_states
                loss.backward()
                pg_loss += float(loss.detach())
        gnorm = clip_grad_norm_(self.adapter.parameters(), self.grad_clip)
        self.opt.step()
        if self.ema:
            self.ema.update(self.adapter)
        if was_training:
            self.adapter.train()
        return {"loss": pg_loss, "kl": kl_val, "grad_norm": float(gnorm)}

    def save(self, path: str, use_ema: bool = True) -> str:
        """Save the RL'd ADAPTER (loads with AdaLNAdapter.load)."""
        if use_ema and self.ema is not None:
            backup = {k: v.detach().clone() for k, v in self.adapter.state_dict().items()}
            self.ema.copy_to(self.adapter)
            out = self.adapter.save(path)
            self.adapter.load_state_dict(backup)
            return out
        return self.adapter.save(path)
