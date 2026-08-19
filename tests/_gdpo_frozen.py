"""
FROZEN COPY of the GDPO trainer stack as of commit 4094123, for the refactor parity
gate (docs/dam_design.md section 7).

`tests/test_rl_parity.py` compares these pre-refactor classes against the live ones in
`defog.core.rl` IN THE SAME PROCESS, which removes the machine, the torch build and the
thread count from the comparison. A committed state_dict hash cannot do that: measured,
the hash differs across 1/4/8/12 CPU threads and again across ATEN_CPU_CAPABILITY
settings, and torch.use_deterministic_algorithms(True) is a no-op on this path.

PROVENANCE. RolloutSampler, RolloutBuffer, GDPOTrainer and AdapterGDPOTrainer were
sliced out of rl.py mechanically (lines 220-296, 297-316, 317-589, 672-826 at commit
4094123). Exactly ONE transformation was applied, because this module lives outside the
`defog` package and cannot resolve package-relative imports:

    from .X import ...   ->   from defog.core.X import ...

Nothing else was changed. Verified at generation time by re-applying that rewrite to
the live source and diffing: zero remaining differences.

Only LEAF helpers are imported from rl.py (eager_logprob, kl_clean, group_advantage,
EMA, adapter_*). The refactor moves class structure and leaves those untouched, so
importing them keeps this file small without weakening the gate.

DO NOT EDIT to keep it in sync with rl.py. Divergence is the whole point: once the
refactor lands, this file is the only surviving record of the old behaviour.
"""

from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from defog.core.data import dense_to_pyg
from defog.core.guidance import _edge_upper_mask
from defog.core.sampler import Sampler
from defog.core.rl import (
    EMA,
    eager_logprob,
    kl_clean,
    group_advantage,
    adapter_eager_logprob,
    adapter_kl_clean,
    _base_uncond_softmax,
    _compose_logmarginals,
    _score_logprob,
    _kl_from_logmarginals,
)


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
        from defog.core.adapter import AdapterComposition, ConditionBranch
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
        # `condition` MUST reach the size distribution. A condition-aware
        # SizeDistribution (defog_megan.sizes.ConceptSizeDistribution) recovers its
        # target from `condition.argmin(-1)` and silently falls back to the base's
        # marginal when it is absent -- so omitting it does not raise, it just draws
        # every rollout graph from the wrong distribution. Measured on the AqSolDB
        # concepts: rollouts averaged 22.5 heavy atoms against 18.1 at evaluation,
        # i.e. RL trained on a +4.5-atom surplus that deployment does not have, and
        # that surplus is exactly what produces macrocycles and disconnected
        # fragments. Callers using a size distribution that ignores `condition`
        # (e.g. CategoricalSizeDistribution) are unaffected.
        sampler.sample(self.rollout_size, num_nodes=self.num_nodes, size_dist=self.size_dist,
                       condition=cond, device=self.device, show_progress=False)
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


# ===========================================================================
# Conditional property rewards
# ===========================================================================
# Promoted verbatim from experiments/adapter_rl_finetune__zinc.py, where they were the
# reward used to produce the shipped first-party adapters. They live here because they are
# training-loop code, and because `molsmith adapter refine` needs them from the library
# rather than from an experiment script.
#
# The CONNECTIVITY-FIRST tiering is the load-bearing part and must not be "simplified":
#
#     invalid (-10)  <  disconnected (-4)  <  connected [-clamp, 0]
#
# `disconnect_reward` sits strictly below `-prop_clamp`, so ANY connected molecule outranks
# ANY disconnected one no matter how far off-target it is, while the property term still
# gives a full gradient among connected molecules. Flatten this ordering and the policy
# learns to hit the target with fragments, which score well on most property functions and
# are not molecules.

