"""
Discrete Adjoint Matching (DAM, arXiv:2602.07132) for DeFoG.

DAM extends Adjoint Matching to CTMCs. Its optimal rate is

    u*_t(y, x) = u^base_t(y, x) * exp(-V_t(y) + V_t(x))

so the reward enters as an exponential of a value DIFFERENCE, never as a gradient --
which is why it works on a discrete state space and on non-differentiable rewards.
The correction is multiplicative on the base rate, and the loss is a generalized-KL
Bregman divergence between rates.

This module supplies the piece that makes that implementable against a DeFoG
checkpoint without touching the sampler: DeFoG's rate is an exact LINEAR functional
of the clean-graph head.

    Rbar_i(x_t -> j) = sum_c p_theta(x_1^i = c | x_t) * R_i(x_t -> j | c, t)

The conditional path p_{t|1}(z|x_1) = t*delta(z,x_1) + (1-t)*p_0(z) factorises over
coordinates and a CTMC jump changes exactly one coordinate, so R_i depends on x_1 only
through x_1^i. R_i(.|c, t) is therefore a network-free basis, obtained by evaluating
the existing RateMatrixDesigner with a one-hot prediction at class c.

Two properties of that identity, both measured (see tests/test_dam_rate.py):

* It is EXACT, not approximate -- relative error 0.0 across noise types, rdb modes
  and t including the regimes where the 1e-8 denominator floor and the >1e5 zeroing
  in _stabilize fire. The reason is structural: the head enters
  compute_rate_matrices only through sample_from_probs, so every downstream
  nonlinearity is applied per class by construction, and the expectation over that
  single draw IS the contraction.
* It is therefore also LOWER VARIANCE than the sampler, which estimates the same
  expectation with one draw.

See docs/dam_design.md sections 3 and 4.
"""

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .data import PlaceHolder
from .noise import sample_from_probs
from .renoise import draw_times, renoise_states
from .rl import (
    AdapterGDPOTrainer,
    _base_uncond_softmax,
    _compose_logmarginals,
    _score_logprob,
)

__all__ = [
    "assert_valid_base_rate",
    "sample_jump",
    "gather_rate",
    "rate_basis",
    "marginal_rate",
    "estimate_neg_value",
    "discrete_adjoint",
    "gkl",
    "AdapterDAMTrainer",
    "REVERSIBLE_RDB",
]

#: detailed-balance designs whose R^DB is reversible w.r.t. p_t, hence admissible as
#: part of a base rate. Measured violation exactly 0.0 for these; 1.0x the full flow
#: scale for rdb="marginal", whose mask 1[limit(j) > limit(x_t)] is one-directional.
REVERSIBLE_RDB = ("general", "column", "entry")


def assert_valid_base_rate(designer, *, check_omega: bool = True) -> None:
    """Refuse rate configurations that break DAM's preconditions.

    DAM needs ``u^base`` to actually generate ``p^base``. Two of DeFoG's sampling
    knobs quietly break that, and both fail in the same way -- a one-directional flow
    that is not reversible w.r.t. ``p_t``, so it does not preserve the marginals:

    * ``omega > 0``. R^TG is an unnormalised additive push toward the sampled clean
      state with no reverse flow. Measured detailed-balance violation equals the FULL
      flow scale. With omega > 0 the rollout endpoints are not draws from p^theta_1
      and the (G1, G_t) joint DAM reconstructs is wrong.
    * ``rdb='marginal'``. Its mask ``1[limit(j) > limit(x_t)]`` is strictly
      one-directional: whenever it fires for x->y it is zero for y->x. Same failure
      mode, same measured violation. ``rdb`` is a free string on
      ``DeFoGModel.__init__`` persisted into every checkpoint, so a foreign
      checkpoint can select it without anyone noticing.

    A third configuration breaks the rate BASIS rather than the base rate:

    * ``rdb='column'`` with ``rdb_crit='p_x1_g_xt'`` reads ``X_1_pred.argmax(-1)``,
      the one place the head enters ``compute_rate_matrices`` without going through
      ``sample_from_probs``. A one-hot probe moves that argmax to class c, so the
      basis no longer describes the sampler (measured Monte-Carlo mean z = 86.5
      against 4000 draws, versus 0.054 for every other criterion).

    This is a DENYLIST on the criterion, not an allowlist: ``rdb_crit='dummy'`` is
    what ``configs/sample/sample_default.yaml`` sets, falls through to the branch
    using ``X_1_sampled``, and measures 0.0 error -- an allowlist would reject the
    repo's own default.
    """
    if check_omega and getattr(designer, "omega", 0.0) != 0:
        raise ValueError(
            f"DAM requires omega == 0, got {designer.omega}. R^TG is not "
            "marginal-preserving (measured detailed-balance violation = the full flow "
            "scale), so with omega > 0 the rate is not a base rate that generates "
            "p^base and the rollout endpoints are not draws from p^theta_1."
        )
    if designer.rdb not in REVERSIBLE_RDB:
        raise ValueError(
            f"DAM requires rdb in {REVERSIBLE_RDB}, got {designer.rdb!r}. "
            "rdb='marginal' uses a one-directional mask 1[limit(j) > limit(x_t)] "
            "whose R^DB is not reversible w.r.t. p_t -- the same failure mode that "
            "forbids omega > 0."
        )
    if designer.rdb == "column" and designer.rdb_crit == "p_x1_g_xt":
        raise ValueError(
            "DAM cannot use rdb='column' with rdb_crit='p_x1_g_xt': that path reads "
            "X_1_pred.argmax(-1) directly instead of going through sample_from_probs, "
            "so the one-hot rate basis does not describe the sampler and the linear "
            "identity behind marginal_rate() no longer holds."
        )


@torch.no_grad()
def rate_basis(
    model,
    X_t: torch.Tensor,
    E_t: torch.Tensor,
    t: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    eta: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The network-free rate basis ``R_i(x_t -> j | c, t)``.

    Returns ``(BX, BE)`` of shape ``(dx, bs, n, dx)`` and ``(de, bs, n, n, de)``,
    where ``B[c]`` is the rate matrix the sampler would produce if the clean-graph
    prediction were one-hot at class ``c`` everywhere.

    Reuses ``model.rate_matrix_designer`` rather than constructing one, so the basis
    is built from exactly the configuration the sampler uses. ``eta`` is set the same
    way ``denoise_step`` sets it (a mutable attribute, restored afterwards);
    ``omega`` is forced to 0 and checked, per :func:`assert_valid_base_rate`.

    Cost is ``dx + de`` calls -- 13 on the real ZINC base -- measured at 32.9 ms
    against 488.7 ms for one network forward at bs=128, n=38.
    """
    designer = model.rate_matrix_designer
    # Validate BEFORE mutating, so a refusal cannot leave the designer disturbed.
    # omega is exempt here because rate_basis sets it itself: the run-level omega
    # check belongs in the trainer's __init__, where the run is configured.
    assert_valid_base_rate(designer, check_omega=False)
    eta_before, omega_before = designer.eta, designer.omega
    designer.eta, designer.omega = float(eta), 0.0
    try:
        dx = model.limit_dist.num_node_classes
        de = model.limit_dist.num_edge_classes
        bs, n = node_mask.shape
        dev = X_t.device

        BX, BE = [], []
        for c in range(dx):
            pX = F.one_hot(torch.full((bs, n), c, device=dev), dx).float()
            pE = F.one_hot(torch.zeros(bs, n, n, dtype=torch.long, device=dev), de).float()
            BX.append(model.rate_matrix_designer.compute_rate_matrices(
                t, node_mask, X_t, E_t, pX, pE)[0])
        for c in range(de):
            pX = F.one_hot(torch.zeros(bs, n, dtype=torch.long, device=dev), dx).float()
            pE = F.one_hot(torch.full((bs, n, n), c, device=dev), de).float()
            BE.append(model.rate_matrix_designer.compute_rate_matrices(
                t, node_mask, X_t, E_t, pX, pE)[1])
    finally:
        designer.eta, designer.omega = eta_before, omega_before

    BX = torch.stack(BX)                                   # (dx, bs, n, dx)
    BE = torch.stack(BE)                                   # (de, bs, n, n, de)

    # Padded entries are garbage by construction: sample_from_probs fills masked
    # nodes and the edge diagonal with a uniform distribution, so their sampled class
    # -- hence their basis row -- is random per call. Zero them so the basis is
    # deterministic everywhere a caller might look.
    nm = node_mask[None, :, :, None]
    em = (node_mask[:, :, None] & node_mask[:, None, :])[None, :, :, :, None]
    return BX * nm, BE * em


def marginal_rate(
    pX: torch.Tensor,
    pE: torch.Tensor,
    BX: torch.Tensor,
    BE: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Contract a clean-graph head against the rate basis: the exact marginal rate.

    ``pX`` ``(bs, n, dx)`` and ``pE`` ``(bs, n, n, de)`` are the network's clean-graph
    probabilities (softmax outputs, NOT logits). Gradient flows through them, which is
    the whole point -- ``RateMatrixDesigner.compute_rate_matrices`` samples the clean
    graph internally and therefore carries no gradient to the head at all.
    """
    return (
        torch.einsum("cbnj,bnc->bnj", BX, pX),
        torch.einsum("cbnmj,bnmc->bnmj", BE, pE),
    )


# ===========================================================================
# The discrete adjoint (DAM Prop 2.4 / Eq. 13) and the matching loss (Eq. 14)
# ===========================================================================
# These take PLAIN TENSORS -- no model, no trainer -- so the tabular gate in
# tests/test_dam_estimator.py exercises the shipped estimator rather than a
# re-implementation of it. Everything is in log space: the importance ratio is a
# product over ~740 coordinates at n=38, so computing it directly overflows.


def estimate_neg_value(log_ratio: torch.Tensor, g: torch.Tensor,
                       dim: int = -1) -> torch.Tensor:
    """Estimate ``-V_t(x) = log E_{p^base_{1|t}(.|x)}[e^{-g}]`` from model samples.

    ``log_ratio[..., k] = log p^base_{1|t}(X1_k|x) - log p^theta_{1|t}(X1_k|x)`` and
    ``g[..., k] = g(X1_k)``, with ``X1_k ~ p^theta_{1|t}(.|x)``. Importance sampling
    then gives ``E_{p^base}[e^{-g}] = E_{p^theta}[(p^base/p^theta) e^{-g}]``, i.e. the
    mean of ``w_k = exp(log_ratio_k - g_k)``.

    THE ``e^{-g}`` IS PART OF THE WEIGHT. Dropping it -- using the bare density ratio
    -- was measured 363x too large on a tabular problem with this repo's reward
    tiering, and 517x with ``p^theta = p^base``, where the two forms coincide and the
    error is invisible. Normalising the weights to sum to one rather than to mean one
    is off by exactly ``1/K``.
    """
    K = log_ratio.shape[dim]
    return torch.logsumexp(log_ratio - g, dim=dim) - math.log(K)


def discrete_adjoint(
    log_ratio_Z: torch.Tensor,
    g_Z: torch.Tensor,
    log_ratio_X1: torch.Tensor,
    g_X1: torch.Tensor,
    *,
    clamp: float = 10.0,
) -> Tuple[torch.Tensor, float]:
    """``log a_hat(y, x)`` -- DAM Eq. (13), in log space.

    Two factors, as in the paper:

    * a single sample ``Z ~ p^theta_{1|t}(.|y)`` estimating
      ``E_{p^base(.|y)}[e^{-g}]`` by plain importance sampling;
    * ``K`` samples ``X1_k ~ p^theta_{1|t}(.|x)`` estimating ``e^{V_t(x)}`` by
      self-normalised importance sampling -- the RECIPROCAL of
      :func:`estimate_neg_value`.

    Returns ``(log_a, clamp_fraction)``.

    ``clamp`` is a saturating envelope, not a health metric. The true adjoint genuinely
    reaches ``e^{10}`` at low ``t``, so a tighter bound would corrupt the target; and
    the fraction is 0 by construction whenever ``lambda * reward_span <= clamp``, which
    is the case at lambda = 1 for ``PropertyMatchReward`` (span [-10, 0]). The metric
    that actually detects mis-tempering is ``E[a_hat]`` on a ``y = x`` control, where
    the true adjoint is exactly 1.0; see docs/dam_design.md section 4.1.
    """
    log_a = (log_ratio_Z - g_Z) - estimate_neg_value(log_ratio_X1, g_X1)
    if clamp is None:
        return log_a, 0.0
    frac = float((log_a.abs() > clamp).float().mean())
    return log_a.clamp(-clamp, clamp), frac


def gkl(u: torch.Tensor, w: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Generalized-KL Bregman divergence ``u - w + w log(w/u)`` -- DAM Eq. (14).

    Elementwise; the caller reduces. Its minimiser in ``u`` is ``u = w``, and unlike an
    l2 norm it respects the nonnegativity that rates must satisfy.

    Because the minimiser matches ``E[w]`` rather than a median or a mode, a
    heavy-tailed adjoint biases the fitted rate by its MEAN -- which is why the ``y = x``
    control above is worth logging every run.
    """
    u = u.clamp_min(eps)
    w = w.clamp_min(eps)
    return u - w + w * (w.log() - u.log())


# ===========================================================================
# Jump sampling (DAM Alg. 1 line 7 / Eq. 11)
# ===========================================================================
def sample_jump(RX: torch.Tensor, RE: torch.Tensor, X_t: torch.Tensor,
                E_t: torch.Tensor, node_mask: torch.Tensor):
    """Sample one jump target ``Y ~ p^u_t(.|x)`` per graph, and return its log-prob.

    A CTMC jump changes exactly one coordinate, so the reachable ``y`` are enumerated
    by the rate tensors themselves: ``(node i -> class j)`` and ``(edge (i,k) -> class
    j)``. Conditioned on a jump occurring, ``y`` is drawn proportional to its rate.

    DAM samples one ``y`` and debiases the objective by ``1 / p^u_t(y|x)`` (Eq. 11)
    rather than summing Eq. (14) over all ``y``. We keep that even though DeFoG's
    reachable set IS enumerable -- roughly 3.2k entries at n=38 -- because the adjoint
    needs a network forward AT ``y``, so enumerating would cost thousands of forwards
    per state.

    Returns ``(logp, pick, X_Y, E_Y)``: the log-probability of the drawn jump, a flat
    index into ``cat([RX, RE])`` so the rate at that jump can be gathered
    differentiably, and the state the jump leads to.
    """
    bs, n, dx = RX.shape
    de = RE.shape[-1]
    dev = RX.device

    # mask: real coordinates only, off-diagonal (a "jump" to the current class is not
    # a jump), upper triangle for edges so each undirected edge is one coordinate.
    cur_x = X_t.argmax(-1, keepdim=True)
    wX = RX.clone()
    wX.scatter_(-1, cur_x, 0.0)
    wX = wX * node_mask[..., None]

    iu = torch.triu(torch.ones(n, n, device=dev, dtype=torch.bool), diagonal=1)
    em = (node_mask[:, :, None] & node_mask[:, None, :]) & iu[None]
    cur_e = E_t.argmax(-1, keepdim=True)
    wE = RE.clone()
    wE.scatter_(-1, cur_e, 0.0)
    wE = wE * em[..., None]

    flat = torch.cat([wX.reshape(bs, -1), wE.reshape(bs, -1)], dim=1).clamp_min(0)
    total = flat.sum(1, keepdim=True)
    probs = torch.where(total > 0, flat / total.clamp_min(1e-30),
                        torch.full_like(flat, 1.0 / flat.shape[1]))
    pick = torch.multinomial(probs, 1).squeeze(1)
    logp = probs.gather(1, pick[:, None]).squeeze(1).clamp_min(1e-30).log()

    X_Y, E_Y = X_t.clone(), E_t.clone()
    n_node = n * dx
    is_node = pick < n_node
    for b in range(bs):
        p = int(pick[b])
        if bool(is_node[b]):
            i, j = divmod(p, dx)
            X_Y[b, i] = F.one_hot(torch.tensor(j, device=dev), dx).float()
        else:
            q = p - n_node
            j = q % de
            ik = q // de
            i, k = divmod(ik, n)
            oh = F.one_hot(torch.tensor(j, device=dev), de).float()
            E_Y[b, i, k] = oh
            E_Y[b, k, i] = oh
    return logp, pick, X_Y, E_Y


def gather_rate(RX: torch.Tensor, RE: torch.Tensor, pick: torch.Tensor) -> torch.Tensor:
    """The rate at the jump ``sample_jump`` picked -- ``u(y, x)`` for that one ``y``.

    Differentiable in ``RX`` / ``RE``, which is the point: this is the quantity the
    gKL loss fits.
    """
    bs = RX.shape[0]
    flat = torch.cat([RX.reshape(bs, -1), RE.reshape(bs, -1)], dim=1)
    return flat.gather(1, pick[:, None]).squeeze(1)


# ===========================================================================
# Trainer
# ===========================================================================
class AdapterDAMTrainer(AdapterGDPOTrainer):
    """Discrete Adjoint Matching fine-tuning of a frozen-base AdaLN adapter.

    Subclasses :class:`AdapterGDPOTrainer` and overrides only ``__init__`` and
    ``update``. The rollout is IDENTICAL across all three arms -- and it is the one
    method in ``rl.py`` carrying a comment about a bug that already cost a run
    (omitting ``condition=cond`` silently draws every rollout graph from the wrong
    size distribution), so it is inherited rather than copied.

    Differences from the GDPO arm:

    * **The target is a distribution, not a direction.** GDPO maximises reward with an
      optional KL guard. DAM fits the rate to ``u^base * a_hat``, whose fixed point is
      ``p^base e^{-g}``. There is no ``kl_coef``: DAM's problem fixes the KL weight at
      1 and folds the temperature into ``g = -lambda * reward``, so ``dam_lambda`` IS
      the inverse temperature and the anchor is structural.
    * **p^base is the PRE-RL COMPOSED POLICY** -- frozen base plus frozen pre-RL
      adapter -- not the unconditional base. ``_base_uncond_softmax`` returns group 0
      of the product-of-experts blend; anchoring to that would target
      ``p_uncond e^{-g}`` and actively pull the adapter's learned conditioning out,
      which is what the GDPO arm's KL term exists to prevent. The frozen reference is
      therefore built UNCONDITIONALLY here, unlike GDPO where it appears only when
      ``kl_coef > 0``.
    * **No advantages.** ``group_advantage``, ``advantage_clip`` and ``positive_only``
      are GDPO/RAM machinery; DAM weights by ``e^{-g}`` with importance correction.
      The rollout still computes them, and they are still reported, but the loss does
      not read them.
    * **Scored at re-noised states**, so ``record_trace`` is off.

    Costs about 2.3x GDPO per scored state: one gradient forward at ``x``, a frozen
    reference forward at ``x``, and both again at the sampled jump target ``y``. The
    ``K + 1`` reward evaluations are ~40 ms per iteration and are not the bottleneck.
    """

    record_trace = False

    def __init__(self, base, adapter, cond_reward, *, ref_adapter=None,
                 dam_k: int = 12, dam_lambda: float = 0.3,
                 adjoint_clamp: float = 10.0, renoise_draws: int = 16,
                 t_sampler: str = "match", debias: str = "snis",
                 lr: float = 1e-5, weight_decay: float = 1e-5, ema_decay=0.999,
                 **gdpo_kw):
        super().__init__(base, adapter, cond_reward, ref_adapter=None, kl_coef=0.0,
                         lr=lr, weight_decay=weight_decay, ema_decay=ema_decay,
                         **gdpo_kw)
        if self.omega != 0:
            raise ValueError(
                f"DAM requires omega == 0, got {self.omega}. R^TG is not "
                "marginal-preserving, so with omega > 0 the rollout endpoints are not "
                "draws from p^theta_1 and u^base does not generate p^base."
            )
        assert_valid_base_rate(self.base.rate_matrix_designer, check_omega=False)
        if debias not in ("snis", "raw"):
            raise ValueError(f"debias must be 'snis' or 'raw', got {debias!r}")

        self.dam_k = int(dam_k)
        self.dam_lambda = float(dam_lambda)
        self.adjoint_clamp = adjoint_clamp
        self.renoise_draws = int(renoise_draws)
        self.t_sampler = t_sampler
        self.debias = debias
        # `_choose_subsample()` counts from self.subsample_steps, and in "match" mode
        # that is what decides how many noise levels we get -- so renoise_draws would
        # be silently ignored there, making the RENOISE_DRAWS sweep in the experiment
        # plan a no-op in the DEFAULT configuration. Point them at the same knob.
        # subsample_steps has no other effect in this arm: it only feeds
        # RolloutSampler's subsample_idx, which record_trace=False ignores.
        self.subsample_steps = self.renoise_draws
        # In "match" mode the levels are distinct grid indices, so the effective
        # count is min(renoise_draws, sample_steps). That binds only on toy fixtures;
        # production runs at sample_steps=250.

        self.ref_adapter = (ref_adapter if ref_adapter is not None
                            else self._frozen_adapter_ref())

    # ------------------------------------------------------------- helpers
    def _composed(self, adapter, noisy, extra, nm, cond, puX, puE):
        return _compose_logmarginals(self.base, adapter, noisy, extra, nm, cond,
                                     puX, puE, self.rollout_weight, self.rollout_mode)

    def _terminal_loss(self, X1, E1, nm, cond):
        """``g = -lambda * reward``, evaluated on the STRIPPED class space the reward
        classes expect (a no-op for marginal noise)."""
        Xr, Er, _ = self.base.limit_dist.ignore_virtual_classes(X1.clone(), E1.clone())
        r = self.cond_reward(Xr, Er, nm, cond).to(self.device).float().reshape(-1)
        return -self.dam_lambda * r

    def _draw_clean(self, logpX, logpE, nm):
        """One clean graph per row from the factorised head, one-hot and masked."""
        s = sample_from_probs(logpX.exp(), logpE.exp(), nm)
        X1 = F.one_hot(s.X, logpX.shape[-1]).float()
        E1 = F.one_hot(s.E, logpE.shape[-1]).float()
        ph = PlaceHolder(X=X1, E=E1, y=torch.zeros(X1.shape[0], 0, device=X1.device))
        ph = ph.mask(nm)
        return ph.X, ph.E

    def _log_ratio(self, refX, refE, polX, polE, X1, E1, nm):
        """``log p^base_{1|t}(G|.) - log p^theta_{1|t}(G|.)`` under the two factorised
        heads. reduction='sum' -- this is a genuine joint log-likelihood, not the
        size-normalised score the eager gradient uses."""
        a = _score_logprob(refX, refE, X1, E1, nm, 1.0, "sum")
        b = _score_logprob(polX, polE, X1, E1, nm, 1.0, "sum")
        return a - b

    # ------------------------------------------------------------- update
    def update(self, buf) -> dict:
        adapter = self.adapter
        was_training = adapter.training
        adapter.eval()

        K = buf.X1.shape[0]
        cond, nm = buf.y, buf.node_mask
        y0 = torch.zeros(K, 0, device=self.device)

        idx = None
        if self.t_sampler == "match":
            # `_choose_subsample()` returns None to mean "every step", which for the
            # match schedule is the full grid rather than "no levels".
            idx = self._choose_subsample()
            if idx is None:
                idx = list(range(self.sample_steps))
        times = draw_times(self.base, K, self.device, mode=self.t_sampler,
                           n_draws=self.renoise_draws, step_indices=idx,
                           sample_steps=self.sample_steps,
                           time_distortion=self.time_distortion)
        states = renoise_states(self.base, buf.X1, buf.E1, y0, nm, times)

        self.opt.zero_grad()
        n_states = max(1, len(states))
        tot, log_a_sum, clamp_sum = 0.0, 0.0, 0.0
        resid_sum = {"resid_ratio": 0.0, "resid_nodes": 0.0, "resid_edges": 0.0}
        resid_n = {k: 0 for k in resid_sum}
        for (X_t, E_t, t) in states:
            loss, d = self._state_loss(X_t, E_t, t, nm, cond)
            (loss / n_states).backward()
            tot += float(loss.detach()) / n_states
            log_a_sum += d["log_a"] / n_states
            clamp_sum += d["clamp_frac"] / n_states
            for k in resid_sum:
                if d[k] == d[k]:                       # skip NaN (no jump of that type)
                    resid_sum[k] += d[k]; resid_n[k] += 1

        gnorm = clip_grad_norm_(adapter.parameters(), self.grad_clip)
        self.opt.step()
        if self.ema:
            self.ema.update(adapter)
        if was_training:
            adapter.train()
        out = {"loss": tot, "kl": 0.0, "grad_norm": float(gnorm),
               "log_adjoint": log_a_sum, "adjoint_clamp_frac": clamp_sum}
        out["resid_gkl_ratio"] = resid_sum["resid_ratio"] / max(resid_n["resid_ratio"], 1)
        out["resid_gkl_nodes"] = resid_sum["resid_nodes"] / max(resid_n["resid_nodes"], 1)
        out["resid_gkl_edges"] = resid_sum["resid_edges"] / max(resid_n["resid_edges"], 1)
        return out

    def _state_loss(self, X_t, E_t, t, nm, cond):
        base = self.base

        # --- at x: one shared unconditional forward, two conditional ones ---------
        puX, puE, noisy, extra = _base_uncond_softmax(base, X_t, E_t, t, nm)
        polX, polE = self._composed(self.adapter, noisy, extra, nm, cond, puX, puE)
        with torch.no_grad():
            refX, refE = self._composed(self.ref_adapter, noisy, extra, nm, cond, puX, puE)

        BX, BE = rate_basis(base, X_t, E_t, t, nm, eta=self.eta)
        uX, uE = marginal_rate(polX.exp(), polE.exp(), BX, BE)
        with torch.no_grad():
            bX, bE = marginal_rate(refX.exp(), refE.exp(), BX, BE)

        with torch.no_grad():
            # --- K clean draws at x: the self-normalised second factor -------------
            lrs, gs = [], []
            for _ in range(self.dam_k):
                X1, E1 = self._draw_clean(polX, polE, nm)
                lrs.append(self._log_ratio(refX, refE, polX, polE, X1, E1, nm))
                gs.append(self._terminal_loss(X1, E1, nm, cond))
            lr_X1 = torch.stack(lrs, -1)
            g_X1 = torch.stack(gs, -1)

            # --- one jump target y, then Z ~ p^theta_{1|t}(.|y) --------------------
            logp_y, pick, X_Y, E_Y = sample_jump(uX.detach(), uE.detach(), X_t, E_t, nm)
            pY, pYE, noisyY, extraY = _base_uncond_softmax(base, X_Y, E_Y, t, nm)
            polYX, polYE = self._composed(self.adapter, noisyY, extraY, nm, cond, pY, pYE)
            refYX, refYE = self._composed(self.ref_adapter, noisyY, extraY, nm, cond, pY, pYE)
            Z_X, Z_E = self._draw_clean(polYX, polYE, nm)
            lr_Z = self._log_ratio(refYX, refYE, polYX, polYE, Z_X, Z_E, nm)
            g_Z = self._terminal_loss(Z_X, Z_E, nm, cond)

            log_a, clamp_frac = discrete_adjoint(lr_Z, g_Z, lr_X1, g_X1,
                                                 clamp=self.adjoint_clamp)
            u_base_Y = gather_rate(bX, bE, pick)
            target = u_base_Y * log_a.exp()

        u_theta_Y = gather_rate(uX, uE, pick)
        per = gkl(u_theta_Y, target)

        # Eq. (11) weights each sampled y by 1/p^u_t(y|x), which makes the estimate an
        # unbiased draw from the sum over y in Eq. (14). Raw, that weight is of order
        # the number of reachable coordinates (~3.2k at n=38) and swamps grad_clip, so
        # the default self-normalises it: same direction, O(1) magnitude, biased
        # O(1/batch). "raw" is Eq. (11) verbatim.
        w = torch.exp(-logp_y)
        loss = ((w * per).sum() / w.sum()) if self.debias == "snis" else (w * per).mean()

        with torch.no_grad():
            noop = gkl(u_base_Y, target)
            n_node = uX.shape[1] * uX.shape[2]
            is_node = (pick < n_node)
            def _ratio(mask):
                if not bool(mask.any()):
                    return float("nan")
                return float((per[mask].detach().sum() / noop[mask].sum().clamp_min(1e-12)).clamp(0, 1e6))
            resid = float((per.sum() / noop.sum().clamp_min(1e-12)).clamp(0, 1e6))
        # Split by coordinate type: calibration on the real base shows the node and
        # edge channels move in OPPOSITE directions with t -- the node gap sits at
        # ~0.66-0.71 everywhere while the edge gap falls to 0.09 by t=0.99 -- so a
        # single pooled number hides the only thing worth watching.
        return loss, {"log_a": float(log_a.mean()),
                      "clamp_frac": float(clamp_frac),
                      "resid_ratio": resid,
                      "resid_nodes": _ratio(is_node),
                      "resid_edges": _ratio(~is_node)}
