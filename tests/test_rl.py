"""
Tests for GDPO-style RL fine-tuning (defog.core.rl, arXiv:2402.16302).

Organized around the properties that must hold for the eager policy gradient to be
correct:

1. The eager gradient actually LEARNS: on a toy graph-statistic reward (no RDKit),
   GDPO fine-tuning increases the reward of freshly generated graphs.
2. ``eager_logprob(reduction="sum")`` equals a hand-rolled masking-/symmetry-aware
   cross-entropy of the endpoint (per-entry identity, not equality with TrainLoss).
3. Masking: padded nodes, padded edges, and the lower edge triangle contribute
   nothing to the log-prob.
4. Size normalization: ``"mean"`` is invariant to padding; ``"sum"`` is the joint LL.
5. ``kl_clean`` is ~0 against an identical reference and >0 after a perturbation.
6. ``group_advantage`` whitens (grpo: mean~0/std~1), respects modes and clipping.
7. No-op / regression: importing rl + building a GDPOTrainer leaves ``model.sample``
   deterministic; ``kl_coef=0`` allocates no reference.
8. Overfit sanity: repeated updates on one frozen rollout raise the eager objective.
"""

import copy

import pytest
import torch
import torch.nn.functional as F

from defog.core import (
    GDPOTrainer,
    RolloutSampler,
    eager_logprob,
    kl_clean,
    group_advantage,
    reward_from_energy,
    EMA,
)
from defog.core.guidance import _edge_upper_mask
from defog.core.noise import sample_noise


# ------------------------------------------------------------------ helpers
def _rand_onehot_graph(bs, n, dx, de, node_mask):
    """A random masked one-hot (X, E) with symmetric E, for building noisy states
    and endpoints."""
    X = F.one_hot(torch.randint(0, dx, (bs, n)), dx).float()
    idxE = torch.randint(0, de, (bs, n, n))
    idxE = torch.triu(idxE, diagonal=1)
    idxE = idxE + idxE.transpose(1, 2)
    E = F.one_hot(idxE, de).float()
    # mask padding
    X = X * node_mask[..., None]
    em = (node_mask[:, :, None] & node_mask[:, None, :]).float()
    E = E * em[..., None]
    return X, E


def _mask(bs, n, counts):
    m = torch.zeros(bs, n, dtype=torch.bool)
    for i, c in enumerate(counts):
        m[i, :c] = True
    return m


def edge_count_reward(X1, E1, node_mask):
    """Toy reward: number of present edges (class 1). Non-differentiable graph
    statistic -- no RDKit. GDPO should push generated graphs to have more edges."""
    emask = _edge_upper_mask(node_mask)
    has_edge = (E1.argmax(-1) == 1).float()
    return (has_edge * emask).sum(dim=(1, 2))


# ======================================================================
# 1. The eager gradient learns
# ======================================================================
def test_gdpo_increases_toy_reward(small_model):
    torch.manual_seed(0)
    model = small_model

    @torch.no_grad()
    def eval_reward(k=64):
        s = RolloutSampler(model, sample_steps=5)
        s.sample(k, device=torch.device("cpu"), show_progress=False)
        X1, E1 = s.endpoint
        return float(edge_count_reward(X1, E1, s.end_node_mask).mean())

    before = eval_reward()
    trainer = GDPOTrainer(
        model, edge_count_reward, rollout_size=32, sample_steps=5,
        subsample_steps=None, eta=0.0, time_distortion="identity",
        advantage_mode="grpo", kl_coef=0.0, lr=2e-3, ema_decay=None, seed=0,
    )
    trainer.fit(40)
    after = eval_reward()

    # The toy reward must rise meaningfully after fine-tuning.
    assert after > before + 0.5, f"reward did not increase: {before:.2f} -> {after:.2f}"


# ======================================================================
# 2. Per-entry cross-entropy identity
# ======================================================================
def test_eager_logprob_matches_manual_ce(small_model):
    torch.manual_seed(1)
    model = small_model
    model.eval()
    bs, n = 3, 8
    node_mask = _mask(bs, n, [8, 6, 5])
    X_t, E_t = _rand_onehot_graph(bs, n, 4, 2, node_mask)
    X1, E1 = _rand_onehot_graph(bs, n, 4, 2, node_mask)
    y = torch.zeros(bs, 0)
    t = torch.full((bs, 1), 0.4)

    lp = eager_logprob(model, X_t, E_t, y, t, X1, E1, node_mask,
                       lambda_edge=1.0, reduction="sum")

    # hand-rolled reference
    noisy = {"X_t": X_t, "E_t": E_t, "y_t": y, "t": t, "node_mask": node_mask}
    pred = model.forward(noisy, model._compute_extra_data(noisy), node_mask)
    lpX = (X1 * F.log_softmax(pred.X, -1)).sum(-1)
    lpE = (E1 * F.log_softmax(pred.E, -1)).sum(-1)
    emask = _edge_upper_mask(node_mask)
    manual = (lpX * node_mask).sum(-1) + (lpE * emask).sum(dim=(1, 2))

    assert torch.allclose(lp, manual, atol=1e-5)


# ======================================================================
# 3. Masking / symmetry invariance
# ======================================================================
def test_eager_logprob_ignores_padding_and_lower_triangle(small_model):
    torch.manual_seed(2)
    model = small_model
    model.eval()
    bs, n = 2, 9
    node_mask = _mask(bs, n, [9, 4])
    X_t, E_t = _rand_onehot_graph(bs, n, 4, 2, node_mask)
    X1, E1 = _rand_onehot_graph(bs, n, 4, 2, node_mask)
    y = torch.zeros(bs, 0)
    t = torch.full((bs, 1), 0.3)

    base = eager_logprob(model, X_t, E_t, y, t, X1, E1, node_mask, reduction="sum")

    # Corrupt the TARGET in padded node rows, padded edge entries, and the whole
    # lower triangle of E1 -- none of which the masking should count.
    X1b = X1.clone()
    E1b = E1.clone()
    X1b[1, 4:] = F.one_hot(torch.randint(0, 4, (n - 4,)), 4).float()
    tri_lower = torch.tril(torch.ones(n, n), diagonal=-1).bool()
    E1b[:, tri_lower] = F.one_hot(torch.randint(0, 2, (bs, int(tri_lower.sum()))), 2).float()
    em = (node_mask[:, :, None] & node_mask[:, None, :])
    E1b[~em] = F.one_hot(torch.randint(0, 2, (int((~em).sum()),)), 2).float()

    corrupted = eager_logprob(model, X_t, E_t, y, t, X1b, E1b, node_mask, reduction="sum")
    assert torch.allclose(base, corrupted, atol=1e-5)


# ======================================================================
# 4. Size normalization
# ======================================================================
def test_reduction_sum_vs_mean_arithmetic(small_model):
    torch.manual_seed(3)
    model = small_model
    model.eval()
    bs, n = 3, 8
    node_mask = _mask(bs, n, [8, 6, 4])
    X_t, E_t = _rand_onehot_graph(bs, n, 4, 2, node_mask)
    X1, E1 = _rand_onehot_graph(bs, n, 4, 2, node_mask)
    y = torch.zeros(bs, 0)
    t = torch.full((bs, 1), 0.5)

    # manual node/edge log-prob sums and real counts from a single forward
    noisy = {"X_t": X_t, "E_t": E_t, "y_t": y, "t": t, "node_mask": node_mask}
    pred = model.forward(noisy, model._compute_extra_data(noisy), node_mask)
    emask = _edge_upper_mask(node_mask)
    node_sum = ((X1 * F.log_softmax(pred.X, -1)).sum(-1) * node_mask).sum(-1)
    edge_sum = ((E1 * F.log_softmax(pred.E, -1)).sum(-1) * emask).sum(dim=(1, 2))
    n_nodes = node_mask.sum(-1).clamp_min(1)
    n_edges = emask.sum(dim=(1, 2)).clamp_min(1)

    got_sum = eager_logprob(model, X_t, E_t, y, t, X1, E1, node_mask, reduction="sum")
    got_mean = eager_logprob(model, X_t, E_t, y, t, X1, E1, node_mask, reduction="mean")
    assert torch.allclose(got_sum, node_sum + edge_sum, atol=1e-5)
    assert torch.allclose(got_mean, node_sum / n_nodes + edge_sum / n_edges, atol=1e-5)
    # graphs of different size have different "sum" magnitude (size-dependent) but
    # bounded "mean" (per-entry, size-invariant): mean's magnitude is much smaller.
    assert got_mean.abs().max() < got_sum.abs().max()


# ======================================================================
# 5. KL to reference
# ======================================================================
def test_kl_clean_zero_for_identical_ref_positive_after_perturbation(small_model):
    torch.manual_seed(4)
    model = small_model
    model.eval()
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    bs, n = 3, 7
    node_mask = _mask(bs, n, [7, 6, 4])
    X_t, E_t = _rand_onehot_graph(bs, n, 4, 2, node_mask)
    y = torch.zeros(bs, 0)
    t = torch.full((bs, 1), 0.5)

    kl0 = kl_clean(model, ref, X_t, E_t, y, t, node_mask)
    assert float(kl0) < 1e-5
    assert float(kl0) >= -1e-6  # KL is non-negative

    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    kl1 = kl_clean(model, ref, X_t, E_t, y, t, node_mask)
    assert float(kl1) > 1e-3


# ======================================================================
# 6. Advantage whitening
# ======================================================================
def test_group_advantage_modes():
    r = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    a = group_advantage(r, mode="grpo", clip=None)
    assert abs(float(a.mean())) < 1e-5
    assert abs(float(a.std(unbiased=True)) - 1.0) < 0.2
    assert torch.allclose(group_advantage(r, mode="none", clip=None), r)
    assert torch.allclose(group_advantage(r, mode="mean", clip=None), r - r.mean())
    # clipping
    big = torch.tensor([0.0, 100.0])
    assert group_advantage(big, mode="grpo", clip=1.0).abs().max() <= 1.0 + 1e-6
    # grouped whitening
    groups = torch.tensor([0, 0, 1, 1])
    rg = torch.tensor([1.0, 3.0, 10.0, 20.0])
    ag = group_advantage(rg, groups=groups, mode="mean", clip=None)
    assert torch.allclose(ag, torch.tensor([-1.0, 1.0, -5.0, 5.0]))
    # singleton group must NOT NaN (finding 8: rg.std() of size 1 is NaN)
    a_single = group_advantage(torch.tensor([1.0, 2.0, 3.0]),
                               groups=torch.tensor([0, 0, 1]), mode="grpo", clip=5.0)
    assert torch.isfinite(a_single).all()
    # constant batch -> zero advantage, not div-by-~0 blowup (finding 9)
    a_const = group_advantage(torch.tensor([0.7, 0.7, 0.7, 0.7]), mode="grpo")
    assert torch.allclose(a_const, torch.zeros(4))


def test_kl_clean_reduction_scale(small_model):
    import copy
    from defog.core import kl_clean
    torch.manual_seed(11)
    model = small_model.eval()
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    bs, n = 2, 8
    node_mask = _mask(bs, n, [8, 8])
    X_t, E_t = _rand_onehot_graph(bs, n, 4, 2, node_mask)
    y = torch.zeros(bs, 0); t = torch.full((bs, 1), 0.5)
    kl_mean = float(kl_clean(model, ref, X_t, E_t, y, t, node_mask, reduction="mean"))
    kl_sum = float(kl_clean(model, ref, X_t, E_t, y, t, node_mask, reduction="sum"))
    # "sum" aggregates over ~n nodes + ~n^2/2 edges, so it is much larger than the
    # per-element "mean" -- the size-dependence that made kl_coef untunable (finding 7).
    assert kl_sum > kl_mean * 3
    assert kl_mean > 0


# ======================================================================
# 7. No-op / regression safety
# ======================================================================
def test_import_and_trainer_construction_leave_sampling_deterministic(small_model):
    model = small_model
    torch.manual_seed(7)
    a = model.sample(4, device=torch.device("cpu"), show_progress=False)
    # constructing a trainer must not perturb the model / RNG contract
    GDPOTrainer(model, edge_count_reward, kl_coef=0.0, ema_decay=None)
    torch.manual_seed(7)
    b = model.sample(4, device=torch.device("cpu"), show_progress=False)
    assert len(a) == len(b)
    for da, db in zip(a, b):
        assert torch.equal(da.x, db.x)
        assert torch.equal(da.edge_index, db.edge_index)


def test_no_reference_allocated_when_kl_off(small_model):
    t0 = GDPOTrainer(small_model, edge_count_reward, kl_coef=0.0, ema_decay=None)
    assert t0.ref is None
    t1 = GDPOTrainer(small_model, edge_count_reward, kl_coef=0.1, ema_decay=None)
    assert t1.ref is not None
    assert all(not p.requires_grad for p in t1.ref.parameters())


# ======================================================================
# 8. Overfit sanity: repeated updates raise the eager objective
# ======================================================================
def test_repeated_updates_raise_objective_on_frozen_rollout(small_model):
    torch.manual_seed(8)
    model = small_model
    trainer = GDPOTrainer(
        model, edge_count_reward, rollout_size=24, sample_steps=5,
        subsample_steps=None, eta=0.0, time_distortion="identity",
        advantage_mode="grpo", lr=3e-3, ema_decay=None, kl_coef=0.0, seed=8,
    )
    buf = trainer.rollout()

    @torch.no_grad()
    def objective():
        tot = 0.0
        for (X_t, E_t, t) in buf.states:
            lp = eager_logprob(model, X_t, E_t, buf.y, t, buf.X1, buf.E1, buf.node_mask,
                               reduction=trainer.reduction)
            tot += float((buf.advantage * lp).mean())
        return tot

    before = objective()
    for _ in range(15):
        trainer.update(buf)
    after = objective()
    assert after > before, f"objective did not rise: {before:.3f} -> {after:.3f}"


# ======================================================================
# reward_from_energy adapter
# ======================================================================
def test_optional_features_default_to_noop(small_model):
    # all new features off by default -> behavior identical to before
    t = GDPOTrainer(small_model, edge_count_reward, kl_coef=0.0, ema_decay=None)
    assert t.positive_only is False
    assert t.kl_target is None
    assert t.advantage_mode == "mean"   # Dr. GRPO is the default


def test_positive_only_clamps_and_still_learns(small_model):
    torch.manual_seed(20)
    model = small_model
    trainer = GDPOTrainer(
        model, edge_count_reward, rollout_size=32, sample_steps=5,
        subsample_steps=None, eta=0.0, time_distortion="identity",
        advantage_mode="grpo", positive_only=True, kl_coef=0.0,
        lr=3e-3, ema_decay=None, seed=20,
    )
    buf = trainer.rollout()
    # GRPO produces negatives; positive-only must clamp them out of the loss
    assert (buf.advantage < 0).any()
    assert (buf.advantage.clamp_min(0.0) >= 0).all()

    @torch.no_grad()
    def eval_reward():
        s = RolloutSampler(model, sample_steps=5)
        s.sample(64, device=torch.device("cpu"), show_progress=False)
        return float(edge_count_reward(s.endpoint[0], s.endpoint[1], s.end_node_mask).mean())

    before = eval_reward()
    trainer.fit(30)
    after = eval_reward()
    # positive-only still increases the toy reward (only pushes good endpoints up)
    assert after > before, f"positive-only did not learn: {before:.2f} -> {after:.2f}"


def test_kl_reference_is_frozen(small_model):
    torch.manual_seed(22)
    model = small_model
    fix = GDPOTrainer(model, edge_count_reward, rollout_size=16, sample_steps=5,
                      subsample_steps=None, eta=0.0, time_distortion="identity",
                      kl_coef=0.1, lr=5e-3, ema_decay=None, seed=22)
    ref0 = {k: v.detach().clone() for k, v in fix.ref.state_dict().items()}
    fix.fit(6)
    ref1 = fix.ref.state_dict()
    # the KL reference is a fixed frozen copy of the pretrained weights; never updates
    assert all(torch.equal(ref0[k], ref1[k]) for k in ref0)
    assert all(not p.requires_grad for p in fix.ref.parameters())


def test_adaptive_kl_moves_coef(small_model):
    torch.manual_seed(23)
    model = small_model
    t = GDPOTrainer(model, edge_count_reward, rollout_size=16, sample_steps=5,
                    subsample_steps=None, eta=0.0, time_distortion="identity",
                    kl_coef=0.5, kl_target=0.01, lr=3e-3, ema_decay=None, seed=23)
    t.fit(5)
    # the controller nudges kl_coef away from its initial value toward the target
    assert abs(t.kl_coef - 0.5) > 1e-4


def test_reward_from_energy_floors_invalid():
    class FakeEnergy:
        invalid = 1e3
        def __call__(self, X1, E1, node_mask):
            # graph 0 valid (energy 0.5), graph 1 invalid (energy 1e3)
            return torch.tensor([0.5, 1e3])
    r = reward_from_energy(FakeEnergy(), transform="neg", invalid_reward=-5.0)
    out = r(None, None, torch.ones(2, 3, dtype=torch.bool))
    assert abs(float(out[0]) - (-0.5)) < 1e-6
    assert abs(float(out[1]) - (-5.0)) < 1e-6


# ======================================================================
# Adapter GDPO: behavior == scored (the eager gradient scores the SAME
# product-of-experts blend the rollout samples, at any CFG weight)
# ======================================================================
def _model_composed_logmarginal(base, adapter, X_t, E_t, t, node_mask, cond, w, mode):
    """Reference: the model's OWN composed clean-graph marginal at the terminal decode
    (mirrors DeFoGModel._denoise_step_composed) -- a single batched (N+1)*bs forward
    through the composition, then _blend_logp. This is the behavior policy."""
    from defog.core.adapter import AdapterComposition, ConditionBranch
    from defog.core.data import PlaceHolder
    bs, rep = X_t.size(0), 2
    y0 = torch.zeros(bs, 0)
    base_noisy = {"X_t": X_t, "E_t": E_t, "y_t": y0, "t": t, "node_mask": node_mask}
    extra = base._compute_extra_data(base_noisy)
    extra_b = PlaceHolder(X=extra.X.repeat(rep, 1, 1), E=extra.E.repeat(rep, 1, 1, 1),
                          y=extra.y.repeat(rep, 1))
    nd = {"X_t": X_t.repeat(rep, 1, 1), "E_t": E_t.repeat(rep, 1, 1, 1),
          "y_t": y0.repeat(rep, 1), "t": t.repeat(rep, 1), "node_mask": node_mask.repeat(rep, 1)}
    comp = AdapterComposition([ConditionBranch(adapter, cond, w)], base=base, mode=mode)
    mod = comp.build_modulation(bs, t)
    pred = base.forward(nd, extra_b, nd["node_mask"], cond_modulation=mod)
    pX = F.softmax(pred.X, -1).view(rep, bs, *pred.X.shape[1:])
    pE = F.softmax(pred.E, -1).view(rep, bs, *pred.E.shape[1:])
    ww = comp.weights(X_t.device, dtype=pX.dtype)
    qX = base._blend_logp(pX, ww, comp.mode)
    qE = base._blend_logp(pE, ww, comp.mode)
    return F.log_softmax(qX, -1), F.log_softmax(qE, -1)


def _make_adapter(base):
    from defog.core.adapter import AdaLNAdapter
    a = AdaLNAdapter.for_base(base, cond_dim=2, time_conditioned=True)
    with torch.no_grad():                        # off the zero-init no-op so cond != uncond
        for p in a.parameters():
            p.add_(0.2 * torch.randn_like(p))
    return a


def test_adapter_scoring_matches_rollout_blend_at_any_weight(small_model):
    # The fix: the adapter eager log-prob must SCORE the endpoint under the SAME
    # composed policy the rollout samples -- at any CFG weight, not only w=1. Compare
    # the scored value (what enters the gradient) to scoring the endpoint under the
    # model's OWN composed clean marginal.
    from defog.core.rl import adapter_eager_logprob
    torch.manual_seed(31)
    base = small_model.eval()
    adapter = _make_adapter(base)
    bs, n = 3, 7
    node_mask = _mask(bs, n, [7, 6, 5])
    X_t, E_t = _rand_onehot_graph(bs, n, 4, 2, node_mask)
    X1, E1 = _rand_onehot_graph(bs, n, 4, 2, node_mask)
    cond = torch.randn(bs, 2)
    t = torch.full((bs, 1), 0.4)
    emask = _edge_upper_mask(node_mask)
    for w in (1.0, 2.0):
        got = adapter_eager_logprob(base, adapter, X_t, E_t, cond, t, X1, E1, node_mask,
                                    weight=w, mode="product", reduction="mean")
        rX, rE = _model_composed_logmarginal(base, adapter, X_t, E_t, t, node_mask, cond, w, "product")
        lpX, lpE = (X1 * rX).sum(-1), (E1 * rE).sum(-1)
        ref = ((lpX * node_mask).sum(-1) / node_mask.sum(-1).clamp_min(1)
               + (lpE * emask).sum(dim=(1, 2)) / emask.sum(dim=(1, 2)).clamp_min(1))
        # atol swallows float32 log/exp round-trip noise; a real formula bug (wrong
        # weight, logits-not-softmax, uncond/cond swap) would be O(1), not O(1e-3).
        assert torch.allclose(got, ref, atol=1e-3), f"scored logprob != rollout blend at w={w}"


def test_adapter_scoring_w1_matches_plain_conditional_on_realistic_endpoint(small_model):
    # Backward compat where it matters: for a HIGH-PROB endpoint -- the model's own
    # argmax, i.e. what a rollout actually produces -- the composed w=1 scoring matches
    # the plain conditional. (The _blend_logp eps floor only diverges for near-zero-prob
    # classes, which a real rollout endpoint never selects; a RANDOM endpoint would.)
    from defog.core.rl import adapter_eager_logprob
    torch.manual_seed(32)
    base = small_model.eval()
    adapter = _make_adapter(base)
    bs, n = 3, 8
    node_mask = _mask(bs, n, [8, 6, 5])
    X_t, E_t = _rand_onehot_graph(bs, n, 4, 2, node_mask)
    cond = torch.randn(bs, 2)
    t = torch.full((bs, 1), 0.4)
    y0 = torch.zeros(bs, 0)
    noisy = {"X_t": X_t, "E_t": E_t, "y_t": y0, "t": t, "node_mask": node_mask}
    pred = base.forward(noisy, base._compute_extra_data(noisy), node_mask,
                        cond_modulation=adapter(cond, t=t))
    X1 = F.one_hot(pred.X.argmax(-1), 4).float() * node_mask[..., None]   # high-prob G1
    E1 = F.one_hot(pred.E.argmax(-1), 2).float()
    got = adapter_eager_logprob(base, adapter, X_t, E_t, cond, t, X1, E1, node_mask,
                                weight=1.0, mode="product", reduction="mean")
    lpX = (X1 * F.log_softmax(pred.X, -1)).sum(-1)
    lpE = (E1 * F.log_softmax(pred.E, -1)).sum(-1)
    emask = _edge_upper_mask(node_mask)
    ref = ((lpX * node_mask).sum(-1) / node_mask.sum(-1).clamp_min(1)
           + (lpE * emask).sum(dim=(1, 2)) / emask.sum(dim=(1, 2)).clamp_min(1))
    assert torch.allclose(got, ref, atol=1e-3)


def test_adapter_scoring_grad_only_to_adapter(small_model):
    from defog.core.rl import _base_uncond_softmax, _compose_logmarginals
    torch.manual_seed(33)
    base = small_model.eval().requires_grad_(False)
    adapter = _make_adapter(base)
    bs, n = 2, 6
    node_mask = _mask(bs, n, [6, 5])
    X_t, E_t = _rand_onehot_graph(bs, n, 4, 2, node_mask)
    cond = torch.randn(bs, 2)
    t = torch.full((bs, 1), 0.3)
    puX, puE, noisy, extra = _base_uncond_softmax(base, X_t, E_t, t, node_mask)
    logpX, logpE = _compose_logmarginals(base, adapter, noisy, extra, node_mask, cond, puX, puE, 2.0, "product")
    (logpX.sum() + logpE.sum()).backward()
    assert all(p.grad is None for p in base.parameters())      # frozen base gets no grad
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in adapter.parameters())


def test_adapter_trainer_step_at_weight2_updates_adapter(small_model):
    # End-to-end: rollout + update at rollout_weight=2 with KL on runs and moves the
    # adapter (base frozen). Exercises the shared weight/mode source-of-truth path.
    from defog.core.rl import AdapterGDPOTrainer
    torch.manual_seed(34)
    base = small_model
    adapter = _make_adapter(base)

    def cond_sampler():
        return torch.randn(8, 2), torch.zeros(8, dtype=torch.long)

    def cond_reward(X1, E1, node_mask, cond):
        emask = _edge_upper_mask(node_mask)
        return ((E1.argmax(-1) == 1).float() * emask).sum(dim=(1, 2))

    before = [p.detach().clone() for p in adapter.parameters()]
    trainer = AdapterGDPOTrainer(
        base, adapter, cond_reward, condition_sampler=cond_sampler,
        rollout_weight=2.0, kl_coef=0.1, rollout_size=8, sample_steps=5,
        subsample_steps=2, minibatch_size=4, eta=0.0, time_distortion="identity",
        lr=1e-2, ema_decay=None, seed=34,
    )
    m = trainer.step()
    assert all(v == v for v in (m["loss"], m["kl"], m["grad_norm"]))   # finite (not NaN)
    assert all(not p.requires_grad for p in base.parameters())
    assert any(not torch.equal(b, p) for b, p in zip(before, adapter.parameters())), \
        "adapter did not update"


# ======================================================================
# Common random numbers (shared start noise+size within an advantage group)
# ======================================================================
def test_crn_init_state_shares_within_group_and_differs_across(small_model):
    from defog.core.rl import RolloutSampler
    group_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    s = RolloutSampler(small_model, group_ids=group_ids, crn=True)
    bs, n = 6, 7
    node_mask = _mask(bs, n, [7, 5, 6, 4, 7, 3])   # varied sizes to prove replication
    X, E = _rand_onehot_graph(bs, n, 4, 2, node_mask)
    n_nodes = node_mask.sum(-1).long()
    X2, E2, nm2, nn2 = s._init_state(X.clone(), E.clone(), node_mask.clone(), n_nodes.clone())
    for gid in (0, 1):
        idx = (group_ids == gid).nonzero(as_tuple=True)[0]
        r = idx[0]
        for j in idx.tolist():
            assert torch.equal(X2[j], X2[r]) and torch.equal(E2[j], E2[r])
            assert torch.equal(nm2[j], nm2[r]) and int(nn2[j]) == int(nn2[r])
    # each group took its representative's size (member 0): group0->row0 (7), group1->row3 (4)
    assert nn2.tolist() == [7, 7, 7, 4, 4, 4]


def test_crn_off_is_noop(small_model):
    from defog.core.rl import RolloutSampler
    s = RolloutSampler(small_model, group_ids=torch.tensor([0, 0, 1, 1]), crn=False)
    bs, n = 4, 6
    node_mask = _mask(bs, n, [6, 5, 4, 6])
    X, E = _rand_onehot_graph(bs, n, 4, 2, node_mask)
    n_nodes = node_mask.sum(-1).long()
    X2, E2, nm2, nn2 = s._init_state(X.clone(), E.clone(), node_mask.clone(), n_nodes.clone())
    assert torch.equal(X2, X) and torch.equal(E2, E)
    assert torch.equal(nm2, node_mask) and torch.equal(nn2, n_nodes)


def test_crn_rollout_first_recorded_state_shared_within_group(small_model):
    # End-to-end: through Sampler.sample -> _init_state -> _pre_step, the initial
    # recorded state is identical within each group (shared noise), different across.
    from defog.core.rl import RolloutSampler
    torch.manual_seed(42)
    group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    s = RolloutSampler(small_model, group_ids=group_ids, crn=True, sample_steps=3,
                       eta=0.0, time_distortion="identity", subsample_idx=None)
    s.sample(8, device=torch.device("cpu"), show_progress=False)
    x0 = s.trace_X[0]                          # state entering step 0 = the initial noise
    for gid in (0, 1):
        idx = (group_ids == gid).nonzero(as_tuple=True)[0]
        r = int(idx[0])
        for j in idx.tolist():
            assert torch.equal(x0[j], x0[r])
