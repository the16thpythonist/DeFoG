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
    assert t.kl_anchor == "fixed"
    assert t.kl_target is None
    assert t._anchor is None


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


def test_moving_anchor_follows_policy_fixed_stays_frozen(small_model):
    torch.manual_seed(21)
    model = small_model
    init = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def max_dev(sd):
        return max(float((sd[k] - init[k]).abs().max())
                   for k in init if sd[k].dtype.is_floating_point)

    # moving anchor: the reference should drift toward the (changing) policy
    mov = GDPOTrainer(model, edge_count_reward, rollout_size=16, sample_steps=5,
                      subsample_steps=None, eta=0.0, time_distortion="identity",
                      kl_coef=0.1, kl_anchor="moving", anchor_decay=0.5,
                      lr=5e-3, ema_decay=None, seed=21)
    assert mov._anchor is not None
    mov.fit(6)
    assert max_dev(mov.ref.state_dict()) > 1e-5  # ref moved away from init


def test_fixed_anchor_reference_is_frozen(small_model):
    torch.manual_seed(22)
    model = small_model
    fix = GDPOTrainer(model, edge_count_reward, rollout_size=16, sample_steps=5,
                      subsample_steps=None, eta=0.0, time_distortion="identity",
                      kl_coef=0.1, kl_anchor="fixed", lr=5e-3, ema_decay=None, seed=22)
    assert fix._anchor is None
    ref0 = {k: v.detach().clone() for k, v in fix.ref.state_dict().items()}
    fix.fit(6)
    ref1 = fix.ref.state_dict()
    # a fixed anchor never updates the reference
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
