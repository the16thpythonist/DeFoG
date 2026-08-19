"""
Tests for defog.core.renoise -- the shared re-noising step behind the RAM and DAM
estimators (docs/dam_design.md section 8).

The properties that must hold:

1. ``renoise_states`` samples from the pretraining kernel
   ``t*delta(G1) + (1-t)*p_0``, for every noise type.
2. Structural invariants survive: E symmetric, padding masked. NOTE the real-node
   DIAGONAL comes back one-hot(class 0), not zero -- ``sample_from_probs`` does
   ``triu(E, 1)`` then mirrors -- so "identity at t=1" is an upper-triangle claim.
3. t=1 reproduces the endpoint; t=0 is the prior and carries no information about it.
4. ``draw_times(mode="match")`` reproduces the exact noise levels a GDPO rollout
   would have scored at -- otherwise the A/B is not isolating the state source.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from defog.core import DeFoGModel
from defog.core.guidance import _edge_upper_mask
from defog.core.renoise import draw_times, renoise_states


SEED = 5


# ------------------------------------------------------------------ fixtures
def _model(noise_type, node_counts):
    torch.manual_seed(SEED)
    kw = dict(num_node_classes=4, num_edge_classes=2, n_layers=2, hidden_dim=32,
              hidden_mlp_dim=64, n_heads=2, dropout=0.0, max_nodes=15, sample_steps=5)
    if noise_type == "marginal":
        kw["node_marginals"] = torch.tensor([0.4, 0.3, 0.2, 0.1])
        kw["edge_marginals"] = torch.tensor([0.85, 0.15])
    return DeFoGModel(**kw, noise_type=noise_type, node_counts=node_counts)


def _endpoint(model, bs, n, counts):
    """A masked one-hot endpoint in the network's OUTPUT class space."""
    dx = model.limit_dist.num_node_classes
    de = model.limit_dist.num_edge_classes
    node_mask = torch.zeros(bs, n, dtype=torch.bool)
    for i, c in enumerate(counts):
        node_mask[i, :c] = True
    X1 = F.one_hot(torch.randint(0, dx, (bs, n)), dx).float()
    idxE = torch.randint(0, de, (bs, n, n))
    idxE = torch.triu(idxE, 1)
    idxE = idxE + idxE.transpose(1, 2)
    E1 = F.one_hot(idxE, de).float()
    X1 = X1 * node_mask[..., None]
    em = (node_mask[:, :, None] & node_mask[:, None, :]).float()
    E1 = E1 * em[..., None]
    return X1, E1, node_mask


ALL_NOISE = ["uniform", "marginal", "absorbing"]


# ============================================================ 1. the kernel
@pytest.mark.parametrize("noise_type", ALL_NOISE)
@pytest.mark.parametrize("t_val", [0.0, 0.25, 0.6, 1.0])
def test_renoise_marginal_matches_kernel(node_counts_distribution, noise_type, t_val):
    """Empirical class frequencies must match t*delta(G1) + (1-t)*p_0."""
    torch.manual_seed(SEED)
    model = _model(noise_type, node_counts_distribution)
    bs, n, R = 3, 5, 8000
    X1, E1, node_mask = _endpoint(model, bs, n, [5, 4, 3])
    y = torch.zeros(bs, 0)

    # one batched call with the endpoint repeated R times, rather than R calls
    Xr, Er = X1.repeat(R, 1, 1), E1.repeat(R, 1, 1, 1)
    nmr, yr = node_mask.repeat(R, 1), y.repeat(R, 1)
    t = torch.full((bs * R, 1), float(t_val))
    (Xt, Et, _), = renoise_states(model, Xr, Er, yr, nmr, [t])

    p0X = model.limit_dist.X.float()
    expX = t_val * X1 + (1 - t_val) * p0X[None, None, :]        # (bs, n, dx)
    obsX = Xt.view(R, bs, n, -1).mean(0)                        # (bs, n, dx)

    live = node_mask[..., None].expand_as(expX)
    dev = (obsX - expX)[live].abs().max().item()
    band = 4.0 * math.sqrt(0.25 / R)                            # 4 sigma, worst-case p=.5
    assert dev < band, f"node kernel off by {dev:.4f} (4-sigma band {band:.4f})"


# ============================================================ 2. structure
@pytest.mark.parametrize("noise_type", ALL_NOISE)
def test_renoise_symmetric_and_masked(node_counts_distribution, noise_type):
    torch.manual_seed(SEED)
    model = _model(noise_type, node_counts_distribution)
    bs, n = 3, 6
    X1, E1, node_mask = _endpoint(model, bs, n, [6, 4, 2])
    y = torch.zeros(bs, 0)
    (Xt, Et, _), = renoise_states(model, X1, E1, y, node_mask,
                                  [torch.full((bs, 1), 0.5)])

    assert torch.equal(Et, Et.transpose(1, 2)), "E is not symmetric after re-noising"
    assert (Xt[~node_mask] == 0).all(), "padded nodes are not masked"
    em = node_mask[:, :, None] & node_mask[:, None, :]
    assert (Et[~em] == 0).all(), "padded edges are not masked"

    # The real-node diagonal is one-hot(class 0) -- a consequence of triu-then-mirror
    # in sample_from_probs, not a bug. Pinned so it cannot regress silently.
    diag = torch.diagonal(Et, dim1=1, dim2=2).transpose(1, 2)   # (bs, n, de)
    real = node_mask
    assert (diag[real].argmax(-1) == 0).all(), "diagonal is not class 0"
    assert (diag[real].sum(-1) == 1).all(), "diagonal is not one-hot"


# ============================================================ 3. endpoints
@pytest.mark.parametrize("noise_type", ALL_NOISE)
def test_renoise_t1_is_identity(node_counts_distribution, noise_type):
    """At t=1 the kernel is a delta, so X_t == X1 and E_t == E1 -- but only on the
    UPPER TRIANGLE of real-node pairs, because the diagonal is rebuilt as class 0."""
    torch.manual_seed(SEED)
    model = _model(noise_type, node_counts_distribution)
    bs, n = 3, 6
    X1, E1, node_mask = _endpoint(model, bs, n, [6, 5, 3])
    y = torch.zeros(bs, 0)
    (Xt, Et, _), = renoise_states(model, X1, E1, y, node_mask,
                                  [torch.ones(bs, 1)])

    assert torch.equal(Xt, X1), "t=1 did not reproduce the node endpoint"
    up = _edge_upper_mask(node_mask)
    assert torch.equal(Et[up], E1[up]), "t=1 did not reproduce the upper-triangle edges"


@pytest.mark.parametrize("noise_type", ALL_NOISE)
def test_renoise_t0_is_prior(node_counts_distribution, noise_type):
    """At t=0 the state must carry no information about the endpoint: two different
    endpoints must produce the same class frequencies."""
    torch.manual_seed(SEED)
    model = _model(noise_type, node_counts_distribution)
    bs, n, R = 1, 5, 8000
    y = torch.zeros(bs * R, 0)
    counts = [5]

    freqs = []
    for seed in (11, 12):
        torch.manual_seed(seed)
        X1, E1, node_mask = _endpoint(model, bs, n, counts)
        assert X1.argmax(-1).tolist() != []                      # sanity
        Xr, Er = X1.repeat(R, 1, 1), E1.repeat(R, 1, 1, 1)
        (Xt, _, _), = renoise_states(model, Xr, Er, y, node_mask.repeat(R, 1),
                                     [torch.zeros(bs * R, 1)])
        freqs.append(Xt.view(R, bs, n, -1).mean(0))

    dev = (freqs[0] - freqs[1]).abs().max().item()
    band = 4.0 * math.sqrt(2 * 0.25 / R)
    assert dev < band, f"t=0 states depend on the endpoint (dev {dev:.4f} > {band:.4f})"


# ============================================================ 4. time modes
def test_draw_times_match_reproduces_the_rollout_grid(small_model):
    """mode='match' must give exactly the noise levels the GDPO rollout recorded.
    If it does not, the A/B is not isolating the state source."""
    from defog.core.rl import RolloutSampler
    idx = [0, 2, 4]
    torch.manual_seed(SEED)
    s = RolloutSampler(small_model, subsample_idx=idx, sample_steps=5,
                       time_distortion="polydec")
    s.sample(3, device=torch.device("cpu"), show_progress=False)
    recorded = torch.cat(s.trace_t).reshape(len(idx), -1)[:, :1]

    got = draw_times(small_model, 1, torch.device("cpu"), mode="match",
                     step_indices=idx, sample_steps=5, time_distortion="polydec")
    got = torch.cat(got)
    assert torch.allclose(recorded, got, rtol=0, atol=0), \
        f"match grid differs from the recorded trace:\n{recorded}\n{got}"


def test_draw_times_match_shares_across_batch_train_does_not(small_model):
    m = draw_times(small_model, 8, torch.device("cpu"), mode="match",
                   step_indices=[1], sample_steps=5, time_distortion="polydec")[0]
    assert m.unique().numel() == 1, "match should share one level across the batch"
    torch.manual_seed(SEED)
    tr = draw_times(small_model, 8, torch.device("cpu"), mode="train", n_draws=1)[0]
    assert tr.unique().numel() > 1, "train should draw independently per trajectory"


def test_draw_times_ram_density():
    """RAM's own recipe is p(t) = 2(1-t), i.e. t = 1 - sqrt(U), mean 1/3 in DeFoG's
    convention where t=1 is data."""
    class _Stub:
        pass
    torch.manual_seed(SEED)
    ts = torch.cat(draw_times(_Stub(), 20000, torch.device("cpu"),
                              mode="ram", n_draws=1))
    assert abs(float(ts.mean()) - 1 / 3) < 0.01, f"mean {float(ts.mean()):.4f} != 1/3"
    assert 0.0 <= float(ts.min()) and float(ts.max()) <= 1.0


def test_draw_times_rejects_bad_config(small_model):
    with pytest.raises(ValueError, match="unknown time mode"):
        draw_times(small_model, 4, torch.device("cpu"), mode="nope", n_draws=1)
    with pytest.raises(ValueError, match="needs step_indices"):
        draw_times(small_model, 4, torch.device("cpu"), mode="match", sample_steps=5)
    with pytest.raises(ValueError, match="needs n_draws"):
        draw_times(small_model, 4, torch.device("cpu"), mode="train")
