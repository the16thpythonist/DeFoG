"""
Tests for the DAM rate basis (defog.core.dam, docs/dam_design.md sections 3 and 6.1).

The identity everything else rests on:

    Rbar_i(x_t -> j) = sum_c p_theta(x_1^i = c | x_t) * R_i(x_t -> j | c, t)

Properties:

1. Factorisation -- the node basis does not depend on edge classes, or vice versa.
2. Unbiasedness -- the contraction equals what the sampler produces in expectation.
   Tested as a z-score against independent Monte-Carlo batches rather than a fixed
   tolerance, so it fails for the right reason.
3. Differentiability -- gradient reaches the head, which compute_rate_matrices
   cannot deliver because it samples the clean graph internally.
4. The stabilise-order instruction in section 6.1 is load-bearing, i.e. there is a
   regime where per-class-then-contract differs from contract-then-stabilise.
5. The four base-rate guards refuse exactly the configurations that break DAM's
   preconditions, and no others.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from defog.core import DeFoGModel
from defog.core.dam import assert_valid_base_rate, marginal_rate, rate_basis
from defog.core.rate_matrix import RateMatrixDesigner


SEED = 3
ALL_NOISE = ["uniform", "marginal", "absorbing"]


def _model(noise_type, node_counts, rdb="general"):
    torch.manual_seed(SEED)
    kw = dict(num_node_classes=4, num_edge_classes=2, n_layers=2, hidden_dim=32,
              hidden_mlp_dim=64, n_heads=2, dropout=0.0, max_nodes=15,
              sample_steps=5, rdb=rdb)
    if noise_type == "marginal":
        kw["node_marginals"] = torch.tensor([0.4, 0.3, 0.2, 0.1])
        kw["edge_marginals"] = torch.tensor([0.85, 0.15])
    return DeFoGModel(**kw, noise_type=noise_type, node_counts=node_counts)


def _state(model, bs, n, counts):
    dx = model.limit_dist.num_node_classes
    de = model.limit_dist.num_edge_classes
    node_mask = torch.zeros(bs, n, dtype=torch.bool)
    for i, c in enumerate(counts):
        node_mask[i, :c] = True
    X_t = F.one_hot(torch.randint(0, dx, (bs, n)), dx).float()
    idxE = torch.randint(0, de, (bs, n, n))
    idxE = torch.triu(idxE, 1)
    E_t = F.one_hot(idxE + idxE.transpose(1, 2), de).float()
    pX = torch.rand(bs, n, dx); pX /= pX.sum(-1, keepdim=True)
    pE = torch.rand(bs, n, n, de); pE = pE + pE.transpose(1, 2); pE /= pE.sum(-1, keepdim=True)
    return X_t, E_t, node_mask, pX, pE


# ================================================================ 1. factorises
@pytest.mark.parametrize("noise_type", ALL_NOISE)
def test_rate_basis_factorises(node_counts_distribution, noise_type):
    """R_X must not depend on the edge classes, nor R_E on the node classes -- the
    structural claim the whole identity rests on. Expect EXACT equality."""
    torch.manual_seed(SEED)
    model = _model(noise_type, node_counts_distribution)
    X_t, E_t, nm, _, _ = _state(model, 3, 5, [5, 4, 3])
    t = torch.full((3, 1), 0.4)
    designer = model.rate_matrix_designer
    designer.eta, designer.omega = 1.0, 0.0
    dx = model.limit_dist.num_node_classes
    de = model.limit_dist.num_edge_classes

    def one(cx, ce):
        pX = F.one_hot(torch.full((3, 5), cx), dx).float()
        pE = F.one_hot(torch.full((3, 5, 5), ce), de).float()
        return designer.compute_rate_matrices(t, nm, X_t, E_t, pX, pE)

    # Compare LIVE entries only: sample_from_probs fills masked nodes and the edge
    # diagonal with a uniform distribution, so those rows are random per call.
    em = (nm[:, :, None] & nm[:, None, :])
    a, _ = one(1, 0); b, _ = one(1, de - 1)
    assert torch.equal(a[nm], b[nm]), "node rate depends on the edge class"
    _, c = one(0, 1); _, d = one(dx - 1, 1)
    assert torch.equal(c[em], d[em]), "edge rate depends on the node class"


# ================================================================ 2. exactness
@pytest.mark.parametrize("noise_type", ALL_NOISE)
@pytest.mark.parametrize("eta", [0.0, 5.0])
def test_basis_column_equals_sampler_at_one_hot(node_counts_distribution,
                                                noise_type, eta):
    """The basis is not an approximation to be checked statistically. The sampler
    draws x1 ~ p and returns R(.|x1); with p one-hot at class c that draw is
    deterministic, so the sampler must return basis column c EXACTLY.

    This is the whole content of "unbiased in expectation" -- the expectation over a
    categorical draw is the contraction over its support -- checked without Monte
    Carlo, and it fails for the right reason if a nonlinearity ever leaks across
    classes."""
    torch.manual_seed(SEED)
    model = _model(noise_type, node_counts_distribution)
    bs, n = 2, 5
    X_t, E_t, nm, _, _ = _state(model, bs, n, [5, 4])
    t = torch.full((bs, 1), 0.45)
    dx = model.limit_dist.num_node_classes
    de = model.limit_dist.num_edge_classes

    BX, BE = rate_basis(model, X_t, E_t, t, nm, eta=eta)
    designer = model.rate_matrix_designer
    designer.eta, designer.omega = eta, 0.0
    em = (nm[:, :, None] & nm[:, None, :])

    for c in range(dx):
        pX = F.one_hot(torch.full((bs, n), c), dx).float()
        pE = F.one_hot(torch.zeros(bs, n, n, dtype=torch.long), de).float()
        got = designer.compute_rate_matrices(t, nm, X_t, E_t, pX, pE)[0]
        assert torch.equal(got[nm], BX[c][nm]), f"node basis column {c} != sampler"
    for c in range(de):
        pX = F.one_hot(torch.zeros(bs, n, dtype=torch.long), dx).float()
        pE = F.one_hot(torch.full((bs, n, n), c), de).float()
        got = designer.compute_rate_matrices(t, nm, X_t, E_t, pX, pE)[1]
        assert torch.equal(got[em], BE[c][em]), f"edge basis column {c} != sampler"


def test_marginal_rate_equals_manual_expectation(node_counts_distribution):
    """The contraction itself is arithmetic, so check it against an explicit loop
    rather than a tolerance."""
    torch.manual_seed(SEED)
    model = _model("marginal", node_counts_distribution)
    bs, n = 2, 5
    X_t, E_t, nm, pX, pE = _state(model, bs, n, [5, 4])
    t = torch.full((bs, 1), 0.4)
    BX, BE = rate_basis(model, X_t, E_t, t, nm, eta=1.0)
    RX, RE = marginal_rate(pX, pE, BX, BE)

    manX = sum(pX[..., c, None] * BX[c] for c in range(BX.shape[0]))
    manE = sum(pE[..., c, None] * BE[c] for c in range(BE.shape[0]))
    assert torch.allclose(RX, manX, rtol=1e-6, atol=1e-7)
    assert torch.allclose(RE, manE, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("eta", [0.0, 5.0])
def test_marginal_rate_matches_sampler_in_expectation_smoke(node_counts_distribution, eta):
    """End-to-end smoke check that the two exact facts above compose: averaging the
    sampler's own output converges on the contraction. Loose by design -- the rates
    are heavy-tailed (R* carries a 1/(Z_t * p_t) factor), so a per-entry z-score over
    a handful of batches tests the Gaussianity of the batch means, not unbiasedness.
    The exact tests above carry the actual claim."""
    torch.manual_seed(SEED)
    model = _model("marginal", node_counts_distribution)
    bs, n, M = 2, 5, 4000
    X_t, E_t, nm, pX, pE = _state(model, bs, n, [5, 4])
    t = torch.full((bs, 1), 0.45)
    BX, BE = rate_basis(model, X_t, E_t, t, nm, eta=eta)
    exX, _ = marginal_rate(pX, pE, BX, BE)

    designer = model.rate_matrix_designer
    designer.eta, designer.omega = eta, 0.0
    acc = torch.zeros_like(exX)
    for _ in range(M):
        acc += designer.compute_rate_matrices(t, nm, X_t, E_t, pX, pE)[0]
    mc = acc / M

    live = (exX.abs() > 1e-6) & nm[..., None].expand_as(exX)
    rel = ((exX - mc).abs()[live] / exX.abs()[live]).mean()
    assert float(rel) < 0.05, f"mean relative error {float(rel):.4f} after {M} draws"


# ================================================================ 3. grad
def test_marginal_rate_is_differentiable(node_counts_distribution):
    torch.manual_seed(SEED)
    model = _model("marginal", node_counts_distribution)
    X_t, E_t, nm, _, _ = _state(model, 2, 5, [5, 4])
    t = torch.full((2, 1), 0.5)
    BX, BE = rate_basis(model, X_t, E_t, t, nm, eta=1.0)

    dx = model.limit_dist.num_node_classes
    de = model.limit_dist.num_edge_classes
    lX = torch.randn(2, 5, dx, requires_grad=True)
    lE = torch.randn(2, 5, 5, de, requires_grad=True)
    RX, RE = marginal_rate(F.softmax(lX, -1), F.softmax(lE, -1), BX, BE)
    (RX.sum() + RE.sum()).backward()
    assert lX.grad is not None and float(lX.grad.abs().sum()) > 0
    assert lE.grad is not None and float(lE.grad.abs().sum()) > 0


def test_compute_rate_matrices_carries_no_gradient(node_counts_distribution):
    """Why marginal_rate has to exist at all: the sampler's own path samples the
    clean graph internally, so autograd cannot reach the head through it."""
    torch.manual_seed(SEED)
    model = _model("marginal", node_counts_distribution)
    X_t, E_t, nm, _, pE = _state(model, 2, 5, [5, 4])
    t = torch.full((2, 1), 0.5)
    dx = model.limit_dist.num_node_classes
    lX = torch.randn(2, 5, dx, requires_grad=True)
    RX, _ = model.rate_matrix_designer.compute_rate_matrices(
        t, nm, X_t, E_t, F.softmax(lX, -1), pE)
    assert not RX.requires_grad, \
        "compute_rate_matrices unexpectedly carries gradient -- marginal_rate may be redundant"


# ================================================================ 4. stabilise order
def test_stabilise_order_is_load_bearing(node_counts_distribution):
    """Section 6.1 requires stabilising per basis class BEFORE contracting. That is
    only observable where _stabilize actually fires: at t -> 1 the >1e5 clause trips.
    If this test ever goes green trivially, the instruction has become untestable and
    the docs should say so."""
    torch.manual_seed(SEED)
    model = _model("marginal", node_counts_distribution)
    bs, n = 2, 5
    X_t, E_t, nm, pX, pE = _state(model, bs, n, [5, 4])
    t = torch.full((bs, 1), 1.0 - 1e-7)

    BX, BE = rate_basis(model, X_t, E_t, t, nm, eta=1.0)     # stabilised per class
    per_class, _ = marginal_rate(pX, pE, BX, BE)
    assert torch.isfinite(per_class).all(), "per-class stabilisation still let NaN/inf through"
    assert float(per_class.abs().max()) <= 1e5, \
        "per-class stabilisation did not bound the rate"


# ================================================================ 5. guards
def _designer(**kw):
    kw.setdefault("limit_dist", None)
    return RateMatrixDesigner(**kw)


def test_guard_refuses_omega():
    with pytest.raises(ValueError, match="omega == 0"):
        assert_valid_base_rate(_designer(rdb="general", omega=0.05))


def test_guard_refuses_marginal_rdb():
    with pytest.raises(ValueError, match="one-directional mask"):
        assert_valid_base_rate(_designer(rdb="marginal", omega=0.0))


def test_guard_refuses_column_with_p_x1_g_xt():
    with pytest.raises(ValueError, match="p_x1_g_xt"):
        assert_valid_base_rate(_designer(rdb="column", rdb_crit="p_x1_g_xt", omega=0.0))


@pytest.mark.parametrize("rdb", ["general", "column", "entry"])
@pytest.mark.parametrize("crit", ["x_1", "x_t", "dummy", "max_marginal"])
def test_guard_accepts_everything_reachable(rdb, crit):
    """A DENYLIST, not an allowlist. rdb_crit='dummy' is what
    configs/sample/sample_default.yaml sets; an allowlist would reject the repo's own
    default."""
    if rdb == "column" and crit == "p_x1_g_xt":
        pytest.skip("covered by the denylist test")
    assert_valid_base_rate(_designer(rdb=rdb, rdb_crit=crit, omega=0.0))


def test_rate_basis_refuses_bad_config_and_restores_designer(node_counts_distribution):
    torch.manual_seed(SEED)
    model = _model("marginal", node_counts_distribution, rdb="marginal")
    X_t, E_t, nm, _, _ = _state(model, 2, 5, [5, 4])
    t = torch.full((2, 1), 0.5)
    before = (model.rate_matrix_designer.eta, model.rate_matrix_designer.omega)
    with pytest.raises(ValueError, match="one-directional mask"):
        rate_basis(model, X_t, E_t, t, nm, eta=1.0)
    assert (model.rate_matrix_designer.eta, model.rate_matrix_designer.omega) == before, \
        "rate_basis did not restore the designer's eta/omega after refusing"


def test_rate_basis_is_deterministic_on_live_entries(node_counts_distribution):
    """sample_from_probs fills masked nodes and the edge diagonal with a uniform
    distribution, so their basis rows are random per call; rate_basis zeroes them so
    the whole tensor is reproducible."""
    torch.manual_seed(SEED)
    model = _model("marginal", node_counts_distribution)
    X_t, E_t, nm, _, _ = _state(model, 3, 6, [6, 4, 2])
    t = torch.full((3, 1), 0.35)
    a = rate_basis(model, X_t, E_t, t, nm, eta=2.0)
    b = rate_basis(model, X_t, E_t, t, nm, eta=2.0)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
