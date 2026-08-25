"""The credit head's algebra (docs/credit_head_design.md).

The load-bearing property is that the loss recovers a CONDITIONAL MEAN. Everything the
design claims -- that m = E[exp(-g) | x1^i=c, xt] is what gets fitted -- rests on it, so
it is tested directly rather than assumed from the Bregman literature.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from defog.core.credit import (
    constant_baseline,
    credit_gkl,
    edge_mask_of,
    gather_log_m,
    guided_logmarginals,
    per_class_baseline,
)

SEED = 17


def test_gkl_minimiser_is_the_conditional_mean():
    """Fit one scalar to heavy-tailed positive samples; it must land on E[w], not the
    median, the mode, or exp(E[log w]). This is the whole justification for the loss."""
    torch.manual_seed(SEED)
    log_w = torch.randn(200_000) * 1.3 - 0.5          # lognormal: mean != median != mode
    log_m = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([log_m], lr=0.05)
    for _ in range(2000):
        opt.zero_grad()
        credit_gkl(log_m.expand_as(log_w), log_w).mean().backward()
        opt.step()
    got, want = float(log_m.exp()), float(log_w.exp().mean())
    assert abs(got - want) / want < 0.02, f"got {got:.4f}, E[w] = {want:.4f}"
    # ...and is NOT the Jensen-biased alternative the design warns about
    assert abs(got - float(log_w.mean().exp())) / want > 0.5


def test_gkl_is_zero_only_at_the_target():
    lw = torch.tensor([-1.0, 0.0, 2.0])
    assert torch.allclose(credit_gkl(lw, lw), torch.zeros(3), atol=1e-6)
    assert (credit_gkl(lw + 0.3, lw) > 0).all()
    assert (credit_gkl(lw - 0.3, lw) > 0).all()


def test_guidance_is_a_noop_at_zero_credit():
    """log m = 0 means m = 1 everywhere, i.e. no reweighting. The steered head must then
    reproduce the base head exactly -- the analogue of the adapter's zero-init no-op."""
    torch.manual_seed(SEED)
    lX = F.log_softmax(torch.randn(3, 5, 9), -1)
    lE = F.log_softmax(torch.randn(3, 5, 5, 4), -1)
    gX, gE = guided_logmarginals(lX, lE, torch.zeros_like(lX), torch.zeros_like(lE))
    assert torch.allclose(gX, lX, atol=1e-6) and torch.allclose(gE, lE, atol=1e-6)


def test_guidance_applies_the_product_rule():
    """p* ~ p_base * m, renormalised -- checked against an explicit computation."""
    torch.manual_seed(SEED)
    lX = F.log_softmax(torch.randn(2, 4, 9), -1)
    lE = F.log_softmax(torch.randn(2, 4, 4, 4), -1)
    lmX, lmE = torch.randn(2, 4, 9) * 0.4, torch.randn(2, 4, 4, 4) * 0.4
    gX, _ = guided_logmarginals(lX, lE, lmX, lmE, scale=1.0)
    want = lX.exp() * lmX.exp()
    want = want / want.sum(-1, keepdim=True)
    assert torch.allclose(gX.exp(), want, atol=1e-6)


def test_guidance_scale_interpolates():
    torch.manual_seed(SEED)
    lX = F.log_softmax(torch.randn(2, 4, 9), -1)
    lE = F.log_softmax(torch.randn(2, 4, 4, 4), -1)
    lmX, lmE = torch.randn(2, 4, 9), torch.randn(2, 4, 4, 4)
    g0, _ = guided_logmarginals(lX, lE, lmX, lmE, scale=0.0)
    assert torch.allclose(g0, lX, atol=1e-6)
    g2, _ = guided_logmarginals(lX, lE, lmX, lmE, scale=2.0)
    gd, _ = guided_logmarginals(lX, lE, 2 * lmX, 2 * lmE, scale=1.0)
    assert torch.allclose(g2, gd, atol=1e-6)


def test_gather_picks_the_observed_class():
    """One (xt, x1) pair supervises exactly the class the endpoint took at each
    coordinate -- not the argmax of the prediction, and not all of them."""
    torch.manual_seed(SEED)
    lmX = torch.randn(2, 6, 9)
    lmE = torch.randn(2, 6, 6, 4)
    iX = torch.randint(0, 9, (2, 6))
    iE = torch.randint(0, 4, (2, 6, 6))
    gX, gE = gather_log_m(lmX, lmE, F.one_hot(iX, 9).float(), F.one_hot(iE, 4).float())
    assert torch.allclose(gX, lmX.gather(-1, iX[..., None]).squeeze(-1), atol=1e-6)
    assert torch.allclose(gE, lmE.gather(-1, iE[..., None]).squeeze(-1), atol=1e-6)


def test_baselines_are_the_right_conditional_means():
    """Gate 1's reference predictors. If these match the network the credit is a
    per-element preference and no per-coordinate machinery is warranted."""
    torch.manual_seed(SEED)
    log_w = torch.randn(4000) * 0.8
    cls = torch.randint(0, 3, (4000,))
    msk = torch.ones(4000, dtype=torch.bool)
    assert abs(math.exp(constant_baseline(log_w)) - float(log_w.exp().mean())) < 1e-4
    pc = per_class_baseline(log_w, cls, 3, msk)
    for c in range(3):
        want = float(log_w[cls == c].exp().mean())
        assert abs(float(pc[c].exp()) - want) < 1e-4


def test_per_class_baseline_handles_an_absent_class():
    """dx=9 on zinc-kek but P and F have near-zero marginals; an unseen class must not
    produce a NaN that silently poisons the gate."""
    log_w = torch.randn(500)
    cls = torch.zeros(500, dtype=torch.long)
    pc = per_class_baseline(log_w, cls, 4, torch.ones(500, dtype=torch.bool))
    assert torch.isfinite(pc[0]) and torch.isinf(pc[1:]).all()


def test_edge_mask_is_upper_triangular_and_masked():
    nm = torch.tensor([[True, True, True, False], [True, True, False, False]])
    em = edge_mask_of(nm)
    assert int(em[0].sum()) == 3 and int(em[1].sum()) == 1     # C(3,2) and C(2,1)
    assert not em[..., 0, 0].any() and not em[:, 1, 0].any()   # no diagonal, no lower


def test_head_starts_as_an_exact_constant():
    """Zero-init gate: at iteration 0 the head must reproduce its per-class bias
    EXACTLY, so Gate 1 starts at the baseline it has to beat. Without this the head
    starts at exp(arbitrary base logits) -- the first smoke run hit |grad| = 4.8e5."""
    from defog.core.credit import CreditHead

    class _Attn:
        dx, de, dy = 8, 4, 2

    class _Layer:
        self_attn = _Attn()

    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tf_layers = [_Layer(), _Layer()]
            self.mlp_in_X = torch.nn.Sequential(torch.nn.Linear(4, 8))

    class _Lim:
        num_node_classes, num_edge_classes = 9, 4

    class _Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self.limit_dist = _Lim()

    torch.manual_seed(SEED)
    h = CreditHead(_Base(), backbone="shared", cond_mean=[0.0], cond_std=[1.0])
    h.init_bias(-1.7)
    assert torch.allclose(h.bias_X, torch.full((9,), -1.7), atol=1e-6)
    assert float(h.gate_X) == 0.0 and float(h.gate_E) == 0.0
    # gate at zero => whatever the backbone says, the output is the bias
    fakeX = torch.randn(2, 5, 9) * 50
    out = h.bias_X + h.gate_X * (fakeX - fakeX.mean(-1, keepdim=True))
    assert torch.allclose(out, torch.full((2, 5, 9), -1.7), atol=1e-6)


def test_per_class_bias_is_expressible():
    """The bias is per-class, not scalar, so the head can start at the STRONGER of the
    two Gate 1 references rather than the weaker one."""
    from defog.core.credit import CreditHead

    class _Attn:
        dx, de, dy = 8, 4, 2

    class _Layer:
        self_attn = _Attn()

    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tf_layers = [_Layer()]
            self.mlp_in_X = torch.nn.Sequential(torch.nn.Linear(4, 8))

    class _Lim:
        num_node_classes, num_edge_classes = 9, 4

    class _Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self.limit_dist = _Lim()

    h = CreditHead(_Base(), backbone="shared", cond_mean=[0.0], cond_std=[1.0])
    want = torch.linspace(-2.0, -0.5, 9)
    h.init_bias(want)
    assert torch.allclose(h.bias_X, want, atol=1e-6)


def test_init_bias_rejects_reusing_node_classes_for_edges():
    """dx=9 and de=4 are different label spaces. A per-class node bias must not be
    silently broadcast onto edges; the scalar mean is used instead."""
    from defog.core.credit import CreditHead

    class _Attn:
        dx, de, dy = 8, 4, 2

    class _Layer:
        self_attn = _Attn()

    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tf_layers = [_Layer()]
            self.mlp_in_X = torch.nn.Sequential(torch.nn.Linear(4, 8))

    class _Lim:
        num_node_classes, num_edge_classes = 9, 4

    class _Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self.limit_dist = _Lim()

    h = CreditHead(_Base(), backbone="shared", cond_mean=[0.0], cond_std=[1.0])
    want = torch.linspace(-2.0, -0.5, 9)
    h.init_bias(want)
    assert torch.allclose(h.bias_X, want, atol=1e-6)
    assert h.bias_E.shape == (4,)
    assert torch.allclose(h.bias_E, torch.full((4,), float(want.mean())), atol=1e-6)
    # an explicit edge vector is honoured
    h.init_bias(want, torch.tensor([-1.0, -2.0, -3.0, -4.0]))
    assert torch.allclose(h.bias_E, torch.tensor([-1.0, -2.0, -3.0, -4.0]), atol=1e-6)


# ===========================================================================
# CreditGuidance: the posterior_transform hook
# ===========================================================================
class _StubHead:
    """A credit head with a fixed, known log m -- so the guidance algebra is tested
    without dragging a transformer into a unit test."""

    def __init__(self, lmX, lmE):
        self.lmX, self.lmE = lmX, lmE

    def __call__(self, X_t, E_t, t, node_mask, cond):
        b = X_t.shape[0]
        return self.lmX[:b], self.lmE[:b]


def _noisy(b, n, dx, de):
    return {"X_t": torch.zeros(b, n, dx), "E_t": torch.zeros(b, n, n, de),
            "t": torch.zeros(b, 1)}


def test_guidance_at_scale_zero_is_bit_identical():
    """The Gate 3 control arm is scale=0. It must reproduce the unguided sampler
    exactly, not approximately -- otherwise the control is not a control."""
    from defog.core.credit import CreditGuidance

    torch.manual_seed(SEED)
    pX = F.softmax(torch.randn(3, 5, 9), -1)
    pE = F.softmax(torch.randn(3, 5, 5, 4), -1)
    g = CreditGuidance(_StubHead(torch.randn(3, 5, 9) * 3, torch.randn(3, 5, 5, 4) * 3),
                       torch.zeros(3, 1), scale=0.0)
    qX, qE = g(pX, pE, _noisy(3, 5, 9, 4), torch.ones(3, 5, dtype=torch.bool))
    assert qX is pX and qE is pE


def test_guidance_matches_the_product_rule_and_stays_normalised():
    from defog.core.credit import CreditGuidance

    torch.manual_seed(SEED)
    pX = F.softmax(torch.randn(2, 4, 9), -1)
    pE = F.softmax(torch.randn(2, 4, 4, 4), -1)
    lmX, lmE = torch.randn(2, 4, 9) * 0.5, torch.zeros(2, 4, 4, 4)
    g = CreditGuidance(_StubHead(lmX, lmE), torch.zeros(2, 1), scale=1.0)
    qX, qE = g(pX, pE, _noisy(2, 4, 9, 4), torch.ones(2, 4, dtype=torch.bool))
    want = pX * lmX.exp()
    want = want / want.sum(-1, keepdim=True)
    assert torch.allclose(qX, want, atol=1e-5)
    assert torch.allclose(qX.sum(-1), torch.ones(2, 4), atol=1e-5)
    assert torch.allclose(qE.sum(-1), torch.ones(2, 4, 4), atol=1e-5)


def test_guidance_keeps_edges_symmetric():
    """DeFoG symmetrises edges; guidance that breaks it corrupts the rate silently."""
    from defog.core.credit import CreditGuidance

    torch.manual_seed(SEED)
    pE = F.softmax(torch.randn(2, 5, 5, 4), -1)
    pE = 0.5 * (pE + pE.transpose(1, 2))
    lmE = torch.randn(2, 5, 5, 4)                       # deliberately ASYMMETRIC
    g = CreditGuidance(_StubHead(torch.zeros(2, 5, 9), lmE), torch.zeros(2, 1))
    _, qE = g(F.softmax(torch.randn(2, 5, 9), -1), pE,
              _noisy(2, 5, 9, 4), torch.ones(2, 5, dtype=torch.bool))
    assert torch.allclose(qE, qE.transpose(1, 2), atol=1e-6)


def test_guidance_refuses_a_condition_that_does_not_divide_the_batch():
    """A cond/marginal batch mismatch guides rows toward the WRONG target and is
    invisible in the output, so it must raise rather than broadcast."""
    from defog.core.credit import CreditGuidance

    g = CreditGuidance(_StubHead(torch.zeros(7, 4, 9), torch.zeros(7, 4, 4, 4)),
                       torch.zeros(3, 1), scale=1.0)
    with pytest.raises(ValueError, match="does not divide"):
        g(F.softmax(torch.randn(7, 4, 9), -1), F.softmax(torch.randn(7, 4, 4, 4), -1),
          _noisy(7, 4, 9, 4), torch.ones(7, 4, dtype=torch.bool))


def test_guidance_repeats_a_condition_that_does_divide():
    """Composition/Feynman-Kac call with rep*bs rows; that must still work."""
    from defog.core.credit import CreditGuidance

    torch.manual_seed(SEED)
    g = CreditGuidance(_StubHead(torch.zeros(6, 4, 9), torch.zeros(6, 4, 4, 4)),
                       torch.zeros(3, 1), scale=1.0)
    qX, _ = g(F.softmax(torch.randn(6, 4, 9), -1), F.softmax(torch.randn(6, 4, 4, 4), -1),
              _noisy(6, 4, 9, 4), torch.ones(6, 4, dtype=torch.bool))
    assert qX.shape == (6, 4, 9)


# ===========================================================================
# Pool assembly across ragged rollout batches
# ===========================================================================
def test_pad_batch_preserves_content_and_masks_the_padding():
    """Rollout batches pad to THEIR OWN max node count, so the pool has to reconcile
    e.g. 35 and 37. This is the bug that killed two 100-minute cluster jobs."""
    from defog.core.credit import pad_batch

    torch.manual_seed(SEED)
    X1 = F.one_hot(torch.randint(0, 9, (3, 5)), 9).float()
    E1 = F.one_hot(torch.randint(0, 4, (3, 5, 5)), 4).float()
    nm = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1] * 5]).bool()
    pX, pE, pM = pad_batch(X1, E1, nm, 8)
    assert pX.shape == (3, 8, 9) and pE.shape == (3, 8, 8, 4) and pM.shape == (3, 8)
    assert torch.equal(pX[:, :5], X1) and torch.equal(pE[:, :5, :5], E1)
    assert torch.equal(pM[:, :5], nm)
    assert not pM[:, 5:].any(), "padded slots must be masked out"
    # padded slots are a VALID one-hot, so argmax cannot yield a silent wrong answer
    assert torch.equal(pX[:, 5:].sum(-1), torch.ones(3, 3))
    assert torch.equal(pE[:, 5:].sum(-1), torch.ones(3, 3, 8))


def test_pad_batch_is_a_noop_at_the_target_size():
    from defog.core.credit import pad_batch

    X1 = F.one_hot(torch.randint(0, 9, (2, 4)), 9).float()
    E1 = F.one_hot(torch.randint(0, 4, (2, 4, 4)), 4).float()
    nm = torch.ones(2, 4, dtype=torch.bool)
    pX, pE, pM = pad_batch(X1, E1, nm, 4)
    assert pX is X1 and pE is E1 and pM is nm


def test_assemble_reconciles_ragged_batches():
    """The exact failure: cat() of batches with 5 and 7 nodes. Must succeed and keep
    every real coordinate, with the pool padded to the widest batch."""
    from defog.core.credit import assemble

    def mk(b, n):
        return {"X1": F.one_hot(torch.randint(0, 9, (b, n)), 9).float(),
                "E1": F.one_hot(torch.randint(0, 4, (b, n, n)), 4).float(),
                "node_mask": torch.ones(b, n, dtype=torch.bool),
                "cond": torch.randn(b, 1), "reward": torch.randn(b)}

    pool = assemble([mk(2, 5), mk(3, 7), mk(1, 6)])
    assert pool["X1"].shape == (6, 7, 9)
    assert pool["E1"].shape == (6, 7, 7, 4)
    assert pool["reward"].shape == (6,)
    assert int(pool["node_mask"].sum()) == 2 * 5 + 3 * 7 + 1 * 6
