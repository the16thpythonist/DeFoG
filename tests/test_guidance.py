"""
Tests for exact posterior-based discrete guidance (arXiv:2509.21912).

The suite is organized around the properties that must hold for the method to be
correct "in principle":

1. The model hook (``posterior_transform``) is a true no-op when off, and is the
   ONLY thing guidance changes -> full backward compatibility.
2. The reweight operator is exactly Theorem 1: ``softmax(g + log p) == h·p/E_p[r]``.
3. DeFoG invariants survive the reweight: edge symmetry, node masking.
4. The guidance network is architecturally decoupled from the base's conditioning
   (``cond_dim=0`` build works on a conditional base; reweight never reads the
   base's ``y_t``); and it is memoized so CFG composition costs one forward pass.
5. The Bregman objective actually learns the density ratio: on a synthetic dataset
   with a KNOWN ``r(x1)``, a trained ``h`` recovers ``E[r|x_t]`` (checked at the
   clean-data limit ``t=1``, where ``E[r|x_t]=r``).
6. The amortized (target-conditioned) network responds to the target value.
7. End-to-end guided sampling runs and returns valid graphs; the frozen base is
   never trained and never serialized into guidance checkpoints.
"""

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl

from defog.core import (
    DeFoGModel,
    GuidedSampler,
    Sampler,
    ExactGuidance,
    GuidanceModule,
    AmortizedPropertyGuidanceModule,
    build_guidance_network,
    bregman_loss,
    DensityRatio,
)
from defog.core.data import to_dense


# ---------------------------------------------------------------- helpers
class CountRatio(DensityRatio):
    """A per-graph density ratio that depends only on x1 in a learnable way:
    ``r = exp(a * #{nodes of class `cls`})``. Analytic, so we can check that a
    trained guidance network recovers it."""

    def __init__(self, cls: int = 0, a: float = 0.5):
        self.cls, self.a = cls, a

    def __call__(self, X1, E1, node_mask):
        cnt = ((X1.argmax(-1) == self.cls).float() * node_mask.float()).sum(-1)
        return torch.exp(self.a * cnt)


def _noisy_from_dense(X, E, node_mask, t_val=0.4):
    """Build the noisy_data dict denoise_step passes to a posterior_transform,
    from clean dense tensors (we only need X_t/E_t/t/node_mask for the reweight)."""
    bs = X.size(0)
    return {
        "X_t": X,
        "E_t": E,
        "y_t": torch.zeros(bs, 0),
        "t": torch.full((bs, 1), float(t_val)),
        "node_mask": node_mask,
    }


def _rand_marginals(bs, n, de):
    """A random symmetric edge-marginal tensor (bs, n, n, de) that sums to 1 over
    the last dim, for testing symmetry preservation."""
    logits = torch.randn(bs, n, n, de)
    logits = 0.5 * (logits + logits.transpose(1, 2))  # symmetric logits
    return F.softmax(logits, dim=-1)


# ======================================================================
# 1. The model hook is off-by-default and a true no-op
# ======================================================================
class TestHookIsNoOp:
    def test_none_transform_matches_no_kwarg(self, small_model, dense_tensors):
        X, E, node_mask, _ = dense_tensors
        bs = X.size(0)
        t = torch.full((bs, 1), 0.4)
        s = torch.full((bs, 1), 0.6)
        y = torch.zeros(bs, 0)

        torch.manual_seed(0)
        a = small_model.denoise_step(t, s, X, E, y, node_mask)
        torch.manual_seed(0)
        b = small_model.denoise_step(t, s, X, E, y, node_mask, posterior_transform=None)

        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    def test_identity_transform_matches_none(self, small_model, dense_tensors):
        X, E, node_mask, _ = dense_tensors
        bs = X.size(0)
        t, s, y = torch.full((bs, 1), 0.4), torch.full((bs, 1), 0.6), torch.zeros(bs, 0)

        identity = lambda pX, pE, nd, nm: (pX, pE)
        torch.manual_seed(1)
        a = small_model.denoise_step(t, s, X, E, y, node_mask, posterior_transform=None)
        torch.manual_seed(1)
        b = small_model.denoise_step(t, s, X, E, y, node_mask, posterior_transform=identity)

        assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])

    def test_transform_actually_changes_output(self, small_model, dense_tensors):
        X, E, node_mask, _ = dense_tensors
        bs = X.size(0)
        t, s, y = torch.full((bs, 1), 0.4), torch.full((bs, 1), 0.6), torch.zeros(bs, 0)

        def force_class0(pX, pE, nd, nm):
            qX = torch.zeros_like(pX); qX[..., 0] = 1.0
            qE = torch.zeros_like(pE); qE[..., 0] = 1.0
            return qX, qE

        torch.manual_seed(2)
        a = small_model.denoise_step(t, s, X, E, y, node_mask, posterior_transform=None)
        torch.manual_seed(2)
        b = small_model.denoise_step(t, s, X, E, y, node_mask, posterior_transform=force_class0)

        # A degenerate transform must change the sampled next state.
        assert not (torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))


# ======================================================================
# 2. Reweight == Theorem 1
# ======================================================================
class TestReweightIdentity:
    def test_softmax_of_g_plus_logp_equals_h_p_normalized(self, small_model, dense_tensors):
        X, E, node_mask, _ = dense_tensors
        bs, n, _ = X.shape
        de = E.size(-1)
        guidance = ExactGuidance(small_model)  # small_model plays the role of h
        noisy = _noisy_from_dense(X, E, node_mask)

        pred_X = F.softmax(torch.randn(bs, n, small_model.num_node_classes), dim=-1)
        pred_E = _rand_marginals(bs, n, de)

        gX, gE = guidance._g(noisy, node_mask)
        qX, qE = guidance.reweight(pred_X, pred_E, noisy, node_mask)

        # softmax(g + log p)
        expect_X = F.softmax(gX + torch.log(pred_X.clamp_min(1e-8)), dim=-1)
        assert torch.allclose(qX, expect_X, atol=1e-6)

        # == h·p / sum(h·p), with h = exp(g)
        hp = gX.exp() * pred_X
        manual_X = hp / hp.sum(-1, keepdim=True)
        assert torch.allclose(qX, manual_X, atol=1e-5)

        # q is a proper distribution
        assert torch.allclose(qX.sum(-1), torch.ones(bs, n), atol=1e-5)
        assert torch.allclose(qE.sum(-1), torch.ones(bs, n, n), atol=1e-5)

    def test_guidance_weight_scales_logratio(self, small_model, dense_tensors):
        X, E, node_mask, _ = dense_tensors
        bs, n, _ = X.shape
        de = E.size(-1)
        guidance = ExactGuidance(small_model, weight=2.0)
        noisy = _noisy_from_dense(X, E, node_mask)
        pred_X = F.softmax(torch.randn(bs, n, small_model.num_node_classes), dim=-1)
        pred_E = _rand_marginals(bs, n, de)

        gX, _ = guidance._g(noisy, node_mask)
        qX, _ = guidance.reweight(pred_X, pred_E, noisy, node_mask)
        assert torch.allclose(qX, F.softmax(2.0 * gX + torch.log(pred_X.clamp_min(1e-8)), dim=-1), atol=1e-6)

        # weight 0 -> guidance off (q == p, up to the clamp)
        guidance.set_weight(0.0); guidance._memo = None
        qX0, _ = guidance.reweight(pred_X, pred_E, noisy, node_mask)
        assert torch.allclose(qX0, pred_X, atol=1e-5)


# ======================================================================
# 3. DeFoG invariants survive the reweight
# ======================================================================
class TestInvariants:
    def test_reweight_preserves_edge_symmetry(self, small_model, dense_tensors):
        X, E, node_mask, _ = dense_tensors
        bs, n, _ = X.shape
        de = E.size(-1)
        guidance = ExactGuidance(small_model)
        noisy = _noisy_from_dense(X, E, node_mask)

        pred_X = F.softmax(torch.randn(bs, n, small_model.num_node_classes), dim=-1)
        pred_E = _rand_marginals(bs, n, de)  # symmetric
        _, qE = guidance.reweight(pred_X, pred_E, noisy, node_mask)

        assert torch.allclose(qE, qE.transpose(1, 2), atol=1e-6)


# ======================================================================
# 4. Decoupling from the base's conditioning + memoization
# ======================================================================
class TestDecoupling:
    def test_build_guidance_network_forces_unconditional(self, small_cond_model):
        # base is conditional (cond_dim=2); the guidance net must be independent.
        assert small_cond_model.cond_dim == 2
        h = build_guidance_network(small_cond_model, cond_dim=0)
        assert h.cond_dim == 0
        assert h.num_node_classes == small_cond_model.num_node_classes
        assert h.num_edge_classes == small_cond_model.num_edge_classes

    def test_reweight_ignores_base_yt(self, small_model, dense_tensors):
        # A cond_dim=0 guidance net must never read the base's y_t: feed a bogus
        # wide y_t and confirm the reweight still runs and matches a clean y_t.
        X, E, node_mask, _ = dense_tensors
        bs, n, _ = X.shape
        de = E.size(-1)
        guidance = ExactGuidance(small_model)
        pred_X = F.softmax(torch.randn(bs, n, small_model.num_node_classes), dim=-1)
        pred_E = _rand_marginals(bs, n, de)

        clean = _noisy_from_dense(X, E, node_mask)
        bogus = dict(clean); bogus["y_t"] = torch.randn(bs, 7)  # wrong width on purpose
        guidance._memo = None
        qX_clean, _ = guidance.reweight(pred_X, pred_E, clean, node_mask)
        guidance._memo = None
        qX_bogus, _ = guidance.reweight(pred_X, pred_E, bogus, node_mask)
        assert torch.allclose(qX_clean, qX_bogus, atol=1e-6)

    def test_reweight_memoizes_within_step(self, small_model, dense_tensors):
        # Within one step (same X_t/E_t/t), reweight must call h.forward once even
        # if invoked for both the cond and uncond CFG branches.
        X, E, node_mask, _ = dense_tensors
        bs, n, _ = X.shape
        de = E.size(-1)
        guidance = ExactGuidance(small_model)
        noisy = _noisy_from_dense(X, E, node_mask)
        pred_X = F.softmax(torch.randn(bs, n, small_model.num_node_classes), dim=-1)
        pred_E = _rand_marginals(bs, n, de)

        calls = {"n": 0}
        orig = guidance.h.forward
        def counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)
        guidance.h.forward = counting

        guidance.reweight(pred_X, pred_E, noisy, node_mask)
        guidance.reweight(pred_X, pred_E, noisy, node_mask)  # same tensors -> memo hit
        assert calls["n"] == 1

        noisy2 = dict(noisy); noisy2["t"] = noisy["t"] * 0.5  # new time -> recompute
        guidance.reweight(pred_X, pred_E, noisy2, node_mask)
        assert calls["n"] == 2


# ======================================================================
# 5. The Bregman objective learns the density ratio
# ======================================================================
class TestBregmanLearnsRatio:
    def test_bregman_loss_shapes_and_finite(self, small_model, dense_tensors):
        X, E, node_mask, _ = dense_tensors
        bs, n, _ = X.shape
        # Fake logits for g with the model's output dims.
        g = type("G", (), {})()
        g.X = torch.randn(bs, n, small_model.num_node_classes)
        g.E = torch.randn(bs, n, n, small_model.num_edge_classes)
        r = torch.rand(bs) + 0.1
        loss = bregman_loss(g, X, E, r, node_mask, lambda_edge=1.0, g_clamp=20.0)
        assert torch.isfinite(loss)

    @pytest.mark.slow
    def test_trained_h_recovers_r(self, small_model, small_dataset):
        torch.manual_seed(0)
        ratio = CountRatio(cls=0, a=0.5)
        module = GuidanceModule(small_model, ratio, n_layers=2, hidden_dim=32, lr=5e-3)
        loader = DataLoader(small_dataset, batch_size=8, shuffle=True)
        trainer = pl.Trainer(
            max_epochs=120, accelerator="cpu", logger=False,
            enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(module, loader)

        # The frozen base must never receive gradients.
        assert all(p.grad is None for p in module.base.parameters())

        # At t=1, x_t = x1, so E[r|x_t] = r exactly; check exp(g@true) ~ r.
        h = module.h.eval()
        batch = Batch.from_data_list(small_dataset)
        d, nm = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        d = d.mask(nm)
        X1, E1 = d.X, d.E
        bs = X1.size(0)
        noisy = {
            "X_t": X1, "E_t": E1, "t": torch.ones(bs, 1),
            "node_mask": nm, "y_t": torch.zeros(bs, 0),
        }
        with torch.no_grad():
            g = h.forward(noisy, h._compute_extra_data(noisy), nm)
        g_true = g.X.gather(-1, X1.argmax(-1, keepdim=True)).squeeze(-1)  # (bs, n)
        pred_r = torch.stack([g_true[i][nm[i]].exp().mean() for i in range(bs)])
        true_r = ratio(X1, E1, nm)

        corr = torch.corrcoef(torch.stack([pred_r, true_r]))[0, 1]
        assert corr > 0.6, f"guidance net did not track r (corr={corr:.3f})"

        # Rank separation: high-r graphs get higher predicted r than low-r graphs.
        order = true_r.argsort()
        k = max(1, bs // 3)
        low_mean = pred_r[order[:k]].mean()
        high_mean = pred_r[order[-k:]].mean()
        assert high_mean > low_mean


# ======================================================================
# 6. Amortized network responds to the target
# ======================================================================
class TestAmortized:
    def test_target_changes_marginals(self, small_model, dense_tensors):
        X, E, node_mask, _ = dense_tensors
        bs, n, _ = X.shape
        de = E.size(-1)
        h = build_guidance_network(small_model, cond_dim=1, n_layers=2, hidden_dim=32)
        guidance = ExactGuidance(h, prop_mean=0.0, prop_std=1.0)
        noisy = _noisy_from_dense(X, E, node_mask)
        pred_X = F.softmax(torch.randn(bs, n, h.num_node_classes), dim=-1)
        pred_E = _rand_marginals(bs, n, de)

        guidance.set_target(-3.0)
        qX_low, _ = guidance.reweight(pred_X, pred_E, noisy, node_mask)
        guidance.set_target(3.0)
        qX_high, _ = guidance.reweight(pred_X, pred_E, noisy, node_mask)

        assert not torch.allclose(qX_low, qX_high, atol=1e-4)

    def test_requires_normalization_stats(self, small_model):
        h = build_guidance_network(small_model, cond_dim=1)
        with pytest.raises(AssertionError):
            ExactGuidance(h)  # conditional h without prop_mean/prop_std must fail


# ======================================================================
# 7. End-to-end guided sampling + checkpoint hygiene
# ======================================================================
class TestEndToEnd:
    def test_plain_sampler_still_works(self, small_model):
        # Regression: default sampling path unaffected by the new hook.
        out = Sampler(small_model, sample_steps=5).sample(3, show_progress=False)
        assert len(out) == 3

    def test_guided_sampler_runs_fixed(self, small_model):
        h = build_guidance_network(small_model, cond_dim=0, n_layers=2, hidden_dim=32)
        guidance = ExactGuidance(h)
        out = GuidedSampler(small_model, guidance, sample_steps=5).sample(4, show_progress=False)
        assert len(out) == 4
        for d in out:
            assert d.x.size(-1) == small_model.num_node_classes

    def test_guided_sampler_runs_amortized(self, small_model):
        h = build_guidance_network(small_model, cond_dim=1, n_layers=2, hidden_dim=32)
        guidance = ExactGuidance(h, prop_mean=0.0, prop_std=1.0).set_target(2.0)
        out = GuidedSampler(small_model, guidance, sample_steps=5).sample(4, show_progress=False)
        assert len(out) == 4

    def test_checkpoint_excludes_base(self, small_model):
        module = GuidanceModule(small_model, CountRatio())
        ckpt = {"state_dict": {
            "base.model.weight": torch.zeros(1),
            "h.model.weight": torch.zeros(1),
        }}
        module.on_save_checkpoint(ckpt)
        assert not any(k.startswith("base.") for k in ckpt["state_dict"])
        assert any(k.startswith("h.") for k in ckpt["state_dict"])

    def test_amortized_module_trains_one_step(self, small_model, small_dataset):
        # Attach a precomputed per-graph property and run a single optimization
        # step end-to-end (exercises the amortized training path).
        import numpy as np
        props = []
        for d in small_dataset:
            val = float((d.x.argmax(-1) == 0).sum())  # e.g. count of class-0 atoms
            d.prop_val = torch.tensor([val])
            props.append(val)
        module = AmortizedPropertyGuidanceModule(
            small_model, prop_values=props, prop_mean=float(np.mean(props)),
            prop_std=float(np.std(props) + 1e-6), gamma=0.5,
            n_layers=2, hidden_dim=32,
        )
        loader = DataLoader(small_dataset, batch_size=8, shuffle=False)
        trainer = pl.Trainer(
            max_steps=2, accelerator="cpu", logger=False,
            enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(module, loader)
        # A usable guidance object comes straight off the module.
        guidance = module.guidance().set_target(float(np.mean(props)))
        assert guidance.h.cond_dim == 1
