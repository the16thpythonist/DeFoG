"""
Tests for the SizeDistribution interface and its concrete implementations,
plus integration with DeFoGModel.sample().
"""

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from defog.core import (
    SizeDistribution,
    EmpiricalSizeDistribution,
    FixedSizeDistribution,
    ExplicitSizeDistribution,
    UniformSizeDistribution,
    CategoricalSizeDistribution,
    ConditionalSizeDistribution,
    LearnedSizeDistribution,
    SizeBranch,
    ComposedSizeDistribution,
    DeFoGModel,
)


class TestBaseClass:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            SizeDistribution()


class TestEmpiricalSizeDistribution:
    def test_sample_within_support(self):
        hist = torch.zeros(11)
        hist[3:8] = 1.0  # sizes 3..7
        d = EmpiricalSizeDistribution(hist)
        s = d.sample(200)
        assert s.shape == (200,)
        assert s.min() >= 3 and s.max() <= 7
        assert d.max_size == 10

    def test_from_dict(self):
        d = EmpiricalSizeDistribution({4: 1, 6: 1})
        s = d.sample(50)
        assert set(s.tolist()) <= {4, 6}

    def test_log_prob(self):
        hist = torch.tensor([0.0, 0.0, 1.0, 1.0])  # sizes 2, 3
        d = EmpiricalSizeDistribution(hist)
        lp = d.log_prob(torch.tensor([2, 3]))
        assert torch.allclose(lp, torch.log(torch.tensor([0.5, 0.5])), atol=1e-5)

    def test_ignores_condition(self):
        d = EmpiricalSizeDistribution(torch.tensor([0.0, 0.0, 0.0, 1.0]))
        s = d.sample(10, condition=torch.randn(10, 2))
        assert torch.all(s == 3)


class TestFixedSizeDistribution:
    def test_all_equal(self):
        d = FixedSizeDistribution(7)
        s = d.sample(20)
        assert torch.all(s == 7)
        assert d.max_size == 7

    def test_rejects_zero(self):
        with pytest.raises(AssertionError):
            FixedSizeDistribution(0)


class TestExplicitSizeDistribution:
    def test_returns_given(self):
        d = ExplicitSizeDistribution([3, 5, 7])
        s = d.sample(3)
        assert s.tolist() == [3, 5, 7]
        assert d.max_size == 7

    def test_length_mismatch_raises(self):
        d = ExplicitSizeDistribution([3, 5, 7])
        with pytest.raises(AssertionError):
            d.sample(4)


class TestUniformSizeDistribution:
    def test_within_range(self):
        d = UniformSizeDistribution(4, 9)
        s = d.sample(500)
        assert s.min() >= 4 and s.max() <= 9
        assert d.max_size == 9

    def test_rejects_bad_range(self):
        with pytest.raises(AssertionError):
            UniformSizeDistribution(5, 3)


class TestCategoricalSizeDistribution:
    def test_within_support(self):
        d = CategoricalSizeDistribution([2, 8], probs=[0.1, 0.9])
        s = d.sample(300)
        assert set(s.tolist()) <= {2, 8}
        # 8 should dominate
        assert (s == 8).float().mean() > 0.7
        assert d.max_size == 8

    def test_uniform_default_probs(self):
        d = CategoricalSizeDistribution([4, 5, 6])
        s = d.sample(300)
        assert set(s.tolist()) <= {4, 5, 6}


class TestConditionalSizeDistribution:
    @staticmethod
    def _correlated_data(n=400):
        # size increases monotonically with a single property.
        torch.manual_seed(0)
        cond = torch.linspace(-2, 2, n).unsqueeze(-1)
        sizes = torch.round(10 + 4 * cond.squeeze(-1)).long().clamp(min=2)
        return cond, sizes

    def test_kernel_tracks_condition(self):
        cond, sizes = self._correlated_data()
        d = ConditionalSizeDistribution(cond, sizes, method="kernel")
        torch.manual_seed(1)
        low = d.sample(200, condition=torch.full((200, 1), -1.5))
        high = d.sample(200, condition=torch.full((200, 1), 1.5))
        assert low.float().mean() < high.float().mean()

    def test_kernel_stays_in_support(self):
        cond, sizes = self._correlated_data()
        d = ConditionalSizeDistribution(cond, sizes, method="kernel")
        s = d.sample(100, condition=torch.full((100, 1), 0.0))
        assert s.min() >= int(sizes.min()) and s.max() <= int(sizes.max())

    def test_regression_extrapolates(self):
        cond, sizes = self._correlated_data()
        # allow the regression to produce sizes above the training max
        d = ConditionalSizeDistribution(
            cond, sizes, method="regression", max_size=100
        )
        torch.manual_seed(2)
        far = d.sample(200, condition=torch.full((200, 1), 6.0))
        # 10 + 4*6 = 34, well above the training max (~18)
        assert far.float().mean() > int(sizes.max())

    def test_condition_none_falls_back_to_marginal(self):
        cond, sizes = self._correlated_data()
        d = ConditionalSizeDistribution(cond, sizes, method="kernel")
        s = d.sample(100, condition=None)
        assert s.min() >= int(sizes.min()) and s.max() <= int(sizes.max())

    def test_wrong_num_samples_raises(self):
        cond, sizes = self._correlated_data()
        d = ConditionalSizeDistribution(cond, sizes, method="kernel")
        with pytest.raises(AssertionError):
            d.sample(5, condition=torch.randn(4, 1))

    def test_from_dataloader(self):
        graphs = []
        for i in range(20):
            n = 4 + (i % 5)
            x = torch.zeros(n, 4)
            x[torch.arange(n), 0] = 1
            graphs.append(Data(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long),
                               edge_attr=torch.zeros(0, 2), y=torch.randn(1, 2)))
        loader = DataLoader(graphs, batch_size=4)
        d = ConditionalSizeDistribution.from_dataloader(loader, method="kernel")
        assert d.conditions.shape == (20, 2)
        assert d.sizes.shape == (20,)
        assert set(d.sizes.tolist()) <= {4, 5, 6, 7, 8}


class TestModelIntegration:
    def test_size_dist_fixed(self, small_model):
        small_model.eval()
        samples = small_model.sample(
            num_samples=3, size_dist=FixedSizeDistribution(6),
            sample_steps=3, show_progress=False,
        )
        for s in samples:
            assert s.x.shape[0] == 6

    def test_num_nodes_int_still_works(self, small_model):
        small_model.eval()
        samples = small_model.sample(
            num_samples=3, num_nodes=5, sample_steps=3, show_progress=False,
        )
        for s in samples:
            assert s.x.shape[0] == 5

    def test_size_dist_overrides_num_nodes(self, small_model):
        small_model.eval()
        samples = small_model.sample(
            num_samples=3, num_nodes=5, size_dist=FixedSizeDistribution(7),
            sample_steps=3, show_progress=False,
        )
        for s in samples:
            assert s.x.shape[0] == 7

    def test_default_uses_marginal(self, small_model):
        small_model.eval()
        samples = small_model.sample(
            num_samples=4, sample_steps=3, show_progress=False,
        )
        assert len(samples) == 4

    def test_size_clamped_to_max_nodes(self, small_model):
        small_model.eval()
        # request more nodes than the model supports; must clamp, not crash.
        big = small_model.max_nodes + 50
        samples = small_model.sample(
            num_samples=2, size_dist=FixedSizeDistribution(big),
            sample_steps=3, show_progress=False,
        )
        for s in samples:
            assert s.x.shape[0] <= small_model.max_nodes

    def test_conditional_size_dist_end_to_end(self, small_cond_model, cond_dim):
        small_cond_model.eval()
        cond = torch.randn(30, cond_dim)
        sizes = torch.randint(3, 8, (30,))
        size_dist = ConditionalSizeDistribution(cond, sizes, method="kernel")
        condition = torch.randn(4, cond_dim)
        samples = small_cond_model.sample(
            num_samples=4, condition=condition, size_dist=size_dist,
            sample_steps=3, show_progress=False,
        )
        assert len(samples) == 4
        for s in samples:
            assert 1 <= s.x.shape[0] <= small_cond_model.max_nodes


# ===========================================================================
# LearnedSizeDistribution / ComposedSizeDistribution
# ===========================================================================
def _marginal(n_bins=11, lo=2, hi=9):
    """A marginal with deliberate hard zeros at both ends, so every test below
    also exercises the 'unsupported sizes stay unreachable' invariant."""
    m = torch.zeros(n_bins)
    m[lo:hi] = torch.tensor([1.0, 3.0, 6.0, 8.0, 6.0, 3.0, 1.0])
    return m


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    kw.setdefault("marginal", _marginal())
    return LearnedSizeDistribution(1, 5, 15, hidden=32, layers=2,
                                   cond_mean=[2.0], cond_std=[1.5], **kw)


class TestLearnedSizeDistribution:
    def test_grid_and_support(self):
        m = _model()
        assert (m.min_size, m.max_size, m.n_bins) == (5, 15, 11)
        assert m.support.tolist() == [False, False] + [True] * 7 + [False, False]
        assert m.sizes().tolist() == list(range(5, 16))

    def test_log_pmf_normalized(self):
        lp = _model().log_pmf(torch.tensor([[1.0], [3.0]]))
        assert lp.shape == (2, 11)
        assert torch.allclose(lp.exp().sum(-1), torch.ones(2), atol=1e-5)

    def test_unsupported_sizes_unreachable(self):
        lp = _model().log_pmf(torch.tensor([[1.0]]))
        assert torch.isinf(lp[:, :2]).all() and torch.isinf(lp[:, -2:]).all()
        assert not torch.isnan(lp).any()

    def test_samples_land_in_support(self):
        s = _model().sample(50, condition=torch.tensor([[1.0]]))
        assert ((s >= 7) & (s <= 13)).all()

    def test_condition_none_falls_back_to_marginal(self):
        m = _model()
        lp = m.log_pmf(None, num_samples=3)
        assert lp.shape == (3, 11)
        assert torch.allclose(lp[0].exp(), m.marginal, atol=1e-6)

    def test_scalar_condition_broadcasts(self):
        m = _model()
        assert m.log_pmf(torch.tensor([[1.0]]), num_samples=4).shape[0] in (1, 4)
        assert m.sample(4, condition=torch.tensor([[1.0]])).numel() == 4

    def test_log_prob_matches_pmf(self):
        m = _model()
        c = torch.tensor([[1.0], [3.0]])
        lp = m.log_pmf(c)
        got = m.log_prob(torch.tensor([9, 10]), condition=c)
        assert torch.allclose(got, torch.stack([lp[0, 4], lp[1, 5]]), atol=1e-6)

    def test_log_prob_outside_grid_is_neg_inf(self):
        assert torch.isinf(_model().log_prob(torch.tensor([99, 1]))).all()

    def test_normalizes_raw_condition_internally(self):
        m = _model()
        raw = torch.tensor([[3.5]])
        assert torch.allclose(m.normalize(raw), (raw - 2.0) / 1.5)

    def test_roundtrip_save_load(self, tmp_path):
        m = _model(seed=3)
        c = torch.tensor([[1.25]])
        before = m.log_pmf(c)
        path = m.save(tmp_path / "size.ckpt")
        back = LearnedSizeDistribution.load(path)
        assert torch.allclose(back.log_pmf(c), before, atol=1e-6)
        # the three buffers must survive, not just the weights
        assert torch.allclose(back.cond_mean, m.cond_mean)
        assert torch.allclose(back.cond_std, m.cond_std)
        assert torch.allclose(back.marginal, m.marginal)
        assert back.config() == m.config()

    def test_from_config_ignores_unknown_keys(self):
        m = _model()
        cfg = dict(m.config(), some_future_field=123)
        back = LearnedSizeDistribution.from_config(cfg, m.state_dict())
        assert back.n_bins == m.n_bins

    def test_check_compatible(self):
        class FakeAdapter:
            cond_dim = 1
            cond_mean = torch.tensor([2.0])
            cond_std = torch.tensor([1.5])

        m = _model()
        assert m.check_compatible(FakeAdapter())
        bad = FakeAdapter()
        bad.cond_std = torch.tensor([9.0])          # fit on differently-scaled targets
        with pytest.raises(AssertionError, match="cond_std differs"):
            m.check_compatible(bad)
        wide = FakeAdapter()
        wide.cond_dim = 7
        with pytest.raises(AssertionError, match="width-1 condition"):
            m.check_compatible(wide)

    def test_encoder_slot_is_declared_but_unwired(self):
        assert "cond_encoder" in _model().config()
        with pytest.raises(NotImplementedError):
            LearnedSizeDistribution(1, 5, 15, cond_encoder=torch.nn.Identity())


class TestComposedSizeDistribution:
    def test_single_branch_is_the_branch(self):
        m = _model(seed=1)
        c = torch.tensor([[1.0]])
        for mode in ("product", "mean"):
            comp = ComposedSizeDistribution([SizeBranch(m, c, 1.0)], mode=mode)
            assert torch.allclose(comp.log_pmf(1), m.log_pmf(c), atol=1e-6)

    def test_self_composition_under_mean_is_idempotent(self):
        m = _model(seed=1)
        c = torch.tensor([[1.0]])
        comp = ComposedSizeDistribution(
            [SizeBranch(m, c, 1.0), SizeBranch(m, c, 1.0)], mode="mean")
        assert torch.allclose(comp.log_pmf(1), m.log_pmf(c), atol=1e-6)

    def test_zero_weight_recovers_the_marginal(self):
        m = _model(seed=1)
        comp = ComposedSizeDistribution([SizeBranch(m, torch.tensor([[1.0]]), 0.0)])
        assert torch.allclose(comp.log_pmf(1).exp()[0], m.marginal, atol=1e-6)

    def test_blend_is_finite_and_normalized(self):
        a, b = _model(seed=1), _model(seed=2)
        comp = ComposedSizeDistribution(
            [SizeBranch(a, torch.tensor([[1.0]]), 1.0),
             SizeBranch(b, torch.tensor([[3.0]]), 1.0)])
        lq = comp.log_pmf(1)
        assert not torch.isnan(lq).any()
        assert torch.allclose(lq.exp().sum(-1), torch.ones(1), atol=1e-5)

    def test_unsupported_sizes_stay_unreachable_through_the_blend(self):
        a, b = _model(seed=1), _model(seed=2)
        comp = ComposedSizeDistribution(
            [SizeBranch(a, torch.tensor([[1.0]]), 2.0),
             SizeBranch(b, torch.tensor([[3.0]]), 2.0)])
        lq = comp.log_pmf(1)
        assert torch.isinf(lq[:, :2]).all() and torch.isinf(lq[:, -2:]).all()
        assert ((comp.sample(40) >= 7) & (comp.sample(40) <= 13)).all()

    def test_rejects_mismatched_grid(self):
        a = _model(seed=1)
        b = LearnedSizeDistribution(1, 4, 15, marginal=torch.ones(12))
        with pytest.raises(AssertionError, match="size grid"):
            ComposedSizeDistribution([SizeBranch(a, torch.tensor([[1.0]])),
                                      SizeBranch(b, torch.tensor([[1.0]]))])

    def test_rejects_a_marginal_from_different_data(self):
        """A materially different P(n) means the log-ratios are not comparable."""
        a = _model(seed=1)
        b = _model(seed=2, marginal=torch.ones(11))          # uniform vs peaked: TV ~0.4
        with pytest.raises(AssertionError, match="total variation from the anchor"):
            ComposedSizeDistribution([SizeBranch(a, torch.tensor([[1.0]])),
                                      SizeBranch(b, torch.tensor([[1.0]]))])

    def test_tolerates_the_sampling_noise_two_real_fits_would_differ_by(self):
        """Two models fit from the same dataset still differ in which molecules were
        dropped and how the split fell; that must not be an error."""
        base = _marginal()
        jittered = base.clone()
        jittered[4] += 0.02 * base.sum()                     # a small, realistic wobble
        a = _model(seed=1, marginal=base)
        b = _model(seed=2, marginal=jittered)
        comp = ComposedSizeDistribution([SizeBranch(a, torch.tensor([[1.0]])),
                                         SizeBranch(b, torch.tensor([[1.0]]))])
        assert torch.allclose(comp._marginal.sum(), torch.tensor(1.0), atol=1e-6)

    def test_explicit_anchor_overrides_the_first_branch(self):
        a, b = _model(seed=1), _model(seed=2)
        anchor = _marginal()
        comp = ComposedSizeDistribution([SizeBranch(a, torch.tensor([[1.0]])),
                                         SizeBranch(b, torch.tensor([[1.0]]))],
                                        marginal=anchor)
        assert torch.allclose(comp._marginal, anchor / anchor.sum(), atol=1e-6)

    def test_diagnostics_report_collapse_and_agreement(self):
        m = _model(seed=1)
        c = torch.tensor([[1.0]])
        d = ComposedSizeDistribution([SizeBranch(m, c, 1.0)]).diagnostics(1)
        # Entropy is bounded by the support size, NOT by the marginal's entropy:
        # conditioning lowers entropy on average over the true conditional, not at every
        # individual condition, and an untrained model is near-uniform over the 7
        # supported bins (log 7 = 1.95) against a peaked marginal (1.73).
        import math
        assert 0.0 < d["entropy"] <= math.log(int(m.support.sum())) + 1e-6
        assert 0.0 <= d["agreement"] <= 1.0 + 1e-6
        assert 5 <= d["modal_size"] <= 15
        assert m.min_size <= d["mean_size"] <= m.max_size
        # Up-weighting extrapolates away from the marginal and, as w grows, concentrates
        # on argmax(log p_i - log P) -- so a large weight must reduce entropy.
        sharp = ComposedSizeDistribution([SizeBranch(m, c, 12.0)]).diagnostics(1)
        assert sharp["entropy"] < d["entropy"]

    def test_log_prob_agrees_with_log_pmf(self):
        a, b = _model(seed=1), _model(seed=2)
        comp = ComposedSizeDistribution(
            [SizeBranch(a, torch.tensor([[1.0]]), 1.0),
             SizeBranch(b, torch.tensor([[3.0]]), 1.0)])
        lq = comp.log_pmf(1)
        assert torch.allclose(comp.log_prob(torch.tensor([9])), lq[:, 4], atol=1e-6)


class TestFitSizeModel:
    def test_recovers_a_real_dependence_and_rejects_a_null_one(self):
        from defog.core.property_head import fit_size_model

        torch.manual_seed(0)
        n_mol = 4000
        y = torch.randn(n_mol) * 2 + 3
        sizes = (18 + 1.6 * y + torch.randn(n_mol) * 2.5).round().clamp(6, 38).long()

        _, good = fit_size_model(y, sizes, min_size=6, max_size=38, epochs=40,
                                 batch_size=512, hidden=64)
        # an informative property must beat the marginal, and must do so by moving the
        # centre (shrink well under 1) rather than by accident
        assert good["gain_nats"] > 0.15, good
        assert good["shrink"] < 0.85, good

        _, null = fit_size_model(torch.randn(n_mol), sizes, min_size=6, max_size=38,
                                 epochs=40, batch_size=512, hidden=64)
        # a property carrying no size information must NOT look like an improvement
        assert null["gain_nats"] < 0.05, null
        assert null["shrink"] > 0.95, null

    def test_metrics_are_finite_when_val_holds_an_unseen_size(self):
        """Regression: a size present only in validation must not give an infinite NLL.

        The marginal is add-one smoothed across the declared grid precisely so that a
        size the training subsample happened to miss stays reachable."""
        from defog.core.property_head import fit_size_model

        torch.manual_seed(0)
        y = torch.randn(500)
        sizes = torch.full((500,), 10, dtype=torch.long)
        sizes[0] = 30                                    # a size that may land in val only
        _, m = fit_size_model(y, sizes, min_size=6, max_size=38, epochs=5, hidden=16)
        assert all(torch.isfinite(torch.tensor(m[k]))
                   for k in ("nll_val", "nll_marginal", "gain_nats"))


class TestLearnedSizeModelIntegration:
    """The learned distribution actually driving generation.

    Everything above tests the distribution in isolation. These tests answer the
    different question of whether a size drawn from it survives the trip through
    ``DeFoGModel.sample`` and comes out as the node count of a real generated graph --
    which is the only thing that makes the feature worth anything.
    """

    @staticmethod
    def _fitted(target_size_for, lo=3, hi=12, n=3000, seed=0):
        """A size model genuinely trained so that condition -> size, not a random one."""
        from defog.core.property_head import fit_size_model

        torch.manual_seed(seed)
        y = torch.rand(n) * 4.0 - 2.0                       # condition in [-2, 2]
        sizes = target_size_for(y).round().clamp(lo, hi).long()
        model, metrics = fit_size_model(y, sizes, min_size=lo, max_size=hi, epochs=60,
                                        batch_size=256, hidden=64, val_frac=0.1)
        assert metrics["gain_nats"] > 0.2, metrics       # the fit must have worked
        return model

    def test_generated_graphs_take_the_learned_size(self, small_model):
        """A low and a high condition must produce visibly different molecule sizes."""
        small_model.eval()
        model = self._fitted(lambda y: 7.5 + 2.0 * y)     # size tracks the condition

        def mean_size(condition):
            samples = small_model.sample(
                num_samples=12, size_dist=model,
                condition=torch.full((12, 1), condition),
                sample_steps=3, show_progress=False,
            )
            return sum(s.x.shape[0] for s in samples) / len(samples), samples

        low_mean, low = mean_size(-1.8)
        high_mean, high = mean_size(1.8)
        for s in low + high:
            assert 3 <= s.x.shape[0] <= 12                # inside the model's own grid
            assert s.x.shape[0] <= small_model.max_nodes
        # the whole point: the condition moved the size of the graphs that came out
        assert high_mean > low_mean + 1.5, (low_mean, high_mean)

    def test_composed_distribution_drives_generation(self, small_model):
        """Two branches, composed, still yield graphs of a sane size."""
        small_model.eval()
        a = self._fitted(lambda y: 7.5 + 2.0 * y, seed=1)
        b = self._fitted(lambda y: 7.5 + 2.0 * y, seed=2)
        # both branches asked for a LARGE molecule; the product should agree on large
        comp = ComposedSizeDistribution(
            [SizeBranch(a, torch.full((10, 1), 1.8), 1.0),
             SizeBranch(b, torch.full((10, 1), 1.8), 1.0)], mode="product")
        samples = small_model.sample(num_samples=10, size_dist=comp,
                                     sample_steps=3, show_progress=False)
        assert len(samples) == 10
        sizes = [s.x.shape[0] for s in samples]
        assert all(3 <= n <= 12 for n in sizes), sizes
        assert sum(sizes) / len(sizes) > 8.0, sizes       # both branches wanted big

    def test_swapping_in_the_learned_dist_changes_nothing_else(self, small_model):
        """Same seed, marginal vs learned: only the sizes may differ, not the plumbing."""
        small_model.eval()
        model = self._fitted(lambda y: 7.5 + 2.0 * y)
        for size_dist in (None, model):
            torch.manual_seed(7)
            samples = small_model.sample(
                num_samples=5, size_dist=size_dist,
                condition=torch.full((5, 1), 0.5) if size_dist is not None else None,
                sample_steps=3, show_progress=False,
            )
            assert len(samples) == 5
            for s in samples:
                assert s.x.shape[0] >= 1
                assert s.edge_index.shape[0] == 2
