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
