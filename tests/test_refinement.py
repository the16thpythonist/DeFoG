"""
Tests for RefinementSampler: SDEdit-style "seed from a guess, denoise the tail".

Covers the t_start -> start_step schedule mapping, the node-count-preserving
refinement invariant, guess-encoding validation, conditional refinement, and a
regression that the base sampler's new start_step offset defaults to full
generation.
"""

import pytest
import torch
from torch_geometric.data import Data

from defog.core import Sampler, RefinementSampler


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _guess(n, edge_prob=0.3, dx=4, de=2):
    """A rough undirected guess graph with one-hot node/edge features."""
    x = torch.zeros(n, dx)
    x[torch.arange(n), torch.randint(0, dx, (n,))] = 1
    adj = (torch.rand(n, n) < edge_prob).float()
    adj = ((adj + adj.t()) > 0).float()
    adj.fill_diagonal_(0)
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    edge_attr = torch.zeros(edge_index.size(1), de)
    edge_attr[:, 1] = 1  # all present edges are the "edge" class
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _edgeless_guess(n, dx=4, de=2):
    x = torch.zeros(n, dx)
    x[torch.arange(n), torch.randint(0, dx, (n,))] = 1
    return Data(
        x=x,
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, de)),
    )


# ---------------------------------------------------------------------------
# schedule mapping
# ---------------------------------------------------------------------------
class TestStartStep:
    def test_mapping_and_clamp(self, small_model):
        # sample_steps=10: t_start scales the offset; clamped to [1, steps-1].
        assert RefinementSampler(small_model, t_start=0.2, sample_steps=10).start_step == 2
        assert RefinementSampler(small_model, t_start=0.8, sample_steps=10).start_step == 8
        # near 0 clamps up to 1 (always run >= 1 step, never step 0)
        assert RefinementSampler(small_model, t_start=0.01, sample_steps=10).start_step == 1
        # near 1 clamps down to steps-1 (always leave >= 1 step to run)
        assert RefinementSampler(small_model, t_start=0.999, sample_steps=10).start_step == 9

    def test_low_t_start_runs_more_steps(self, small_model):
        lo = RefinementSampler(small_model, t_start=0.2, sample_steps=10).start_step
        hi = RefinementSampler(small_model, t_start=0.8, sample_steps=10).start_step
        assert lo < hi  # lower trust -> earlier start -> more denoising steps

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_t_start_raises(self, small_model, bad):
        with pytest.raises(ValueError):
            RefinementSampler(small_model, t_start=bad)


# ---------------------------------------------------------------------------
# end-to-end refinement
# ---------------------------------------------------------------------------
class TestRefine:
    def test_runs_and_preserves_node_counts(self, small_model):
        torch.manual_seed(0)
        guesses = [_guess(4), _guess(7), _edgeless_guess(5), _guess(10)]
        out = RefinementSampler(small_model, t_start=0.7, sample_steps=5).refine(
            guesses, show_progress=False
        )
        assert len(out) == len(guesses)
        for g, r in zip(guesses, out):
            # refinement keeps the node set fixed; only types/edges may change
            assert r.x.size(0) == g.x.size(0)
            assert r.x.size(-1) == small_model.num_node_classes
            assert r.edge_attr.size(-1) == small_model.num_edge_classes

    def test_high_t_start_light_touch_still_valid_output(self, small_model):
        # t_start near 1 runs a single denoise step; output must still be well-formed.
        torch.manual_seed(1)
        guesses = [_guess(6) for _ in range(3)]
        out = RefinementSampler(small_model, t_start=0.95, sample_steps=5).refine(
            guesses, show_progress=False
        )
        assert len(out) == 3
        assert all(r.x.size(0) == 6 for r in out)

    def test_omega_eta_compose(self, small_model):
        # the inherited sampling knobs must still be honoured on the refine path
        torch.manual_seed(2)
        guesses = [_guess(5), _guess(6)]
        out = RefinementSampler(
            small_model, t_start=0.5, sample_steps=5, eta=10.0, omega=0.1
        ).refine(guesses, show_progress=False)
        assert len(out) == 2


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
class TestGuessValidation:
    def test_empty_guesses_raise(self, small_model):
        with pytest.raises(TypeError):
            RefinementSampler(small_model, sample_steps=5).refine([], show_progress=False)

    def test_wrong_node_class_width_raises(self, small_model):
        # 5 node classes but the model expects 4
        bad = [_guess(4, dx=5)]
        with pytest.raises(ValueError, match="num_node_classes"):
            RefinementSampler(small_model, sample_steps=5).refine(bad, show_progress=False)

    def test_oversize_guess_raises(self, small_model):
        # small_model.max_nodes == 15
        big = [_guess(small_model.max_nodes + 1)]
        with pytest.raises(ValueError, match="max_nodes"):
            RefinementSampler(small_model, sample_steps=5).refine(big, show_progress=False)


# ---------------------------------------------------------------------------
# conditional
# ---------------------------------------------------------------------------
class TestConditionalRefine:
    def test_conditional_refine_runs(self, small_cond_model, cond_dim):
        torch.manual_seed(3)
        guesses = [_guess(5), _guess(6), _guess(7)]
        condition = torch.randn(len(guesses), cond_dim)
        out = RefinementSampler(small_cond_model, t_start=0.6, sample_steps=5).refine(
            guesses, condition=condition, show_progress=False
        )
        assert len(out) == 3


# ---------------------------------------------------------------------------
# regression: base sampler unaffected by the start_step offset
# ---------------------------------------------------------------------------
class TestBaseSamplerRegression:
    def test_base_start_step_defaults_zero(self, small_model):
        assert Sampler(small_model, sample_steps=5).start_step == 0

    def test_base_generation_still_works(self, small_model):
        out = Sampler(small_model, sample_steps=5).sample(3, show_progress=False)
        assert len(out) == 3
