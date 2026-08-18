"""Blending branches in probability space instead of rate space.

`AdapterComposition.blend_space` selects WHERE the product-of-experts blend is applied:

  "prob"  blend the clean-graph marginals, derive ONE rate matrix from the result
          (the default since 2026-08-17)
  "rate"  build a rate matrix per branch, blend those
          (historical; retained so pre-2026-08-17 numbers reproduce)

The form of the blend (geometric / PoE) is identical either way, so these tests are
about placement only.

The load-bearing test is `test_single_branch_w1_agrees_with_rate_space`: with one branch
at w=1 both paths reduce algebraically to R(p_cond), so they MUST agree. If they do not,
the prob-space path is wrong. Everything else follows from that.
"""

import pytest
import torch

from defog.core import (AdaLNAdapter, AdapterComposition, AdaptedSampler,
                        ConditionBranch, DeFoGModel)
from defog.core.data import to_dense

from experiments.utils import build_encoders, smiles_to_pyg_data
from torch_geometric.loader import DataLoader


def build_tiny_model():
    atom_enc, atom_dec, bond_enc, bond_dec = build_encoders(["C", "N", "O"], ["SINGLE", "DOUBLE"])
    smis = ["CCO", "CCN", "CCC", "CNO", "OCC", "NCC"]
    graphs = [g for g in (smiles_to_pyg_data(s, atom_enc, bond_enc) for s in smis) if g is not None]
    loader = DataLoader(graphs, batch_size=3, shuffle=False)
    model = DeFoGModel.from_dataloader(
        loader, n_layers=2, hidden_dim=32, hidden_mlp_dim=64, n_heads=2, dropout=0.0,
        noise_type="marginal", extra_features_type="rrwp", rrwp_steps=3,
        molecular_features=False,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, loader


def a_state(model, loader, seed=0):
    """One noisy graph state to step from."""
    batch = next(iter(loader))
    dense, mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    dense = dense.mask(mask)
    bs = dense.X.size(0)
    torch.manual_seed(seed)
    noisy = model._apply_noise(dense.X, dense.E, torch.zeros(bs, 0), mask)
    return noisy, mask, bs


def a_trained_adapter(model, seed=0):
    """An adapter whose gates are NOT zero -- a zero-gate adapter is an exact no-op,
    which would make every blend trivially agree and prove nothing."""
    torch.manual_seed(seed)
    adapter = AdaLNAdapter.for_base(model, cond_dim=1, hidden=16, time_conditioned=True).eval()
    with torch.no_grad():
        for lay in adapter.gate:
            for s in lay:
                lay[s].weight.normal_(0, 0.05)
                lay[s].bias.normal_(0, 0.05)
    return adapter


def step(model, comp, noisy, mask, t_val=0.5, dt=0.05):
    bs = noisy["X_t"].size(0)
    t = torch.full((bs, 1), t_val)
    s = t + dt
    torch.manual_seed(123)                      # the step itself samples; pin it
    return model.denoise_step(t, s, noisy["X_t"], noisy["E_t"], noisy["y_t"], mask,
                              eta=0.0, omega=0.0, composition=comp)


def blended_rate(model, comp, noisy, mask, t_val=0.5, pin_draw=True):
    """The blended rate matrix a step would actually use.

    ``compute_rate_matrices`` IS stochastic -- it draws its own X_1 sample internally.
    The rate path calls it N+1 times and the prob path once, so the two consume the RNG
    differently and their outputs differ even where the algebra says they should not.
    ``pin_draw`` reseeds immediately before every call so all of them share one draw,
    which is the only way to compare the two placements rather than two noise
    realisations.
    """
    bs = noisy["X_t"].size(0)
    t = torch.full((bs, 1), t_val)
    designer = model.rate_matrix_designer
    real = designer.compute_rate_matrices
    if pin_draw:
        def pinned(*a, **kw):
            torch.manual_seed(4242)
            return real(*a, **kw)
        designer.compute_rate_matrices = pinned
    got = {}
    real_step = model._compute_step_probs

    def spy(RX, RE, X_t, E_t, dt):
        got["RX"], got["RE"] = RX.clone(), RE.clone()
        return real_step(RX, RE, X_t, E_t, dt)

    model._compute_step_probs = spy
    try:
        torch.manual_seed(123)
        model.denoise_step(t, t + 0.05, noisy["X_t"], noisy["E_t"], noisy["y_t"], mask,
                           eta=0.0, omega=0.0, composition=comp)
    finally:
        model._compute_step_probs = real_step
        designer.compute_rate_matrices = real
    return got["RX"], got["RE"]


class TestPlacementEquivalence:
    def test_at_w1_the_paths_differ_only_by_the_base_veto(self):
        """THE correctness test, and it is not the equivalence I first assumed.

        At w=1 the rate-space blend is exp(log R_unc + (log R_cond - log R_unc)) = R_cond
        -- EXCEPT for the last line of _blend_rates, `where(R[0] == 0, 0, Rb)`, which
        forces every transition the UNCONDITIONAL model assigns zero rate to stay zero.
        The prob-space path derives one rate matrix from the blended marginals and carries
        no such veto.

        So the two agree exactly on entries the base allows, and differ exactly on the
        entries it forbids. That is a real semantic change, not a rounding difference:
        prob-space blending lets the adapter re-open transitions the frozen base has
        ruled out.
        """
        model, loader = build_tiny_model()
        noisy, mask, bs = a_state(model, loader)
        adapter = a_trained_adapter(model)
        cond = torch.full((bs, 1), 0.7)

        Rr, _ = blended_rate(model, AdapterComposition(
            [ConditionBranch(adapter, cond, 1.0)], blend_space="rate"), noisy, mask)
        Rp, _ = blended_rate(model, AdapterComposition(
            [ConditionBranch(adapter, cond, 1.0)], blend_space="prob"), noisy, mask)

        vetoed = Rr == 0                       # what the rate path zeroed out
        allowed = ~vetoed
        assert torch.allclose(Rr[allowed], Rp[allowed], atol=1e-4), \
            "on entries the base allows, w=1 must give the same rate matrix"

    def test_the_difference_is_distributional_not_pointwise(self):
        """Why the equivalence test above has to pin the RNG, and what it hides.

        `rate_matrix.py` builds R from a DISCRETE SAMPLE of X_1, not from the marginal:
        `sampled = sample_from_probs(X_1_pred, E_1_pred, node_mask)`. So R is a step
        function of p, and pinning one uniform draw makes it locally constant -- which is
        exactly why the pinned comparison collapses to equality.

        The real difference is which distribution X_1 is drawn from:
          rate  -- draw from p_uncond AND from p_cond, geometrically average the two
                   resulting rate matrices
          prob  -- blend the marginals, draw ONCE from the guided distribution
        The second is what classifier-free guidance means. At w=1 the guided distribution
        IS p_cond, so both reduce to one draw from p_cond and the two are distributionally
        identical. At w!=1 they are not.
        """
        model, loader = build_tiny_model()
        noisy, mask, bs = a_state(model, loader)
        adapter = a_trained_adapter(model)
        cond = torch.full((bs, 1), 0.7)

        def mean_rate(space, w, draws=60):
            comp = AdapterComposition([ConditionBranch(adapter, cond, w)], blend_space=space)
            acc = None
            for s in range(draws):
                R, _ = blended_rate(model, comp, noisy, mask, pin_draw=False)
                acc = R.clone() if acc is None else acc + R
            return acc / draws

        torch.manual_seed(0)
        same_w1 = (mean_rate("rate", 1.0) - mean_rate("prob", 1.0)).abs().max()
        torch.manual_seed(0)
        diff_w3 = (mean_rate("rate", 3.0) - mean_rate("prob", 3.0)).abs().max()

        # w=1: both are one draw from p_cond, so the expected rate matrices agree.
        # w=3: genuinely different computations, so they must not.
        assert diff_w3 > same_w1, (
            f"expected the placements to diverge at w=3 (got {diff_w3:.4f}) more than they "
            f"do at w=1 (got {same_w1:.4f}); if not, the flag is doing nothing")

    def test_rate_path_spends_more_random_draws_than_prob_path(self):
        """The estimator-noise argument, measured rather than asserted.

        compute_rate_matrices draws an X_1 sample per call. Blending N+1 rate matrices
        therefore mixes N+1 independent draws; blending marginals first spends one."""
        model, loader = build_tiny_model()
        noisy, mask, bs = a_state(model, loader)
        adapter = a_trained_adapter(model)
        cond = torch.full((bs, 1), 0.7)
        designer = model.rate_matrix_designer
        real = designer.compute_rate_matrices
        counts = {}
        for space in ("rate", "prob"):
            n = [0]

            def counted(*a, _n=n, **kw):
                _n[0] += 1
                return real(*a, **kw)

            designer.compute_rate_matrices = counted
            try:
                comp = AdapterComposition([ConditionBranch(adapter, cond, 1.0)],
                                          blend_space=space)
                t = torch.full((bs, 1), 0.5)
                torch.manual_seed(123)
                model.denoise_step(t, t + 0.05, noisy["X_t"], noisy["E_t"], noisy["y_t"],
                                   mask, eta=0.0, omega=0.0, composition=comp)
            finally:
                designer.compute_rate_matrices = real
            counts[space] = n[0]
        assert counts["rate"] == 2, counts        # uncond + one conditional branch
        assert counts["prob"] == 1, counts


class TestDefaults:
    def test_default_is_prob_space(self):
        """Flipped on 2026-08-17. This assertion IS the ship decision: prob-space
        blending was measured, recorded in PLAN.md Wave 2, and then left unmerged for
        three days while every downstream caller kept silently using rate space."""
        assert AdapterComposition([]).blend_space == "prob"

    def test_default_branch_weight_is_two(self):
        """The weight default moved with the blend space, and only makes sense with it.
        w=2 in rate space collapses logP to MAE 5.59 / validity 0.526; in prob space it
        is the optimum (0.5420 against 0.6410 at w=1)."""
        model, _ = build_tiny_model()
        adapter = a_trained_adapter(model)
        assert ConditionBranch(adapter, torch.tensor([[0.7]])).weight == 2.0

    def test_default_path_is_byte_identical_to_explicit_prob(self):
        model, loader = build_tiny_model()
        noisy, mask, bs = a_state(model, loader)
        adapter = a_trained_adapter(model)
        cond = torch.full((bs, 1), 0.7)
        default = AdapterComposition([ConditionBranch(adapter, cond, 1.5)])
        explicit = AdapterComposition([ConditionBranch(adapter, cond, 1.5)], blend_space="prob")
        Xa, Ea, _ = step(model, default, noisy, mask)
        Xb, Eb, _ = step(model, explicit, noisy, mask)
        assert torch.equal(Xa, Xb) and torch.equal(Ea, Eb)

    def test_rate_space_is_still_reachable(self):
        """Reproducing a pre-2026-08-17 number requires the old path to still run.

        Reachability only. It is tempting to also assert the two paths give DIFFERENT
        graphs here, and that assertion fails: on this toy model a pinned single step
        puts the two rate matrices within 1e-6 and samples the same graph. That is not a
        bug, it is the point of `test_the_difference_is_distributional_not_pointwise` --
        R is built from a discrete draw, so pinning the draw makes it locally constant.
        Dispatch is proven by `test_rate_path_spends_more_random_draws_than_prob_path`
        (2 calls vs 1), which is a structural fact rather than a sampling outcome.
        """
        model, loader = build_tiny_model()
        noisy, mask, bs = a_state(model, loader)
        adapter = a_trained_adapter(model)
        cond = torch.full((bs, 1), 0.7)
        comp = AdapterComposition([ConditionBranch(adapter, cond, 2.0)], blend_space="rate")
        assert comp.blend_space == "rate"
        X, E, _ = step(model, comp, noisy, mask)
        assert torch.isfinite(X.float()).all() and torch.isfinite(E.float()).all()

    def test_rejects_unknown_blend_space(self):
        with pytest.raises(AssertionError, match="unknown blend_space"):
            AdapterComposition([], blend_space="probability")


class TestEndToEnd:
    @pytest.mark.parametrize("space", ["rate", "prob"])
    def test_sampling_produces_valid_graphs(self, space):
        model, loader = build_tiny_model()
        adapter = a_trained_adapter(model)
        comp = AdapterComposition(
            [ConditionBranch(adapter, torch.tensor([[0.7]]), 1.5)], blend_space=space)
        torch.manual_seed(7)
        out = AdaptedSampler(model, comp, sample_steps=5, eta=0.0, omega=0.0).sample(
            num_samples=3, show_progress=False)
        assert len(out) == 3
        for g in out:
            assert g.x.shape[0] >= 1
            assert g.edge_index.shape[0] == 2
            assert torch.isfinite(g.x.float()).all()

    def test_two_branch_composition_runs_in_prob_space(self):
        model, loader = build_tiny_model()
        a1, a2 = a_trained_adapter(model, seed=1), a_trained_adapter(model, seed=2)
        comp = AdapterComposition(
            [ConditionBranch(a1, torch.tensor([[0.7]]), 1.0),
             ConditionBranch(a2, torch.tensor([[-0.3]]), 1.0)], blend_space="prob")
        torch.manual_seed(7)
        out = AdaptedSampler(model, comp, sample_steps=5, eta=0.0, omega=0.0).sample(
            num_samples=3, show_progress=False)
        assert len(out) == 3
