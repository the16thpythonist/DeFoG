"""
Correctness tests for the frozen-base AdaLN/FiLM CFG-adapter feature.

Run directly (prints PASS/FAIL):
    PYTHONPATH=. .venv/bin/python tests/test_adapter.py
"""
import os
import sys
import tempfile

import torch
from torch_geometric.loader import DataLoader

from experiments.utils import build_encoders, smiles_to_pyg_data
from defog.core import (DeFoGModel, AdaLNAdapter, AdapterComposition, ConditionBranch,
                        AdaptedSampler, Sampler, FeynmanKacSampler)
from defog.core.data import to_dense


def build_tiny_model():
    atom_enc, atom_dec, bond_enc, bond_dec = build_encoders(["C", "N", "O"], ["SINGLE", "DOUBLE"])
    smis = ["CCO", "CCN", "CCC", "CNO", "OCC", "NCC"]
    graphs = [smiles_to_pyg_data(s, atom_enc, bond_enc) for s in smis]
    graphs = [g for g in graphs if g is not None]
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


def a_noisy(model, loader):
    batch = next(iter(loader))
    dense, mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    dense = dense.mask(mask)
    X1, E1 = dense.X, dense.E
    bs = X1.size(0)
    y0 = torch.zeros(bs, 0)
    torch.manual_seed(0)
    noisy = model._apply_noise(X1, E1, y0, mask)
    extra = model._compute_extra_data(noisy)
    return noisy, extra, mask, bs


def test_null_equals_base():
    """A fresh (gate zero-init) adapter modulation must be an EXACT no-op."""
    model, loader = build_tiny_model()
    noisy, extra, mask, bs = a_noisy(model, loader)
    adapter = AdaLNAdapter.for_base(model, cond_dim=2, hidden=32, time_conditioned=True).eval()
    pred_base = model.forward(noisy, extra, mask)
    c = torch.randn(bs, 2)
    mod = adapter(c, t=noisy["t"])
    pred_mod = model.forward(noisy, extra, mask, cond_modulation=mod)
    okX = torch.allclose(pred_base.X, pred_mod.X, atol=1e-6)
    okE = torch.allclose(pred_base.E, pred_mod.E, atol=1e-6)
    assert okX and okE, f"null!=base (X {okX}, E {okE}, maxdX={ (pred_base.X-pred_mod.X).abs().max()})"
    return "null=base exact (gate zero-init)"


def test_modulation_actually_moves_after_perturb():
    """After perturbing the gate, the modulation must change the output (sanity:
    the wiring is live, not silently ignored)."""
    model, loader = build_tiny_model()
    noisy, extra, mask, bs = a_noisy(model, loader)
    adapter = AdaLNAdapter.for_base(model, cond_dim=2, hidden=32).eval()
    with torch.no_grad():
        for g in adapter.gate:
            for s in g:
                g[s].bias.add_(0.5)  # un-zero the gate
    pred_base = model.forward(noisy, extra, mask)
    mod = adapter(torch.randn(bs, 2), t=noisy["t"])
    pred_mod = model.forward(noisy, extra, mask, cond_modulation=mod)
    moved = (pred_base.X - pred_mod.X).abs().max().item()
    assert moved > 1e-4, f"perturbed adapter did not move output (maxdX={moved})"
    return f"adapter is live (maxdX={moved:.3g} after gate perturb)"


def test_batched_composition_bypass():
    """The (N+1)*bs batched forward with zero-init adapters: every group's
    prediction must equal the base's unconditional prediction."""
    model, loader = build_tiny_model()
    noisy, extra, mask, bs = a_noisy(model, loader)
    a1 = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval()
    a2 = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval()
    comp = AdapterComposition(
        [ConditionBranch(a1, torch.zeros(bs, 1), 1.0), ConditionBranch(a2, torch.ones(bs, 1), 1.0)],
        base=model, mode="mean")
    import torch.nn.functional as F
    rep = len(comp) + 1
    mod = comp.build_modulation(bs, noisy["t"])
    nd = {"X_t": noisy["X_t"].repeat(rep, 1, 1), "E_t": noisy["E_t"].repeat(rep, 1, 1, 1),
          "y_t": noisy["y_t"].repeat(rep, 1), "t": noisy["t"].repeat(rep, 1),
          "node_mask": mask.repeat(rep, 1)}
    from defog.core.data import PlaceHolder
    extra_b = PlaceHolder(X=extra.X.repeat(rep, 1, 1), E=extra.E.repeat(rep, 1, 1, 1), y=extra.y.repeat(rep, 1))
    pred = model.forward(nd, extra_b, nd["node_mask"], cond_modulation=mod)
    pX = F.softmax(pred.X, -1).view(rep, bs, *pred.X.shape[1:])
    base_pred = F.softmax(model.forward(noisy, extra, mask).X, -1)
    for g in range(rep):
        assert torch.allclose(pX[g], base_pred, atol=1e-6), f"group {g} != base uncond"
    return f"batched (N+1)*bs bypass: all {rep} groups == base uncond"


def test_empty_composition_fallthrough():
    """AdaptedSampler with an EMPTY composition must be sample-identical to a plain
    Sampler under a fixed seed (falls through to the untouched legacy body)."""
    model, loader = build_tiny_model()
    torch.manual_seed(123)
    s1 = Sampler(model, sample_steps=5, eta=0.0, omega=0.0).sample(4, device="cpu", show_progress=False)
    torch.manual_seed(123)
    s2 = AdaptedSampler(model, AdapterComposition([]), sample_steps=5, eta=0.0, omega=0.0).sample(
        4, device="cpu", show_progress=False)
    ok = len(s1) == len(s2) and all(
        torch.equal(a.x, b.x) and torch.equal(a.edge_index, b.edge_index)
        for a, b in zip(s1, s2))
    assert ok, "empty-composition AdaptedSampler != plain Sampler"
    return "empty composition falls through == plain Sampler (fixed seed)"


def test_adapted_sampler_runs_and_steers_shapewise():
    """A 2-branch composition samples without error and returns valid graphs."""
    model, loader = build_tiny_model()
    a1 = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval()
    a2 = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval()
    comp = AdapterComposition(
        [ConditionBranch(a1, torch.tensor([0.5]), 2.0), ConditionBranch(a2, torch.tensor([-0.5]), 2.0)],
        base=model, mode="mean")
    samp = AdaptedSampler(model, comp, sample_steps=5, eta=0.0, omega=0.0)
    out = samp.sample(4, device="cpu", show_progress=False)
    assert len(out) == 4 and all(d.x.size(0) >= 1 for d in out)
    return f"AdaptedSampler(2 branches) sampled {len(out)} graphs"


def test_save_load_roundtrip():
    model, loader = build_tiny_model()
    a = AdaLNAdapter.for_base(model, cond_dim=3, hidden=32,
                              cond_mean=[0.1, 0.2, 0.3], cond_std=[1.0, 2.0, 0.5], name="tst").eval()
    with torch.no_grad():
        for g in a.gate:
            for s in g:
                g[s].weight.add_(torch.randn_like(g[s].weight) * 0.01)
    c, t = torch.randn(2, 3), torch.rand(2, 1)
    m0 = a(c, t=t).layers[0]["gateX"]
    with tempfile.TemporaryDirectory() as d:
        p = a.save(os.path.join(d, "ad"))
        b = AdaLNAdapter.load(p)
    m1 = b(c, t=t).layers[0]["gateX"]
    assert torch.allclose(m0, m1, atol=1e-6), "save/load changed modulation"
    assert b.name == "tst" and torch.allclose(b.cond_std, torch.tensor([1.0, 2.0, 0.5]))
    return "save/load round-trip reproduces modulation + stats"


def test_interior_null_equals_base():
    """Interior (L4/L10) adapters must ALSO be exact no-ops at null (gate zero-init)."""
    model, loader = build_tiny_model()
    noisy, extra, mask, bs = a_noisy(model, loader)
    pred_base = model.forward(noisy, extra, mask)
    for kw, tag in [(dict(interior_ff=True), "L4"),
                    (dict(interior_attn=True), "L10"),
                    (dict(interior_ff=True, interior_attn=True), "L4+L10")]:
        adapter = AdaLNAdapter.for_base(model, cond_dim=2, hidden=32, **kw).eval()
        mod = adapter(torch.randn(bs, 2), t=noisy["t"])
        pred = model.forward(noisy, extra, mask, cond_modulation=mod)
        dX = (pred_base.X - pred.X).abs().max()
        assert torch.allclose(pred_base.X, pred.X, atol=1e-6) and \
               torch.allclose(pred_base.E, pred.E, atol=1e-6), f"{tag} not no-op (maxdX={dX})"
    return "interior L4/L10/both = base exact at null"


def test_interior_live():
    """Perturbing the interior gates must move the output (wiring is live, not ignored)."""
    model, loader = build_tiny_model()
    noisy, extra, mask, bs = a_noisy(model, loader)
    pred_base = model.forward(noisy, extra, mask)
    moved = {}
    for kw, tag, attr in [(dict(interior_ff=True), "L4", "ff"),
                          (dict(interior_attn=True), "L10", "attn")]:
        adapter = AdaLNAdapter.for_base(model, cond_dim=2, hidden=32, **kw).eval()
        with torch.no_grad():
            for ld in getattr(adapter, attr):
                for k in ld:
                    if k.startswith("gate"):
                        ld[k].bias.add_(0.5)
        mod = adapter(torch.randn(bs, 2), t=noisy["t"])
        pred = model.forward(noisy, extra, mask, cond_modulation=mod)
        m = (pred_base.X - pred.X).abs().max().item()
        # live threshold 1e-5 is 10x above the verified <1e-6 no-op floor; the L10
        # attention-logit path is more dampened than L4's direct node FiLM in the toy model.
        assert m > 1e-5, f"{tag} did not move output (maxdX={m})"
        moved[tag] = m
    return f"interior live: L4 dX={moved['L4']:.2g}, L10 dX={moved['L10']:.2g}"


def test_stack_groups_heterogeneous():
    """stack_groups must handle adapters with DIFFERENT key sets (interior vs output-
    only) in BOTH orders: no crash, union keys, (N+1)*bs rows, group-0 zero."""
    from defog.core.adapter import Modulation
    model, loader = build_tiny_model()
    noisy, extra, mask, bs = a_noisy(model, loader)
    a_out = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval()
    a_int = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32, interior_ff=True, interior_attn=True).eval()
    for order in ([a_int, a_out], [a_out, a_int]):
        mods = [ad(torch.zeros(bs, 1), t=noisy["t"]) for ad in order]
        stk = Modulation.stack_groups(mods, bs, "cpu")
        want = set(mods[0].layers[0]) | set(mods[1].layers[0])
        assert set(stk.layers[0]) == want, "stack_groups did not union keys"
        for k, v in stk.layers[0].items():
            assert v.shape[0] == (len(order) + 1) * bs, f"{k} wrong batch dim"
            assert torch.allclose(v[:bs], torch.zeros_like(v[:bs])), f"group-0 not zero for {k}"
    return "stack_groups unions heterogeneous keys + group-0 bypass (both orders)"


def test_interior_composability_bypass():
    """Compose an INTERIOR adapter with an OUTPUT-ONLY adapter (the swappable+stackable
    case that crashed pre-fix): every group of the (N+1)*bs forward must == base uncond."""
    import torch.nn.functional as F
    from defog.core.data import PlaceHolder
    model, loader = build_tiny_model()
    noisy, extra, mask, bs = a_noisy(model, loader)
    a_out = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval()
    a_int = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32, interior_ff=True, interior_attn=True).eval()
    comp = AdapterComposition(
        [ConditionBranch(a_int, torch.zeros(bs, 1), 1.0), ConditionBranch(a_out, torch.ones(bs, 1), 1.0)],
        base=model, mode="mean")
    rep = len(comp) + 1
    mod = comp.build_modulation(bs, noisy["t"])
    nd = {"X_t": noisy["X_t"].repeat(rep, 1, 1), "E_t": noisy["E_t"].repeat(rep, 1, 1, 1),
          "y_t": noisy["y_t"].repeat(rep, 1), "t": noisy["t"].repeat(rep, 1),
          "node_mask": mask.repeat(rep, 1)}
    extra_b = PlaceHolder(X=extra.X.repeat(rep, 1, 1), E=extra.E.repeat(rep, 1, 1, 1), y=extra.y.repeat(rep, 1))
    pred = model.forward(nd, extra_b, nd["node_mask"], cond_modulation=mod)
    pX = F.softmax(pred.X, -1).view(rep, bs, *pred.X.shape[1:])
    base_pred = F.softmax(model.forward(noisy, extra, mask).X, -1)
    for g in range(rep):
        assert torch.allclose(pX[g], base_pred, atol=1e-6), f"group {g} != base (heterogeneous compose)"
    return f"heterogeneous compose (interior+output-only): all {rep} groups == base"


def test_interior_save_load():
    """Interior adapter round-trips: flags in config, heads in state_dict."""
    model, loader = build_tiny_model()
    a = AdaLNAdapter.for_base(model, cond_dim=2, hidden=32, interior_ff=True,
                              interior_attn=True, name="int").eval()
    with torch.no_grad():
        for ld in a.attn:
            ld["gate"].weight.add_(torch.randn_like(ld["gate"].weight) * 0.01)
    c, t = torch.randn(2, 2), torch.rand(2, 1)
    m0 = a(c, t=t).layers[0]["gate_emul"]
    with tempfile.TemporaryDirectory() as d:
        p = a.save(os.path.join(d, "ai"))
        b = AdaLNAdapter.load(p)
    assert b.interior_ff and b.interior_attn, "interior flags lost on load"
    m1 = b(c, t=t).layers[0]["gate_emul"]
    assert torch.allclose(m0, m1, atol=1e-6), "interior save/load changed modulation"
    return "interior adapter save/load round-trip (flags + heads)"


def test_fk_over_adapter_runs():
    """FeynmanKacSampler with an AdapterComposition as the proposal (FK refinement over
    adapter conditioning): the composition is wired in, and sampling returns valid graphs."""
    model, loader = build_tiny_model()
    a = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval()
    comp = AdapterComposition([ConditionBranch(a, torch.tensor([0.5]), 1.0)], base=model, mode="product")
    energy = lambda X1, E1, node_mask: E1[..., 1:].sum(dim=(1, 2, 3)).float()  # toy per-graph energy
    fk = FeynmanKacSampler(model, energy, beta=1.0, warmup_frac=0.4, sample_steps=6,
                           eta=0.0, omega=0.0, composition=comp)
    assert fk.composition is comp and "+adapter" in fk._desc(), "composition not wired into FK"
    out = fk.sample(4, device="cpu", show_progress=False)
    assert len(out) == 4 and all(d.x.size(0) >= 1 for d in out)
    return f"FK over adapter composition runs ({fk._desc()}, {len(out)} graphs)"


if __name__ == "__main__":
    tests = [
        test_null_equals_base,
        test_modulation_actually_moves_after_perturb,
        test_batched_composition_bypass,
        test_empty_composition_fallthrough,
        test_adapted_sampler_runs_and_steers_shapewise,
        test_save_load_roundtrip,
        test_interior_null_equals_base,
        test_interior_live,
        test_stack_groups_heterogeneous,
        test_interior_composability_bypass,
        test_interior_save_load,
        test_fk_over_adapter_runs,
    ]
    fails = 0
    for t in tests:
        try:
            msg = t()
            print(f"PASS  {t.__name__}: {msg}")
        except Exception as e:
            fails += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
