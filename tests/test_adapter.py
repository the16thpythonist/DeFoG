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
                        AdaptedSampler, Sampler, FeynmanKacSampler,
                        FingerprintEncoder, SpectrumEncoder, GuideBranch,
                        NodeConditionCrossAttention)
from defog.core.data import to_dense
from defog.core.layers import timestep_embedding


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


def test_fingerprint_encoder_shape_and_config():
    """The encoder must reshape cond_dim -> out_dim, and _config() must rebuild it
    exactly. A _config() that drops a field would rebuild a DIFFERENT architecture and
    then fail load_state_dict -- or worse, succeed with mismatched semantics."""
    enc = FingerprintEncoder(in_dim=2048, out_dim=512, hidden=1024, n_blocks=2)
    out = enc(torch.randn(4, 2048))
    assert out.shape == (4, 512), f"expected (4,512), got {tuple(out.shape)}"

    cfg = enc._config()
    assert cfg["kind"] == "mlp", "config must carry its kind or the registry cannot dispatch"
    rebuilt = FingerprintEncoder(**{k: v for k, v in cfg.items() if k != "kind"})
    rebuilt.load_state_dict(enc.state_dict())      # fails loudly if the shapes differ
    enc.eval(); rebuilt.eval()
    c = torch.randn(3, 2048)
    assert torch.allclose(enc(c), rebuilt(c), atol=1e-6), "_config() rebuild is not faithful"

    try:
        enc(torch.randn(2, 1024))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong input width must raise, not silently broadcast")
    return "FingerprintEncoder: shape, width guard, faithful _config()"


def test_cond_encoder_registry_dispatch():
    """kind='mlp' must build a FingerprintEncoder; an unknown kind must fail loudly;
    and a config with NO kind must still build a SpectrumEncoder -- every adapter saved
    before `kind` existed depends on that default."""
    model, _ = build_tiny_model()
    a = AdaLNAdapter.for_base(model, cond_dim=64, hidden=32,
                              cond_encoder=dict(kind="mlp", in_dim=64, out_dim=16, hidden=32,
                                                n_blocks=1))
    assert isinstance(a.cond_encoder, FingerprintEncoder)

    legacy = AdaLNAdapter.for_base(model, cond_dim=8, hidden=32,
                                   cond_encoder=dict(n_bins=4, out_dim=16))
    assert isinstance(legacy.cond_encoder, SpectrumEncoder), \
        "a kind-less config must still resolve to SpectrumEncoder (backward compat)"

    try:
        AdaLNAdapter.for_base(model, cond_dim=64, hidden=32,
                              cond_encoder=dict(kind="no_such_encoder", in_dim=64))
    except ValueError as exc:
        assert "no_such_encoder" in str(exc)
    else:
        raise AssertionError("unknown encoder kind must raise, not build something else")
    return "cond_encoder registry: mlp dispatches, legacy defaults, unknown raises"


def test_encoder_adapter_null_equals_base_and_roundtrips():
    """With an encoder in front, the adapter must STILL be an exact no-op at init (the
    zero-init gates sit downstream of it), and must survive save/load -- including
    rebuilding the encoder from the serialized config rather than dropping it."""
    model, loader = build_tiny_model()
    noisy, extra, mask, bs = a_noisy(model, loader)
    a = AdaLNAdapter.for_base(model, cond_dim=64, hidden=32, interior_ff=True,
                              cond_encoder=dict(kind="mlp", in_dim=64, out_dim=16,
                                                hidden=32, n_blocks=1)).eval()
    pred_base = model.forward(noisy, extra, mask)
    c = torch.randn(bs, 64)
    pred_mod = model.forward(noisy, extra, mask, cond_modulation=a(c, t=noisy["t"]))
    assert torch.allclose(pred_base.X, pred_mod.X, atol=1e-6) and \
        torch.allclose(pred_base.E, pred_mod.E, atol=1e-6), \
        "encoder broke the exact-no-op-at-init property"

    with torch.no_grad():                       # perturb so a dropped encoder would show
        for g in a.gate:
            for s in g:
                g[s].weight.add_(torch.randn_like(g[s].weight) * 0.05)
    m0 = a(c, t=noisy["t"]).layers[0]["gateX"]
    with tempfile.TemporaryDirectory() as d:
        b = AdaLNAdapter.load(a.save(os.path.join(d, "ad")))
    b.eval()
    assert isinstance(b.cond_encoder, FingerprintEncoder), "encoder lost on load"
    m1 = b(c, t=noisy["t"]).layers[0]["gateX"]
    assert torch.allclose(m0, m1, atol=1e-6), "save/load with encoder changed modulation"
    return "encoder adapter: exact no-op at init, encoder survives save/load"


def test_zero_gate_is_noop_in_prob_space_at_any_weight():
    """The exact-no-op-at-init property must survive the new blend path, and it must
    hold at EVERY weight -- w only multiplies the log-ratio, which is identically zero
    while the gates are zero. Tested at the blend itself rather than through sampling
    so the assertion is deterministic and free of RNG-alignment confounds."""
    import torch.nn.functional as F
    from defog.core.data import PlaceHolder
    model, loader = build_tiny_model()
    noisy, extra, mask, bs = a_noisy(model, loader)
    a = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval()
    comp = AdapterComposition([ConditionBranch(a, torch.full((bs, 1), 0.7), 1.0)],
                              base=model, mode="product")
    rep = len(comp) + 1
    mod = comp.build_modulation(bs, noisy["t"])
    nd = {"X_t": noisy["X_t"].repeat(rep, 1, 1), "E_t": noisy["E_t"].repeat(rep, 1, 1, 1),
          "y_t": noisy["y_t"].repeat(rep, 1), "t": noisy["t"].repeat(rep, 1),
          "node_mask": mask.repeat(rep, 1)}
    extra_b = PlaceHolder(X=extra.X.repeat(rep, 1, 1), E=extra.E.repeat(rep, 1, 1, 1),
                          y=extra.y.repeat(rep, 1))
    pred = model.forward(nd, extra_b, nd["node_mask"], cond_modulation=mod)
    pX = F.softmax(pred.X, -1).view(rep, bs, *pred.X.shape[1:])
    p_uncond = F.log_softmax(torch.log(pX[0] + 1e-8), dim=-1)
    for w_val in (1.0, 2.0, 5.0):
        w = pX.new_tensor([w_val])
        q = F.log_softmax(model._blend_logp(pX, w, "product"), dim=-1)
        assert torch.allclose(q, p_uncond, atol=1e-5), f"zero-gate adapter moved the blend at w={w_val}"
    return "zero-gate adapter is an exact no-op in prob space at w in {1, 2, 5}"


# ===========================================================================
# Autoguidance: group 0 is a degraded conditional instead of the frozen base
# ===========================================================================
def _wake_gates(adapter, seed, scale=0.5):
    """Zero-init gates make an adapter an exact no-op. Give it real ones so it moves.

    Without this every autoguidance test passes trivially -- guide and base would be
    the same model -- which is exactly the inert-control failure `dam_result.md`
    records. The tests below that need a LIVE guide all go through here."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in adapter.gate:
            for s in layer:
                layer[s].weight.normal_(0.0, scale, generator=g)
                layer[s].bias.normal_(0.0, scale, generator=g)
    return adapter


def _blend_of(model, comp, noisy, extra, mask, bs, w_val):
    """(per-group clean marginals, PoE-blended log q) for one composition."""
    import torch.nn.functional as F
    from defog.core.data import PlaceHolder
    rep = len(comp) + 1
    mod = comp.build_modulation(bs, noisy["t"])
    nd = {"X_t": noisy["X_t"].repeat(rep, 1, 1), "E_t": noisy["E_t"].repeat(rep, 1, 1, 1),
          "y_t": noisy["y_t"].repeat(rep, 1), "t": noisy["t"].repeat(rep, 1),
          "node_mask": mask.repeat(rep, 1)}
    extra_b = PlaceHolder(X=extra.X.repeat(rep, 1, 1), E=extra.E.repeat(rep, 1, 1, 1),
                          y=extra.y.repeat(rep, 1))
    pred = model.forward(nd, extra_b, nd["node_mask"], cond_modulation=mod)
    pX = F.softmax(pred.X, -1).view(rep, bs, *pred.X.shape[1:])
    w = pX.new_tensor([w_val])
    return pX, F.log_softmax(model._blend_logp(pX, w, "product"), dim=-1)


def test_no_guide_group0_is_exactly_zero():
    """Regression on the stack_groups refactor: with no guide, every group-0 row must
    still be exactly zero, i.e. an exact bypass to the frozen base."""
    model, loader = build_tiny_model()
    noisy, extra, mask, bs = a_noisy(model, loader)
    a = _wake_gates(AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval(), seed=1)
    comp = AdapterComposition([ConditionBranch(a, torch.full((bs, 1), 0.7), 2.0)], base=model)
    mod = comp.build_modulation(bs, noisy["t"])
    n_checked = 0
    for L, d in enumerate(mod.layers):
        for k, v in d.items():
            assert torch.count_nonzero(v[:bs]) == 0, f"layer {L} key {k}: group 0 not zero"
            n_checked += 1
    assert n_checked > 0, "no modulation keys were checked"
    return f"no-guide group 0 exactly zero across {n_checked} keys"


def test_guide_at_init_is_plain_cfg():
    """THE invariant. A zero-init guide IS the frozen base, so autoguidance with an
    untrained guide must reduce to standard CFG bit-for-bit. If this ever fails, the
    guide is being wired in somewhere other than group 0."""
    model, loader = build_tiny_model()
    noisy, extra, mask, bs = a_noisy(model, loader)
    a = _wake_gates(AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval(), seed=2)
    cond = torch.full((bs, 1), 0.7)
    plain = AdapterComposition([ConditionBranch(a, cond, 2.0)], base=model)
    fresh_guide = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval()   # gates zero
    guided = AdapterComposition([ConditionBranch(a, cond, 2.0)], base=model,
                                guide=GuideBranch(fresh_guide, cond))
    pX_p, q_p = _blend_of(model, plain, noisy, extra, mask, bs, 2.0)
    pX_g, q_g = _blend_of(model, guided, noisy, extra, mask, bs, 2.0)
    assert torch.allclose(pX_p[0], pX_g[0], atol=1e-6), "group 0 moved under a zero-init guide"
    assert torch.allclose(q_p, q_g, atol=1e-6), "blend moved under a zero-init guide"
    return "zero-init guide == plain CFG (group 0 and blend both unchanged)"


def test_live_guide_changes_group0_and_the_blend():
    """A guide with real gates must move group 0 AND the blend. The paired assertion
    matters: moving group 0 without moving the blend would mean the negative branch is
    computed and then discarded."""
    model, loader = build_tiny_model()
    noisy, extra, mask, bs = a_noisy(model, loader)
    a = _wake_gates(AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval(), seed=3)
    guide = _wake_gates(AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval(), seed=4)
    cond = torch.full((bs, 1), 0.7)
    plain = AdapterComposition([ConditionBranch(a, cond, 2.0)], base=model)
    guided = AdapterComposition([ConditionBranch(a, cond, 2.0)], base=model,
                                guide=GuideBranch(guide, cond))
    pX_p, q_p = _blend_of(model, plain, noisy, extra, mask, bs, 2.0)
    pX_g, q_g = _blend_of(model, guided, noisy, extra, mask, bs, 2.0)
    assert not torch.allclose(pX_p[0], pX_g[0], atol=1e-5), "live guide left group 0 unchanged"
    assert torch.allclose(pX_p[1], pX_g[1], atol=1e-6), "the CONDITIONAL branch moved; only group 0 should"
    assert not torch.allclose(q_p, q_g, atol=1e-5), "live guide left the blend unchanged"
    d = (q_p - q_g).abs().max().item()
    return f"live guide moves group 0 and the blend (max |dlog q| = {d:.4f}), branch 1 untouched"


def test_guide_condition_is_used():
    """The guide is conditioned, not just a second unconditional pass: changing its
    target must change group 0."""
    model, loader = build_tiny_model()
    noisy, extra, mask, bs = a_noisy(model, loader)
    a = _wake_gates(AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval(), seed=5)
    guide = _wake_gates(AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval(), seed=6)
    cond = torch.full((bs, 1), 0.7)
    lo = AdapterComposition([ConditionBranch(a, cond, 2.0)], base=model,
                            guide=GuideBranch(guide, torch.full((bs, 1), -2.0)))
    hi = AdapterComposition([ConditionBranch(a, cond, 2.0)], base=model,
                            guide=GuideBranch(guide, torch.full((bs, 1), 2.0)))
    pX_lo, _ = _blend_of(model, lo, noisy, extra, mask, bs, 2.0)
    pX_hi, _ = _blend_of(model, hi, noisy, extra, mask, bs, 2.0)
    assert not torch.allclose(pX_lo[0], pX_hi[0], atol=1e-5), "guide ignored its own condition"
    return "guide honours its condition (group 0 differs between targets)"


def test_guide_broadcasts_a_scalar_condition():
    """molsmith passes torch.tensor([target]) -- shape (1,), not (bs, 1). The guide must
    broadcast it the same way a branch does, or the batched forward silently mis-shapes."""
    model, loader = build_tiny_model()
    noisy, extra, mask, bs = a_noisy(model, loader)
    a = _wake_gates(AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval(), seed=7)
    guide = _wake_gates(AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval(), seed=8)
    comp = AdapterComposition([ConditionBranch(a, torch.tensor([0.7]), 2.0)], base=model,
                              guide=GuideBranch(guide, torch.tensor([0.7])))
    mod = comp.build_modulation(bs, noisy["t"])
    rep = len(comp) + 1
    for L, d in enumerate(mod.layers):
        for k, v in d.items():
            assert v.size(0) == rep * bs, f"layer {L} key {k}: got {v.size(0)}, want {rep*bs}"
    return f"scalar (1,) guide condition broadcasts to {rep}*{bs} rows"


def test_guide_rejects_multi_branch_and_rate_space():
    """Both guards must fire. A guide with N>1 has no defined target, and in rate space
    `_blend_rates` would derive the forbidden-transition set from the guide."""
    model, loader = build_tiny_model()
    a1 = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval()
    a2 = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval()
    g = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval()
    cond = torch.tensor([0.7])
    try:
        AdapterComposition([ConditionBranch(a1, cond, 2.0), ConditionBranch(a2, cond, 2.0)],
                           base=model, guide=GuideBranch(g, cond))
        raise AssertionError("multi-branch guide was accepted")
    except ValueError as e:
        assert "single branch" in str(e), f"wrong error: {e}"
    try:
        AdapterComposition([ConditionBranch(a1, cond, 2.0)], base=model,
                           blend_space="rate", guide=GuideBranch(g, cond))
        raise AssertionError("rate-space guide was accepted")
    except ValueError as e:
        assert "blend_space" in str(e), f"wrong error: {e}"
    return "guide guards fire on N>1 and on blend_space='rate'"


def test_guided_sampler_runs():
    """End to end: AdaptedSampler with a guide samples without error."""
    model, loader = build_tiny_model()
    a = _wake_gates(AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval(), seed=9)
    guide = _wake_gates(AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval(), seed=10)
    cond = torch.tensor([0.7])
    comp = AdapterComposition([ConditionBranch(a, cond, 2.0)], base=model,
                              guide=GuideBranch(guide, cond))
    torch.manual_seed(0)
    graphs = AdaptedSampler(model, comp, sample_steps=5, eta=0.0, omega=0.0).sample(
        4, device="cpu", show_progress=False)
    assert len(graphs) == 4, f"got {len(graphs)} graphs"
    return f"guided AdaptedSampler produced {len(graphs)} graphs"



# ===========================================================================
# Fourier condition features + node->condition cross-attention
# ===========================================================================
def _wake_xattn(adapter, seed, scale=0.3):
    """Zero-init output projections make cross-attention an exact no-op. Wake them."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in adapter.xattn:
            m.out.weight.normal_(0.0, scale, generator=g)
            m.out.bias.normal_(0.0, scale, generator=g)
    return adapter


def test_fourier_and_xattn_are_exact_noops_at_init():
    """Both new mechanisms must reproduce the frozen base bit-for-bit at init, alone
    and together. The whole product-of-experts story rests on this."""
    model, loader = build_tiny_model()
    noisy, extra, mask, bs = a_noisy(model, loader)
    base_pred = model.forward(noisy, extra, mask)
    out = []
    for kw, label in [(dict(cond_fourier=6), "fourier"),
                      (dict(xattn_tokens=8, xattn_dim=32, xattn_heads=4), "xattn"),
                      (dict(cond_fourier=6, xattn_tokens=8, xattn_dim=32, xattn_heads=4), "both")]:
        a = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32, **kw).eval()
        mod = a(torch.full((bs, 1), 0.7), t=noisy["t"])
        pred = model.forward(noisy, extra, mask, cond_modulation=mod)
        d = (pred.X - base_pred.X).abs().max().item()
        assert d == 0.0, f"{label}: not an exact no-op at init (max|dX|={d:.3e})"
        out.append(label)
    return "exact no-op at init for " + ", ".join(out)


def test_fourier_widens_the_condition_path_and_keeps_the_raw_scalar():
    """cond_in must grow by exactly 2*cond_dim*n_bands, and the raw normalised scalar
    must still be there -- turning the feature on cannot REMOVE information."""
    model, _ = build_tiny_model()
    plain = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32, cond_fourier=0)
    ff = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32, cond_fourier=6)
    w0 = plain.trunk[0].in_features
    w1 = ff.trunk[0].in_features
    assert w1 - w0 == 2 * 1 * 6, f"cond_in grew by {w1-w0}, want {2*6}"
    c = torch.tensor([[0.3]])
    feats = ff._fourier(ff.normalize(c))
    assert feats.shape == (1, 12), f"fourier feature shape {tuple(feats.shape)}"
    assert torch.isfinite(feats).all()
    return f"cond_in {w0} -> {w1} (+{w1-w0}), raw scalar retained alongside 6 bands"


def test_fourier_resolves_nearby_targets_better_than_a_raw_scalar():
    """The POINT of the feature: the map c -> modulation must be able to separate two
    nearby targets. With a raw scalar the first layer is linear in c, so the response to
    a small delta is proportional to it; the Fourier bands make it possible to respond
    much more sharply. Measured on the trunk output, which is what every FiLM head reads."""
    model, _ = build_tiny_model()
    t = torch.full((2, 1), 0.5)
    c1, c2 = torch.tensor([[2.8]]), torch.tensor([[2.9]])       # 0.1 logP apart
    ratios = {}
    for n_bands, label in [(0, "raw"), (3, "fourier")]:
        torch.manual_seed(7)
        a = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32, cond_fourier=n_bands,
                                  cond_mean=[2.8], cond_std=[1.16]).eval()
        far = torch.tensor([[6.0]])
        def trunk_of(c):
            cn = a.normalize(c.float())
            parts = [cn] + ([a._fourier(cn)] if n_bands else [])
            parts.append(timestep_embedding(t[:1].reshape(-1, 1), a.time_emb_dim))
            return a.trunk(torch.cat(parts, dim=-1))
        near = (trunk_of(c2) - trunk_of(c1)).norm().item()
        wide = (trunk_of(far) - trunk_of(c1)).norm().item()
        ratios[label] = near / max(wide, 1e-9)
    assert ratios["fourier"] > ratios["raw"], (
        f"fourier did not improve near/far sensitivity ratio: {ratios}")
    return (f"near/far trunk sensitivity ratio: raw {ratios['raw']:.4f} -> "
            f"fourier {ratios['fourier']:.4f}")


def test_fourier_bandwidth_keeps_neighbours_correlated():
    """Pins the frequency bank. The band vector must stay SIMILAR for nearby targets --
    that similarity is what lets the adapter interpolate to targets it never trained on --
    while separating far ones. Measured on the features themselves, so it does not depend
    on a random trunk init. Fails if the default bandwidth is raised into aliasing."""
    model, _ = build_tiny_model()
    def cos_at(n, v0, v1):
        a = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32, cond_fourier=n,
                                  cond_mean=[2.8], cond_std=[1.158])
        f = lambda v: a._fourier(a.normalize(torch.tensor([[v]])))[0]
        return torch.nn.functional.cosine_similarity(f(v0), f(v1), dim=0).item()
    N = 3                                   # the default this test exists to protect
    near, far = cos_at(N, 2.8, 2.9), cos_at(N, 2.8, 5.3)
    assert near > 0.6, f"n={N}: neighbours 0.1 logP apart decorrelated (cos {near:.3f}) -- aliasing"
    assert near - far > 0.25, f"n={N}: near {near:.3f} and far {far:.3f} not separated"
    aliased = cos_at(8, 2.8, 2.9)
    assert aliased < near, f"n=8 should be worse than n={N}, got {aliased:.3f} vs {near:.3f}"
    return (f"n={N}: cos(0.1 logP)={near:.3f}, cos(2.5 logP)={far:.3f}; "
            f"n=8 aliases to {aliased:.3f}")


def test_fourier_refuses_encoder_and_high_dimensional_conditions():
    """Fourier features are a LOW-dimensional-input result. Expanding a fingerprint
    into bands would be meaningless and enormous, so both cases must raise."""
    model, _ = build_tiny_model()
    for kw, why in [(dict(cond_dim=512, cond_fourier=4,
                          cond_encoder={"kind": "mlp", "in_dim": 512, "out_dim": 64}), "encoder"),
                    (dict(cond_dim=32, cond_fourier=4), "high-dim")]:
        try:
            AdaLNAdapter.for_base(model, hidden=32, **kw)
            raise AssertionError(f"{why}: cond_fourier was accepted")
        except ValueError:
            pass
    return "cond_fourier refuses a cond_encoder and cond_dim > 8"


def test_xattn_is_content_addressed_not_a_broadcast():
    """The claim the mechanism rests on. FiLM applies ONE diagonal affine map to every
    atom; cross-attention lets each atom SELECT what it reads. So the per-node attention
    distributions must not all be identical -- if they were, this would be an expensive
    way to reproduce a broadcast."""
    torch.manual_seed(3)
    xa = NodeConditionCrossAttention(dx=32, d_tok=16, n_heads=4)
    X = torch.randn(2, 7, 32) * 2.0                 # genuinely different node states
    tokens = torch.randn(2, 8, 16)
    h, dh = xa.n_heads, xa.dx // xa.n_heads
    q = xa.q(xa.norm(X)).view(2, 7, h, dh).transpose(1, 2)
    k = xa.k(tokens).view(2, -1, h, dh).transpose(1, 2)
    att = torch.softmax(q @ k.transpose(-1, -2) / (dh ** 0.5), dim=-1)   # (2,h,7,m)
    spread = (att - att.mean(dim=2, keepdim=True)).abs().max().item()
    assert spread > 1e-3, f"attention identical across nodes (spread {spread:.2e})"
    # ...and identical node states must attend identically, or it is not a function of content
    Xsame = X.clone(); Xsame[:, 1] = Xsame[:, 0]
    q2 = xa.q(xa.norm(Xsame)).view(2, 7, h, dh).transpose(1, 2)
    att2 = torch.softmax(q2 @ k.transpose(-1, -2) / (dh ** 0.5), dim=-1)
    assert torch.allclose(att2[:, :, 0], att2[:, :, 1], atol=1e-6), \
        "identical node states attended differently"
    return f"attention is content-addressed (per-node spread {spread:.3f}, ties respected)"


def test_xattn_masks_padded_nodes():
    """A padded row must receive exactly zero delta, or junk padding states leak into
    the next step's RRWP features."""
    model, loader = build_tiny_model()
    noisy, extra, mask, bs = a_noisy(model, loader)
    a = _wake_xattn(AdaLNAdapter.for_base(model, cond_dim=1, hidden=32, xattn_tokens=8,
                                          xattn_dim=32, xattn_heads=4).eval(), seed=11)
    mod = a(torch.full((bs, 1), 0.7), t=noisy["t"])
    n = mask.size(1)
    # The tiny fixture's molecules are all the same size, so `mask` may have NO padding
    # and the assertion below would pass vacuously -- the inert-control failure mode.
    # Force padding, and check there is some.
    forced = mask.clone()
    forced[:, -1] = False
    n_pad = int((~forced).sum().item())
    assert n_pad > 0, "test would be vacuous: no padded nodes"
    X = torch.randn(bs, n, a.dims["dx"])
    x_mask = forced.unsqueeze(-1)
    checked = 0
    for sl, fn in mod.xattn[0]:
        delta = fn(X[sl], x_mask[sl])
        pad = delta[~forced[sl]]
        assert pad.numel() > 0, "no padded entries selected"
        assert pad.abs().max().item() == 0.0, "padded node got a non-zero delta"
        # and the unpadded ones must NOT be zero, or the mask is zeroing everything
        assert delta[forced[sl]].abs().max().item() > 0.0, "every delta was zero"
        checked += pad.numel()
    return f"padded nodes get exactly zero delta ({n_pad} pads, {checked} entries checked)"


def test_xattn_only_touches_its_own_group_in_a_stacked_forward():
    """THE containment test. In the batched (N+1)*bs forward, group 0 is the negative
    branch and MUST stay the frozen base. If the closure ran over all rows instead of its
    own slice, group 0 would be silently conditioned and every CFG blend would be wrong
    while still looking plausible."""
    import torch.nn.functional as F
    from defog.core.data import PlaceHolder
    model, loader = build_tiny_model()
    noisy, extra, mask, bs = a_noisy(model, loader)
    a = _wake_xattn(AdaLNAdapter.for_base(model, cond_dim=1, hidden=32, xattn_tokens=8,
                                          xattn_dim=32, xattn_heads=4).eval(), seed=12)
    comp = AdapterComposition([ConditionBranch(a, torch.tensor([0.7]), 2.0)], base=model)
    rep = len(comp) + 1
    mod = comp.build_modulation(bs, noisy["t"])
    assert mod.xattn is not None and len(mod.xattn[0]) == 1, "xattn entry missing after stacking"
    sl, _ = mod.xattn[0][0]
    assert sl == slice(bs, 2 * bs), f"branch closure landed on {sl}, want rows {bs}..{2*bs}"
    nd = {"X_t": noisy["X_t"].repeat(rep, 1, 1), "E_t": noisy["E_t"].repeat(rep, 1, 1, 1),
          "y_t": noisy["y_t"].repeat(rep, 1), "t": noisy["t"].repeat(rep, 1),
          "node_mask": mask.repeat(rep, 1)}
    extra_b = PlaceHolder(X=extra.X.repeat(rep, 1, 1), E=extra.E.repeat(rep, 1, 1, 1),
                          y=extra.y.repeat(rep, 1))
    pred = model.forward(nd, extra_b, nd["node_mask"], cond_modulation=mod)
    pX = F.softmax(pred.X, -1).view(rep, bs, *pred.X.shape[1:])
    base_pred = F.softmax(model.forward(noisy, extra, mask).X, -1)
    assert torch.allclose(pX[0], base_pred, atol=1e-6), "group 0 was contaminated by xattn"
    assert not torch.allclose(pX[1], base_pred, atol=1e-5), "group 1 was NOT modulated"
    return "stacked forward: group 0 == frozen base, group 1 modulated"


def test_xattn_bypass_rows_silences_it():
    """Condition dropout must produce a genuinely UNCONDITIONAL row. Zeroing the FiLM
    gates is not enough once cross-attention exists -- its delta passes through no gate."""
    model, loader = build_tiny_model()
    noisy, extra, mask, bs = a_noisy(model, loader)
    a = _wake_xattn(AdaLNAdapter.for_base(model, cond_dim=1, hidden=32, xattn_tokens=8,
                                          xattn_dim=32, xattn_heads=4).eval(), seed=13)
    mod = a(torch.full((bs, 1), 0.7), t=noisy["t"])
    drop = torch.ones(bs, dtype=torch.bool)                 # drop every row
    dropped = mod.bypass_rows(drop)
    pred = model.forward(noisy, extra, mask, cond_modulation=dropped)
    base_pred = model.forward(noisy, extra, mask)
    d = (pred.X - base_pred.X).abs().max().item()
    assert d < 1e-6, f"fully-dropped rows still differ from the base (max|dX|={d:.3e})"
    return "bypass_rows silences cross-attention as well as the FiLM gates"


def test_xattn_stacking_is_refused():
    """Two cross-attention branches is untested, so it must raise rather than run."""
    model, _ = build_tiny_model()
    kw = dict(cond_dim=1, hidden=32, xattn_tokens=4, xattn_dim=16, xattn_heads=4)
    a1 = AdaLNAdapter.for_base(model, **kw).eval()
    a2 = AdaLNAdapter.for_base(model, **kw).eval()
    c = torch.tensor([0.7])
    try:
        AdapterComposition([ConditionBranch(a1, c, 2.0), ConditionBranch(a2, c, 2.0)], base=model)
        raise AssertionError("stacked cross-attention adapters were accepted")
    except NotImplementedError as e:
        assert "untested" in str(e)
    # one xattn + one FiLM-only must still compose
    plain = AdaLNAdapter.for_base(model, cond_dim=1, hidden=32).eval()
    AdapterComposition([ConditionBranch(a1, c, 2.0), ConditionBranch(plain, c, 2.0)], base=model)
    return "2x xattn refused; xattn + FiLM-only still composes"


def test_fourier_xattn_save_load_roundtrip():
    """The new fields must survive save/load, or a reloaded adapter is a different model
    and state_dict loading fails (or worse, silently rebuilds without them)."""
    model, loader = build_tiny_model()
    noisy, extra, mask, bs = a_noisy(model, loader)
    a = _wake_xattn(AdaLNAdapter.for_base(
        model, cond_dim=1, hidden=32, cond_fourier=6, xattn_tokens=8, xattn_dim=32,
        xattn_heads=4, cond_mean=[2.8], cond_std=[1.16], name="rt").eval(), seed=14)
    c = torch.full((bs, 1), 3.1)
    before = a(c, t=noisy["t"])
    with tempfile.TemporaryDirectory() as d:
        path = a.save(os.path.join(d, "rt"))
        b = AdaLNAdapter.load(path)
        cfg = b._config()
        for k, v in [("cond_fourier", 6), ("xattn_tokens", 8), ("xattn_dim", 32), ("xattn_heads", 4)]:
            assert cfg[k] == v, f"config lost {k}: {cfg[k]} != {v}"
        after = b(c, t=noisy["t"])
        for L in range(len(before.layers)):
            for k in before.layers[L]:
                assert torch.allclose(before.layers[L][k], after.layers[L][k], atol=1e-6), \
                    f"layer {L} key {k} changed across save/load"
        X = torch.randn(bs, mask.size(1), a.dims["dx"])
        xm = mask.unsqueeze(-1)
        d0 = before.xattn[0][0][1](X, xm)
        d1 = after.xattn[0][0][1](X, xm)
        assert torch.allclose(d0, d1, atol=1e-6), "cross-attention delta changed across save/load"
        # from_config is the safetensors path and must accept the new keys too
        AdaLNAdapter.from_config(cfg, b.state_dict())
    return "cond_fourier/xattn_* survive save/load and from_config; deltas reproduce"



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
        test_fingerprint_encoder_shape_and_config,
        test_cond_encoder_registry_dispatch,
        test_encoder_adapter_null_equals_base_and_roundtrips,
        test_zero_gate_is_noop_in_prob_space_at_any_weight,
        test_no_guide_group0_is_exactly_zero,
        test_guide_at_init_is_plain_cfg,
        test_live_guide_changes_group0_and_the_blend,
        test_guide_condition_is_used,
        test_guide_broadcasts_a_scalar_condition,
        test_guide_rejects_multi_branch_and_rate_space,
        test_guided_sampler_runs,
        test_fourier_and_xattn_are_exact_noops_at_init,
        test_fourier_widens_the_condition_path_and_keeps_the_raw_scalar,
        test_fourier_resolves_nearby_targets_better_than_a_raw_scalar,
        test_fourier_bandwidth_keeps_neighbours_correlated,
        test_fourier_refuses_encoder_and_high_dimensional_conditions,
        test_xattn_is_content_addressed_not_a_broadcast,
        test_xattn_masks_padded_nodes,
        test_xattn_only_touches_its_own_group_in_a_stacked_forward,
        test_xattn_bypass_rows_silences_it,
        test_xattn_stacking_is_refused,
        test_fourier_xattn_save_load_roundtrip,
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
