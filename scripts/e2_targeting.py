#!/usr/bin/env python
"""
E2: property targeting under FreeGress's protocol.

Protocol (docs/targeting-protocol.md §1, matching FreeGress Tab. 2 on ZINC250k):

  1. take 100 molecules from the dataset; each molecule's own measured property
     value becomes a target y_i
  2. generate 10 molecules per target -> 1000 generated molecules
  3. measure the property on every generated molecule with RDKit
  4. MAE = mean |y_i - yhat_ij| over all 1000
  5. report chemical validity in the same row

WHY 10 PER TARGET AND NOT 1-OF-K
Under Feynman-Kac the ten come from ONE particle system of K=10, and all ten are
kept. It is tempting to instead run ten systems of K=8 and keep the best particle
from each -- the outputs would be independent, which superficially matches the
baseline better. It would also be best-of-8 selection: an 8x compute advantage
over a method that simply draws ten times. Keeping all ten spends the same budget
the baseline spends.

The cost is that resampling COUPLES the ten: it culls low-weight particles and
duplicates high-weight ones, so a badly tuned FK run can return ten copies of one
molecule and post an excellent MAE. That is why uniqueness is reported beside MAE
here rather than left to a follow-up, and why FK's warmup_frac / ess_frac / beta
are the knobs to tune -- they control how hard and how early the system collapses.

SPLIT DISCIPLINE. --split validation for anything that informs a choice; --split
test exactly once, with the configuration already frozen. The 100 targets are
drawn with an explicit --seed, and both split and seed are recorded in the output
so the caption can state them, as the protocol requires.

Usage:
    python scripts/e2_targeting.py --adapter molsmith/clogp@1.2.0 --property logp \\
        --split validation --method adapter --weight 1.0 --out e2_val_adapter.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/media/ssd2/Programming/defog-web")

import numpy as np  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Crippen, Descriptors, QED  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# Synthetic accessibility, from RDKit's contrib SA_Score. Not importable by default --
# it lives outside the installed package tree -- so the path is added explicitly rather
# than left to fail at first use, hours into a training run.
def _load_sascorer():
    import os, sys
    from rdkit import RDConfig
    p = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if p not in sys.path:
        sys.path.append(p)
    import sascorer
    return sascorer


_SASCORER = None


def _sa_score(m):
    """SA score, roughly 1 (easy) to 10 (hard). Unlike logp/tpsa it is a heuristic over
    fragment frequencies plus complexity penalties, so it is bounded in practice to about
    [1, 8] on drug-like molecules and is NOT symmetric -- most of ZINC sits near 2-3."""
    global _SASCORER
    if _SASCORER is None:
        _SASCORER = _load_sascorer()
    return float(_SASCORER.calculateScore(m))


PROP_FNS = {
    "logp": lambda m: float(Crippen.MolLogP(m)),
    "qed": lambda m: float(QED.qed(m)),
    "tpsa": lambda m: float(Descriptors.TPSA(m)),
    "sascore": _sa_score,
}


# TOLERANCE. 1e-6, not the 1e-3 that AdaLNAdapter.check_compatible warns at. base_token
# is a deterministic sum over a fixed weight, so a matching base matches EXACTLY -- there
# is no reason to allow slack, and 1e-3 allows 0.095 here while ckpts/zinc_e1_seed42_kek
# (the base the launchers name as the one to avoid) sits just 0.0607 away. It passed.
# This is the same tolerance the training launcher's own preflight uses.
_TOKEN_RTOL = 1e-6


def draw_targets(split: str, n: int, seed: int, prop_fn):
    """The 100 target values, from real molecules in `split`.

    Targets are the molecules' OWN measured property, not quantiles of the
    distribution -- that is the difference between this and the percentile mode the
    training experiment uses, and it is what makes the number comparable to
    FreeGress's.
    """
    from defog.data import zinc_reference as zref
    s = zref.load_reference_split()
    pool = {"validation": s.val_smiles, "test": s.test_smiles}[split]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pool))
    out = []
    for i in idx:
        smi = pool[int(i)]
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        try:
            out.append((smi, float(prop_fn(m))))
        except Exception:                                   # noqa: BLE001
            continue
        if len(out) >= n:
            break
    return out


class _TargetedSize:
    """A :class:`SizeDistribution` that pins one target value into a size model.

    ``molsmith`` hands ``size_dist.sample`` whatever condition the *adapter* pipeline
    computed, which is not the raw property value the size model wants. This wrapper
    carries the target instead, so the two cannot drift apart.
    """

    def __init__(self, model, target: float, n: int):
        import torch
        self.model = model
        self.condition = torch.full((n, model.cond_dim), float(target))

    def sample(self, num_samples, condition=None, device=None, generator=None):
        c = self.condition[:num_samples] if self.condition.size(0) >= num_samples else \
            self.condition[:1].expand(num_samples, -1)
        return self.model.sample(num_samples, condition=c, device=device,
                                 generator=generator)

    @property
    def max_size(self) -> int:
        return self.model.max_size

    def log_prob(self, sizes, condition=None):
        return self.model.log_prob(sizes, condition=self.condition[:sizes.numel()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="molsmith/zinc-kek")
    ap.add_argument("--adapter", default=None, help="store ref, e.g. molsmith/clogp@1.2.0")
    # A RAW .ckpt, for an adapter that is not a shippable package yet -- a new
    # architecture under test. Same precedent molsmith documents for `size_dist` and
    # `guide`: an experiment's own model has nothing to publish, so the caller supplies
    # the module and molsmith keys it by whatever string the caller uses. The cost is
    # that it skips the store's compatibility gate, so the base check below is the only
    # one it gets -- and it is made strict, not a warning.
    ap.add_argument("--adapter-ckpt", default=None,
                    help="Path to an AdaLNAdapter .ckpt to steer with, instead of a "
                         "packaged --adapter. For architectures not yet packaged.")
    ap.add_argument("--property", required=True, choices=sorted(PROP_FNS))
    ap.add_argument("--split", required=True, choices=("validation", "test"),
                    help="validation for anything that informs a choice; test once, frozen")
    ap.add_argument("--method", required=True, choices=("adapter", "fk"))
    ap.add_argument("--n-targets", type=int, default=100)
    ap.add_argument("--per-target", type=int, default=10)
    ap.add_argument("--weight", type=float, default=2.0, help="adapter guidance weight")
    # Kept reachable, not merely documented: every E2 number before 2026-08-17 was
    # measured in rate space, and re-deriving one means being able to ask for it.
    ap.add_argument("--blend-space", default="prob", choices=("prob", "rate"),
                    help="where CFG is applied: 'prob' blends clean-graph marginals "
                         "(FreeGress Eq. 10/11, the default), 'rate' blends rate matrices "
                         "(historical; breaks down above w=1)")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eta", type=float, default=25.0)
    ap.add_argument("--omega", type=float, default=0.0)
    ap.add_argument("--time-distortion", default="polydec")
    # FK knobs. These carry two jobs at once: pull toward the target, and avoid
    # collapsing the ten particles into copies of one molecule.
    ap.add_argument("--fk-beta", type=float, default=2.5)
    ap.add_argument("--fk-warmup", type=float, default=0.6)
    ap.add_argument("--fk-ess", type=float, default=0.5,
                    help="resample only when effective sample size < ess*K; lower "
                         "means less culling and more surviving diversity")
    ap.add_argument("--fk-rejuvenate", action="store_true",
                    help="MCMC moves after each resample -- the standard SMC remedy "
                         "for particle impoverishment. Regenerates duplicated "
                         "particles rather than leaving copies, which is the failure "
                         "the uniqueness column exists to catch.")
    ap.add_argument("--fk-jump", type=int, default=10)
    # How many nodes each generated graph gets. DEFAULT IS `marginal`, which is what every
    # E2 number before this flag existed used -- so those numbers stay reproducible.
    #
    # This is an ABLATION AXIS, not a free improvement. FreeGress Tab. 3 shows conditioned
    # node inference alone moving MW MAE by -70%, i.e. capable of dominating the column it
    # appears in. Folded silently into one row it reads as "the adapter got better". Run
    # the pair.
    ap.add_argument("--size-mode", default="marginal",
                    choices=("marginal", "learned"),
                    help="marginal: the base's P(n) (the historical default). "
                         "learned: P(n|target) from --size-model.")
    ap.add_argument("--size-model", default=None,
                    help="Path to a LearnedSizeDistribution ckpt (--size-mode learned)")
    # AUTOGUIDANCE. Replaces the blend's negative branch (normally the frozen
    # unconditional base) with a DELIBERATELY WORSE conditional model, so guidance pushes
    # away from a weak version of the same thing rather than away from no conditioning at
    # all. Karras et al.; MolGuidance reports it improving structural validity where CFG
    # costs it, which is the reason to try it here -- CFG's validity collapses as w rises
    # (0.982 / 0.898 / 0.466 at w = 2 / 3 / 4), and the question is whether autoguidance
    # buys headroom to push w further.
    #
    # A RAW .ckpt path, not a store reference: a deliberately undertrained guide is not a
    # shippable artefact and has no package. The cost is that it skips molsmith's
    # compatibility gate, so this script checks the base token itself, strictly, below.
    ap.add_argument("--guide", default=None,
                    help="Path to an AdaLNAdapter .ckpt to use as the autoguidance "
                         "negative branch, in place of the unconditional base.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if bool(args.adapter) == bool(args.adapter_ckpt):
        sys.exit("give exactly one of --adapter (a store ref) or --adapter-ckpt (a raw .ckpt)")
    if args.adapter_ckpt and args.method == "fk":
        sys.exit("REFUSING: --adapter-ckpt with --method fk. FK needs the property head "
                 "the package bundles; a raw checkpoint has none, and FK would silently "
                 "reduce to plain adapter sampling.")
    if args.size_mode == "learned" and not args.size_model:
        sys.exit("--size-mode learned needs --size-model PATH")
    if args.guide and args.method == "fk":
        sys.exit("REFUSING: --guide with --method fk. FeynmanKacSampler assigns the "
                 "composition without calling check_compatible on anything "
                 "(defog/core/feynman_kac.py), so a guide would reach the sampler with no "
                 "structural validation at all. The adapter path validates it; use "
                 "--method adapter, or add the check to FeynmanKacSampler first.")
    if args.guide and args.blend_space != "prob":
        sys.exit(f"--guide requires --blend-space prob, got {args.blend_space!r}. In rate "
                 f"space the forbidden-transition guard would be derived from the guide "
                 f"instead of the base, and w>1 is broken there anyway -- which makes the "
                 f"w sweep this flag exists for impossible.")

    from molsmith import sample as ms

    # Loaded before any config is built so every config -- including the probe -- is
    # identical to the ones that sample. `adapter_key` is what molsmith uses as the key
    # into Loaded.adapters; for a packaged adapter it is the store ref, for a raw
    # checkpoint it is a path-derived string that resolve_package must never see.
    adapter_module = None
    adapter_key = args.adapter
    if args.adapter_ckpt:
        from defog.core import AdaLNAdapter
        adapter_module = AdaLNAdapter.load(args.adapter_ckpt, device="cpu")
        adapter_key = f"ckpt:{Path(args.adapter_ckpt).resolve()}"
        cfg_dump = adapter_module._config()
        print(f"adapter (raw ckpt): {args.adapter_ckpt}\n"
              f"  params={sum(p.numel() for p in adapter_module.parameters()):,} "
              f"hidden={cfg_dump['hidden']} n_layers={cfg_dump['n_layers']} "
              f"cond_fourier={cfg_dump.get('cond_fourier')} "
              f"xattn_tokens={cfg_dump.get('xattn_tokens')} "
              f"interior_ff={cfg_dump.get('interior_ff')} "
              f"interior_attn={cfg_dump.get('interior_attn')}\n"
              f"  cond_mean={adapter_module.cond_mean.tolist()} "
              f"cond_std={adapter_module.cond_std.tolist()}")
        # _fill_from_packages short-circuits only when property is set AND scale != 1.0.
        # If it does not short-circuit it calls resolve_package on `adapter_key`, which
        # is not a package, and the run dies mid-loop rather than here.
        if float(adapter_module.cond_std.reshape(-1)[0]) == 1.0:
            sys.exit("REFUSING: this adapter's cond_std is exactly 1.0, which collides "
                     "with molsmith's 'unfilled' sentinel for AdapterTarget.scale, so "
                     "molsmith would try to resolve the raw checkpoint as a package.")

    prop_fn = PROP_FNS[args.property]
    targets = draw_targets(args.split, args.n_targets, args.seed, prop_fn)
    tvals = np.array([v for _, v in targets])
    print(f"{len(targets)} targets from {args.split} (seed {args.seed}); "
          f"{args.property} range [{tvals.min():.3f}, {tvals.max():.3f}] "
          f"mean {tvals.mean():.3f}")

    size_model = None
    if args.size_mode == "learned":
        from defog.core import LearnedSizeDistribution
        size_model = LearnedSizeDistribution.load(args.size_model)
        print(f"size model: {args.size_model}  grid {size_model.min_size}.."
              f"{size_model.max_size}  property={size_model.property_name!r} "
              f"from={size_model.property_from!r}")

    # GUARD: SamplingConfig is a plain dataclass, so `config.size_dist = ...` silently
    # succeeds even against a molsmith build that never reads it -- the run then uses
    # the marginal while this script's own JSON records "learned". That happened once;
    # it is not allowed to happen quietly again.
    if size_model is not None and "size_dist" not in ms.SamplingConfig.__dataclass_fields__:
        sys.exit(
            "REFUSING: this molsmith build has no SamplingConfig.size_dist field, so a "
            "learned size distribution would be silently ignored and the results would "
            "be mislabelled as 'learned'. Update molsmith before re-running."
        )
    # The same failure mode, for the blend space: asking for "prob" against a molsmith
    # that predates the field would sample in rate space and record "prob".
    if "blend_space" not in ms.SamplingConfig.__dataclass_fields__:
        sys.exit(
            f"REFUSING: this molsmith build has no SamplingConfig.blend_space field, so "
            f"--blend-space {args.blend_space!r} would be ignored and the output would be "
            f"mislabelled. Update molsmith before re-running."
        )

    # Same failure mode as the two guards above: a molsmith without the field would run
    # plain CFG while this script's JSON recorded a guide.
    if args.guide and "guide" not in ms.SamplingConfig.__dataclass_fields__:
        sys.exit(
            "REFUSING: this molsmith build has no SamplingConfig.guide field, so --guide "
            "would be silently ignored -- the run would be plain CFG and the output would "
            "be labelled autoguidance. Update molsmith before re-running."
        )

    # Loaded BEFORE the first config is built so that every config, including probe_cfg,
    # carries it. Loading it later would leave the probe on a different code path from the
    # runs, which is how the FK scale=1.0 bug survived a passing probe.
    guide_module = None
    if args.guide:
        from defog.core import AdaLNAdapter
        guide_module = AdaLNAdapter.load(args.guide, device="cpu")
        print(f"guide: {args.guide}  name={guide_module.name!r} "
              f"cond_type={guide_module.cond_type!r} "
              f"params={sum(p.numel() for p in guide_module.parameters()):,}")

    # Pre-filled for a raw checkpoint so _fill_from_packages never reaches the store.
    raw_target_kw = {}
    if adapter_module is not None:
        raw_target_kw = dict(property=args.property,
                             scale=float(adapter_module.cond_std.reshape(-1)[0]),
                             mean=float(adapter_module.cond_mean.reshape(-1)[0]))

    def cfg_for(target: float, seed: int):
        c = ms.SamplingConfig(
            base=args.base, n=args.per_target, seed=seed, steps=args.steps,
            eta=args.eta, omega=args.omega, time_distortion=args.time_distortion,
            adapters=[ms.AdapterTarget(package=adapter_key, target=target,
                                       weight=args.weight, **raw_target_kw)],
            blend_space=args.blend_space,
            method="fk" if args.method == "fk" else "none")
        if guide_module is not None:
            c.guide = guide_module
        if size_model is not None:
            # A ready-made SizeDistribution, bypassing size_mode. The condition rides in
            # the branch here rather than through SamplingConfig, because the target is
            # per-call and the model is not.
            c.size_dist = _TargetedSize(size_model, target, args.per_target)
        if args.method == "fk":
            c.fk = ms.FeynmanKac(beta=args.fk_beta, warmup_frac=args.fk_warmup,
                                 ess_frac=args.fk_ess,
                                 rejuvenate=args.fk_rejuvenate,
                                 jump_length=args.fk_jump)
        return c

    probe_cfg = cfg_for(float(tvals[0]), args.seed)
    if adapter_module is None:
        loaded = ms.load(probe_cfg)                 # fills probe_cfg.adapters[*] in place
    else:
        # ms.load resolves every adapter ref against the store, so the raw checkpoint is
        # withheld for that call and injected afterwards, keyed exactly as the configs
        # name it. sample() only does `loaded.adapters.get(spec.package)`, which molsmith
        # documents as a supported entry point for a caller holding pre-loaded modules.
        held, probe_cfg.adapters = probe_cfg.adapters, []
        loaded = ms.load(probe_cfg)
        probe_cfg.adapters = held
        base_dev = next(loaded.base.parameters()).device
        adapter_module.to(base_dev).eval()
        adapter_module.check_compatible(loaded.base)     # dims / n_layers, raises
        from defog.core.adapter import _base_token as _bt
        tok = _bt(loaded.base)
        if adapter_module.base_token is None:
            sys.exit(f"REFUSING: {args.adapter_ckpt} records no base_token, so there is no "
                     f"way to tell which base it was trained on. Retrain it with a current "
                     f"AdaLNAdapter.for_base, which records one.")
        if abs(tok - adapter_module.base_token) > _TOKEN_RTOL * (1 + abs(adapter_module.base_token)):
            sys.exit(f"REFUSING: {args.adapter_ckpt} was trained on a different base "
                     f"(token {adapter_module.base_token} != {tok} for {args.base}). "
                     f"A raw checkpoint skips the store's compatibility gate, so this is "
                     f"the only thing standing between a mismatch and a plausible-looking "
                     f"number.")
        loaded.adapters[adapter_key] = adapter_module
        loaded.heads[adapter_key] = None
        loaded.size_models[adapter_key] = None
        print(f"adapter base check: OK (token {tok:.8g}); injected as {adapter_key!r}")

    if guide_module is not None:
        # The guide did not go through the store's compatibility gate, and
        # AdaLNAdapter.check_compatible only WARNS on a base-token mismatch. A warning is
        # not enough here: a guide trained on a different base does not share this base's
        # flaws, so the blend would not be autoguidance at all -- it would be pushing away
        # from an unrelated model, and the run would still produce plausible molecules and
        # a plausible number. Hard-fail instead.
        from defog.core.adapter import _base_token
        base_dev = next(loaded.base.parameters()).device
        guide_module.to(base_dev).eval()
        guide_module.check_compatible(loaded.base)   # dims / n_layers, raises on mismatch
        tok = _base_token(loaded.base)
        if guide_module.base_token is None:
            sys.exit(f"REFUSING: guide {args.guide} carries no base_token, so there is no "
                     f"way to tell whether it was trained on {args.base}. Retrain it with a "
                     f"current AdaLNAdapter.for_base, which records one.")
        if abs(tok - guide_module.base_token) > _TOKEN_RTOL * (1 + abs(guide_module.base_token)):
            sys.exit(f"REFUSING: guide {args.guide} was trained on a DIFFERENT base "
                     f"(token {guide_module.base_token:.8g} != {tok:.8g} for {args.base}). "
                     f"Autoguidance requires the guide to share the model's flaws; a guide "
                     f"over another base does not, and the result would be uninterpretable.")

        # The guide must also read the target on the SAME SCALE as the adapter it negates.
        # AdaLNAdapter normalises internally with its own cond_mean/cond_std buffers, so a
        # guide fitted on a different normalisation is conditioned on a different VALUE
        # while every log line still says the same target. Nothing downstream would notice.
        import torch as _torch
        main_adapter = loaded.adapters.get(adapter_key)
        if main_adapter is None:
            sys.exit(f"REFUSING: adapter {adapter_key} did not load, so the guide's "
                     f"conditioning scale cannot be checked against it.")
        for field in ("cond_mean", "cond_std"):
            g_v = getattr(guide_module, field).detach().cpu()
            m_v = getattr(main_adapter, field).detach().cpu()
            if g_v.shape != m_v.shape or not _torch.allclose(g_v, m_v, atol=1e-4):
                sys.exit(
                    f"REFUSING: guide {field} {g_v.tolist()} != adapter {field} "
                    f"{m_v.tolist()}. The two would normalise the same target to different "
                    f"values, so the guide would not be a degraded version of THIS adapter "
                    f"at THIS target. Retrain the guide on the same property and data.")
        print(f"guide base check: OK (token {tok:.8g}); "
              f"cond_mean/std match adapter ({main_adapter.cond_mean.item():.6f} / "
              f"{main_adapter.cond_std.item():.6f})")
    if args.method == "fk":
        h = loaded.heads.get(adapter_key)
        if h is None:
            sys.exit(f"REFUSING: --method fk but {adapter_key} bundles no property head. "
                     f"LearnedPropertyEnergy needs one; without it FK has nothing to score "
                     f"and would silently reduce to plain adapter sampling.")
        # The scale is the property's own std, and the energy is divided by its square, so
        # beta is dimensionless and means the same thing across properties. Printing it (and
        # the raw-units beta it corresponds to) is what lets a later reader tell a run made
        # before that normalisation existed from one made after. `probe_cfg` has been through
        # ms.load, which fills the field in place -- a freshly built config would still read
        # 1.0 and this check would pass while the real runs went un-normalised.
        scale = float(getattr(probe_cfg.adapters[0], "scale", 1.0))
        if scale == 1.0:
            sys.exit("REFUSING: adapter spec has scale=1.0, so the FK energy would be in raw "
                     "property units and beta would not be comparable across properties. "
                     "Expected _fill_from_packages to set it from the package prop_std.")
        print(f"FK energy: head from {adapter_key}  "
              f"beta={args.fk_beta} (dimensionless; scale={scale:.4f}, "
              f"= raw-units beta {args.fk_beta / (scale * scale):.4g})  "
              f"warmup={args.fk_warmup} ess={args.fk_ess} "
              f"rejuvenate={args.fk_rejuvenate}")

    # The field-presence guard above is the weaker half of the FK guard's lesson: a
    # molsmith whose SamplingConfig HAS `guide` but whose sample() predates the wiring
    # would pass it, run plain CFG, and write "guide": "<path>". Nothing else in either
    # repo covers that hop -- molsmith has no test for the guide path, and every DeFoG
    # test builds AdapterComposition directly. So watch the constructor itself.
    # molsmith imports AdapterComposition INSIDE sample(), so patching the attribute here
    # is picked up on the next call.
    # Installed for the raw-checkpoint path too, not just for --guide. That path
    # deliberately bypasses the store's compatibility gate, and if the injected key ever
    # failed to match, molsmith would set composition=None and fall through to a plain
    # UNGUIDED Sampler -- the run completes, the MAE is plausible, and the JSON still
    # records "adapter_ckpt": "<path>". This is the one hop in this file that was not
    # probed, in a file whose whole idiom is to probe the hop.
    _probe = None
    if guide_module is not None or adapter_module is not None:
        import defog.core as _dc
        _probe = {"real": _dc.AdapterComposition, "built": False, "guided": False,
                  "adapter_is_ours": None}

        def _probe_ctor(*a, **kw):
            _probe["built"] = True
            _probe["guided"] = kw.get("guide") is not None
            branches = a[0] if a else kw.get("branches", [])
            if adapter_module is not None and branches:
                _probe["adapter_is_ours"] = branches[0].adapter is adapter_module
            return _probe["real"](*a, **kw)

        _dc.AdapterComposition = _probe_ctor

    rows, all_err, t0 = [], [], time.time()
    for k, (smi, tgt) in enumerate(targets):
        cfg = cfg_for(tgt, args.seed + k)          # a different draw per target
        res = ms.sample(cfg, loaded)
        if k == 0 and _probe is not None:
            import defog.core as _dc
            _dc.AdapterComposition = _probe["real"]          # restore before anything else
            if not _probe["built"]:
                sys.exit("REFUSING: no AdapterComposition was constructed during sampling, "
                         "so the run was NOT steered at all -- molsmith fell through to a "
                         "plain unguided Sampler. The output would have looked normal.")
            if adapter_module is not None and _probe["adapter_is_ours"] is not True:
                sys.exit(f"REFUSING: the composition's branch adapter is not the module "
                         f"loaded from {args.adapter_ckpt}. The injected key "
                         f"{adapter_key!r} did not reach the sampler.")
            if guide_module is not None and not _probe["guided"]:
                sys.exit("REFUSING: molsmith's sample() built the composition WITHOUT the "
                         "guide -- this run is plain CFG and would have been recorded as "
                         "autoguidance. The installed molsmith has the SamplingConfig.guide "
                         "field but does not pass it through. Update molsmith.")
            what = []
            if adapter_module is not None:
                what.append("the raw-ckpt adapter")
            if guide_module is not None:
                what.append("the guide")
            print(f"wiring check: AdapterComposition received {' and '.join(what)}", flush=True)
        if k == 0 and args.method == "fk" and float(cfg.adapters[0].scale) == 1.0:
            # The per-target configs are built fresh, so the one that just sampled is the
            # only thing that proves the energy was normalised. Checking probe_cfg alone
            # would pass on a molsmith whose sample() does not fill the field.
            sys.exit("REFUSING: the config that just sampled still has scale=1.0, so the FK "
                     "energy ran in raw property units. Update molsmith: sample() must call "
                     "_fill_from_packages.")
        smis = [s for s in res.smiles if s]
        achieved, ok = [], []
        for s in smis:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            try:
                achieved.append(float(prop_fn(m)))
                ok.append(s)
            except Exception:                               # noqa: BLE001
                continue
        a = np.array(achieved)
        err = np.abs(a - tgt) if a.size else np.array([])
        all_err.extend(err.tolist())
        rows.append({
            "target_smiles": smi, "target": tgt,
            "n_requested": args.per_target, "n_valid": len(ok),
            # validity is over what was ASKED for, not over what parsed
            "validity": len(ok) / args.per_target,
            # uniqueness catches FK particle collapse: ten copies of one molecule
            # post an excellent MAE and are worthless
            "uniqueness": (len(set(ok)) / len(ok)) if ok else float("nan"),
            "achieved_mean": float(a.mean()) if a.size else float("nan"),
            "achieved_sd": float(a.std()) if a.size else float("nan"),
            "mae": float(err.mean()) if err.size else float("nan"),
        })
        if (k + 1) % 10 == 0:
            done = np.array([r["mae"] for r in rows if np.isfinite(r["mae"])])
            print(f"  {k+1}/{len(targets)} targets  running MAE {done.mean():.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    finite = np.array([r["mae"] for r in rows if np.isfinite(r["mae"])])
    val = np.array([r["validity"] for r in rows])
    uniq = np.array([r["uniqueness"] for r in rows if np.isfinite(r["uniqueness"])])
    # MAE across the RANGE, not only pooled: the protocol asks for it, and a model
    # that nails mid-range while failing the ends posts a fine pooled number.
    order = np.argsort([r["target"] for r in rows])
    thirds = np.array_split(order, 3)
    def _third_mae(part):
        # A third can be empty (n < 3) or entirely dead (every target produced zero valid
        # molecules), and np.nanmean warns and returns NaN for both. NaN is the honest
        # answer; the warning is noise that has appeared in real runs.
        vals = [rows[i]["mae"] for i in part if np.isfinite(rows[i]["mae"])]
        return float(np.mean(vals)) if vals else float("nan")

    by_third = [_third_mae(part) for part in thirds]

    summary = {
        "adapter": args.adapter or adapter_key, "adapter_ckpt": args.adapter_ckpt,
        "adapter_config": (adapter_module._config() if adapter_module is not None else None),
        "base": args.base, "property": args.property,
        "split": args.split, "method": args.method, "seed": args.seed,
        "n_targets": len(rows), "per_target": args.per_target,
        "sampling": {"weight": args.weight, "steps": args.steps, "eta": args.eta,
                     "omega": args.omega, "time_distortion": args.time_distortion,
                     "blend_space": args.blend_space},
        # None means the negative branch was the frozen unconditional base (plain CFG).
        "guide": args.guide,
        # Recorded so a pair of runs can be read as an ablation without anyone having to
        # remember which was which.
        "size": {"mode": args.size_mode, "model": args.size_model,
                 "grid": ([size_model.min_size, size_model.max_size]
                          if size_model is not None else None)},
        "fk": ({"beta": args.fk_beta, "warmup_frac": args.fk_warmup,
                "ess_frac": args.fk_ess, "rejuvenate": args.fk_rejuvenate,
                "jump_length": args.fk_jump} if args.method == "fk" else None),
        "mae_pooled": float(np.mean(all_err)) if all_err else float("nan"),
        "mae_per_target_mean": float(finite.mean()) if finite.size else float("nan"),
        "mae_low_third": by_third[0], "mae_mid_third": by_third[1],
        "mae_high_third": by_third[2],
        "validity": float(val.mean()),
        "uniqueness": float(uniq.mean()) if uniq.size else float("nan"),
        "per_target": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))

    print()
    print(f"=== E2 {args.property} / {args.method} / {args.split} / "
          f"size={args.size_mode} / "
          f"{'autoguidance' if args.guide else 'CFG'} w={args.weight} ===")
    print(f"  MAE (pooled over {len(all_err)} molecules) {summary['mae_pooled']:.4f}")
    print(f"  MAE by target third   low {by_third[0]:.4f}  mid {by_third[1]:.4f}  "
          f"high {by_third[2]:.4f}")
    print(f"  validity   {summary['validity']:.4f}")
    print(f"  uniqueness {summary['uniqueness']:.4f}"
          + ("   <- FK collapse check; well below 1.0 means duplicated particles"
             if args.method == "fk" else ""))
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
