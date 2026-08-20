#!/usr/bin/env python
"""
Head-to-head comparison of N fingerprint adapters at MATCHED metric widths.

Why this exists
---------------
The v3 run changed two things at once: the conditioning encoding (binary ->
log1p counts) and the fingerprint width (512 -> 1024). I was careful to keep the
Tanimoto METRIC binary so v2 and v3 would stay comparable -- and that precaution
was incomplete. The metric is binary in both, but it is computed at the
adapter's own width, and Tanimoto at 1024 bits is systematically lower than at
512 because fewer hash collisions mean less spurious overlap. The v2 and v3
baselines duly differ, 0.150 against 0.128.

So "v3 lift +0.194 against v2 +0.173" compares two numbers on different scales
and means nothing on its own. Worse, the size-vs-lift correlation is affected
too: higher resolution removes proportionally more spurious overlap from large
molecules, which is exactly the axis under test.

This script fixes that by generating from both adapters on the SAME targets and
scoring every sample at BOTH widths. Each adapter is still CONDITIONED the way
it was trained -- that is a property of the adapter, not of the comparison --
while the metric is held fixed across both.

The same trap reappeared in the bottleneck ablation (job 1288793): its in-job control
runs at 1024 bits while the other three arms run at 2048, and the experiment scores
each arm at its own FP_BITS. An in-job control removes every cross-job difference
except the one the control is defined by, so this script is still needed.

Usage:
    python scripts/compare_fp_adapters.py \\
        --base ckpts/zinc_rl2_seed42/best_model \\
        --adapter ctrl:<path>/fp_adapter \\
        --adapter enc:<path>/fp_adapter \\
        --n-per-target 64
Bit width defaults to each checkpoint's own cond_dim, so it cannot be mistyped.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from defog.core import (  # noqa: E402
    AdaLNAdapter, AdapterComposition, AdaptedSampler, ConditionBranch,
    DeFoGModel, Sampler,
)
from defog.data import vocabulary, zinc_reference as zref  # noqa: E402
from defog.domains.molecule import pyg_data_to_mol, mol_to_smiles  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# The six held-out targets both runs used. Hard-coded so the comparison cannot
# drift with a reshuffled holdout -- they are what the two runs were scored on.
TARGETS = [
    "COCCNC(=O)c1ccc2c(c1)OCO2",
    "Cc1ccc(CN(Cc2ccco2)S(=O)(=O)c2c(C)noc2C)s1",
    "CCc1nnc(NC(=O)C(C)(C)Nc2ccc([N+](=O)[O-])cc2)s1",
    "C[NH+]1CCN(Cn2cc(Br)cn2)[C@H](c2ccccc2)C1",
    "Cc1ccc(C(=O)N[C@@H](C)c2cn(C)nc2C)cc1[N+](=O)[O-]",
    "O=C(CCn1cnc2c(-c3ccccc3)noc2c1=O)Nc1ccccc1Cl",
]
#: Metric widths every adapter is scored at. Absolute Tanimoto is not comparable
#: across these -- only rows WITHIN one width may be compared -- but scoring every
#: adapter at every width is what makes the comparison possible at all. Widened at
#: runtime to include each adapter's own width.
METRIC_WIDTHS = (512, 1024, 2048)


def condition_vector(mol, bits, radius, counts):
    """Exactly what the adapter was trained on -- must match molsmith's
    morgan_bits and the training script's mol_morgan_bits."""
    arr = np.zeros((bits,), dtype=np.float32)
    if counts:
        cv = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=bits)
        DataStructs.ConvertToNumpyArray(cv, arr)
        return np.log1p(arr).astype(np.float32)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def binary_fp(mol, bits, radius=2):
    """The METRIC. Always binary, at an explicitly chosen width."""
    arr = np.zeros((bits,), dtype=np.float32)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def tanimoto(mat, target):
    if mat.size == 0:
        return np.zeros((0,), dtype=np.float32)
    inter = mat @ target
    return inter / np.clip(mat.sum(1) + target.sum() - inter, 1e-8, None)


def decode(samples, atom_decoder, bond_decoder):
    """(molecules, disconnected fraction).

    Disconnected molecules are KEPT in the scored pool. That is not an endorsement --
    it preserves comparability with every number already measured on this axis, all of
    which were computed this way. What changes is that the fraction is now REPORTED
    instead of invisible: "validity" here means "decodes to a parseable SMILES", and a
    multi-fragment result passes that test while being useless as a molecule.

    It matters more than it sounds: the frozen base emits fragments ~1.6% of the time
    unconditionally, and fingerprint conditioning pushes that to ~15%. A validity
    column alone reports ~0.98 for both and hides the entire effect.
    """
    mols, n_disc = [], 0
    for s in samples:
        mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
        smi = mol_to_smiles(mol) if mol is not None else None
        if smi and Chem.MolFromSmiles(smi) is not None:
            mols.append(Chem.MolFromSmiles(smi))
            n_disc += "." in smi
    return mols, (n_disc / len(mols) if mols else float("nan"))


def sample_n(sampler, n, chunk, device, num_nodes=None):
    """``num_nodes`` pins every generated graph to that many heavy atoms.

    Used only by the SIZE-MATCHED companion metric. It is a strictly easier task:
    bit count correlates with molecule size, so "how big should this be" is part of
    what the fingerprint is supposed to convey, and supplying it externally hands
    the model an answer it would otherwise have to infer. That is why this is an
    additional column and never the primary number.
    """
    out, rem = [], n
    while rem > 0:
        cur = min(chunk, rem)
        out += sampler.sample(cur, num_nodes=num_nodes, device=device, show_progress=False)
        rem -= cur
    return out


def parse_adapter(spec: str):
    """``name:path[:bits[:counts]]``.

    bits defaults to the adapter's own ``cond_dim`` -- the checkpoint knows its width,
    and a hand-typed one that disagrees would silently feed the adapter a condition of
    the wrong length. counts defaults to true, which is what every adapter trained
    since the encoding switch uses; pass ``:0`` for a legacy binary one.
    """
    parts = spec.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            f"--adapter wants name:path[:bits[:counts]], got {spec!r}")
    name, path = parts[0], parts[1]
    bits = int(parts[2]) if len(parts) > 2 and parts[2] else None
    counts = True if len(parts) < 4 else parts[3] not in ("0", "false", "False", "binary")
    return name, path, bits, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", action="append", type=parse_adapter, default=[],
                    metavar="NAME:PATH[:BITS[:COUNTS]]",
                    help="Repeatable. Every adapter is CONDITIONED as it was trained "
                         "but SCORED at every metric width, which is the entire point.")
    # Back-compatible two-adapter form.
    ap.add_argument("--v2", default=None); ap.add_argument("--v2-bits", type=int, default=512)
    ap.add_argument("--v2-counts", action="store_true")
    ap.add_argument("--v3", default=None); ap.add_argument("--v3-bits", type=int, default=1024)
    ap.add_argument("--v3-counts", action="store_true")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--n-per-target", type=int, default=64)
    ap.add_argument("--n-baseline", type=int, default=256)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eta", type=float, default=25.0)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--size-matched", action="store_true",
                    help="ALSO report Tanimoto with every generated molecule pinned to "
                         "the target's heavy-atom count. An additional column, never the "
                         "primary number: supplying the size hands the model information "
                         "the fingerprint is meant to convey, so it measures an easier "
                         "task and is not comparable to the free-size figures.")
    ap.add_argument("--n-size-baseline", type=int, default=128,
                    help="Unconditional samples drawn AT EACH TARGET'S SIZE for the "
                         "size-matched baseline. Cannot reuse the shared baseline: a "
                         "size-matched numerator over a free-size denominator would book "
                         "the size effect as steering.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = DeFoGModel.load(args.base, device="cpu").to(device).eval()
    atoms, bonds, adec, bdec, _rep, msg = vocabulary.resolve_and_check(
        zref, base, None, what=args.base)
    print(msg)

    specs = list(args.adapter)
    if args.v2:
        specs.append(("v2", args.v2, args.v2_bits, args.v2_counts))
    if args.v3:
        specs.append(("v3", args.v3, args.v3_bits, args.v3_counts))
    if len(specs) < 2:
        ap.error("need at least two adapters (--adapter name:path, repeatable)")

    adapters = {}
    for name, path, bits, counts in specs:
        ad = AdaLNAdapter.load(path).to(device).eval()
        cond_dim = int(getattr(ad, "cond_dim", 0) or 0)
        if bits is None:
            bits = cond_dim
        elif cond_dim and bits != cond_dim:
            # Conditioning an adapter at the wrong width raises nothing useful; it
            # feeds the trunk a vector of the wrong length or, worse, the right length
            # of wrong numbers. The checkpoint is the authority.
            raise SystemExit(f"{name}: --adapter says {bits} bits but the checkpoint's "
                             f"cond_dim is {cond_dim}; refusing to guess")
        adapters[name] = (ad, bits, counts)
        enc = getattr(ad, "cond_encoder", None)
        print(f"{name}: {path}\n     bits={bits} counts={counts} hidden={getattr(ad,'hidden','?')} "
              f"interior_ff={getattr(ad,'interior_ff','?')} "
              f"encoder={type(enc).__name__ if enc is not None else 'none'}")

    tmols = [Chem.MolFromSmiles(s) for s in TARGETS]

    print(f"\nunconditional baseline: {args.n_baseline} samples")
    bsamp = sample_n(Sampler(base, eta=args.eta, omega=0.0, sample_steps=args.steps,
                             time_distortion="polydec"),
                     args.n_baseline, args.chunk, device)
    bmols, base_disc = decode(bsamp, adec, bdec)
    print(f"  valid {len(bmols)}/{args.n_baseline}  disconnected {base_disc:.3f}"
          f"   <- the frozen base's own fragment rate; conditioning is measured against this")

    # Size-matched baselines depend only on the FROZEN base and the atom count, not
    # on which adapter is being scored -- so draw each size once and share it. With
    # six adapters this is the difference between 6x and 1x the baseline sampling.
    size_baseline_cache = {}

    def size_baseline(n_at):
        if n_at not in size_baseline_cache:
            mols, _ = decode(
                sample_n(Sampler(base, eta=args.eta, omega=0.0, sample_steps=args.steps,
                                 time_distortion="polydec"),
                         args.n_size_baseline, args.chunk, device, num_nodes=n_at),
                adec, bdec)
            size_baseline_cache[n_at] = mols
            print(f"    (size baseline @{n_at} atoms: {len(mols)} valid, cached)")
        return size_baseline_cache[n_at]

    widths = tuple(sorted(set(METRIC_WIDTHS) | {b for _, b, _ in adapters.values()}))
    print(f"scoring every adapter at metric widths {widths}")
    results = {"widths": list(widths), "targets": TARGETS, "per_adapter": {}}
    for name, (ad, bits, counts) in adapters.items():
        print(f"\n=== {name} ===")
        agg = {w: {"baseline": [], "steered": [], "sm_baseline": [], "sm_steered": []}
               for w in widths}
        per_target = []
        for ti, (smi, tmol) in enumerate(zip(TARGETS, tmols)):
            cond = condition_vector(tmol, bits, args.radius, counts)
            comp = AdapterComposition(
                [ConditionBranch(ad, torch.as_tensor(cond, dtype=torch.float32), 1.0)],
                base=base, mode="product")
            samp = AdaptedSampler(base, comp, eta=args.eta, omega=0.0,
                                  sample_steps=args.steps, time_distortion="polydec")
            gmols, gdisc = decode(sample_n(samp, args.n_per_target, args.chunk, device),
                                  adec, bdec)
            n_heavy = tmol.GetNumHeavyAtoms()
            row = {"smiles": smi, "n_heavy": n_heavy,
                   "validity": len(gmols) / args.n_per_target,
                   "disconnected": gdisc}

            # --- companion metric: everything pinned to the target's atom count ---
            # Both the steered sample AND its baseline are re-drawn at this size. A
            # size-matched numerator against a free-size baseline would fold the size
            # effect into the "lift" and report it as steering.
            smols = bmols_sm = None
            if args.size_matched:
                smols, sdisc = decode(
                    sample_n(samp, args.n_per_target, args.chunk, device, num_nodes=n_heavy),
                    adec, bdec)
                bmols_sm = size_baseline(n_heavy)
                row["sm_validity"] = len(smols) / args.n_per_target
                row["sm_disconnected"] = sdisc
                row["sm_mean_size"] = float(np.mean([m.GetNumHeavyAtoms() for m in smols])) \
                    if smols else float("nan")
            # What size does the model choose on its own? Never measured before, and
            # it is the more interesting question: if the free-size generations already
            # match the target, pinning the size cannot buy anything.
            row["free_mean_size"] = float(np.mean([m.GetNumHeavyAtoms() for m in gmols])) \
                if gmols else float("nan")

            for w in widths:
                tgt = binary_fp(tmol, w, args.radius)
                gen = np.stack([binary_fp(m, w, args.radius) for m in gmols]) \
                    if gmols else np.zeros((0, w), dtype=np.float32)
                bse = np.stack([binary_fp(m, w, args.radius) for m in bmols])
                st = tanimoto(gen, tgt); bl = tanimoto(bse, tgt)
                agg[w]["steered"].extend(st.tolist()); agg[w]["baseline"].extend(bl.tolist())
                row[f"mean_T@{w}"] = float(st.mean()) if st.size else float("nan")
                row[f"baseline_T@{w}"] = float(bl.mean())
                row[f"lift@{w}"] = row[f"mean_T@{w}"] - row[f"baseline_T@{w}"]
                if args.size_matched:
                    sgen = np.stack([binary_fp(m, w, args.radius) for m in smols]) \
                        if smols else np.zeros((0, w), dtype=np.float32)
                    sbse = np.stack([binary_fp(m, w, args.radius) for m in bmols_sm]) \
                        if bmols_sm else np.zeros((0, w), dtype=np.float32)
                    sst = tanimoto(sgen, tgt); sbl = tanimoto(sbse, tgt)
                    agg[w]["sm_steered"].extend(sst.tolist())
                    agg[w]["sm_baseline"].extend(sbl.tolist())
                    row[f"sm_mean_T@{w}"] = float(sst.mean()) if sst.size else float("nan")
                    row[f"sm_baseline_T@{w}"] = float(sbl.mean()) if sbl.size else float("nan")
                    row[f"sm_lift@{w}"] = row[f"sm_mean_T@{w}"] - row[f"sm_baseline_T@{w}"]
            per_target.append(row)
            msg = (f"  [{ti}] {n_heavy:2d} atoms  valid {row['validity']:.3f}  "
                   f"disc {row['disconnected']:.3f}  gen_size {row['free_mean_size']:.1f}  "
                   + "  ".join(f"lift@{w}={row[f'lift@{w}']:+.3f}" for w in widths))
            if args.size_matched:
                msg += ("   || size-matched: "
                        + "  ".join(f"lift@{w}={row[f'sm_lift@{w}']:+.3f}" for w in widths))
            print(msg)
        results["per_adapter"][name] = {
            "bits": bits, "counts": counts, "per_target": per_target,
            "baseline_disconnected": base_disc,
            "disconnected": float(np.nanmean([r["disconnected"] for r in per_target])),
            "mean_free_size": float(np.nanmean([r["free_mean_size"] for r in per_target])),
            "mean_target_size": float(np.mean([r["n_heavy"] for r in per_target])),
            "aggregate": {str(w): {
                "baseline": float(np.mean(agg[w]["baseline"])),
                "steered": float(np.mean(agg[w]["steered"])),
                "lift": float(np.mean(agg[w]["steered"]) - np.mean(agg[w]["baseline"])),
                **({"sm_baseline": float(np.mean(agg[w]["sm_baseline"])),
                    "sm_steered": float(np.mean(agg[w]["sm_steered"])),
                    "sm_lift": float(np.mean(agg[w]["sm_steered"])
                                     - np.mean(agg[w]["sm_baseline"]))}
                   if agg[w]["sm_steered"] else {}),
            } for w in widths},
        }

    print("\n" + "=" * 66)
    print("MATCHED-WIDTH COMPARISON (metric width held fixed across adapters)")
    print("=" * 66)
    print(f"{'metric':>8s}{'adapter':>10s}{'baseline':>10s}{'steered':>10s}{'lift':>9s}"
          f"{'validity':>10s}{'disc':>8s}{'corr(size,lift)':>18s}")
    for w in widths:
        rows = []
        for name in adapters:
            a = results["per_adapter"][name]
            agg = a["aggregate"][str(w)]
            sizes = np.array([r["n_heavy"] for r in a["per_target"]])
            lifts = np.array([r[f"lift@{w}"] for r in a["per_target"]])
            c = float(np.corrcoef(sizes, lifts)[0, 1]) if len(sizes) > 1 else float("nan")
            val = float(np.mean([r["validity"] for r in a["per_target"]]))
            rows.append((name, agg, c, val, a["disconnected"]))
        best = max(rows, key=lambda r: r[1]["lift"])[0]
        for name, agg, c, val, disc in rows:
            mark = " *" if name == best and len(rows) > 1 else "  "
            print(f"{w:>8d}{name:>10s}{agg['baseline']:>10.4f}{agg['steered']:>10.4f}"
                  f"{agg['lift']:>+9.4f}{val:>10.4f}{disc:>8.3f}{c:>16.3f}{mark}")
    print()
    print("Read the rows WITHIN a metric width. Across widths the numbers are not")
    print("comparable -- that is the whole reason this script exists. (* = best lift")
    print("at that width; it is a pointer, not a significance claim -- run-to-run")
    print("variation for an IDENTICAL config has been measured at ~0.012 lift.)")
    print()
    if args.size_matched:
        print()
        print("=" * 78)
        print("SIZE-MATCHED COMPANION METRIC (generation pinned to the target's atom count)")
        print("=" * 78)
        print(f"{'metric':>8s}{'adapter':>10s}{'free lift':>11s}{'sm lift':>10s}"
              f"{'delta':>9s}{'free size':>11s}{'target':>8s}")
        for w in widths:
            for name in adapters:
                a = results["per_adapter"][name]; g = a["aggregate"][str(w)]
                if "sm_lift" not in g:
                    continue
                print(f"{w:>8d}{name:>10s}{g['lift']:>+11.4f}{g['sm_lift']:>+10.4f}"
                      f"{g['sm_lift']-g['lift']:>+9.4f}"
                      f"{a['mean_free_size']:>11.1f}{a['mean_target_size']:>8.1f}")
        print()
        # Mean-vs-mean is not evidence: a model that always emits ~22-atom molecules
        # matches the AVERAGE target size while missing every individual target. The
        # per-target correlation is the actual test of whether size is being inferred.
        for name in adapters:
            a = results["per_adapter"][name]
            tn = np.array([r["n_heavy"] for r in a["per_target"]], dtype=float)
            gn = np.array([r["free_mean_size"] for r in a["per_target"]], dtype=float)
            ok = np.isfinite(gn)
            c = float(np.corrcoef(tn[ok], gn[ok])[0, 1]) if ok.sum() > 1 else float("nan")
            a["corr_target_vs_generated_size"] = c
            print(f"  {name:>8s}: corr(target size, generated size) = {c:+.3f}"
                  + ("   <- size IS being inferred from the fingerprint" if c > 0.5 else
                     "   <- size is NOT being inferred; generation ignores target size"
                     if c < 0.2 else "   <- size only weakly inferred"))
        print()
        print("NOT comparable to the free-size lift: pinning the size supplies information")
        print("the fingerprint is supposed to carry, so this measures an easier task. Read")
        print("`delta` as the headroom size mismatch was costing, and `free size` vs")
        print("`target` as whether the model was inferring the right size on its own.")
        print()
    print("disc = fraction of the SCORED molecules that are multi-fragment. They are")
    print("counted as valid and are included in the Tanimoto pool, as in every earlier")
    print("measurement on this axis -- so the lifts stay comparable, but a high disc")
    print("means the validity column is flattering the adapter.")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
