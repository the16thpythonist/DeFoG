#!/usr/bin/env python
"""
Head-to-head comparison of two fingerprint adapters at a MATCHED metric width.

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

Usage:
    python scripts/compare_fp_adapters.py \\
        --base ckpts/zinc_rl2_seed42/best_model \\
        --v2 <path>/fp_adapter --v2-bits 512 \\
        --v3 <path>/fp_adapter --v3-bits 1024 --v3-counts \\
        --n-per-target 64
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
METRIC_WIDTHS = (512, 1024)


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
    mols = []
    for s in samples:
        mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
        smi = mol_to_smiles(mol) if mol is not None else None
        if smi and Chem.MolFromSmiles(smi) is not None:
            mols.append(Chem.MolFromSmiles(smi))
    return mols


def sample_n(sampler, n, chunk, device):
    out, rem = [], n
    while rem > 0:
        cur = min(chunk, rem)
        out += sampler.sample(cur, device=device, show_progress=False)
        rem -= cur
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--v2", required=True); ap.add_argument("--v2-bits", type=int, default=512)
    ap.add_argument("--v2-counts", action="store_true")
    ap.add_argument("--v3", required=True); ap.add_argument("--v3-bits", type=int, default=1024)
    ap.add_argument("--v3-counts", action="store_true")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--n-per-target", type=int, default=64)
    ap.add_argument("--n-baseline", type=int, default=256)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eta", type=float, default=25.0)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = DeFoGModel.load(args.base, device="cpu").to(device).eval()
    atoms, bonds, adec, bdec, _rep, msg = vocabulary.resolve_and_check(
        zref, base, None, what=args.base)
    print(msg)

    adapters = {}
    for name, path, bits, counts in (("v2", args.v2, args.v2_bits, args.v2_counts),
                                     ("v3", args.v3, args.v3_bits, args.v3_counts)):
        ad = AdaLNAdapter.load(path).to(device).eval()
        adapters[name] = (ad, bits, counts)
        print(f"{name}: {path}  bits={bits} counts={counts} "
              f"cond_dim={getattr(ad, 'cond_dim', '?')}")

    tmols = [Chem.MolFromSmiles(s) for s in TARGETS]

    print(f"\nunconditional baseline: {args.n_baseline} samples")
    bsamp = sample_n(Sampler(base, eta=args.eta, omega=0.0, sample_steps=args.steps,
                             time_distortion="polydec"),
                     args.n_baseline, args.chunk, device)
    bmols = decode(bsamp, adec, bdec)
    print(f"  valid {len(bmols)}/{args.n_baseline}")

    results = {"widths": list(METRIC_WIDTHS), "targets": TARGETS, "per_adapter": {}}
    for name, (ad, bits, counts) in adapters.items():
        print(f"\n=== {name} ===")
        agg = {w: {"baseline": [], "steered": []} for w in METRIC_WIDTHS}
        per_target = []
        for ti, (smi, tmol) in enumerate(zip(TARGETS, tmols)):
            cond = condition_vector(tmol, bits, args.radius, counts)
            comp = AdapterComposition(
                [ConditionBranch(ad, torch.as_tensor(cond, dtype=torch.float32), 1.0)],
                base=base, mode="product")
            samp = AdaptedSampler(base, comp, eta=args.eta, omega=0.0,
                                  sample_steps=args.steps, time_distortion="polydec")
            gmols = decode(sample_n(samp, args.n_per_target, args.chunk, device),
                           adec, bdec)
            row = {"smiles": smi, "n_heavy": tmol.GetNumHeavyAtoms(),
                   "validity": len(gmols) / args.n_per_target}
            for w in METRIC_WIDTHS:
                tgt = binary_fp(tmol, w, args.radius)
                gen = np.stack([binary_fp(m, w, args.radius) for m in gmols]) \
                    if gmols else np.zeros((0, w), dtype=np.float32)
                bse = np.stack([binary_fp(m, w, args.radius) for m in bmols])
                st = tanimoto(gen, tgt); bl = tanimoto(bse, tgt)
                agg[w]["steered"].extend(st.tolist()); agg[w]["baseline"].extend(bl.tolist())
                row[f"mean_T@{w}"] = float(st.mean()) if st.size else float("nan")
                row[f"baseline_T@{w}"] = float(bl.mean())
                row[f"lift@{w}"] = row[f"mean_T@{w}"] - row[f"baseline_T@{w}"]
            per_target.append(row)
            print(f"  [{ti}] {tmol.GetNumHeavyAtoms():2d} atoms  valid {row['validity']:.3f}  "
                  + "  ".join(f"lift@{w}={row[f'lift@{w}']:+.3f}" for w in METRIC_WIDTHS))
        results["per_adapter"][name] = {
            "bits": bits, "counts": counts, "per_target": per_target,
            "aggregate": {str(w): {
                "baseline": float(np.mean(agg[w]["baseline"])),
                "steered": float(np.mean(agg[w]["steered"])),
                "lift": float(np.mean(agg[w]["steered"]) - np.mean(agg[w]["baseline"])),
            } for w in METRIC_WIDTHS},
        }

    print("\n" + "=" * 66)
    print("MATCHED-WIDTH COMPARISON (metric width held fixed across adapters)")
    print("=" * 66)
    print(f"{'metric':>8s}{'adapter':>9s}{'baseline':>10s}{'steered':>10s}{'lift':>9s}"
          f"{'corr(size,lift)':>18s}")
    for w in METRIC_WIDTHS:
        for name in ("v2", "v3"):
            a = results["per_adapter"][name]
            agg = a["aggregate"][str(w)]
            sizes = np.array([r["n_heavy"] for r in a["per_target"]])
            lifts = np.array([r[f"lift@{w}"] for r in a["per_target"]])
            c = float(np.corrcoef(sizes, lifts)[0, 1])
            print(f"{w:>8d}{name:>9s}{agg['baseline']:>10.4f}{agg['steered']:>10.4f}"
                  f"{agg['lift']:>+9.4f}{c:>18.3f}")
    print()
    print("Read the rows WITHIN a metric width. Across widths the numbers are not")
    print("comparable -- that is the whole reason this script exists.")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
