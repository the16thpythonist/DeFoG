"""
Calibrate the E1 validity harness against DeFoG's *published* samples.

This is protocol section 8's "cheap insurance", in its strongest form. Rather
than training a model and hoping its numbers land near a published row -- which
confounds harness correctness with model quality -- this scores DeFoG's own
released ``generated_samples.pkl`` with our metric code and checks we reproduce
the numbers in their own ``results_report.txt``.

If the numbers match, the decode path, both validity conventions, the
largest-fragment rule and the uniqueness denominator are all verified against
the reference implementation, with no GPU and no training involved. If they do
not match, we have found a harness bug before it silently became a paper claim.

Samples come from https://drive.switch.ch/index.php/s/MG7y2EZoithAywE (the link
in CLAUDE.md); each dataset directory holds ``generated_samples.pkl`` and the
matching ``results_report.txt``.

Note the archive's reports carry ``fcd: (-1.0, 0.0)`` -- the sentinel written
when FCD is disabled -- so FCD cannot be calibrated from this source.

Usage:
    python scripts/calibrate_e1_validity.py --samples <path/to/generated_samples.pkl> \
        [--report <path/to/results_report.txt>] [--folds 5] [--limit N]
"""

from __future__ import annotations

import argparse
import pickle
import re
import statistics
import sys

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from defog.data import zinc_reference
from defog.domains.molecule import build_encoders, validity_report


def dense_to_pyg(atom_types: torch.Tensor, edge_types: torch.Tensor, num_atom_classes: int,
                 num_bond_classes: int) -> Data:
    """``[atom_types(n), edge_types(n,n)]`` -> PyG ``Data`` with one-hot features.

    DeFoG stores samples as dense class indices; our metric code consumes the
    one-hot PyG form the training path produces. Masked-out nodes are encoded as
    -1 by ``mask(collapse=True)``, so they are dropped here rather than being
    read as class -1.
    """
    keep = atom_types >= 0
    atom_types = atom_types[keep]
    edge_types = edge_types[keep][:, keep]
    n = atom_types.numel()

    x = F.one_hot(atom_types.long(), num_classes=num_atom_classes).float()

    # Upper triangle only, then mirrored -- the graph is undirected and storing
    # both directions from a symmetric matrix would double-count nothing but is
    # what the encoder produces, so mirror explicitly.
    iu = torch.triu_indices(n, n, offset=1)
    et = edge_types[iu[0], iu[1]]
    present = et > 0
    src, dst, cls = iu[0][present], iu[1][present], et[present]

    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    edge_attr = F.one_hot(torch.cat([cls, cls]).long(), num_classes=num_bond_classes).float()
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def parse_report(path) -> dict:
    """Pull ``key: (mean, std)`` pairs out of a DeFoG results_report.txt."""
    out = {}
    pattern = re.compile(r"^([^:]+):\s*\(([-\d.eE+]+),\s*([-\d.eE+]+)\)")
    with open(path) as fh:
        for line in fh:
            m = pattern.match(line.strip())
            if m:
                out[m.group(1).strip()] = (float(m.group(2)), float(m.group(3)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--folds", type=int, default=5,
                    help="DeFoG reports mean/std over num_sample_fold folds.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Score only the first N samples (quick check).")
    ap.add_argument("--tolerance", type=float, default=0.005,
                    help="Absolute agreement required, in metric units.")
    args = ap.parse_args()

    _, atom_decoder, _, bond_decoder = build_encoders(
        zinc_reference.ATOM_TYPES, zinc_reference.BOND_TYPES
    )
    n_atom_cls = len(atom_decoder)
    n_bond_cls = len(bond_decoder)

    with open(args.samples, "rb") as fh:
        samples = pickle.load(fh)
    if args.limit:
        samples = samples[: args.limit]
    print(f"loaded {len(samples)} samples from {args.samples}")

    folds = max(1, args.folds if not args.limit else 1)
    per_fold = len(samples) // folds
    print(f"scoring {folds} fold(s) x {per_fold}")

    keys = ("validity_strict_largest_frag", "validity_relaxed_largest_frag", "uniqueness")
    collected = {k: [] for k in keys}

    for f in range(folds):
        chunk = samples[f * per_fold : (f + 1) * per_fold]
        graphs = [dense_to_pyg(a, e, n_atom_cls, n_bond_cls) for a, e in chunk]
        rep = validity_report(graphs, atom_decoder, bond_decoder)
        for k in keys:
            collected[k].append(rep[k])
        print(f"  fold {f}: strict={rep['validity_strict_largest_frag']:.5f} "
              f"relaxed={rep['validity_relaxed_largest_frag']:.5f} "
              f"uniq={rep['uniqueness']:.5f}")

    ours = {k: (statistics.fmean(v),
                statistics.pstdev(v) if len(v) > 1 else 0.0)
            for k, v in collected.items()}

    print("\n" + "=" * 74)
    print(f"{'metric':<34}{'ours':>13}{'published':>13}{'delta':>12}")
    print("=" * 74)

    if not args.report:
        for k, (m, s) in ours.items():
            print(f"{k:<34}{m:>13.5f}{'-':>13}{'-':>12}")
        return 0

    pub = parse_report(args.report)
    mapping = {
        "validity_strict_largest_frag": "Validity",
        "validity_relaxed_largest_frag": "Relaxed Validity",
        "uniqueness": "Uniqueness",
    }
    worst, rows = 0.0, 0
    for ours_key, pub_key in mapping.items():
        m, s = ours[ours_key]
        if pub_key not in pub:
            print(f"{ours_key:<34}{m:>13.5f}{'ABSENT':>13}{'-':>12}")
            continue
        pm, _ = pub[pub_key]
        delta = m - pm
        worst = max(worst, abs(delta))
        rows += 1
        flag = "" if abs(delta) <= args.tolerance else "   <-- MISMATCH"
        print(f"{ours_key:<34}{m:>13.5f}{pm:>13.5f}{delta:>+12.5f}{flag}")

    print("=" * 74)
    if rows and worst <= args.tolerance:
        print(f"CALIBRATED: all {rows} metrics within {args.tolerance} "
              f"(worst {worst:.5f}).")
        return 0
    print(f"NOT CALIBRATED: worst deviation {worst:.5f} exceeds {args.tolerance}. "
          f"Resolve before trusting any harness number.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
