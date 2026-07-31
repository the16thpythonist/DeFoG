"""
Sweep sampling parameters (steps / eta / omega) on the VALIDATION split.

Protocol section 5. DeFoG's central design property is that the rate matrix is
assembled at sampling time, so steps, stochasticity eta and target-guidance
omega are tunable without retraining. That freedom is also a way to overfit a
benchmark, hence the rule this script exists to enforce:

    sweep on validation -> freeze the winner -> ONE evaluation pass on test

**This script never touches the test split.** It samples from a trained
checkpoint at each grid point and writes the SMILES out; scoring happens
afterwards against the VALIDATION reference, via scripts/e1_metrics.py in the
isolated metrics environment.

The split is deliberate. Sampling needs a GPU and the training env; FCD, NSPDK
and the GuacaMol/MOSES suites need the metrics env, which is x86-only and would
not install on JUPITER's aarch64. Dumping SMILES in between also makes scoring
re-runnable without re-sampling -- so a metric added later can be applied to a
sweep that already ran.

One caveat on comparing FCD across grid points: FCD is strongly biased by sample
count (measured ~1/n), so absolute values from a sweep at n=1000 are not
comparable to a final table at n=10000. They ARE comparable to each other, since
every grid point uses the same n -- which is all a sweep needs.

Usage (one GPU, a slice of the grid):
    python scripts/sweep_sampling.py --ckpt ckpts/zinc_e1_seed42/best_model \\
        --dataset zinc --slice 0/4 --num-samples 1000 --out-dir sweep_zinc
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import torch

from defog.core import DeFoGModel
from defog.data import guacamol_reference as gmref
from defog.data import moses_reference as mref
from defog.data import zinc_reference as zref
from defog.domains.molecule import build_encoders, validity_report

REFERENCES = {"zinc": zref, "guacamol": gmref, "moses": mref}

#: Default grid. steps covers the two operating points DeFoG reports (50 and
#: 500); eta and omega bracket the ranges its appendix explores. 32 points.
DEFAULT_STEPS = [50, 500]
DEFAULT_ETA = [0.0, 5.0, 25.0, 50.0]
DEFAULT_OMEGA = [0.0, 0.05, 0.1, 0.25]


def parse_slice(text: str):
    idx, total = text.split("/")
    idx, total = int(idx), int(total)
    if not 0 <= idx < total:
        raise ValueError(f"bad slice {text}")
    return idx, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Checkpoint WITHOUT the .ckpt suffix, e.g. "
                         "ckpts/zinc_e1_seed42/best_model")
    ap.add_argument("--dataset", required=True, choices=sorted(REFERENCES))
    ap.add_argument("--slice", default="0/1", help="i/N: which part of the grid")
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--steps", default=None, help="comma-separated override")
    ap.add_argument("--eta", default=None)
    ap.add_argument("--omega", default=None)
    ap.add_argument("--time-distortion", default="polydec")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    steps = [int(x) for x in args.steps.split(",")] if args.steps else DEFAULT_STEPS
    etas = [float(x) for x in args.eta.split(",")] if args.eta else DEFAULT_ETA
    omegas = [float(x) for x in args.omega.split(",")] if args.omega else DEFAULT_OMEGA

    grid = list(itertools.product(steps, etas, omegas))
    idx, total = parse_slice(args.slice)
    mine = grid[idx::total]
    print(f"grid={len(grid)} points; this slice ({args.slice}) runs {len(mine)}",
          flush=True)

    mod = REFERENCES[args.dataset]
    _, atom_decoder, _, bond_decoder = build_encoders(mod.ATOM_TYPES, mod.BOND_TYPES)

    # The VALIDATION split is the only reference this script loads. Novelty is
    # measured against train, as usual; test is never read.
    split = mod.load_reference_split()
    val_smiles = split.val_smiles
    train_ref = set(split.train_smiles)
    os.makedirs(args.out_dir, exist_ok=True)
    val_path = os.path.join(args.out_dir, "_validation_reference.smi")
    if not os.path.exists(val_path):
        with open(val_path, "w") as fh:
            fh.write("\n".join(val_smiles) + "\n")
        print(f"wrote validation reference ({len(val_smiles)}) -> {val_path}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeFoGModel.load(args.ckpt).to(device).eval()
    print(f"loaded {args.ckpt} on {device}", flush=True)

    for n_steps, eta, omega in mine:
        tag = f"steps{n_steps}_eta{eta:g}_omega{omega:g}"
        out_json = os.path.join(args.out_dir, f"{tag}.json")
        out_smi = os.path.join(args.out_dir, f"{tag}.smi")
        if os.path.exists(out_json):
            print(f"[skip] {tag} already done", flush=True)
            continue

        t0 = time.time()
        samples, remaining = [], args.num_samples
        while remaining > 0:
            cur = min(args.chunk, remaining)
            samples += model.sample(num_samples=cur, sample_steps=n_steps,
                                    eta=eta, omega=omega,
                                    time_distortion=args.time_distortion,
                                    device=device, show_progress=False)
            remaining -= cur
        sample_s = time.time() - t0

        rep = validity_report(samples, atom_decoder, bond_decoder,
                              reference_smiles=train_ref)
        smiles = rep.pop("smiles")
        with open(out_smi, "w") as fh:
            fh.write("\n".join(smiles) + "\n")

        record = {
            "dataset": args.dataset, "ckpt": args.ckpt,
            "sample_steps": n_steps, "eta": eta, "omega": omega,
            "time_distortion": args.time_distortion,
            "num_samples": args.num_samples,
            "sample_seconds": round(sample_s, 1),
            "smiles_file": os.path.basename(out_smi),
            # Scored against VALIDATION downstream; test is untouched.
            "scored_against": "validation",
            "metrics": rep,
        }
        with open(out_json, "w") as fh:
            json.dump(record, fh, indent=2)
        print(f"[done] {tag}: validity={rep['validity_relaxed_largest_frag']:.4f} "
              f"uniq={rep['uniqueness']:.4f} "
              f"novelty={rep.get('novelty', float('nan')):.4f} "
              f"({sample_s:.0f}s)", flush=True)

    print("slice complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
