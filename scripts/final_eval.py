"""
The single evaluation pass on TEST, with a frozen sampling configuration.

Protocol section 5, step 3. Everything before this happens on validation; this
script is the one place the test split is read, and it is meant to run once per
dataset. If it has to run again, that has to be said in the paper.

Guard rails, because a one-shot evaluation is exactly where a silent default
does the most damage:

* ``--sample-steps``, ``--eta`` and ``--omega`` are REQUIRED, with no defaults.
  A frozen configuration has to be stated, not inherited -- the whole point is
  that these were chosen on validation, and a script that could quietly fall
  back to a built-in value would make that unverifiable.
* ``--sweep-dir`` is recorded so the run carries a pointer to the evidence its
  configuration came from.
* Output records ``sampling_config_frozen: true`` and ``split: test``, in
  contrast to the training runs, which record ``false`` precisely so nothing
  produced before this point can be mistaken for a table row.

Like the sweep, this only SAMPLES and writes SMILES. FCD, NSPDK and scaffold
similarity are computed afterwards in the metrics environment against the test
reference this script also writes out.

Usage:
    python scripts/final_eval.py --ckpt ckpts/zinc_e1_seed42/best_model \\
        --dataset zinc --sample-steps 500 --eta 25 --omega 0 \\
        --num-samples 10000 --out-dir final_zinc --tag seed42
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Checkpoint WITHOUT .ckpt suffix")
    ap.add_argument("--dataset", required=True, choices=sorted(REFERENCES))
    ap.add_argument("--tag", required=True, help="e.g. seed42; names the outputs")
    # No defaults on purpose -- see the module docstring.
    ap.add_argument("--sample-steps", type=int, required=True)
    ap.add_argument("--eta", type=float, required=True)
    ap.add_argument("--omega", type=float, required=True)
    ap.add_argument("--time-distortion", default="polydec")
    ap.add_argument("--num-samples", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--split", choices=("test", "validation"), default="test",
                    help="Which held-out split to score against. Use 'validation' "
                         "for anything that feeds a SELECTION decision -- picking a "
                         "model by its test numbers is tuning on test, however "
                         "defensible each individual pass looks.")
    ap.add_argument("--sweep-dir", default=None,
                    help="Where the frozen config was chosen; recorded for provenance")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    mod = REFERENCES[args.dataset]
    _, atom_decoder, _, bond_decoder = build_encoders(mod.ATOM_TYPES, mod.BOND_TYPES)
    os.makedirs(args.out_dir, exist_ok=True)

    split = mod.load_reference_split()
    # The reference this run is scored against. Selection work must use
    # validation; test is reserved for reporting a model that is already chosen.
    test_smiles = split.test_smiles if args.split == "test" else split.val_smiles
    train_ref = set(split.train_smiles)
    ref_path = os.path.join(args.out_dir, f"_{args.split}_reference.smi")
    if not os.path.exists(ref_path):
        with open(ref_path, "w") as fh:
            fh.write("\n".join(test_smiles) + "\n")
        print(f"wrote {args.split} reference ({len(test_smiles)}) -> {ref_path}",
              flush=True)
    # MOSES additionally requires FCD against test_scaffolds (test split only).
    if args.split == "test" and hasattr(split, "test_scaffolds_smiles"):
        sf_path = os.path.join(args.out_dir, "_test_scaffolds_reference.smi")
        if not os.path.exists(sf_path):
            with open(sf_path, "w") as fh:
                fh.write("\n".join(split.test_scaffolds_smiles) + "\n")
            print(f"wrote test_scaffolds reference "
                  f"({len(split.test_scaffolds_smiles)}) -> {sf_path}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeFoGModel.load(args.ckpt).to(device).eval()
    print(f"loaded {args.ckpt} on {device}", flush=True)
    print(f"FROZEN CONFIG: steps={args.sample_steps} eta={args.eta} "
          f"omega={args.omega} distortion={args.time_distortion} "
          f"n={args.num_samples}", flush=True)

    t0 = time.time()
    samples, remaining = [], args.num_samples
    while remaining > 0:
        cur = min(args.chunk, remaining)
        samples += model.sample(num_samples=cur, sample_steps=args.sample_steps,
                                eta=args.eta, omega=args.omega,
                                time_distortion=args.time_distortion,
                                device=device, show_progress=False)
        remaining -= cur
        if (args.num_samples - remaining) % 2500 == 0:
            print(f"  {args.num_samples - remaining}/{args.num_samples} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    sample_s = time.time() - t0

    rep = validity_report(samples, atom_decoder, bond_decoder,
                          reference_smiles=train_ref)
    smiles = rep.pop("smiles")
    smi_path = os.path.join(args.out_dir, f"{args.tag}.smi")
    with open(smi_path, "w") as fh:
        fh.write("\n".join(smiles) + "\n")

    record = {
        "dataset": args.dataset,
        "tag": args.tag,
        "ckpt": args.ckpt,
        "split": args.split,
        "sampling_config_frozen": True,
        "frozen_config": {
            "sample_steps": args.sample_steps, "eta": args.eta,
            "omega": args.omega, "time_distortion": args.time_distortion,
        },
        "chosen_on": "validation",
        "sweep_dir": args.sweep_dir,
        "num_samples": args.num_samples,
        "sample_seconds": round(sample_s, 1),
        "n_reference": len(test_smiles),
        "smiles_file": os.path.basename(smi_path),
        # Novelty is against TRAIN, as everywhere else.
        "novelty_reference": "train",
        "metrics": rep,
    }
    out_json = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(out_json, "w") as fh:
        json.dump(record, fh, indent=2)

    print(f"\nvalidity (relaxed, largest frag) = "
          f"{rep['validity_relaxed_largest_frag']:.4f}")
    print(f"validity (strict, no correction) = {rep['validity_strict_largest_frag']:.4f}")
    print(f"validity (whole molecule)        = {rep['validity_whole_molecule']:.4f}")
    print(f"per-valid : uniqueness={rep['uniqueness']:.4f} "
          f"novelty={rep.get('novelty', float('nan')):.4f}")
    print(f"cumulative: V={rep['v']:.4f} V.U.={rep['v_u']:.4f} "
          f"V.U.N.={rep.get('v_u_n', float('nan')):.4f}")
    print(f"wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
