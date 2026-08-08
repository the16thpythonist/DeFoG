#!/usr/bin/env python
"""
Does encoding a molecule as a graph and decoding it back give the same molecule?

This is the losslessness premise that has to hold before a representation is
worth training on. It was measured for MOSES before ``kekulized_v2`` was
written (50,000 molecules, zero failures); this script generalises that check
so the same question can be asked of ChEMBL, where the chemistry is dirtier
because the prep applies no drug-likeness filter.

**Why a control matters here.** Formal charge is not a generated channel in any
of these representations -- it is reconstructed at decode time by the relaxed
N/O/S repair in ``pyg_data_to_mol``. So a charged molecule can fail to round
trip under the *currently shipped* aromatic vocabulary too. Measuring the
kekulized one alone would attribute that pre-existing loss to kekulization. By
default this runs every representation the dataset declares and reports them
side by side, so the decisive number is the *difference*, not the absolute.

Failure buckets, in the order they are tested:

    encode_skip     smiles_to_pyg_data returned None (unknown atom/bond, or the
                    molecule would not kekulize)
    decode_none     graph -> RWMol reconstruction failed
    sanitize_fail   the decoded molecule will not sanitize
    charge_mismatch canonical SMILES differ AND the input carried a formal
                    charge -- the known lossy class, not a new defect
    smiles_mismatch canonical SMILES differ for any other reason (the one that
                    would actually block a representation)
    ok              identical canonical SMILES

Usage:
    python scripts/check_representation_roundtrip.py --dataset chembl \
        --smiles data/chembl/chembl_train.smiles --n 50000
    python scripts/check_representation_roundtrip.py --dataset chembl \
        --smiles data/chembl/chembl_train.smiles          # all of them
"""
import argparse
import collections
import importlib
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem, RDLogger  # noqa: E402

from defog.domains.molecule import (  # noqa: E402
    build_encoders, pyg_data_to_mol, smiles_to_pyg_data,
)

REFERENCES = {"zinc": "zinc_reference", "guacamol": "guacamol_reference",
              "moses": "moses_reference", "chembl": "chembl_reference"}

_WORKER = {}


def _init_worker(atom_types, bond_types, kekulize):
    RDLogger.DisableLog("rdApp.*")
    ae, ad, be, bd = build_encoders(list(atom_types), list(bond_types))
    _WORKER.update(atom_encoder=ae, atom_decoder=ad, bond_encoder=be,
                   bond_decoder=bd, kekulize=kekulize)


def roundtrip_one(smiles):
    """(bucket, detail) for one SMILES under the worker's representation."""
    w = _WORKER
    ref = Chem.MolFromSmiles(smiles)
    if ref is None:
        return ("unparsable", "")
    # Canonicalize the input the same way the output will be canonicalized, so
    # a difference means the graph lost something -- not that the source file
    # was written in a different SMILES dialect.
    want = Chem.MolToSmiles(ref)
    charged = any(a.GetFormalCharge() != 0 for a in ref.GetAtoms())

    data = smiles_to_pyg_data(smiles, w["atom_encoder"], w["bond_encoder"],
                              kekulize=w["kekulize"])
    if data is None:
        return ("encode_skip", "")

    mol = pyg_data_to_mol(data, w["atom_decoder"], w["bond_decoder"],
                          charge_correction=True)
    if mol is None:
        return ("decode_none", "")
    try:
        probe = Chem.Mol(mol)
        Chem.SanitizeMol(probe)
    except Exception as exc:                                    # noqa: BLE001
        return ("sanitize_fail", type(exc).__name__)

    got = Chem.MolToSmiles(probe)
    if got == want:
        return ("ok", "")
    return ("charge_mismatch" if charged else "smiles_mismatch", f"{want}\t{got}")


def read_smiles(path, n, seed):
    smiles = [ln.strip() for ln in open(path) if ln.strip()]
    if n and n < len(smiles):
        random.Random(seed).shuffle(smiles)
        smiles = smiles[:n]
    return smiles


def run(rep, smiles, workers, chunksize, n_examples):
    counts = collections.Counter()
    details = collections.Counter()
    examples = collections.defaultdict(list)
    with Pool(workers, initializer=_init_worker,
              initargs=(rep.atom_types, rep.bond_types, rep.kekulize)) as pool:
        for bucket, detail in pool.imap_unordered(roundtrip_one, smiles,
                                                  chunksize=chunksize):
            counts[bucket] += 1
            if detail:
                details[detail] += 1
                if len(examples[bucket]) < n_examples:
                    examples[bucket].append(detail)
    return counts, details, examples


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(REFERENCES))
    ap.add_argument("--smiles", required=True, help="one SMILES per line")
    ap.add_argument("--representation", default=None,
                    help="only this one (default: every representation the "
                         "dataset declares, so the comparison has a control)")
    ap.add_argument("--n", type=int, default=None, help="random subsample size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--chunksize", type=int, default=2000)
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--out", default=None, help="write the report as JSON")
    args = ap.parse_args()

    import os
    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)

    mod = importlib.import_module(f"defog.data.{REFERENCES[args.dataset]}")
    if args.representation:
        reps = [mod.get_representation(args.representation)]
    elif hasattr(mod, "REPRESENTATIONS"):
        reps = list(mod.REPRESENTATIONS.values())
    else:
        from defog.data.vocabulary import Representation
        reps = [Representation(name="default", atom_types=list(mod.ATOM_TYPES),
                               bond_types=list(mod.BOND_TYPES),
                               kekulize="AROMATIC" not in mod.BOND_TYPES)]

    smiles = read_smiles(args.smiles, args.n, args.seed)
    print(f"{args.dataset}: {len(smiles):,} molecules from {args.smiles}, "
          f"{workers} workers")

    report = {"dataset": args.dataset, "smiles_file": args.smiles,
              "n": len(smiles), "seed": args.seed, "representations": {}}

    for rep in reps:
        print(f"\n=== {rep.name} "
              f"({len(rep.atom_types)} atom / {len(rep.bond_types) + 1} edge, "
              f"kekulize={rep.kekulize}) ===")
        counts, details, examples = run(rep, smiles, workers, args.chunksize,
                                        args.examples)
        total = sum(counts.values())
        for bucket, c in counts.most_common():
            print(f"  {bucket:16s}{c:>10,}{c / total:>10.5f}")
        lossy = total - counts["ok"]
        print(f"  {'-' * 36}")
        print(f"  {'LOSSY':16s}{lossy:>10,}{lossy / total:>10.5f}")
        # Which RDKit exception the sanitize failures are is the whole question
        # for an aromatic vocabulary: KekulizeException means the decoder cannot
        # recover the ring system, AtomValenceException means something else.
        sanitize_kinds = {k: c for k, c in details.items() if "\t" not in k}
        for kind, c in sorted(sanitize_kinds.items(), key=lambda kv: -kv[1]):
            print(f"      sanitize_fail/{kind}: {c:,}")
        for bucket, exs in examples.items():
            if bucket in ("charge_mismatch", "smiles_mismatch"):
                print(f"  examples [{bucket}] (want / got):")
                for ex in exs:
                    want, got = ex.split("\t")
                    print(f"    {want}\n    {got}")
        report["representations"][rep.name] = {
            "atom_types": list(rep.atom_types), "bond_types": list(rep.bond_types),
            "kekulize": rep.kekulize, "counts": dict(counts),
            "lossy": lossy, "lossy_frac": lossy / total,
            "examples": {k: v for k, v in examples.items()},
        }

    # The decisive comparison: kekulizing is only a cost where it loses
    # molecules the shipped representation keeps.
    names = list(report["representations"])
    if len(names) > 1:
        print("\n" + "=" * 60)
        print(f"{'representation':20s}{'lossy':>12s}{'share':>12s}")
        print("=" * 60)
        for name in names:
            r = report["representations"][name]
            print(f"{name:20s}{r['lossy']:>12,}{r['lossy_frac']:>12.5f}")
        base = report["representations"][names[0]]["lossy_frac"]
        for name in names[1:]:
            delta = report["representations"][name]["lossy_frac"] - base
            print(f"delta vs {names[0]}: {delta:+.5f} for {name}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
