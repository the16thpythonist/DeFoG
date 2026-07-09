#!/usr/bin/env python3
"""
Prepare the AqSolDB dataset for conditional DeFoG training.

For every molecule in ``aqsoldb.csv`` this:
  - parses the SMILES with RDKit (drops invalid),
  - keeps only molecules with <= MAX_HEAVY_ATOMS heavy atoms and >= 1 bond
    whose bonds are all in the supported set (SINGLE/DOUBLE/TRIPLE/AROMATIC),
  - recomputes logP (Crippen.MolLogP) and SAS (RDKit Contrib sascorer),

and writes ``data/aqsoldb_conditional.csv`` with columns ``smiles, logp, sas``.
It also reports the set of atom elements observed in the kept molecules (used as
the model's atom vocabulary) and basic size / property statistics.

Usage:
    python scripts/prepare_aqsoldb.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, RDConfig

# RDKit Contrib synthetic-accessibility scorer.
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

RDLogger.DisableLog("rdApp.*")

PROJECT_DIR = Path(__file__).parent.parent.resolve()
INPUT_CSV = PROJECT_DIR / "aqsoldb.csv"
OUTPUT_CSV = PROJECT_DIR / "data" / "aqsoldb_conditional.csv"

MAX_HEAVY_ATOMS = 32
# Keep only elements observed at least this many times; the rare-metal tail in
# AqSolDB (~47 elements below this) yields un-learnable node classes.
MIN_ATOM_COUNT = 100
SUPPORTED_BONDS = {
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
}


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    # -- Pass 1: parse, apply size/bond filters, compute properties, count atoms.
    candidates = []
    atom_symbols = {}
    n_invalid = n_too_big = n_bad_bond = n_no_bond = 0

    for smiles in df["SMILES"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_invalid += 1
            continue
        if mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
            n_too_big += 1
            continue
        if mol.GetNumBonds() == 0:
            n_no_bond += 1
            continue
        if any(b.GetBondType() not in SUPPORTED_BONDS for b in mol.GetBonds()):
            n_bad_bond += 1
            continue

        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        for sym in symbols:
            atom_symbols[sym] = atom_symbols.get(sym, 0) + 1
        candidates.append(
            {
                "smiles": Chem.MolToSmiles(mol),
                "logp": float(Crippen.MolLogP(mol)),
                "sas": float(sascorer.calculateScore(mol)),
                "size": mol.GetNumHeavyAtoms(),
                "_symbols": set(symbols),
            }
        )

    # -- Determine the atom vocabulary and drop molecules with rarer elements.
    vocab = {sym for sym, count in atom_symbols.items() if count >= MIN_ATOM_COUNT}
    kept = [c for c in candidates if c["_symbols"] <= vocab]
    n_rare_atom = len(candidates) - len(kept)

    rows = [
        {"smiles": c["smiles"], "logp": c["logp"], "sas": c["sas"]} for c in kept
    ]
    sizes = [c["size"] for c in kept]

    out = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    print("\n=== Filtering summary ===")
    print(f"  kept:            {len(out)}")
    print(f"  invalid SMILES:  {n_invalid}")
    print(f"  > {MAX_HEAVY_ATOMS} heavy atoms: {n_too_big}")
    print(f"  no bonds:        {n_no_bond}")
    print(f"  unsupported bond:{n_bad_bond}")
    print(f"  rare atom (<{MIN_ATOM_COUNT}): {n_rare_atom}")

    print("\n=== Atom vocabulary (>= "
          f"{MIN_ATOM_COUNT} occurrences) ===")
    ordered = sorted(
        [(s, c) for s, c in atom_symbols.items() if s in vocab], key=lambda kv: -kv[1]
    )
    print("  ATOM_TYPES =", [sym for sym, _ in ordered])
    for sym, count in ordered:
        print(f"    {sym:>3}: {count}")

    sizes = np.array(sizes)
    print("\n=== Size (heavy atoms) ===")
    print(f"  min={sizes.min()}  max={sizes.max()}  mean={sizes.mean():.1f}")

    for name in ("logp", "sas"):
        vals = out[name].values
        p10, p50, p90 = np.percentile(vals, [10, 50, 90])
        print(f"\n=== {name} ===")
        print(f"  min={vals.min():.2f}  max={vals.max():.2f}  mean={vals.mean():.2f}")
        print(f"  p10={p10:.2f}  p50(median)={p50:.2f}  p90={p90:.2f}")

    print(f"\nWrote {len(out)} molecules to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
