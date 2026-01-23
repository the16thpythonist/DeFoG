#!/usr/bin/env python
"""
Script to compute Crippen logP values for all molecules in aqsoldb.csv
and add them as a new column.

Usage:
    python scripts/add_logp_to_csv.py

This will:
1. Read aqsoldb.csv
2. Compute Crippen logP for each SMILES
3. Add a 'logp' column
4. Save to aqsoldb.csv (backup created as aqsoldb_backup.csv)
5. Print statistics (mean, std) for normalization
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen
from tqdm import tqdm
import shutil
import os


def compute_logp(smiles: str) -> float:
    """Compute Crippen logP for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan
    try:
        return Crippen.MolLogP(mol)
    except Exception:
        return np.nan


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    csv_path = os.path.join(repo_root, "aqsoldb.csv")
    backup_path = os.path.join(repo_root, "aqsoldb_backup.csv")

    print(f"Reading CSV from: {csv_path}")

    # Create backup
    if not os.path.exists(backup_path):
        shutil.copy(csv_path, backup_path)
        print(f"Backup created at: {backup_path}")
    else:
        print(f"Backup already exists at: {backup_path}")

    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} molecules")

    # Check if logp column already exists
    if 'logp' in df.columns:
        print("'logp' column already exists. Recomputing...")

    # Compute logP for all molecules
    print("Computing Crippen logP values...")
    logp_values = []
    failed_count = 0

    for smiles in tqdm(df['SMILES'], desc="Computing logP"):
        logp = compute_logp(smiles)
        if np.isnan(logp):
            failed_count += 1
        logp_values.append(logp)

    df['logp'] = logp_values

    # Statistics
    valid_logp = df['logp'].dropna()
    print(f"\n{'='*60}")
    print("LogP Statistics:")
    print(f"{'='*60}")
    print(f"  Total molecules:     {len(df)}")
    print(f"  Failed to compute:   {failed_count}")
    print(f"  Valid logP values:   {len(valid_logp)}")
    print(f"  Mean:                {valid_logp.mean():.4f}")
    print(f"  Std:                 {valid_logp.std():.4f}")
    print(f"  Min:                 {valid_logp.min():.4f}")
    print(f"  Max:                 {valid_logp.max():.4f}")
    print(f"  Median:              {valid_logp.median():.4f}")
    print(f"{'='*60}")

    # Save updated CSV
    df.to_csv(csv_path, index=False)
    print(f"\nSaved updated CSV to: {csv_path}")

    # Print normalization parameters for config
    print(f"\n{'='*60}")
    print("Use these values in your config for normalization:")
    print(f"{'='*60}")
    print(f"  logp_mean: {valid_logp.mean():.4f}")
    print(f"  logp_std:  {valid_logp.std():.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
