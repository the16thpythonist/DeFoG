"""
Utility functions for molecular graph experiments.

Provides conversions between SMILES strings, PyG Data objects, and RDKit molecules.
"""

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from typing import Dict, List, Optional

# SMILES<->graph encoding and graph->molecule reconstruction are centralized in
# the packaged library (defog.domains.molecule) as the single source of truth.
# They are re-exported here so existing `from experiments.utils import ...` call
# sites keep working. The dependency direction is one-way (experiments -> defog);
# the library never imports from the experiments/ scripts directory.
from defog.domains.molecule import (  # noqa: F401  (re-export)
    pyg_data_to_mol,
    mol_to_smiles,
    build_encoders,
    smiles_to_pyg_data,
    molecular_metrics,
    ring_sizes_ok,
    property_distributions,
    BOND_RDKIT_TYPES,
    BOND_NAME_TO_RDKIT,
)


def make_generation_metrics_fn(atom_decoder, bond_decoder, train_smiles):
    """
    Build a ``metrics_fn(samples) -> {validity, uniqueness, novelty}`` for the
    TrainingMonitorCallback, given how to decode graphs and the reference set of
    training SMILES (for novelty).

    - validity   = valid molecules / total sampled
    - uniqueness = distinct canonical SMILES / valid molecules
    - novelty    = unique molecules not in ``train_smiles`` / unique molecules
    """
    train_set = set(train_smiles)

    def metrics_fn(samples):
        valid = []
        for s in samples:
            mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
            smi = mol_to_smiles(mol) if mol is not None else None
            if smi is not None and Chem.MolFromSmiles(smi) is not None:
                valid.append(smi)
        n = len(samples)
        n_valid = len(valid)
        unique = set(valid)
        n_unique = len(unique)
        n_novel = sum(1 for smi in unique if smi not in train_set)
        return {
            "validity": n_valid / n if n else 0.0,
            "uniqueness": n_unique / n_valid if n_valid else 0.0,
            "novelty": n_novel / n_unique if n_unique else 0.0,
        }

    return metrics_fn


def tag_generated_smiles(samples, atom_decoder, bond_decoder, train_smiles=None):
    """
    Decode generated graphs to canonical SMILES and tag every sample.

    Persists ALL generated samples (not just valid/unique ones) so the run's
    output can be recovered/re-analyzed without re-sampling. Returns one record
    per sample:
        {"index": i,
         "smiles": <canonical SMILES> | None (None if invalid),
         "valid":  bool,
         "unique": bool,   # first occurrence of this canonical SMILES among valid
         "novel":  bool | None}   # valid & not in train_smiles (None if no ref)

    Because ``unique`` flags the first occurrence, ``sum(r["unique"])`` equals the
    number of distinct molecules. Commit the result via ``e.commit_json(...)``.
    Callers may add extra per-sample fields (e.g. conditioning target) afterwards.
    """
    ref = set(train_smiles) if train_smiles is not None else None
    seen = set()
    records = []
    for i, s in enumerate(samples):
        mol = pyg_data_to_mol(s, atom_decoder, bond_decoder)
        smi = mol_to_smiles(mol) if mol is not None else None
        valid = smi is not None and Chem.MolFromSmiles(smi) is not None
        rec = {"index": i, "smiles": smi if valid else None, "valid": valid,
               "unique": False, "novel": None}
        if valid:
            rec["unique"] = smi not in seen
            seen.add(smi)
            rec["novel"] = (smi not in ref) if ref is not None else None
        records.append(rec)
    return records
