"""
Utility functions for molecular graph experiments.

Provides conversions between SMILES strings, PyG Data objects, and RDKit molecules.
"""

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from typing import Dict, List, Optional

# Graph->molecule reconstruction is centralized in defog (single source of
# truth); re-exported here for backward compatibility with existing imports.
from defog.domains.molecule import pyg_data_to_mol, mol_to_smiles

# RDKit bond type mapping: index -> BondType (index 0 = no bond)
BOND_RDKIT_TYPES = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

# String name -> RDKit BondType
BOND_NAME_TO_RDKIT = {
    "SINGLE": Chem.rdchem.BondType.SINGLE,
    "DOUBLE": Chem.rdchem.BondType.DOUBLE,
    "TRIPLE": Chem.rdchem.BondType.TRIPLE,
    "AROMATIC": Chem.rdchem.BondType.AROMATIC,
}


def build_encoders(atom_types: List[str], bond_types: List[str]):
    """
    Build atom and bond encoder/decoder dicts from type lists.

    Args:
        atom_types: List of atom symbols, e.g. ["C", "N", "O", "F"]
        bond_types: List of bond type names, e.g. ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

    Returns:
        Tuple of (atom_encoder, atom_decoder, bond_encoder, bond_decoder):
        - atom_encoder: {"C": 0, "N": 1, ...}
        - atom_decoder: ["C", "N", ...]
        - bond_encoder: {BondType.SINGLE: 0, BondType.DOUBLE: 1, ...}
        - bond_decoder: [None, BondType.SINGLE, BondType.DOUBLE, ...]
            (index 0 = no-bond, indices 1+ = bond types)
    """
    atom_encoder = {atom: i for i, atom in enumerate(atom_types)}
    atom_decoder = list(atom_types)

    bond_encoder = {}
    bond_decoder = [None]  # Index 0 = no bond
    for i, name in enumerate(bond_types):
        bt = BOND_NAME_TO_RDKIT[name]
        bond_encoder[bt] = i  # 0-indexed within bond_encoder
        bond_decoder.append(bt)

    return atom_encoder, atom_decoder, bond_encoder, bond_decoder


def smiles_to_pyg_data(
    smiles: str,
    atom_encoder: Dict[str, int],
    bond_encoder: Dict,
) -> Optional[Data]:
    """
    Convert a SMILES string to a PyG Data object.

    Based on the pattern from src/datasets/moses_dataset.py.

    Args:
        smiles: SMILES string
        atom_encoder: Mapping from atom symbol to index, e.g. {"C": 0, "N": 1}
        bond_encoder: Mapping from RDKit BondType to index (0-based),
            e.g. {BondType.SINGLE: 0, BondType.DOUBLE: 1}

    Returns:
        PyG Data with one-hot x and edge_attr, or None if invalid.
        - x: (N, num_atom_types) one-hot node features
        - edge_index: (2, num_edges) edge indices
        - edge_attr: (num_edges, num_bond_types + 1) one-hot (class 0 = no-edge)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_atom_classes = len(atom_encoder)
    num_bond_classes = len(bond_encoder) + 1  # +1 for no-edge class

    # Extract atom types
    type_idx = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in atom_encoder:
            return None  # Unknown atom type
        type_idx.append(atom_encoder[symbol])

    N = len(type_idx)

    # Extract bonds (bidirectional)
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        if bt not in bond_encoder:
            return None  # Unknown bond type
        # +1 to reserve index 0 for no-edge
        bond_idx = bond_encoder[bt] + 1
        row += [start, end]
        col += [end, start]
        edge_type += [bond_idx, bond_idx]

    if len(row) == 0:
        return None  # No bonds

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=num_bond_classes).float()

    # Sort edges canonically
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]

    # One-hot node features
    x = F.one_hot(torch.tensor(type_idx), num_classes=num_atom_classes).float()

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


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
