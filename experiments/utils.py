"""
Utility functions for molecular graph experiments.

Provides conversions between SMILES strings, PyG Data objects, and RDKit molecules.
"""

import re
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from typing import Dict, List, Optional


# Valence table for formal charge correction (atomic_num -> max_valence)
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

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


def _check_valency(mol):
    """
    Check if molecule has valid valences.

    Returns:
        Tuple of (is_valid, [atom_idx, valence]) or (True, None)
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r"\d+", e_sub)))
        return False, atomid_valence


def pyg_data_to_mol(
    data: Data,
    atom_decoder: List[str],
    bond_decoder: List,
) -> Optional[Chem.Mol]:
    """
    Convert a PyG Data object to an RDKit molecule.

    Based on the pattern from src/analysis/rdkit_functions.py:625-673.
    Handles valence errors by adding formal charges to N, O, S.
    RDKit warnings are suppressed during reconstruction.

    Args:
        data: PyG Data with one-hot x and edge_attr
        atom_decoder: List of atom symbols, e.g. ["C", "N", "O"]
        bond_decoder: List of RDKit BondTypes, index 0 = None (no bond),
            e.g. [None, BondType.SINGLE, BondType.DOUBLE, ...]

    Returns:
        RDKit Mol object, or None if reconstruction fails
    """
    try:
        # Suppress RDKit warnings during reconstruction
        RDLogger.DisableLog("rdApp.*")

        # Get atom types from one-hot node features
        if data.x.dim() == 2:
            atom_types = torch.argmax(data.x, dim=-1)
        else:
            atom_types = data.x.long()

        n = atom_types.shape[0]

        # Build adjacency matrix from edge_index and edge_attr
        adj = torch.zeros(n, n, dtype=torch.long)
        if data.edge_index.numel() > 0:
            if data.edge_attr.dim() == 2:
                edge_classes = torch.argmax(data.edge_attr, dim=-1)
            else:
                edge_classes = data.edge_attr.long()
            adj[data.edge_index[0], data.edge_index[1]] = edge_classes

        # Build molecule
        mol = Chem.RWMol()

        # Add atoms
        for i in range(n):
            idx = atom_types[i].item()
            if idx >= len(atom_decoder):
                return None
            a = Chem.Atom(atom_decoder[idx])
            mol.AddAtom(a)

        # Add bonds (upper triangular only to avoid duplicates)
        edge_types = torch.triu(adj)
        edge_types[edge_types >= len(bond_decoder)] = 0
        all_bonds = torch.nonzero(edge_types)

        for bond in all_bonds:
            i, j = bond[0].item(), bond[1].item()
            if i == j:
                continue
            bt_idx = edge_types[i, j].item()
            if bt_idx == 0 or bt_idx >= len(bond_decoder):
                continue
            bt = bond_decoder[bt_idx]
            if bt is None:
                continue

            mol.AddBond(i, j, bt)

            # Handle valence errors by adding formal charges
            flag, atomid_valence = _check_valency(mol)
            if not flag and atomid_valence is not None and len(atomid_valence) == 2:
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY.get(an, 0)) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)

        return mol

    except Exception:
        return None

    finally:
        RDLogger.EnableLog("rdApp.*")


def mol_to_smiles(mol: Chem.Mol) -> Optional[str]:
    """
    Convert an RDKit molecule to a canonical SMILES string.

    Based on src/analysis/rdkit_functions.py:409-414.

    Args:
        mol: RDKit Mol object

    Returns:
        Canonical SMILES string, or None if sanitization fails
    """
    if mol is None:
        return None
    try:
        RDLogger.DisableLog("rdApp.*")
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    finally:
        RDLogger.EnableLog("rdApp.*")
    return Chem.MolToSmiles(mol)
