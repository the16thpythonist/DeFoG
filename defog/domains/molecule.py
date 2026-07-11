"""
RDKit-backed molecule domain for DeFoG.

Interprets DeFoG's categorical graphs as molecules: node classes are atom types
(via ``atom_decoder``) and edge classes are bond types (via ``bond_decoder``,
index 0 = no bond). Provides the three :class:`~defog.core.domain.GraphDomain`
concerns for molecules:

* **decode**    - reconstruct a sanitized RDKit ``Mol`` (or None if invalid);
* **visualize** - draw a proper 2D depiction with RDKit; for graphs that don't
                  sanitize into a valid molecule it falls back to a best-effort
                  skeleton (relaxed valence, no kekulization) so early-training
                  samples still show structure;
* **evaluate**  - validity / uniqueness / novelty via canonical SMILES.

This module is also the single source of truth for graph->molecule
reconstruction: :func:`pyg_data_to_mol` and :func:`mol_to_smiles` live here and
``experiments.utils`` re-exports them.
"""
from __future__ import annotations

import io
import re
import warnings
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

from defog.core.domain import GraphDomain, _num_nodes, generation_metrics


# Valence table for formal-charge correction (atomic_num -> max_valence)
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

# RDKit bond-type mapping: index -> BondType (index 0 = no bond). This module is
# the single source of truth for the SMILES<->graph encoding; experiments.utils
# re-exports build_encoders/smiles_to_pyg_data from here (never the reverse), so
# the packaged library never depends on the experiments/ scripts directory.
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


def build_encoders(atom_types, bond_types):
    """
    Build atom and bond encoder/decoder dicts from type lists.

    Args:
        atom_types: List of atom symbols, e.g. ["C", "N", "O", "F"]
        bond_types: List of bond type names, e.g. ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

    Returns:
        (atom_encoder, atom_decoder, bond_encoder, bond_decoder):
        - atom_encoder: {"C": 0, "N": 1, ...}
        - atom_decoder: ["C", "N", ...]
        - bond_encoder: {BondType.SINGLE: 0, BondType.DOUBLE: 1, ...}
        - bond_decoder: [None, BondType.SINGLE, ...] (index 0 = no-bond)
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


def smiles_to_pyg_data(smiles: str, atom_encoder, bond_encoder):
    """
    Convert a SMILES string to a PyG ``Data`` object (one-hot x and edge_attr).

    Args:
        smiles: SMILES string.
        atom_encoder: {symbol: index}.
        bond_encoder: {RDKit BondType: 0-based index} (as from build_encoders).

    Returns:
        PyG Data with one-hot x (N, num_atom_types) and edge_attr
        (num_edges, num_bond_types + 1; class 0 = no-edge), or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_atom_classes = len(atom_encoder)
    num_bond_classes = len(bond_encoder) + 1  # +1 for no-edge class

    type_idx = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in atom_encoder:
            return None  # Unknown atom type
        type_idx.append(atom_encoder[symbol])

    N = len(type_idx)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        if bt not in bond_encoder:
            return None  # Unknown bond type
        bond_idx = bond_encoder[bt] + 1  # +1 to reserve index 0 for no-edge
        row += [start, end]
        col += [end, start]
        edge_type += [bond_idx, bond_idx]

    if len(row) == 0:
        return None  # No bonds

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=num_bond_classes).float()

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]

    x = F.one_hot(torch.tensor(type_idx), num_classes=num_atom_classes).float()

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _check_valency(mol):
    """Check molecule valences. Returns (is_valid, [atom_idx, valence]) or (True, None)."""
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r"\d+", e_sub)))
        return False, atomid_valence


def pyg_data_to_mol(data, atom_decoder: List[str], bond_decoder: List) -> Optional[Chem.Mol]:
    """Convert a PyG ``Data`` (one-hot ``x``/``edge_attr``) to an RDKit molecule.

    Handles valence errors by adding formal charges to N, O, S. RDKit warnings
    are suppressed during reconstruction. Returns the (un-sanitized) RWMol, or
    None if reconstruction fails.

    Args:
        data: PyG Data with one-hot x and edge_attr.
        atom_decoder: class idx -> atom symbol, e.g. ["C", "N", "O"].
        bond_decoder: class idx -> RDKit BondType, index 0 = None (no bond),
            e.g. [None, BondType.SINGLE, BondType.DOUBLE, ...].
    """
    try:
        RDLogger.DisableLog("rdApp.*")

        # RDKit reconstruction is CPU/numpy-only. Move tensors to CPU up front so
        # a GPU-sampled graph decodes correctly (otherwise a CPU adjacency matrix
        # indexed by a CUDA edge_index raises a device mismatch that the broad
        # except would silently turn into "invalid" -> 0% validity).
        data = data.cpu()

        if data.x.dim() == 2:
            atom_types = torch.argmax(data.x, dim=-1)
        else:
            atom_types = data.x.long()

        n = atom_types.shape[0]

        adj = torch.zeros(n, n, dtype=torch.long)
        if data.edge_index.numel() > 0:
            if data.edge_attr.dim() == 2:
                edge_classes = torch.argmax(data.edge_attr, dim=-1)
            else:
                edge_classes = data.edge_attr.long()
            adj[data.edge_index[0], data.edge_index[1]] = edge_classes

        mol = Chem.RWMol()

        for i in range(n):
            idx = atom_types[i].item()
            if idx >= len(atom_decoder):
                return None
            mol.AddAtom(Chem.Atom(atom_decoder[idx]))

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

            flag, atomid_valence = _check_valency(mol)
            if not flag and atomid_valence is not None and len(atomid_valence) == 2:
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY.get(an, 0)) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)

        return mol

    except (ValueError, RuntimeError, Chem.rdchem.AtomValenceException,
            Chem.rdchem.KekulizeException, Chem.rdchem.MolSanitizeException):
        # Genuine "this graph is not a valid molecule" outcomes from RDKit.
        return None
    except Exception as exc:
        # Anything else (device mismatch, shape/type bug, ...) is a programming
        # error, NOT an invalid molecule -- surface it rather than hide it.
        warnings.warn(f"pyg_data_to_mol: unexpected {type(exc).__name__}: {exc}")
        return None
    finally:
        RDLogger.EnableLog("rdApp.*")


def mol_to_smiles(mol: Chem.Mol) -> Optional[str]:
    """Canonical SMILES for a molecule, or None if it fails to sanitize."""
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


class MoleculeDomain(GraphDomain):
    """Molecule domain: RDKit reconstruction, depiction, and SMILES metrics.

    Args:
        atom_decoder: class idx -> atom symbol, e.g. ["C", "N", "O", ...].
        bond_decoder: class idx -> RDKit BondType, index 0 = None (no bond),
            e.g. [None, BondType.SINGLE, ...]. This is exactly the 4th value
            returned by ``experiments.utils.build_encoders``.
        reference_smiles: optional iterable of canonical training SMILES, used
            as the default reference set for the novelty metric.
        image_size: pixel size of each square RDKit depiction.
        show_smiles: whether :meth:`caption` returns the SMILES (or "invalid").
        max_caption_len: SMILES longer than this are truncated in captions.
    """

    def __init__(
        self,
        atom_decoder,
        bond_decoder,
        *,
        reference_smiles=None,
        image_size: int = 320,
        show_smiles: bool = True,
        max_caption_len: int = 40,
    ):
        self.atom_decoder = list(atom_decoder)
        self.bond_decoder = list(bond_decoder)
        self.reference = set(reference_smiles) if reference_smiles is not None else None
        self.image_size = image_size
        self.show_smiles = show_smiles
        self.max_caption_len = max_caption_len

    # ------------------------------------------------------------------ encode
    def _atom_encoder(self):
        return {sym: i for i, sym in enumerate(self.atom_decoder)}

    def _bond_class(self):
        """{RDKit BondType -> class index}, matching bond_decoder (class 0 = none)."""
        return {bt: i for i, bt in enumerate(self.bond_decoder) if bt is not None}

    def _mol_to_core(self, mol, keep=None):
        """Dense one-hot core (X_core (k,dx), E_core (k,k,de)) for an RDKit mol.

        Freezes atom classes AND the full k x k internal connectivity: bonded
        pairs get their bond class, all other pairs (and the diagonal) are class
        0 (no bond). ``keep`` selects a subset of atom indices (subgraph route);
        None keeps the whole molecule.
        """
        atom_encoder = self._atom_encoder()
        bond_class = self._bond_class()
        dx = len(self.atom_decoder)
        de = len(self.bond_decoder)

        if keep is None:
            keep = list(range(mol.GetNumAtoms()))
        keep = list(keep)
        pos = {a: i for i, a in enumerate(keep)}
        k = len(keep)
        if k == 0:
            raise ValueError("core is empty (no atoms kept)")

        X = torch.zeros(k, dx)
        for new_i, a in enumerate(keep):
            sym = mol.GetAtomWithIdx(int(a)).GetSymbol()
            if sym not in atom_encoder:
                raise ValueError(
                    f"atom '{sym}' is not in the model vocabulary {self.atom_decoder}"
                )
            X[new_i, atom_encoder[sym]] = 1.0

        E = torch.zeros(k, k, de)
        E[:, :, 0] = 1.0  # default every pair (incl. diagonal) to no-bond
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in pos and j in pos:
                bt = bond.GetBondType()
                if bt not in bond_class:
                    raise ValueError(f"bond type {bt} is not in the model vocabulary")
                cls = bond_class[bt]
                ni, nj = pos[i], pos[j]
                E[ni, nj] = 0.0; E[ni, nj, cls] = 1.0
                E[nj, ni] = 0.0; E[nj, ni, cls] = 1.0
        return X, E

    def encode(self, smiles: str):
        """Encode a whole SMILES molecule into a dense one-hot core (X_core, E_core)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
        return self._mol_to_core(mol, keep=None)

    def encode_subgraph(self, smiles_or_mol, keep_indices):
        """Encode a subgraph of a molecule (keep the given atom indices) as a core.

        Warns loudly if ``keep_indices`` splits an aromatic ring: a partial
        aromatic ring almost never sanitizes, so such a core will very likely
        produce ~0% valid completions.
        """
        mol = (
            Chem.MolFromSmiles(smiles_or_mol)
            if isinstance(smiles_or_mol, str)
            else smiles_or_mol
        )
        if mol is None:
            raise ValueError("RDKit could not parse the molecule for encode_subgraph")

        keep_set = set(int(i) for i in keep_indices)
        for ring in mol.GetRingInfo().AtomRings():
            inter = keep_set.intersection(ring)
            if inter and len(inter) != len(ring):
                is_arom = any(
                    mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring
                )
                warnings.warn(
                    f"encode_subgraph: keep_indices splits a ring {ring} "
                    f"(kept {sorted(inter)}). "
                    + ("This ring is aromatic; a partial aromatic ring will almost "
                       "certainly fail RDKit sanitization -> ~0%% valid completions. "
                       if is_arom else "")
                    + "Prefer keeping whole rings."
                )
        return self._mol_to_core(mol, keep=sorted(keep_set))

    def core_valence_report(self, X_core, E_core):
        """Per-core-atom attachment-viability diagnostic.

        For each core atom returns {symbol, nominal, used, free}: nominal maximum
        valence, valence already consumed by frozen internal bonds, and the
        remaining free valence. Atoms with free <= 0 cannot accept an attachment
        bond, so a core with no free-valence atoms will very likely yield ~0%%
        valid completions (this is the dominant inpainting failure mode).
        """
        bond_order = {  # class index -> valence contribution
            i: {None: 0.0,
                Chem.rdchem.BondType.SINGLE: 1.0,
                Chem.rdchem.BondType.DOUBLE: 2.0,
                Chem.rdchem.BondType.TRIPLE: 3.0,
                Chem.rdchem.BondType.AROMATIC: 1.5}[bt]
            for i, bt in enumerate(self.bond_decoder)
        }
        k = X_core.shape[0]
        x_idx = X_core.argmax(-1).tolist()
        e_idx = E_core.argmax(-1)
        report = []
        for i in range(k):
            sym = self.atom_decoder[x_idx[i]]
            atomic_num = Chem.Atom(sym).GetAtomicNum()
            nominal = float(ATOM_VALENCY.get(atomic_num, Chem.GetPeriodicTable().GetDefaultValence(atomic_num)))
            used = sum(bond_order[int(e_idx[i, j])] for j in range(k) if j != i)
            report.append({
                "atom": i, "symbol": sym,
                "nominal": nominal, "used": round(used, 2),
                "free": round(nominal - used, 2),
            })
        return report

    # ------------------------------------------------------------------ decode
    def _valid_smiles(self, data) -> Optional[str]:
        """Canonical SMILES iff the graph is a genuinely valid molecule.

        Matches the validity notion used by the training metrics: reconstructs,
        sanitizes to SMILES, and round-trips through ``MolFromSmiles``.
        """
        mol = pyg_data_to_mol(data, self.atom_decoder, self.bond_decoder)
        smi = mol_to_smiles(mol) if mol is not None else None
        if smi is not None and Chem.MolFromSmiles(smi) is not None:
            return smi
        return None

    def decode(self, data) -> Optional[Chem.Mol]:
        smi = self._valid_smiles(data)
        return Chem.MolFromSmiles(smi) if smi is not None else None

    def is_valid(self, data) -> bool:
        return self._valid_smiles(data) is not None

    def identity(self, data) -> Optional[str]:
        return self._valid_smiles(data)

    # --------------------------------------------------------------- visualize
    def _draw(self, mol, kekulize: bool):
        from PIL import Image
        from rdkit.Chem.Draw import rdMolDraw2D

        d = rdMolDraw2D.MolDraw2DCairo(self.image_size, self.image_size)
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, kekulize=kekulize)
        d.FinishDrawing()
        return np.array(Image.open(io.BytesIO(d.GetDrawingText())))

    def _depiction(self, data):
        """A 2D depiction as an RGBA array: valid molecule, else best-effort
        skeleton (relaxed valence, no kekulization), else None."""
        RDLogger.DisableLog("rdApp.*")
        try:
            raw = pyg_data_to_mol(data, self.atom_decoder, self.bond_decoder)
            if raw is None:
                return None
            # 1) proper depiction of a sanitizable molecule
            try:
                m = Chem.Mol(raw)
                Chem.SanitizeMol(m)
                return self._draw(m, kekulize=True)
            except Exception:
                pass
            # 2) best-effort skeleton: relaxed props, ring perception, no kekulize
            try:
                m = Chem.Mol(raw)
                m.UpdatePropertyCache(strict=False)
                Chem.FastFindRings(m)
                return self._draw(m, kekulize=False)
            except Exception:
                return None
        finally:
            RDLogger.EnableLog("rdApp.*")

    def render(self, ax, data) -> None:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        img = self._depiction(data)
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "invalid", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="crimson")

    def caption(self, data) -> Optional[str]:
        if not self.show_smiles:
            return None
        smi = self._valid_smiles(data)
        if smi is None:
            return "invalid"
        if len(smi) > self.max_caption_len:
            return smi[: self.max_caption_len - 1] + "…"
        return smi

    def summarize(self, samples) -> str:
        smis = [self._valid_smiles(s) for s in samples]
        n = len(samples)
        n_valid = sum(s is not None for s in smis)
        atom_counts = [_num_nodes(s) for s in samples]
        avg_atoms = float(np.mean(atom_counts)) if atom_counts else 0.0
        pct = (100.0 * n_valid / n) if n else 0.0
        return f"valid {n_valid}/{n} ({pct:.0f}%) | avg atoms {avg_atoms:.1f}"

    # ----------------------------------------------------------------- metrics
    def metrics(self, samples, reference: Optional[set] = None):
        ref = reference if reference is not None else self.reference
        return generation_metrics(self, samples, ref)
