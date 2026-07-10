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
from rdkit import Chem, RDLogger

from defog.core.domain import GraphDomain, _num_nodes, generation_metrics


# Valence table for formal-charge correction (atomic_num -> max_valence)
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


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
