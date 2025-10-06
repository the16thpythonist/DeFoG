from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
import pandas as pd

from src import utils
from analysis.rdkit_functions import (
    mol2smiles,
    build_molecule_with_partial_charges,
    compute_molecular_metrics,
)
from datasets.abstract_dataset import AbstractDatasetInfos, MolecularDataModule


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


# Use same atom types as MOSES
atom_decoder = ["C", "N", "S", "O", "F", "Cl", "Br", "H"]


class AqSolDBDataset(InMemoryDataset):
    """
    AqSolDB (Aqueous Solubility Database) dataset.

    Contains ~11k molecules with solubility measurements.
    Split: 0=test (1291 samples), 1=train (9733 samples, will be split into train/val)
    """

    def __init__(
        self,
        stage,
        root,
        csv_path,
        filter_dataset: bool,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.stage = stage
        self.atom_decoder = atom_decoder
        self.filter_dataset = filter_dataset
        self.csv_path = csv_path

        if self.stage == "train":
            self.file_idx = 0
        elif self.stage == "val":
            self.file_idx = 1
        else:  # test
            self.file_idx = 2

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx], weights_only=False)

    @property
    def raw_file_names(self):
        return ["aqsoldb.csv"]

    @property
    def processed_file_names(self):
        if self.filter_dataset:
            return [
                "train_filtered.pt",
                "val_filtered.pt",
                "test_filtered.pt",
            ]
        else:
            return ["train.pt", "val.pt", "test.pt"]

    def download(self):
        """Dataset is already downloaded, just copy to raw dir if needed"""
        raw_aqsoldb_path = osp.join(self.raw_dir, "aqsoldb.csv")
        if not osp.exists(raw_aqsoldb_path):
            # Copy from repository root to raw directory
            import shutil
            if osp.exists(self.csv_path):
                shutil.copy(self.csv_path, raw_aqsoldb_path)
            else:
                raise FileNotFoundError(
                    f"AqSolDB CSV file not found at {self.csv_path}. "
                    "Please ensure aqsoldb.csv is in the repository root."
                )

    def process(self):
        RDLogger.DisableLog("rdApp.*")
        types = {atom: i for i, atom in enumerate(self.atom_decoder)}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        # Read the CSV file
        csv_path = osp.join(self.raw_dir, "aqsoldb.csv")
        df = pd.read_csv(csv_path)

        # Filter by stage
        # split column: 0=test, 1=train+val (we'll split this 90/10)
        if self.stage == "test":
            df = df[df["split"] == 0]
        else:  # train or val
            train_val_df = df[df["split"] == 1]
            # Split train_val into train (90%) and val (10%)
            n_train = int(0.9 * len(train_val_df))

            # Use a fixed random seed for reproducibility
            train_val_df = train_val_df.sample(frac=1, random_state=42).reset_index(drop=True)

            if self.stage == "train":
                df = train_val_df.iloc[:n_train]
            else:  # val
                df = train_val_df.iloc[n_train:]

        smiles_list = df["SMILES"].values
        solubility_list = df["Solubility"].values

        data_list = []
        smiles_kept = []

        for i, (smile, solubility) in enumerate(tqdm(zip(smiles_list, solubility_list),
                                                      total=len(smiles_list),
                                                      desc=f"Processing {self.stage}")):
            mol = Chem.MolFromSmiles(smile)

            if mol is None:
                continue

            N = mol.GetNumAtoms()

            type_idx = []
            for atom in mol.GetAtoms():
                atom_symbol = atom.GetSymbol()
                if atom_symbol not in types:
                    # Skip molecules with atoms not in our decoder
                    break
                type_idx.append(types[atom_symbol])

            if len(type_idx) != N:
                # Molecule contains atoms we don't support
                continue

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]

            if len(row) == 0:
                # Skip molecules with no bonds
                continue

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()

            # Store solubility as target (not used in unconditional generation but available)
            y = torch.zeros(size=(1, 0), dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

            if self.filter_dataset:
                # Try to build the molecule again from the graph. If it fails, do not add it
                dense_data, node_mask = utils.to_dense(
                    data.x, data.edge_index, data.edge_attr, data.batch
                )
                dense_data = dense_data.mask(node_mask, collapse=True)
                X, E = dense_data.X, dense_data.E

                assert X.size(0) == 1
                atom_types = X[0]
                edge_types = E[0]
                mol = build_molecule_with_partial_charges(
                    atom_types, edge_types, atom_decoder
                )
                smiles = mol2smiles(mol)
                if smiles is not None:
                    try:
                        mol_frags = Chem.rdmolops.GetMolFrags(
                            mol, asMols=True, sanitizeFrags=True
                        )
                        if len(mol_frags) == 1:
                            data_list.append(data)
                            smiles_kept.append(smiles)

                    except Chem.rdchem.AtomValenceException:
                        print("Valence error in GetmolFrags")
                    except Chem.rdchem.KekulizeException:
                        print("Can't kekulize molecule")
            else:
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

        if self.filter_dataset:
            smiles_save_path = osp.join(
                pathlib.Path(self.raw_paths[0]).parent, f"new_{self.stage}.smiles"
            )
            print(smiles_save_path)
            with open(smiles_save_path, "w") as f:
                f.writelines("%s\n" % s for s in smiles_kept)
            print(f"Number of molecules kept: {len(smiles_kept)} / {len(smiles_list)}")


class AqSolDBDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.remove_h = False
        self.datadir = cfg.dataset.datadir
        self.filter_dataset = cfg.dataset.filter
        self.train_smiles = []

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        # Path to the CSV file (should be in repository root)
        csv_path = os.path.join(base_path, "aqsoldb.csv")

        datasets = {
            "train": AqSolDBDataset(
                stage="train", root=root_path, csv_path=csv_path,
                filter_dataset=self.filter_dataset
            ),
            "val": AqSolDBDataset(
                stage="val", root=root_path, csv_path=csv_path,
                filter_dataset=self.filter_dataset
            ),
            "test": AqSolDBDataset(
                stage="test", root=root_path, csv_path=csv_path,
                filter_dataset=self.filter_dataset
            ),
        }
        super().__init__(cfg, datasets)


class AqSolDBinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False, meta=None):
        self.name = "AqSolDB"
        self.input_dims = None
        self.output_dims = None
        self.remove_h = False
        self.compute_fcd = cfg.dataset.compute_fcd

        self.atom_decoder = atom_decoder
        self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
        # Atomic weights
        self.atom_weights = {0: 12, 1: 14, 2: 32, 3: 16, 4: 19, 5: 35.4, 6: 79.9, 7: 1}
        # Valencies
        self.valencies = [4, 3, 4, 2, 1, 1, 1, 1]
        self.num_atom_types = len(self.atom_decoder)
        self.max_weight = 350

        meta_files = dict(
            n_nodes=f"{self.name}_n_counts.txt",
            node_types=f"{self.name}_atom_types.txt",
            edge_types=f"{self.name}_edge_types.txt",
            valency_distribution=f"{self.name}_valencies.txt",
        )

        # Initialize with reasonable defaults (will be computed from data if needed)
        # These are placeholder values - will be computed from the actual data
        self.n_nodes = None
        self.max_n_nodes = None
        self.node_types = None
        self.edge_types = None
        self.valency_distribution = None

        if meta is None:
            meta = dict(
                n_nodes=None,
                node_types=None,
                edge_types=None,
                valency_distribution=None,
            )
        assert set(meta.keys()) == set(meta_files.keys())

        for k, v in meta_files.items():
            if (k not in meta or meta[k] is None) and os.path.exists(v):
                meta[k] = np.loadtxt(v)
                setattr(self, k, meta[k])

        if recompute_statistics or self.n_nodes is None:
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt(meta_files["n_nodes"], self.n_nodes.numpy())
            self.max_n_nodes = len(self.n_nodes) - 1

        if recompute_statistics or self.node_types is None:
            self.node_types = datamodule.node_types()
            print("Distribution of node types", self.node_types)
            np.savetxt(meta_files["node_types"], self.node_types.numpy())

        if recompute_statistics or self.edge_types is None:
            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt(meta_files["edge_types"], self.edge_types.numpy())

        if recompute_statistics or self.valency_distribution is None:
            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt(meta_files["valency_distribution"], valencies.numpy())
            self.valency_distribution = valencies

        # Complete infos after we have the data
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)


def get_smiles(raw_dir, filter_dataset):
    """Get SMILES strings for train, val, test sets"""
    if filter_dataset:
        smiles_save_paths = {
            "train": osp.join(raw_dir, "new_train.smiles"),
            "val": osp.join(raw_dir, "new_val.smiles"),
            "test": osp.join(raw_dir, "new_test.smiles"),
        }
        train_smiles = open(smiles_save_paths["train"]).readlines()
        val_smiles = open(smiles_save_paths["val"]).readlines()
        test_smiles = open(smiles_save_paths["test"]).readlines()
    else:
        # Load from processed CSV
        csv_path = osp.join(raw_dir, "aqsoldb.csv")
        df = pd.read_csv(csv_path)

        # Split same way as in dataset processing
        test_df = df[df["split"] == 0]
        train_val_df = df[df["split"] == 1]
        train_val_df = train_val_df.sample(frac=1, random_state=42).reset_index(drop=True)
        n_train = int(0.9 * len(train_val_df))

        train_smiles = train_val_df.iloc[:n_train]["SMILES"].tolist()
        val_smiles = train_val_df.iloc[n_train:]["SMILES"].tolist()
        test_smiles = test_df["SMILES"].tolist()

    return {
        "train": train_smiles,
        "val": val_smiles,
        "test": test_smiles,
    }
