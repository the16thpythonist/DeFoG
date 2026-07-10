"""
Tests for the graph-domain visualization adapters:
- MoleculeDomain (RDKit decode / depiction / SMILES metrics, skeleton fallback)
- GenericGraphDomain (node-link render, WL-hash identity, generic metrics)
- SampleVisualizationCallback domain wiring + round-epoch labeling

All tests are CPU-only and construct PyG Data tensors directly (no model, no
sampling), so they run fast and offline.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data
from rdkit import Chem

from defog.core.domain import GenericGraphDomain, generation_metrics
from defog.domains import MoleculeDomain


# Atom/bond vocab used across the molecule tests.
ATOM_DECODER = ["C", "O", "N", "F"]
# index 0 = no bond (matches build_encoders output)
BOND_DECODER = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
NUM_ATOM_CLASSES = len(ATOM_DECODER)
NUM_BOND_CLASSES = len(BOND_DECODER)  # 5


def make_data(atom_classes, bonds):
    """Build a PyG Data with one-hot x/edge_attr. bonds = [(i, j, bond_class)]."""
    x = torch.zeros(len(atom_classes), NUM_ATOM_CLASSES)
    for i, c in enumerate(atom_classes):
        x[i, c] = 1.0
    rows, cols, attrs = [], [], []
    for (i, j, bc) in bonds:
        for a, b in ((i, j), (j, i)):
            rows.append(a)
            cols.append(b)
            oh = torch.zeros(NUM_BOND_CLASSES)
            oh[bc] = 1.0
            attrs.append(oh)
    if rows:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.stack(attrs)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, NUM_BOND_CLASSES)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def ethanol():
    # C-C-O, all single bonds -> valid molecule "CCO"
    return make_data([0, 0, 1], [(0, 1, 1), (1, 2, 1)])


def oversaturated_carbon():
    # central C single-bonded to 5 carbons -> valence 5 -> invalid
    return make_data([0, 0, 0, 0, 0, 0], [(0, i, 1) for i in range(1, 6)])


# --------------------------------------------------------------------------- #
# MoleculeDomain
# --------------------------------------------------------------------------- #

def test_molecule_domain_decodes_valid():
    dom = MoleculeDomain(ATOM_DECODER, BOND_DECODER)
    data = ethanol()
    assert dom.is_valid(data)
    assert dom.identity(data) == Chem.CanonSmiles("CCO")
    mol = dom.decode(data)
    assert mol is not None and mol.GetNumAtoms() == 3
    assert dom.caption(data) == Chem.CanonSmiles("CCO")


def test_molecule_domain_invalid():
    dom = MoleculeDomain(ATOM_DECODER, BOND_DECODER)
    data = oversaturated_carbon()
    assert not dom.is_valid(data)
    assert dom.identity(data) is None
    assert dom.decode(data) is None
    assert dom.caption(data) == "invalid"


def test_molecule_domain_render_valid_produces_image():
    dom = MoleculeDomain(ATOM_DECODER, BOND_DECODER, image_size=120)
    fig, ax = plt.subplots()
    dom.render(ax, ethanol())
    assert len(ax.images) == 1  # imshow of the RDKit depiction
    plt.close(fig)


def test_molecule_domain_render_invalid_does_not_raise():
    dom = MoleculeDomain(ATOM_DECODER, BOND_DECODER, image_size=120)
    fig, ax = plt.subplots()
    dom.render(ax, oversaturated_carbon())  # best-effort skeleton or "invalid"
    assert len(ax.images) == 1 or len(ax.texts) >= 1
    plt.close(fig)


def test_molecule_domain_summary_and_metrics():
    dom = MoleculeDomain(ATOM_DECODER, BOND_DECODER, reference_smiles=[Chem.CanonSmiles("CCO")])
    samples = [ethanol(), ethanol(), oversaturated_carbon()]
    summary = dom.summarize(samples)
    assert "valid 2/3" in summary
    m = dom.metrics(samples)
    assert m["validity"] == 2 / 3
    assert m["uniqueness"] == 1 / 2  # 1 unique SMILES among 2 valid
    assert m["novelty"] == 0.0       # the one unique SMILES is in the reference


# --------------------------------------------------------------------------- #
# GenericGraphDomain
# --------------------------------------------------------------------------- #

def test_generic_domain_render_and_identity():
    dom = GenericGraphDomain()
    data = ethanol()
    fig, ax = plt.subplots()
    dom.render(ax, data)  # networkx node-link plot; must not raise
    plt.close(fig)
    # Identity is a stable WL hash; isomorphic graphs share it.
    assert dom.identity(data) == dom.identity(ethanol())
    assert dom.identity(data) != dom.identity(oversaturated_carbon())


def test_generic_generation_metrics_all_valid():
    dom = GenericGraphDomain()
    samples = [ethanol(), oversaturated_carbon()]
    m = generation_metrics(dom, samples)
    assert m["validity"] == 1.0          # generic: every graph is "valid"
    assert m["uniqueness"] == 1.0        # two distinct WL hashes


# --------------------------------------------------------------------------- #
# SampleVisualizationCallback: domain wiring + round-epoch labeling
# --------------------------------------------------------------------------- #

class _FakeTrainer:
    def __init__(self, sanity_checking, current_epoch):
        self.sanity_checking = sanity_checking
        self.current_epoch = current_epoch


class _FakeModule:
    """Minimal stand-in for a LightningModule that just returns fixed samples."""

    def __init__(self, samples):
        self._samples = samples
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def sample(self, **kwargs):
        return self._samples


def _make_callback():
    from defog.core import SampleVisualizationCallback
    captured = {}
    cb = SampleVisualizationCallback(
        num_samples=2,
        every_k_epochs=25,
        figure_callback=lambda fig: captured.__setitem__("title", fig._suptitle.get_text()),
    )
    return cb, captured


def test_callback_skips_sanity_check():
    cb, captured = _make_callback()
    module = _FakeModule([ethanol(), ethanol()])
    cb.on_validation_epoch_end(_FakeTrainer(sanity_checking=True, current_epoch=0), module)
    assert cb._val_epoch_count == 0  # sanity pass not counted
    assert "title" not in captured


def test_callback_round_epoch_labeling():
    cb, captured = _make_callback()
    module = _FakeModule([ethanol(), ethanol()])
    # 24 real validation epochs (current_epoch 0..23): no preview yet.
    for ep in range(24):
        cb.on_validation_epoch_end(_FakeTrainer(False, ep), module)
    assert "title" not in captured
    # 25th real epoch (current_epoch=24) fires -> labelled "Epoch 25".
    cb.on_validation_epoch_end(_FakeTrainer(False, 24), module)
    assert captured["title"].startswith("Epoch 25")
