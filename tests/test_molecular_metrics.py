"""
Tests for the extended molecular evaluation suite in ``defog.domains.molecule``:
- connected / disconnected  (valid-but-fragmented molecules)
- sanity  (valid AND connected AND rings within [3, 8])
- ring-size checks
- logP / TPSA / QED KL divergence (Gaussian-KDE, GuacaMol-style)

CPU-only; graphs are built from SMILES via the same converter training uses.
"""
import math

import numpy as np
from rdkit import Chem

from defog.domains.molecule import (
    build_encoders,
    smiles_to_pyg_data,
    molecular_metrics,
    ring_sizes_ok,
    continuous_kl,
    property_distributions,
)

ATOM_DECODER = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]
ATOM_ENC, ATOM_DEC, BOND_ENC, BOND_DEC = build_encoders(
    ATOM_DECODER, ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
)


def _samples(smiles_list):
    return [smiles_to_pyg_data(s, ATOM_ENC, BOND_ENC) for s in smiles_list]


def test_ring_sizes_ok():
    assert ring_sizes_ok(Chem.MolFromSmiles("c1ccccc1"))          # 6-ring
    assert ring_sizes_ok(Chem.MolFromSmiles("C1CC1"))             # 3-ring
    assert ring_sizes_ok(Chem.MolFromSmiles("C1CCCCCCC1"))        # 8-ring
    assert not ring_sizes_ok(Chem.MolFromSmiles("C1CCCCCCCC1"))   # 9-ring (wonky)
    assert ring_sizes_ok(Chem.MolFromSmiles("CCO"))              # acyclic -> ok


def test_connected_and_sanity_split():
    # benzene: connected + sane; two benzenes: valid but disconnected;
    # cyclononane: connected but ring size 9 -> not sane; aspirin: sane.
    smiles = ["c1ccccc1", "c1ccccc1.c1ccccc1", "C1CCCCCCCC1",
              "CC(=O)Oc1ccccc1C(=O)O"]
    m = molecular_metrics(_samples(smiles), ATOM_DEC, BOND_DEC, compute_kl=False)
    assert m["num_valid"] == 4
    assert m["validity"] == 1.0
    assert math.isclose(m["connected"], 0.75)
    assert math.isclose(m["disconnected"], 0.25)
    assert math.isclose(m["wonky_ring_frac"], 0.25)
    # sane = benzene + aspirin = 2 of 4 total samples
    assert math.isclose(m["sanity"], 0.5)


def test_novelty_against_reference():
    smiles = ["c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]  # benzene, aspirin
    ref = ["CC(=O)Oc1ccccc1C(=O)O", "CCO"]           # aspirin is known
    m = molecular_metrics(_samples(smiles), ATOM_DEC, BOND_DEC,
                          reference_smiles=ref, compute_kl=False)
    # benzene novel, aspirin not -> 1/2
    assert math.isclose(m["novelty"], 0.5)


def test_kl_identical_distributions_is_near_zero():
    # KL(P||P) ~ 0 for identical samples.
    x = np.random.RandomState(0).normal(size=500)
    assert continuous_kl(x, x) < 1e-6


def test_kl_disjoint_distributions_is_positive():
    a = np.random.RandomState(1).normal(0, 1, size=500)
    b = np.random.RandomState(2).normal(8, 1, size=500)
    assert continuous_kl(a, b) > 1.0


def test_kl_degenerate_returns_nan():
    assert math.isnan(continuous_kl([1.0], [1.0, 2.0]))       # too few
    assert math.isnan(continuous_kl([2.0, 2.0], [1.0, 2.0]))  # zero variance


def test_full_suite_with_kl_keys_present():
    smiles = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1O", "CCN(CC)CC", "c1ccncc1"]
    ref = property_distributions(
        ["CCO", "CCN", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O",
         "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "c1ccncc1", "CCCCCCCC"], max_n=1000
    )
    m = molecular_metrics(_samples(smiles), ATOM_DEC, BOND_DEC,
                          reference_descriptors=ref, compute_kl=True)
    for key in ("kl_logp", "kl_tpsa", "kl_qed", "kl_score"):
        assert key in m
