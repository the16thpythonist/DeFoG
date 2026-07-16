"""Unit tests for the structural-sanity reward logic in experiments/gdpo_sanity.py.

Covers the NEW, risky pieces: per-molecule structural features (ring systems, spiro,
bridgehead), envelope construction (every reference molecule passes at quantile 1.0),
envelope.check() flagging out-of-support features, and ring-histogram divergence.
The GDPO machinery itself is covered by tests/test_rl.py.
"""
import numpy as np
import pytest
from rdkit import Chem

from experiments.gdpo_sanity import (
    largest_ring_system, mol_features, SanityEnvelope, build_envelope,
    _upper_bound, ring_size_histogram, hist_divergence,
)

# canonical test molecules
BENZENE = "c1ccccc1"                       # 1 ring, size 6
NAPHTHALENE = "c1ccc2ccccc2c1"             # 2 fused rings, size 6
BIPHENYL = "c1ccc(-c2ccccc2)cc1"           # 2 non-fused rings, size 6
SPIRO = "C1CCC2(CC1)CCCCC2"                # spiro[5.5]undecane: 2 rings sharing 1 atom
ANTHRACENE = "c1ccc2cc3ccccc3cc2c1"        # 3 linearly-fused rings, size 6
NORBORNANE = "C1CC2CCC1C2"                 # bridged bicyclic (bridgehead atoms)
CYCLOPROPANE = "C1CC1"                     # ring size 3


def feats(smi):
    return mol_features(Chem.MolFromSmiles(smi))


def test_largest_ring_system_fused_vs_separate():
    assert largest_ring_system(Chem.MolFromSmiles(BENZENE)) == 1
    assert largest_ring_system(Chem.MolFromSmiles(NAPHTHALENE)) == 2   # fused -> one system
    assert largest_ring_system(Chem.MolFromSmiles(BIPHENYL)) == 1      # two separate systems
    assert largest_ring_system(Chem.MolFromSmiles(SPIRO)) == 2         # spiro share -> one system
    assert largest_ring_system(Chem.MolFromSmiles(ANTHRACENE)) == 3
    assert largest_ring_system(Chem.MolFromSmiles("CCO")) == 0         # acyclic


def test_mol_features_basic():
    f = feats(BENZENE)
    assert f["num_rings"] == 1 and f["ring_sizes"] == [6]
    assert feats(NAPHTHALENE)["num_rings"] == 2
    assert feats(SPIRO)["num_spiro"] == 1
    assert feats(NORBORNANE)["num_bridgehead"] >= 2


def test_upper_bound_quantile():
    assert _upper_bound([1, 2, 3, 10], 1.0) == 10      # strict max at q=1.0
    assert _upper_bound([1, 2, 3, 4], 0.5) == 3        # ceil(median 2.5)
    assert _upper_bound([], 1.0) == 0


def test_envelope_admits_every_reference_molecule():
    """Guarantee: with quantile 1.0, every molecule used to build the envelope passes."""
    ref = [BENZENE, NAPHTHALENE, BIPHENYL, SPIRO, ANTHRACENE, NORBORNANE]
    env = build_envelope([feats(s) for s in ref], quantile=1.0, ring_min_count=1)
    for smi in ref:
        ok, reason = env.check(Chem.MolFromSmiles(smi))
        assert ok, f"{smi} rejected by its own envelope: {reason}"


def test_envelope_flags_out_of_support_ring_size():
    env = build_envelope([feats(BENZENE), feats(NAPHTHALENE), feats(BIPHENYL)],
                         quantile=1.0, ring_min_count=1)
    assert env.allowed_ring_sizes == {6}
    ok, reason = env.check(Chem.MolFromSmiles(CYCLOPROPANE))   # size-3 ring
    assert not ok and reason == "ring_size:3"


def test_envelope_flags_too_many_rings():
    # benzene(1) + biphenyl/naphthalene(2) -> max_num_rings = 2; anthracene has 3
    env = build_envelope([feats(BENZENE), feats(BIPHENYL), feats(NAPHTHALENE)],
                         quantile=1.0, ring_min_count=1)
    assert env.max_num_rings == 2
    ok, reason = env.check(Chem.MolFromSmiles(ANTHRACENE))
    assert not ok and reason == "num_rings"


def test_ring_min_count_trims_rare_sizes():
    # one 3-membered ring molecule among many 6-membered ones; min_count=2 drops size 3
    fs = [feats(BENZENE)] * 3 + [feats(CYCLOPROPANE)]
    env = build_envelope(fs, quantile=1.0, ring_min_count=2)
    assert 3 not in env.allowed_ring_sizes and 6 in env.allowed_ring_sizes


def test_ring_histogram_and_divergence():
    h, bins = ring_size_histogram([BENZENE], ring_hist_max=12)
    assert bins[3] == 6                      # bins = [3,4,5,6,...,12,'overflow']
    assert h[3] == pytest.approx(1.0)        # all mass on size-6 rings
    assert h.sum() == pytest.approx(1.0)
    tv, mae = hist_divergence(h, h)
    assert tv == pytest.approx(0.0) and mae == pytest.approx(0.0)
    # disjoint distributions -> TV = 1.0
    h2, _ = ring_size_histogram([CYCLOPROPANE], ring_hist_max=12)
    tv2, _ = hist_divergence(h, h2)
    assert tv2 == pytest.approx(1.0)


def test_macrocycle_overflow_bin():
    macro = "C1CCCCCCCCCCCCCCC1"            # 16-membered ring -> overflow bin
    h, bins = ring_size_histogram([macro], ring_hist_max=12)
    assert bins[-1] == "overflow"
    assert h[-1] == pytest.approx(1.0)


def test_envelope_roundtrip_dict():
    env = build_envelope([feats(BENZENE), feats(NAPHTHALENE)], quantile=1.0)
    env2 = SanityEnvelope.from_dict(env.to_dict())
    assert env2.allowed_ring_sizes == env.allowed_ring_sizes
    assert env2.max_num_rings == env.max_num_rings
