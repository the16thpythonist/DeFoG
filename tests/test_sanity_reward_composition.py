"""
Tests for the shaped reward composition in experiments/gdpo_sanity.py.

The penalty terms are tested on their own in test_distribution_penalty.py. What
is tested here is the *composition*, on real decoded molecules, because that is
where the two ways of getting this wrong live:

1. The penalty is applied to invalid samples, breaking the "invalid is strictly
   worst" ordering the graded sanity reward depends on. If a valid but atypical
   molecule can score below an unparseable one, the gradient points at
   invalidity.
2. The weights do not actually reach the reward, so a sweep arm labelled
   ``beta=0.5`` silently trains the control. That failure produces a clean,
   plausible, completely uninformative sweep.

The local MOSES checkpoint is a 481 KB debug stub that generates nothing valid,
so an end-to-end GPU smoke test cannot exercise these paths. These build the
dense tensors from known SMILES instead, which reaches the same code with
molecules whose penalties can be computed independently.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("rdkit")

ROOT = Path(__file__).resolve().parents[1]


def _load_experiment_module():
    spec = importlib.util.spec_from_file_location(
        "gdpo_sanity_mod", ROOT / "experiments" / "gdpo_sanity.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gdpo_sanity_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


GS = _load_experiment_module()

from defog.core.data import to_dense  # noqa: E402
from defog.core.distribution_penalty import (  # noqa: E402
    FragmentTypicalityPenalty,
    FragmentVocabulary,
    MMDPenalty,
)
from defog.domains import MoleculeDomain  # noqa: E402
from defog.domains.molecule import build_encoders  # noqa: E402

ATOMS = ["C", "N", "S", "O", "F", "Cl", "Br", "H"]
BONDS = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

# Drug-like, connected, all rings in [3,8] -> every one should score sanity 3.
MOLECULES = [
    "CC(=O)Nc1ccc(O)cc1",
    "CCOc1ccc(N)cc1",
    "O=C(N)c1cccnc1",
    "Clc1ccccc1CN1CCOCC1",
    "COc1ccc(CCN)cc1OC",
    "CN1CCCC1c1cccnc1",
]

REFERENCE = MOLECULES + [
    "CC(=O)Oc1ccccc1C(=O)O", "Fc1ccc(cc1)C(=O)N1CCNCC1",
    "CCCCOC(=O)c1ccccc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
]


@pytest.fixture(scope="module")
def domain():
    _, atom_decoder, _, bond_decoder = build_encoders(ATOMS, BONDS)
    return MoleculeDomain(atom_decoder, bond_decoder, reference_smiles=set(REFERENCE))


@pytest.fixture(scope="module")
def batch(domain):
    """Dense (X1, E1, node_mask) for MOLECULES, the way a rollout supplies them.

    ``MoleculeDomain.encode`` already returns dense one-hot cores, so these are
    padded to a common size by hand rather than routed through PyG batching.
    Padded nodes are left all-zero and excluded by ``node_mask``, which is the
    same convention the sampler produces.
    """
    cores = [domain.encode(s) for s in MOLECULES]
    n_max = max(x.shape[0] for x, _ in cores)
    n_x, n_e = cores[0][0].shape[-1], cores[0][1].shape[-1]

    X = torch.zeros(len(cores), n_max, n_x)
    E = torch.zeros(len(cores), n_max, n_max, n_e)
    node_mask = torch.zeros(len(cores), n_max, dtype=torch.bool)
    for i, (x, e) in enumerate(cores):
        n = x.shape[0]
        X[i, :n], E[i, :n, :n], node_mask[i, :n] = x, e, True
    return X, E, node_mask


@pytest.fixture(scope="module")
def penalties():
    """min_count=2 on purpose.

    The reference set contains the test molecules, so at min_count=1 every
    fragment is in-vocabulary and the fragment penalty is a correct but
    identically-zero vector -- which would make "does this weight reach the
    reward" untestable. At 2 the singleton fragments fall out and the penalties
    land in (0, 1).
    """
    vocab = FragmentVocabulary.build(REFERENCE, max_molecules=0, log=lambda *_: None)
    frag = FragmentTypicalityPenalty(vocab, min_count=2)
    mmd = MMDPenalty(REFERENCE, n_reference=0, kernel="descriptor",
                     log=lambda *_: None)
    return frag, mmd


def _decoded(domain, batch):
    from defog.core.data import dense_to_pyg

    X1, E1, node_mask = batch
    datas = dense_to_pyg(X1, E1, None, node_mask, node_mask.sum(-1))
    return [domain.identity(d) for d in datas]


def test_the_test_molecules_actually_decode(domain, batch):
    """Guard the premise: if these do not decode, every assertion below is
    vacuously about a batch of zeros -- which is exactly how the stub checkpoint
    made the end-to-end smoke test look like it passed."""
    smiles = _decoded(domain, batch)
    assert sum(s is not None for s in smiles) >= 4


def test_zero_weights_reproduce_the_unshaped_reward(domain, batch, penalties):
    """The control arm must be the original code, not an approximation."""
    frag, mmd = penalties
    plain = GS.SanityReward(domain)(*batch)
    with_zero = GS.SanityReward(domain, frag_penalty=frag, alpha=0.0,
                                mmd_penalty=mmd, beta=0.0)(*batch)
    assert torch.equal(plain, with_zero)


def test_shaped_reward_equals_sanity_minus_weighted_penalties(domain, batch, penalties):
    frag, mmd = penalties
    alpha, beta = 0.3, 0.4
    smiles = _decoded(domain, batch)

    base = GS.SanityReward(domain)(*batch)
    shaped = GS.SanityReward(domain, frag_penalty=frag, alpha=alpha,
                             mmd_penalty=mmd, beta=beta)(*batch)
    expected = (base
                - alpha * torch.as_tensor(frag(smiles), dtype=base.dtype)
                - beta * torch.as_tensor(mmd(smiles), dtype=base.dtype))
    assert torch.allclose(shaped, expected, atol=1e-6)


def test_each_weight_moves_the_reward_on_its_own(domain, batch, penalties):
    """Catches a wiring slip where one weight is read but the other is dropped."""
    frag, mmd = penalties
    base = GS.SanityReward(domain)(*batch)
    only_frag = GS.SanityReward(domain, frag_penalty=frag, alpha=0.5)(*batch)
    only_mmd = GS.SanityReward(domain, mmd_penalty=mmd, beta=0.5)(*batch)
    assert not torch.allclose(only_frag, base)
    assert not torch.allclose(only_mmd, base)


def test_invalid_samples_stay_at_exactly_zero(domain, batch, penalties):
    """The ordering invariant: no penalty may be charged to a sample that did
    not decode, or invalid stops being strictly worst."""
    frag, mmd = penalties
    X1, E1, node_mask = batch
    smiles = _decoded(domain, batch)

    reward = GS.SanityReward(domain, frag_penalty=frag, alpha=0.4,
                             mmd_penalty=mmd, beta=0.4)
    out = reward(X1, E1, node_mask)
    for i, s in enumerate(smiles):
        if s is None:
            assert out[i].item() == 0.0


def test_valid_stays_above_invalid_while_alpha_plus_beta_below_one(domain, batch, penalties):
    """The documented safe region for the weights, asserted rather than assumed."""
    frag, mmd = penalties
    reward = GS.SanityReward(domain, frag_penalty=frag, alpha=0.4,
                             mmd_penalty=mmd, beta=0.4)
    out = reward(*batch)
    smiles = _decoded(domain, batch)
    valid = [out[i].item() for i, s in enumerate(smiles) if s is not None]
    assert valid, "need at least one valid molecule for this to mean anything"
    assert min(valid) > 0.0


def test_raw_targeted_rates_are_unaffected_by_the_weights(domain, batch, penalties):
    """``sanity_frac`` and friends must keep meaning the raw failure rate, so the
    training curves stay comparable between a control arm and a penalised one."""
    frag, mmd = penalties
    plain = GS.SanityReward(domain)
    shaped = GS.SanityReward(domain, frag_penalty=frag, alpha=0.5,
                             mmd_penalty=mmd, beta=0.5)
    plain(*batch)
    shaped(*batch)
    for key in ("valid_frac", "connected_frac", "rings_ok_frac", "sanity_frac",
                "disconnected_frac", "wonky_ring_frac"):
        assert plain.last[key] == pytest.approx(shaped.last[key])


def test_shaped_stats_are_recorded_only_when_shaping(domain, batch, penalties):
    frag, mmd = penalties
    plain = GS.SanityReward(domain)
    plain(*batch)
    assert "shaped_reward_mean" not in plain.last

    shaped = GS.SanityReward(domain, mmd_penalty=mmd, beta=0.2)
    shaped(*batch)
    assert "shaped_reward_mean" in shaped.last
    assert "mmd_sim_sibling" in shaped.last


def test_penalties_are_finite_on_real_molecules(domain, batch, penalties):
    frag, mmd = penalties
    out = GS.SanityReward(domain, frag_penalty=frag, alpha=0.3,
                          mmd_penalty=mmd, beta=0.3)(*batch)
    assert torch.all(torch.isfinite(out))
