"""
Tests for the RL distribution-fidelity penalties (defog.core.distribution_penalty).

These terms exist to stop a reward hack, so the tests are organized around the
properties that make them capable of that -- not merely around their shapes:

1. Fragment typicality is **one-sided**: it is invariant to which in-vocabulary
   fragments a molecule uses, so it exerts no pull toward the mode. This is the
   property that keeps the anti-hacking term from causing its own collapse.
2. Fragment typicality fires on positive evidence only: invalid and
   unassessable molecules score 0, never a default penalty.
3. The MMD sibling term excludes the self-comparison. Verified by exact
   arithmetic, because including the diagonal reintroduces an ``O(1/n)``
   sample-size artefact -- the very thing that disqualifies FCD at rollout size.
4. MMD is **anti-collapse**: a batch of identical molecules scores worse than a
   diverse batch drawn from the same reference.
5. MMD tracks the reference: with the batch held fixed (so the sibling term is
   identical by construction), swapping in a matching reference lowers the score.
6. ``mmd2`` reproduces the textbook unbiased MMD^2 estimator exactly.
7. Degenerate batches (one valid, none valid) do not crash and do not silently
   produce a penalty.
"""

import gzip
import json

import numpy as np
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem, DataStructs  # noqa: E402

from defog.core.distribution_penalty import (  # noqa: E402
    FragmentTypicalityPenalty,
    FragmentVocabulary,
    MMDPenalty,
    brics_fragments,
    morgan_fingerprints,
)

# Chosen so their BRICS decompositions overlap in three of four fragments and
# differ in exactly one -- which is what makes the one-sidedness test sharp.
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"          # [1*]C(C)=O [16*]c1ccccc1[16*] [3*]O[3*] [6*]C(=O)O
ACETAMIDO = "CC(=O)Nc1ccccc1C(=O)O"        # ... [5*]N[5*] instead of [3*]O[3*]
ETHOXY = "CCOc1ccccc1C(=O)O"               # ... [4*]CC instead of [1*]C(C)=O

F_ACETYL = "[1*]C(C)=O"
F_PHENYL = "[16*]c1ccccc1[16*]"
F_ETHER = "[3*]O[3*]"
F_ACID = "[6*]C(=O)O"
F_AMINE = "[5*]N[5*]"
F_ETHYL = "[4*]CC"

DIVERSE = [
    "CC(=O)Oc1ccccc1C(=O)O", "CCOc1ccc(N)cc1", "Clc1ccccc1CN1CCOCC1",
    "O=C(N)c1cccnc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "c1ccc2[nH]ccc2c1",
    "COc1ccc(CCN)cc1OC", "CN1CCC[C@H]1c1cccnc1", "OCC1OC(O)C(O)C(O)C1O",
    "CC(=O)Nc1ccc(O)cc1", "Fc1ccc(cc1)C(=O)N1CCNCC1", "CCCCOC(=O)c1ccccc1",
]

ALIEN = [
    "CCCCCCCCCCCCCCCCCC", "C1CC2CCC1CC2", "N#CC#N", "OO", "S=C=S",
    "C1CC1C1CC1", "FC(F)(F)C(F)(F)F", "P(=O)(O)(O)O", "[SiH4]", "B(O)(O)O",
    "C1CCCCCCC1", "CCCCCCCCCC=O",
]


def _vocab(counts):
    return FragmentVocabulary(dict(counts))


# ===========================================================================
# BRICS plumbing
# ===========================================================================
def test_brics_returns_whole_molecule_when_unbreakable():
    """No BRICS bond is not an error -- RDKit hands back the molecule itself."""
    assert brics_fragments(Chem.MolFromSmiles("c1ccccc1")) == ["c1ccccc1"]


def test_brics_fragments_are_stable_and_sorted():
    frags = brics_fragments(Chem.MolFromSmiles(ASPIRIN))
    assert frags == sorted(frags)
    assert set(frags) == {F_ACETYL, F_PHENYL, F_ETHER, F_ACID}


# ===========================================================================
# Fragment vocabulary
# ===========================================================================
def test_vocabulary_build_counts_occurrences_not_molecules():
    v = FragmentVocabulary.build([ASPIRIN, ASPIRIN, ETHOXY], max_molecules=0, log=lambda *_: None)
    assert v.counts[F_PHENYL] == 3      # in all three
    assert v.counts[F_ACETYL] == 2      # aspirin only, twice
    assert v.counts[F_ETHYL] == 1       # ethoxy only
    assert v.provenance["n_decomposed"] == 3


def test_vocabulary_common_and_coverage_respect_threshold():
    v = _vocab({"a": 100, "b": 10, "c": 1})
    assert v.common(1) == {"a", "b", "c"}
    assert v.common(10) == {"a", "b"}
    assert v.common(101) == set()
    assert v.coverage(1) == pytest.approx(1.0)
    assert v.coverage(10) == pytest.approx(110 / 111)
    # Coverage must not increase as the threshold rises.
    cov = [v.coverage(t) for t in (1, 2, 10, 50, 200)]
    assert all(b <= a + 1e-12 for a, b in zip(cov, cov[1:]))


def test_vocabulary_roundtrips_through_disk(tmp_path):
    v = _vocab({F_PHENYL: 7, F_ACID: 3})
    v.provenance["marker"] = "kept"
    path = v.save(tmp_path / "vocab.json.gz")
    back = FragmentVocabulary.load(path)
    assert back.counts == v.counts
    assert back.provenance["marker"] == "kept"
    with gzip.open(path, "rt") as fh:      # readable without the class
        assert "counts" in json.load(fh)


def test_cache_path_changes_with_the_reference_content(tmp_path):
    """A different split under the same dataset name must miss the cache.

    Otherwise the vocabulary silently describes a set the generations are not
    being compared against, and nothing downstream would reveal it.
    """
    a = FragmentVocabulary.cache_path("moses", DIVERSE, max_molecules=10, seed=0,
                                      cache_dir=tmp_path)
    same = FragmentVocabulary.cache_path("moses", DIVERSE, max_molecules=10, seed=0,
                                         cache_dir=tmp_path)
    other = FragmentVocabulary.cache_path("moses", ALIEN, max_molecules=10, seed=0,
                                          cache_dir=tmp_path)
    assert a == same
    assert a != other


def test_build_or_load_uses_the_cache_the_second_time(tmp_path):
    calls = []
    v1 = FragmentVocabulary.build_or_load("t", [ASPIRIN, ETHOXY], max_molecules=0,
                                          cache_dir=tmp_path, log=calls.append)
    v2 = FragmentVocabulary.build_or_load("t", [ASPIRIN, ETHOXY], max_molecules=0,
                                          cache_dir=tmp_path, log=calls.append)
    assert v1.counts == v2.counts
    assert any("from cache" in c for c in calls)


# ===========================================================================
# Fragment typicality penalty
# ===========================================================================
def test_all_fragments_known_scores_zero():
    pen = FragmentTypicalityPenalty(
        _vocab({F_ACETYL: 9, F_PHENYL: 9, F_ETHER: 9, F_ACID: 9}), min_count=5)
    assert pen([ASPIRIN])[0] == pytest.approx(0.0)


def test_penalty_is_the_fraction_of_unknown_fragments():
    """One of aspirin's four fragments missing from the vocabulary -> 0.25."""
    pen = FragmentTypicalityPenalty(
        _vocab({F_ACETYL: 9, F_PHENYL: 9, F_ACID: 9}), min_count=5)   # no F_ETHER
    assert pen([ASPIRIN])[0] == pytest.approx(0.25)

    pen_two = FragmentTypicalityPenalty(_vocab({F_ACETYL: 9, F_PHENYL: 9}), min_count=5)
    assert pen_two([ASPIRIN])[0] == pytest.approx(0.5)


def test_penalty_is_one_sided_in_fragment_frequency():
    """THE anti-collapse property of this term.

    Two molecules, each built entirely from in-vocabulary fragments, but one
    using a fragment that is 2000x rarer. Both must score exactly 0: the term
    may say "this is not chemistry from the data", never "be more average".
    """
    counts = {F_ACETYL: 20_000, F_PHENYL: 20_000, F_ETHER: 20_000, F_ACID: 20_000,
              F_ETHYL: 10}
    pen = FragmentTypicalityPenalty(_vocab(counts), min_count=5)
    common_mol, rare_mol = pen([ASPIRIN]), pen([ETHOXY])
    assert common_mol[0] == pytest.approx(0.0)
    assert rare_mol[0] == pytest.approx(0.0)


def test_min_count_turns_a_rare_fragment_into_an_unknown_one():
    counts = {F_ACETYL: 9, F_PHENYL: 9, F_ETHER: 9, F_ACID: 9, F_ETHYL: 3}
    lenient = FragmentTypicalityPenalty(_vocab(counts), min_count=2)
    strict = FragmentTypicalityPenalty(_vocab(counts), min_count=5)
    assert lenient([ETHOXY])[0] == pytest.approx(0.0)
    assert strict([ETHOXY])[0] == pytest.approx(0.25)


def test_invalid_molecules_are_neutral_not_penalized():
    """The sanity reward already floors invalid at 0; penalising again would
    reorder failures that are all equally unusable."""
    pen = FragmentTypicalityPenalty(_vocab({F_PHENYL: 9}), min_count=5)
    out = pen([None, "", "not a smiles", "C1CC(((", ASPIRIN])
    assert out[0] == 0.0 and out[1] == 0.0 and out[2] == 0.0 and out[3] == 0.0
    assert out[4] > 0.0
    assert pen.last["frag_scored_frac"] == pytest.approx(1 / 5)


def test_unbreakable_molecule_is_scored_against_itself():
    pen = FragmentTypicalityPenalty(_vocab({F_PHENYL: 9}), min_count=5)
    assert pen(["c1ccccc1"])[0] == pytest.approx(1.0)          # not a known fragment
    known = FragmentTypicalityPenalty(_vocab({"c1ccccc1": 9}), min_count=5)
    assert known(["c1ccccc1"])[0] == pytest.approx(0.0)


def test_penalty_is_bounded_and_shaped():
    pen = FragmentTypicalityPenalty(_vocab({F_PHENYL: 9}), min_count=5)
    out = pen(DIVERSE)
    assert out.shape == (len(DIVERSE),)
    assert out.dtype == np.float64
    assert np.all((out >= 0.0) & (out <= 1.0))


# ===========================================================================
# MMD penalty
# ===========================================================================
def _tanimoto(a, b):
    fa, fb = morgan_fingerprints([a, b])
    return DataStructs.TanimotoSimilarity(fa, fb)


def _pen(reference, kernel="tanimoto"):
    return MMDPenalty(reference, n_reference=0, kernel=kernel, log=lambda *_: None)


# The estimator is written against the kernel interface, so every property that
# is a property of the *estimator* must hold for both kernels.
BOTH_KERNELS = pytest.mark.parametrize("kernel", ["tanimoto", "descriptor"])


def test_sibling_term_excludes_the_self_comparison():
    """Exact arithmetic: for batch [A, A, B] the sibling mean of sample 0 is
    ``(1 + t)/2``, not ``(1 + 1 + t)/3``.

    Including the diagonal adds a spurious ``1/n`` that shrinks with batch size
    -- exactly the sample-size artefact that rules FCD out of the reward.
    """
    a, b = DIVERSE[0], ALIEN[0]
    t = _tanimoto(a, b)
    pen = _pen(DIVERSE)
    pen([a, a, b])
    # Per sample: A sees {A, B} -> (1+t)/2 twice; B sees {A, A} -> t.
    correct = (2 * (1.0 + t) / 2 + t) / 3
    # If the diagonal leaked in, every sample would also see itself at 1.0.
    with_diagonal = (2 * (2.0 + t) / 3 + (2 * t + 1.0) / 3) / 3
    assert pen.last["mmd_sim_sibling"] == pytest.approx(correct)
    assert pen.last["mmd_sim_sibling"] != pytest.approx(with_diagonal)


def test_collapsed_batch_scores_worse_than_a_diverse_one():
    """THE anti-collapse property of this term.

    Both batches are drawn from the reference itself, so neither is atypical;
    the only difference is that one has repeated itself. A term that only
    rewarded reference-similarity would rank these equally, or backwards.
    """
    pen = _pen(DIVERSE)
    diverse = pen(DIVERSE[:8])
    collapsed = pen([DIVERSE[0]] * 8)
    assert collapsed.mean() > diverse.mean()


def test_reference_similarity_lowers_the_penalty_with_the_batch_held_fixed():
    """Same batch through two references: the sibling term is identical by
    construction, so any difference isolates the reference term."""
    batch = DIVERSE[:6]
    matching = _pen(DIVERSE)
    mismatched = _pen(ALIEN)
    m_out, x_out = matching(batch), mismatched(batch)
    assert matching.last["mmd_sim_sibling"] == pytest.approx(mismatched.last["mmd_sim_sibling"])
    assert matching.last["mmd_sim_ref"] > mismatched.last["mmd_sim_ref"]
    assert m_out.mean() < x_out.mean()


def test_mmd2_matches_the_textbook_unbiased_estimator():
    ref, batch = DIVERSE, ALIEN[:7]
    pen = _pen(ref)
    got = pen.mmd2(batch)

    xf = [f for f in morgan_fingerprints(batch) if f is not None]
    yf = [f for f in morgan_fingerprints(ref) if f is not None]
    n, m = len(xf), len(yf)
    kxx = sum(sum(DataStructs.BulkTanimotoSimilarity(f, xf)) - 1.0 for f in xf) / (n * (n - 1))
    kyy = sum(sum(DataStructs.BulkTanimotoSimilarity(f, yf)) - 1.0 for f in yf) / (m * (m - 1))
    kxy = sum(sum(DataStructs.BulkTanimotoSimilarity(f, yf)) for f in xf) / (n * m)
    assert got == pytest.approx(kxx - 2 * kxy + kyy, abs=1e-9)


def test_mmd2_is_smaller_within_a_distribution_than_across_two():
    """The property that matters: same-distribution scores lower than different.

    Note it is *not* tested by feeding one set as both X and Y. The unbiased
    estimator drops the diagonal from both self-terms but not from the cross
    term, so identical sets give a large negative value rather than zero -- it
    is unbiased for independent draws, which identical sets are not.
    """
    half_a, half_b = DIVERSE[::2], DIVERSE[1::2]
    within = _pen(half_b).mmd2(half_a)
    across = _pen(ALIEN).mmd2(half_a)
    assert within < across


def test_invalid_entries_score_zero_and_do_not_join_the_sibling_set():
    pen = _pen(DIVERSE)
    out = pen([DIVERSE[0], None, "garbage(((", DIVERSE[1]])
    assert out[1] == 0.0 and out[2] == 0.0
    assert out[0] != 0.0 and out[3] != 0.0
    assert pen.last["mmd_valid_frac"] == pytest.approx(0.5)
    # Sibling mean over the two valid entries only, not diluted by the failures.
    assert pen.last["mmd_sim_sibling"] == pytest.approx(_tanimoto(DIVERSE[0], DIVERSE[1]))


def test_single_valid_sample_has_no_sibling_term():
    pen = _pen(DIVERSE)
    out = pen([DIVERSE[0], None])
    assert pen.last["mmd_sim_sibling"] == 0.0
    assert out[0] == pytest.approx(-2.0 * pen.last["mmd_sim_ref"])


def test_all_invalid_batch_is_all_zeros():
    pen = _pen(DIVERSE)
    out = pen([None, "nonsense((("])
    assert np.all(out == 0.0)
    assert pen.last["mmd_valid_frac"] == 0.0


def test_reference_subsampling_is_deterministic():
    a = MMDPenalty(DIVERSE, n_reference=5, seed=3, kernel="tanimoto", log=lambda *_: None)
    b = MMDPenalty(DIVERSE, n_reference=5, seed=3, kernel="tanimoto", log=lambda *_: None)
    assert a(DIVERSE[:4]) == pytest.approx(b(DIVERSE[:4]))
    assert a.n_reference == 5


def test_empty_reference_is_rejected_loudly():
    with pytest.raises(ValueError):
        MMDPenalty(["garbage((("], n_reference=0, kernel="tanimoto", log=lambda *_: None)


def test_unknown_kernel_name_is_rejected():
    with pytest.raises(ValueError):
        MMDPenalty(DIVERSE, n_reference=0, kernel="euclidean", log=lambda *_: None)


# ===========================================================================
# Kernel-agnostic estimator properties
# ===========================================================================
@BOTH_KERNELS
def test_estimator_properties_hold_for_every_kernel(kernel):
    """The estimator is written against the kernel interface, so collapse
    detection and reference tracking must not depend on which kernel is used."""
    pen = _pen(DIVERSE, kernel)
    assert pen([DIVERSE[0]] * 8).mean() > pen(DIVERSE[:8]).mean()   # anti-collapse

    matching, mismatched = _pen(DIVERSE, kernel), _pen(ALIEN, kernel)
    batch = DIVERSE[:6]
    assert matching(batch).mean() < mismatched(batch).mean()        # tracks reference


@BOTH_KERNELS
def test_degenerate_batches_are_safe_for_every_kernel(kernel):
    pen = _pen(DIVERSE, kernel)
    assert np.all(pen([None, "nonsense((("]) == 0.0)
    out = pen([DIVERSE[0], None])
    assert pen.last["mmd_sim_sibling"] == 0.0
    assert out[1] == 0.0


@BOTH_KERNELS
def test_within_distribution_mmd2_is_lower_for_every_kernel(kernel):
    half_a, half_b = DIVERSE[::2], DIVERSE[1::2]
    assert _pen(half_b, kernel).mmd2(half_a) < _pen(ALIEN, kernel).mmd2(half_a)


# ===========================================================================
# Descriptor kernel
# ===========================================================================
def test_descriptor_kernel_standardizes_by_the_reference():
    """Without standardisation, molecular weight (~300) would swamp a fraction
    in [0,1] and the kernel would be a molecular-weight detector."""
    from defog.core.distribution_penalty import DescriptorRBFKernel

    k = DescriptorRBFKernel()
    feats = k.featurize(DIVERSE)
    k.fit(feats)
    assert k.mean.shape == (len(k.descriptors),)
    assert np.all(k.scale > 0)          # no zero divisor even for constant columns
    assert k.gamma > 0


def test_descriptor_kernel_is_one_on_the_diagonal_and_decays():
    from defog.core.distribution_penalty import DescriptorRBFKernel

    k = DescriptorRBFKernel()
    feats = [f for f in k.featurize(DIVERSE) if f is not None]
    k.fit(feats)
    g = k.gram(feats, feats)
    assert np.allclose(np.diag(g), 1.0)
    assert np.all((g >= 0.0) & (g <= 1.0))
    assert g.sum() < g.size            # not saturated at 1 everywhere


def test_descriptor_kernel_drops_unfeaturizable_molecules():
    from defog.core.distribution_penalty import DescriptorRBFKernel

    k = DescriptorRBFKernel()
    feats = k.featurize([DIVERSE[0], None, "garbage((("])
    assert feats[0] is not None and feats[1] is None and feats[2] is None


def test_descriptor_kernel_sees_a_composition_shift_that_tanimoto_misses():
    """Why this kernel exists, in miniature.

    A binary Morgan fingerprint records which substructures are *present*, not
    how many times. Hexane and eicosane therefore score 0.875 Tanimoto despite
    a 3x difference in size, so a batch that drifts in composition barely moves
    the Tanimoto MMD while moving the descriptor MMD a lot.

    That is exactly the shape of the MOSES reward hack: the policy did not
    invent new substructure, it changed how much of the ordinary kind it
    emitted (aromatic rings 1.88 -> 1.63, sp3 fraction 0.35 -> 0.40). FCD saw
    it and doubled; the Tanimoto kernel nearly did not.
    """
    ref = ["CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCC", "CCCCCCCCC", "CCCCCCCCCC"]
    matching = ["CCCCCC", "CCCCCCC", "CCCCCCCC"]
    shifted = ["CCCCCCCCCCCCCCCCCCCC", "CCCCCCCCCCCCCCCCCCCCCC",
               "CCCCCCCCCCCCCCCCCCCCCCCC"]

    # The premise: the fingerprint really is near-blind to this.
    fa, fb = morgan_fingerprints([ref[0], shifted[0]])
    assert DataStructs.TanimotoSimilarity(fa, fb) > 0.8

    deltas = {}
    for kern in ("tanimoto", "descriptor"):
        pen = _pen(ref, kern)
        deltas[kern] = pen(shifted).mean() - pen(matching).mean()
    assert deltas["tanimoto"] > 0        # right direction, but faint
    assert deltas["descriptor"] > 5 * deltas["tanimoto"]
