"""
Distribution-fidelity penalties for RL reward shaping.

Why this module exists
----------------------
The MOSES sanity-RL run raised validity from 0.885 to 0.939 on 4/4 seeds while
FCD against the validation set went from 0.863 to 1.706, also on 4/4 seeds. The
policy found a corner of chemical space that is reliably valid and reliably
narrow. Every quantity the reward could see improved; the one it could not see
collapsed. Of the three datasets, the largest apparent gain turned out to be the
fraudulent one -- which is the whole argument for putting a distribution term
*inside* the reward rather than only in the post-hoc report.

Why not FCD itself
------------------
Two reasons, both fatal at rollout scale:

1. **Bias.** FCD is a Frechet distance between Gaussians fitted to ChemNet
   activations. The covariance estimate is badly biased at small n. Measured on
   this project's own reference data, the same distribution scores 5.18 at
   n=500, 1.40 at n=2000, 0.455 at n=6000 and 0.218 at n=12443. At the rollout
   size of 128 the bias would dwarf any real signal.
2. **It is not per-sample.** FCD is a property of a *set*. GDPO needs a scalar
   per rollout to weight that rollout's log-probability. There is no exact
   decomposition of a Frechet distance into per-sample contributions.

Both penalties here are chosen to survive n=128: the fragment term is a
per-molecule lookup with no set-level statistics at all, and the MMD term uses
the *unbiased* (i != j) estimator, whose variance grows at small n but whose
expectation does not move.

Keeping FCD honest
------------------
Both penalties take their reference from the **train** split. FCD in evaluation
is scored against **validation**. That separation is deliberate and load-bearing:
it leaves FCD an independent verdict rather than a quantity the policy has been
optimised toward. A run that improves the penalty while FCD degrades is still a
failure, and still detectable -- which is exactly the property that caught the
MOSES hack in the first place.

The two terms
-------------
``FragmentTypicalityPenalty`` -- fraction of a molecule's BRICS fragments that
do not appear (often enough) in the train vocabulary. One-sided: it is invariant
to *which* common fragments a molecule uses, so it can be driven to zero by any
molecule built from known parts. There is no gradient toward the most common
fragment, and therefore no collapse pressure. It catches "the policy invented a
motif the data does not contain".

``MMDPenalty`` -- per-sample decomposition of the maximum mean discrepancy under
a Tanimoto kernel on Morgan fingerprints. Unlike the fragment term this *does*
see the batch: its sibling term repels a sample from the rest of its own rollout
batch, so a collapsing policy is penalised even when every molecule it emits is
individually typical. It catches "the policy kept making the same safe thing".

They are complementary, and neither subsumes the other. A policy emitting 128
copies of a perfectly ordinary drug scores 0 on the fragment term and badly on
MMD; a policy emitting 128 diverse molecules full of invented motifs scores well
on MMD's sibling term and badly on fragments.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "_fragcache"

# Morgan settings. radius 2 / 2048 bits is the ECFP4 convention that MOSES,
# GuacaMol and the FCD literature all use, so the kernel sees molecules the same
# way the reported metrics do.
MORGAN_RADIUS = 2
MORGAN_BITS = 2048


# ===========================================================================
# Fingerprints
# ===========================================================================
def _morgan_generator(radius: int = MORGAN_RADIUS, bits: int = MORGAN_BITS):
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)


def morgan_fingerprints(smiles: Sequence[Optional[str]], *,
                        radius: int = MORGAN_RADIUS, bits: int = MORGAN_BITS):
    """SMILES -> list of RDKit bit vectors, ``None`` where the molecule fails.

    ``None`` is propagated rather than dropped so the caller keeps positional
    correspondence with its reward vector -- silently compacting the list is how
    a reward gets attached to the wrong rollout.
    """
    from rdkit import Chem

    gen = _morgan_generator(radius, bits)
    out = []
    for smi in smiles:
        if not smi:
            out.append(None)
            continue
        mol = Chem.MolFromSmiles(smi)
        out.append(None if mol is None else gen.GetFingerprint(mol))
    return out


# ===========================================================================
# Fragment typicality
# ===========================================================================
def brics_fragments(mol) -> Optional[List[str]]:
    """BRICS leaf fragments of ``mol`` as canonical SMILES, or ``None`` on failure.

    Returns ``None`` -- not an empty list -- when decomposition raises, so the
    caller can treat "we could not assess this" separately from "this molecule
    has no atypical fragments". Conflating the two would turn every RDKit
    hiccup into a silent penalty of either 0 or 1 depending on which way the
    conflation went.

    Note that a molecule with **no breakable BRICS bond is not a failure**:
    RDKit returns the molecule itself as a single fragment, and this passes that
    through. So a bare ring or a plain alkane is scored as one fragment equal to
    the whole molecule, which is atypical unless that exact species is a common
    fragment of the reference. That is intended -- "the policy started emitting
    bare cyclohexanes" is precisely a collapse the penalty should see -- but it
    does make the term harsher on small molecules than on large ones.
    """
    from rdkit.Chem import BRICS

    try:
        frags = BRICS.BRICSDecompose(mol, returnMols=False)
    except Exception:
        return None
    if not frags:
        return None
    return sorted(str(f) for f in frags)


class FragmentVocabulary:
    """BRICS fragment counts from a reference (train) set.

    Stores raw counts rather than a thresholded set so ``min_count`` stays a
    load-time decision: re-tuning the rarity threshold must not require an
    8-minute rebuild, or it will not get re-tuned.
    """

    def __init__(self, counts: Dict[str, int], provenance: Optional[Dict] = None):
        self.counts = counts
        self.provenance = provenance or {}

    # -- construction --------------------------------------------------------
    @classmethod
    def build(cls, smiles: Sequence[str], *, max_molecules: int = 250_000,
              seed: int = 0, log_every: int = 25_000, log=print) -> "FragmentVocabulary":
        """Decompose up to ``max_molecules`` reference molecules and count fragments.

        The subsample exists because BRICS runs at roughly 1-2 ms/molecule and
        MOSES train has 1.58M rows -- 40 minutes for a vocabulary whose *common*
        entries are already pinned down long before that. Rare fragments are
        exactly what the ``min_count`` threshold discards anyway, so the tail we
        skip is the tail we would drop. Drawn with a fixed seed so the vocabulary
        is reproducible.
        """
        from rdkit import Chem

        smiles = list(smiles)
        if 0 < max_molecules < len(smiles):
            idx = np.random.default_rng(seed).choice(len(smiles), max_molecules,
                                                     replace=False)
            subset = [smiles[int(i)] for i in sorted(idx)]
        else:
            subset = smiles

        counts: Dict[str, int] = {}
        n_ok = n_failed = 0
        for i, smi in enumerate(subset):
            if log_every and i and i % log_every == 0:
                log(f"  BRICS {i}/{len(subset)} -- {len(counts)} distinct fragments")
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                n_failed += 1
                continue
            frags = brics_fragments(mol)
            if frags is None:
                n_failed += 1
                continue
            n_ok += 1
            for f in frags:
                counts[f] = counts.get(f, 0) + 1

        return cls(counts, provenance={
            "n_reference_smiles": len(smiles),
            "n_decomposed": n_ok,
            "n_failed": n_failed,
            "max_molecules": max_molecules,
            "seed": seed,
            "n_distinct_fragments": len(counts),
        })

    # -- persistence ---------------------------------------------------------
    @staticmethod
    def cache_path(dataset: str, smiles: Sequence[str], *, max_molecules: int,
                   seed: int, cache_dir=None) -> Path:
        """Cache filename keyed by the reference set's own content.

        The digest covers the reference SMILES themselves (via count plus a
        sample), so pointing the same dataset name at a different split -- the
        mistake that would silently score generations against the wrong
        vocabulary -- produces a different file rather than a stale hit.
        """
        h = hashlib.sha256()
        h.update(f"{dataset}|{len(smiles)}|{max_molecules}|{seed}|".encode())
        step = max(1, len(smiles) // 1000)
        for s in list(smiles)[::step][:1000]:
            h.update(s.encode())
        cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        return cache_dir / f"brics_{dataset}_{h.hexdigest()[:16]}.json.gz"

    def save(self, path) -> Path:
        """Write the vocabulary, atomically.

        The rename matters: the sweep launches four RL arms at once and they
        share a cache path, so a plain write leaves a window in which one arm
        reads a half-written file. os.replace is atomic within a filesystem, so
        a late writer replaces a complete file with an identical complete file
        and a reader sees one or the other, never a fragment.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump({"counts": self.counts, "provenance": self.provenance}, fh)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path) -> "FragmentVocabulary":
        with gzip.open(Path(path), "rt", encoding="utf-8") as fh:
            blob = json.load(fh)
        return cls(blob["counts"], blob.get("provenance", {}))

    @classmethod
    def build_or_load(cls, dataset: str, smiles: Sequence[str], *,
                      max_molecules: int = 250_000, seed: int = 0,
                      cache_dir=None, log=print) -> "FragmentVocabulary":
        path = cls.cache_path(dataset, smiles, max_molecules=max_molecules,
                              seed=seed, cache_dir=cache_dir)
        if path.exists():
            log(f"fragment vocabulary from cache {path}")
            return cls.load(path)
        log(f"building fragment vocabulary for {dataset} (no cache at {path})")
        vocab = cls.build(smiles, max_molecules=max_molecules, seed=seed, log=log)
        vocab.save(path)
        log(f"  {vocab.provenance['n_distinct_fragments']} distinct fragments -> {path}")
        return vocab

    # -- queries -------------------------------------------------------------
    def common(self, min_count: int) -> set:
        return {f for f, c in self.counts.items() if c >= min_count}

    def coverage(self, min_count: int) -> float:
        """Share of total fragment *occurrences* retained at this threshold.

        The useful diagnostic when picking ``min_count``: distinct-fragment
        counts collapse dramatically with any threshold (most fragments are
        singletons) while occurrence coverage barely moves, and it is coverage
        that determines how often a legitimate molecule gets penalised.
        """
        total = sum(self.counts.values())
        if not total:
            return 0.0
        return sum(c for c in self.counts.values() if c >= min_count) / total


class FragmentTypicalityPenalty:
    """Fraction of a molecule's BRICS fragments that are not in the vocabulary.

    ``__call__(smiles) -> np.ndarray`` in [0, 1], one entry per input. Higher is
    worse. Invalid inputs (``None``) score 0.0, and so do molecules BRICS cannot
    decompose: the penalty fires only on positive evidence of atypicality, never
    on absence of evidence. Invalid molecules are already floored by the sanity
    reward, so penalising them again would only distort the ordering among
    failures that are all equally unusable.

    **One-sided by construction.** The score depends only on how many fragments
    fall outside the vocabulary, never on how common the in-vocabulary ones are.
    A molecule of entirely ordinary parts scores 0 whether those parts are the
    single most frequent motif in the data or the 5000th. That is what keeps the
    term from doubling as a pull toward the mode -- it can say "this is not
    drug-like chemistry" without ever saying "be more like the average".
    """

    def __init__(self, vocab: FragmentVocabulary, min_count: int = 5):
        self.vocab = vocab
        self.min_count = min_count
        self._common = vocab.common(min_count)
        self.last: Dict[str, float] = {}

    def __len__(self) -> int:
        return len(self._common)

    def score_mol(self, mol) -> Optional[float]:
        """Penalty for one RDKit molecule, or ``None`` if it cannot be assessed."""
        frags = brics_fragments(mol)
        if not frags:
            return None
        n_bad = sum(1 for f in frags if f not in self._common)
        return n_bad / len(frags)

    def __call__(self, smiles: Sequence[Optional[str]]) -> np.ndarray:
        from rdkit import Chem

        out = np.zeros(len(smiles), dtype=np.float64)
        n_scored = n_unassessable = 0
        for i, smi in enumerate(smiles):
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            score = self.score_mol(mol)
            if score is None:
                n_unassessable += 1
                continue
            out[i] = score
            n_scored += 1
        k = max(1, n_scored)
        self.last = {
            "frag_penalty_mean": float(out.sum() / k),
            "frag_penalty_max": float(out.max()) if len(out) else 0.0,
            "frag_scored_frac": n_scored / max(1, len(smiles)),
            "frag_unassessable_frac": n_unassessable / max(1, len(smiles)),
        }
        return out


# ===========================================================================
# Kernels
# ===========================================================================
# A kernel supplies two things: ``featurize(smiles) -> list`` with ``None`` for
# molecules it cannot handle, and ``gram(A, B) -> (len(A), len(B)) ndarray``.
# Everything in MMDPenalty is written against that pair, so swapping the kernel
# changes what the penalty can see without touching the estimator.


class TanimotoKernel:
    """Tanimoto similarity on Morgan (ECFP4) fingerprints.

    The obvious first choice, and empirically the wrong one for the MOSES
    reward hack. Measured on the base-vs-hacked sample sets, this kernel put
    MMD^2 at 0.0008 against 0.0020 -- correctly ordered, but small enough that
    a 128-sample batch could barely tell them apart (AUC 0.71), while FCD moved
    0.863 -> 1.706 on the same data.

    The reason is structural. The hacked policy did not invent new
    substructures; it shifted global composition, emitting fewer aromatic rings
    (1.88 -> 1.63) and more sp3 carbon (0.35 -> 0.40). Tanimoto compares which
    substructure bits are set and normalises by their union, so a moderate
    change in how *often* ordinary rings appear barely moves it. Kept because it
    is the right tool for a different failure -- a policy that starts emitting
    genuinely novel substructure -- which the descriptor kernel would miss.
    """

    name = "tanimoto"

    def __init__(self, radius: int = MORGAN_RADIUS, bits: int = MORGAN_BITS):
        self.radius, self.bits = radius, bits

    def featurize(self, smiles: Sequence[Optional[str]]) -> List:
        return morgan_fingerprints(smiles, radius=self.radius, bits=self.bits)

    def fit(self, reference_features) -> None:
        return None

    def gram(self, a, b) -> np.ndarray:
        from rdkit import DataStructs

        out = np.empty((len(a), len(b)))
        for i, fp in enumerate(a):
            out[i] = DataStructs.BulkTanimotoSimilarity(fp, list(b))
        return out


# Chosen to span the axes that distinguish a molecular distribution: size,
# lipophilicity, polarity, ring content, flexibility and heteroatom makeup.
# These are the same descriptor families the MOSES and GuacaMol
# distribution-learning benchmarks report, which is what makes them a
# reasonable stand-in for ChemNet's view in FCD. QED is deliberately absent --
# it is a function of several of the others, so it adds cost and correlation
# rather than a new axis.
DESCRIPTORS = (
    "MolWt", "MolLogP", "TPSA", "NumRings", "NumAromaticRings",
    "NumRotatableBonds", "NumHBD", "NumHBA", "FractionCSP3", "NumHeavyAtoms",
    "nN", "nO", "nS", "nHalogen",
)


def _descriptor_funcs():
    from rdkit.Chem import Crippen, Descriptors
    from rdkit.Chem import rdMolDescriptors as rdmd

    def _count(symbols):
        return lambda m: float(sum(1 for a in m.GetAtoms() if a.GetSymbol() in symbols))

    return {
        "MolWt": Descriptors.MolWt,
        "MolLogP": Crippen.MolLogP,
        "TPSA": rdmd.CalcTPSA,
        "NumRings": rdmd.CalcNumRings,
        "NumAromaticRings": rdmd.CalcNumAromaticRings,
        "NumRotatableBonds": rdmd.CalcNumRotatableBonds,
        "NumHBD": rdmd.CalcNumHBD,
        "NumHBA": rdmd.CalcNumHBA,
        "FractionCSP3": rdmd.CalcFractionCSP3,
        "NumHeavyAtoms": lambda m: float(m.GetNumHeavyAtoms()),
        "nN": _count({"N"}),
        "nO": _count({"O"}),
        "nS": _count({"S"}),
        "nHalogen": _count({"F", "Cl", "Br", "I"}),
    }


class DescriptorRBFKernel:
    """RBF kernel on physicochemical descriptors, standardised by the reference.

    Built specifically because the Tanimoto kernel could not see the MOSES
    reward hack. The hack was a shift in global composition, and these
    descriptors measure exactly that -- a 0.31-sigma drop in aromatic ring count
    is roughly three standard errors of a 128-sample batch mean, so it is
    detectable at rollout scale where a fingerprint kernel's signal is not.

    Standardisation uses the reference set's own mean and standard deviation, so
    every descriptor contributes on a comparable scale and no single large-valued
    one (molecular weight, ~307 against a fraction in [0,1]) dominates the
    distance. The bandwidth follows the median heuristic on reference pairs,
    which keeps the kernel in its informative range rather than saturating at 0
    or 1.
    """

    name = "descriptor"

    def __init__(self, descriptors: Sequence[str] = DESCRIPTORS, *,
                 whiten: bool = False, shrinkage: float = 0.1):
        self.descriptors = tuple(descriptors)
        self.whiten = whiten
        self.shrinkage = shrinkage
        self._funcs = None
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.transform: Optional[np.ndarray] = None
        self.gamma: Optional[float] = None

    def featurize(self, smiles: Sequence[Optional[str]]) -> List:
        from rdkit import Chem

        if self._funcs is None:
            self._funcs = _descriptor_funcs()
        out = []
        for smi in smiles:
            if not smi:
                out.append(None)
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                out.append(None)
                continue
            try:
                vec = np.array([float(self._funcs[d](mol)) for d in self.descriptors])
            except Exception:
                out.append(None)
                continue
            # A non-finite descriptor would poison every distance it appears in,
            # so such a molecule is dropped rather than clamped to a fiction.
            out.append(vec if np.all(np.isfinite(vec)) else None)
        return out

    def fit(self, reference_features) -> None:
        """Learn the standardisation (or whitening) and bandwidth from the reference."""
        ref = np.asarray([f for f in reference_features if f is not None])
        if ref.size == 0:
            raise ValueError("no reference molecule produced descriptors")
        self.mean = ref.mean(axis=0)
        std = ref.std(axis=0)
        # A constant descriptor carries no information; scaling it by ~0 would
        # turn float noise into a huge distance, so it is neutralised instead.
        self.scale = np.where(std > 1e-8, std, 1.0)

        if self.whiten:
            # Standardise first so the covariance is a correlation matrix and
            # the shrinkage target (the identity) is meaningful regardless of
            # each descriptor's natural units.
            zs = (ref - self.mean) / self.scale
            cov = np.cov(zs, rowvar=False)
            d = cov.shape[0]
            # Ledoit-Wolf-style shrinkage toward a scaled identity. Whitening
            # divides by sqrt(eigenvalue), so the near-null directions of a
            # correlated descriptor set would otherwise be amplified into pure
            # noise -- shrinkage is what keeps that from dominating the metric.
            cov = (1.0 - self.shrinkage) * cov + \
                self.shrinkage * (np.trace(cov) / d) * np.eye(d)
            evals, evecs = np.linalg.eigh(cov)
            evals = np.maximum(evals, 1e-8)
            self.transform = evecs @ np.diag(evals ** -0.5) @ evecs.T
        else:
            self.transform = None

        z = self._project(ref)
        sub = z[:2000]
        d2 = np.maximum(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1), 0.0)
        median = np.median(d2[np.triu_indices(len(sub), k=1)]) if len(sub) > 1 else 1.0
        self.gamma = 1.0 / max(median, 1e-8)

    def _project(self, x) -> np.ndarray:
        z = (np.asarray(x) - self.mean) / self.scale
        return z if self.transform is None else z @ self.transform

    def gram(self, a, b) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("DescriptorRBFKernel.fit must run before gram")
        A, B = self._project(a), self._project(b)
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-self.gamma * d2)


class WhitenedDescriptorKernel(DescriptorRBFKernel):
    """Descriptor RBF on a *whitened* (Mahalanobis) distance.

    **Do not use this for the MOSES validity hack -- it is measurably worse
    than the plain descriptor kernel for that failure.** Kept because it is the
    right tool for the opposite situation, and because the measurement below is
    worth not having to repeat.

    It was added on the theory that per-axis standardisation under-weights a
    correlated cluster by presenting it as several individually-small shifts.
    That theory was wrong, and the reason is worth stating plainly: whitening
    divides by the spread *along each principal direction*, so it shrinks
    shifts along high-variance directions and amplifies shifts along
    low-variance ones.

    Decomposing the measured hack (base -> unpenalised RL mean shift) in the
    reference covariance eigenbasis, essentially all the loading sits on the
    top eigenvalues (3.26, 2.54, 1.80, 1.50, 1.28):

        ||shift||^2  standardised (Euclidean)  = 0.349
        ||shift||^2  whitened (Mahalanobis)    = 0.173     -> 0.50x

    So whitening HALVES the signal here. Chemically that is coherent: the hack
    travels along the dominant polarity/aromaticity axis of drug-like space --
    the direction real molecules vary most -- so it is genuinely *less
    surprising* in Mahalanobis terms even though FCD still penalises it. The
    lesson generalises: whiten when a hack hides in a rigid, low-variance
    direction; standardise when it rides the dominant one.

    The residual it was meant to fix was also *not* evidence of missing
    descriptors: every residual axis is already in the set, and the standardised
    kernel's own MMD^2 still read 3.7x base, so it could see the gap. Widening
    the descriptor set would not have helped either.
    """

    name = "descriptor_whitened"

    def __init__(self, descriptors: Sequence[str] = DESCRIPTORS,
                 shrinkage: float = 0.1):
        super().__init__(descriptors, whiten=True, shrinkage=shrinkage)


KERNELS = {
    "tanimoto": TanimotoKernel,
    "descriptor": DescriptorRBFKernel,
    "descriptor_whitened": WhitenedDescriptorKernel,
}


# ===========================================================================
# MMD
# ===========================================================================
class MMDPenalty:
    """Per-sample decomposition of MMD^2 under a Tanimoto/Morgan kernel.

    For generated batch ``x`` (size n) and reference ``y`` (size m),

        MMD^2 = E k(x,x') - 2 E k(x,y) + E k(y,y')

    Only the first two terms depend on the policy; ``E k(y,y')`` is a constant
    of the reference set. Splitting what is left across samples gives

        penalty_i = mean_{j != i} k(x_i, x_j)  -  2 * mean_l k(x_i, y_l)

    whose batch mean is exactly the policy-dependent part of MMD^2. The factor
    2 is not a tuning knob; it is the cross-term coefficient, and dropping it
    would optimise a different objective than the one named.

    Two properties earn this its place in the reward:

    * **The sibling term is anti-collapse.** ``mean_{j != i} k(x_i, x_j)`` rises
      when a sample resembles the rest of its own batch, so a policy converging
      on one safe molecule is penalised even though each molecule it emits is
      individually typical. A pure "look like the reference" term would happily
      reward exactly that collapse.
    * **It is unbiased at small n.** The ``j != i`` exclusion is what makes the
      estimator unbiased; the diagonal ``k(x_i, x_i) = 1`` would otherwise add a
      spurious ``1/n`` that shrinks as the batch grows, reintroducing precisely
      the sample-size artefact that rules FCD out.

    Higher is worse, and the range is [-2, 1]. It is *not* normalised to [0, 1]
    and does not need to be: GDPO subtracts the batch mean when forming
    advantages, so any constant offset cancels. Only the spread within a batch
    reaches the gradient.

    Invalid samples score 0.0 and are excluded from the sibling set -- they have
    no fingerprint, and letting them dilute the sibling mean would make the
    penalty on valid samples depend on how many neighbours happened to fail.
    """

    def __init__(self, reference_smiles: Sequence[str], *, n_reference: int = 4096,
                 seed: int = 0, kernel="descriptor", log=print):
        if isinstance(kernel, str):
            if kernel not in KERNELS:
                raise ValueError(f"unknown kernel {kernel!r}; have {sorted(KERNELS)}")
            kernel = KERNELS[kernel]()
        self.kernel = kernel

        ref = list(reference_smiles)
        if 0 < n_reference < len(ref):
            idx = np.random.default_rng(seed).choice(len(ref), n_reference,
                                                     replace=False)
            ref = [ref[int(i)] for i in sorted(idx)]
        feats = self.kernel.featurize(ref)
        self.ref_feats = [f for f in feats if f is not None]
        if not self.ref_feats:
            raise ValueError("no reference molecule produced features")
        self.kernel.fit(self.ref_feats)
        self.n_reference = len(self.ref_feats)
        self.last: Dict[str, float] = {}
        self.last_valid: List[int] = []
        self._k_yy: Optional[float] = None
        log(f"MMD reference: {self.n_reference} molecules, kernel={self.kernel.name}")

    def __call__(self, smiles: Sequence[Optional[str]]) -> np.ndarray:
        feats = self.kernel.featurize(smiles)
        valid = [i for i, f in enumerate(feats) if f is not None]
        out = np.zeros(len(smiles), dtype=np.float64)
        if not valid:
            self.last_valid = []
            self.last = {"mmd_penalty_mean": 0.0, "mmd_sim_ref": 0.0,
                         "mmd_sim_sibling": 0.0, "mmd_valid_frac": 0.0}
            return out

        self.last_valid = valid
        gen = [feats[i] for i in valid]
        sim_ref = self.kernel.gram(gen, self.ref_feats).mean(axis=1)
        n_sib = len(gen) - 1
        if n_sib > 0:
            k_xx = self.kernel.gram(gen, gen)
            # The gram diagonal is the self-comparison; subtracting it is the
            # j != i exclusion that keeps this estimator unbiased.
            sim_sib = (k_xx.sum(axis=1) - np.diag(k_xx)) / n_sib
        else:
            sim_sib = np.zeros(len(gen))

        out[valid] = sim_sib - 2.0 * sim_ref
        self.last = {
            "mmd_penalty_mean": float(out[valid].mean()),
            "mmd_sim_ref": float(sim_ref.mean()),
            "mmd_sim_sibling": float(sim_sib.mean()),
            "mmd_valid_frac": len(valid) / max(1, len(smiles)),
        }
        return out

    def reference_self_similarity(self) -> float:
        """``E k(y,y')`` -- the constant term, computed once and cached.

        Only needed to turn the per-sample penalties back into a true MMD^2 for
        reporting. The reward path never touches it, because a constant added
        to every sample cancels in the group-relative advantage.
        """
        if self._k_yy is not None:
            return self._k_yy
        m = len(self.ref_feats)
        if m < 2:
            self._k_yy = 0.0
            return self._k_yy
        # Blocked so a large reference does not need an m x m gram in memory.
        total = 0.0
        block = 512
        for start in range(0, m, block):
            chunk = self.ref_feats[start:start + block]
            g = self.kernel.gram(chunk, self.ref_feats)
            total += float(g.sum()) - float(np.trace(
                g[:, start:start + len(chunk)]))
        self._k_yy = total / (m * (m - 1))
        return self._k_yy

    def mmd2(self, smiles: Sequence[Optional[str]]) -> float:
        """Set-level MMD^2, for reporting and for the offline validation gate.

        Averages the per-sample decomposition over the samples that actually
        carry a value (the valid ones) and adds the constant reference term.
        """
        per_sample = self(smiles)
        if not self.last_valid:
            return float("nan")
        return float(per_sample[self.last_valid].mean() + self.reference_self_similarity())
