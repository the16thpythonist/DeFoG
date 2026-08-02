"""
E1 distribution-quality metrics: FCD, NSPDK, scaffold similarity, GuacaMol KL.

Runs under ``.venv_metrics``, NOT the training environment, and talks to the
training side only through SMILES files. That separation is deliberate:

* ``guacamol`` declares ``rdkit-pypi``, a different distribution from the
  ``rdkit`` the training env uses, and installing it there risks the working
  install. Here it goes in with ``--no-deps`` against official rdkit.
* JUPITER is aarch64 and may not have wheels for these at all. Evaluation never
  needs to run there -- it consumes SMILES, not checkpoints.
* Scoring becomes re-runnable without re-sampling. Sweep once, keep the SMILES,
  re-score as metrics are added.

Setup:
    uv venv --python 3.10 .venv_metrics
    uv pip install --python .venv_metrics/bin/python torch rdkit numpy scipy joblib tqdm
    uv pip install --python .venv_metrics/bin/python FCD eden-kernel
    uv pip install --python .venv_metrics/bin/python --no-deps guacamol

Usage:
    .venv_metrics/bin/python scripts/e1_metrics.py \\
        --generated gen.smi --reference test.smi --dataset zinc --out metrics.json

    # Sanity mode: score a reference set against itself. Every metric has a
    # known answer, so this catches a broken harness without needing a model.
    .venv_metrics/bin/python scripts/e1_metrics.py --self-check --reference test.smi

DIRECTIONS DIFFER BETWEEN DATASETS AND THIS IS THE CLASSIC WAY TO GET E1 WRONG:
``fcd_raw`` is a distance (lower better, ZINC/MOSES); ``fcd_guacamol`` is
``exp(-0.2*FCD)`` in [0,1] (HIGHER better, GuacaMol). Both are emitted with an
explicit ``direction`` field so a table cannot silently mix them.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Dict, List, Optional

import numpy as np

# guacamol 0.5.5 does `from scipy import histogram`, an alias SciPy removed.
# It was always exactly numpy.histogram (used once, as
# `histogram(X, bins=10, density=True)`), so restoring it is faithful rather
# than a guess. Done here instead of editing site-packages so it survives a
# reinstall and stays visible.
import scipy

if not hasattr(scipy, "histogram"):
    scipy.histogram = np.histogram


def _install_molsets_shims() -> None:
    """Three one-line restorations so molsets 0.3.1 (2020) runs on a modern stack.

    Each replaces an alias its upstream removed, with the documented equivalent --
    none of them changes what a metric computes. Applied here rather than in
    site-packages so they survive a reinstall and stay visible.

    Worth recording why molsets is usable at all: its setup.py pins
    ``pomegranate==0.12.0``, which does not build on a current Python, and that
    made it look unavailable. But ``pomegranate`` is imported by exactly one
    file -- ``moses/baselines/hmm.py``, an HMM baseline model -- and
    ``moses/__init__.py`` imports only dataset/metrics/utils. Installing with
    ``--no-deps`` therefore gives the whole official metric suite.
    """
    import sys
    import types

    # 1. rdkit.six: a py2/3 compat shim RDKit removed. sascorer uses one symbol,
    #    and six.iteritems(d) is iter(d.items()) on Python 3.
    if "rdkit.six" not in sys.modules:
        _six = types.ModuleType("rdkit.six")
        _six.iteritems = lambda d: iter(d.items())
        sys.modules["rdkit.six"] = _six

    # 2. DataFrame.append: removed in pandas 2.0, and always concat underneath.
    #    Shimmed rather than pinning pandas<2, which would drag numpy back to
    #    1.x and break the torch/rdkit/FCD stack this env shares.
    import pandas as pd

    if not hasattr(pd.DataFrame, "append"):
        pd.DataFrame.append = lambda self, other, **kw: pd.concat([self, other], **kw)

    # 3. Integer overflow in cos_similarity, which backs FragMetric and
    #    ScafMetric. It builds count vectors with np.array(list_of_ints), so
    #    numpy infers int64; scipy's cosine then computes uu*vv where uu = sum
    #    r^2 and vv = sum g^2. Against a full MOSES reference (176,074 mols) the
    #    fragment counts are large enough that uu*vv exceeds int64, wraps
    #    negative, and math.sqrt raises "math domain error".
    #
    #    It errored rather than wrapping quietly, which was lucky -- a silent
    #    wrap would have produced a plausible-looking wrong number. Casting to
    #    float64 removes the overflow; cosine similarity is mathematically
    #    identical either way, so this changes nothing except the intermediate
    #    dtype. Verified to reproduce the unpatched value wherever the unpatched
    #    code succeeds.
    import moses.metrics.metrics as _mm

    if not getattr(_mm, "_defog_cos_float64", False):
        def _cos_similarity_float64(ref_counts, gen_counts):
            if len(ref_counts) == 0 or len(gen_counts) == 0:
                return np.nan
            keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
            ref_vec = np.array([ref_counts.get(k, 0) for k in keys], dtype=np.float64)
            gen_vec = np.array([gen_counts.get(k, 0) for k in keys], dtype=np.float64)
            return 1 - _mm.cos_distance(ref_vec, gen_vec)

        _mm.cos_similarity = _cos_similarity_float64
        _mm._defog_cos_float64 = True


# ===========================================================================
# IO
# ===========================================================================
def read_smiles(path: str, limit: Optional[int] = None, seed: int = 42) -> List[str]:
    """One SMILES per line, or a JSON list (as the training runs emit)."""
    if path.endswith(".json"):
        with open(path) as fh:
            data = json.load(fh)
        smiles = [s for s in data if isinstance(s, str)]
    else:
        with open(path) as fh:
            smiles = [line.strip().split()[0] for line in fh if line.strip()]
    if limit and len(smiles) > limit:
        smiles = random.Random(seed).sample(smiles, limit)
    return smiles


def canonical(smiles: List[str], keep_stereo: bool = True) -> List[str]:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out.append(Chem.MolToSmiles(m, isomericSmiles=keep_stereo))
    return out


# ===========================================================================
# FCD
# ===========================================================================
def compute_fcd(generated: List[str], reference: List[str], device: str = "cpu") -> Dict:
    """Raw Frechet ChemNet Distance, plus GuacaMol's normalised form.

    One ChemNet forward pass serves both: ``fcd_raw`` is what ZINC and MOSES
    report (lower better) and ``fcd_guacamol`` is ``exp(-0.2 * FCD)`` (higher
    better), the transform read directly from
    ``guacamol/frechet_benchmark.py``.

    guacamol's own FrechetBenchmark is NOT used, because it loads ChemNet as
    ``ChemNet_v0.13_pretrained.h5`` via pkgutil while FCD 1.2.2 ships the torch
    ``.pt``. The underlying network is the same v0.13 ChemNet, so only the
    loading path differs -- the metric is not reimplemented.
    """
    from fcd import get_fcd, load_ref_model

    model = load_ref_model()
    raw = float(get_fcd(generated, reference, model=model, device=device))
    return {
        "fcd_raw": raw,
        "fcd_raw_direction": "lower_is_better",
        "fcd_guacamol": float(np.exp(-0.2 * raw)),
        "fcd_guacamol_direction": "higher_is_better",
        "n_generated": len(generated),
        "n_reference": len(reference),
    }


# ===========================================================================
# GuacaMol KL
# ===========================================================================
class _FixedGenerator:
    """Adapts a SMILES list to guacamol's DistributionMatchingGenerator.

    Using the official benchmark object rather than reimplementing its ten KL
    terms is the point (protocol section 1); this class exists only so a list of
    already-generated molecules can be handed to an interface designed to call a
    live model.
    """

    def __init__(self, smiles: List[str]):
        self.smiles = list(smiles)

    def generate(self, number_samples: int) -> List[str]:
        if number_samples <= len(self.smiles):
            return self.smiles[:number_samples]
        # Never pad: a short list must show up as a penalised score, which is
        # what guacamol does on its own, rather than being silently topped up.
        return self.smiles


def compute_guacamol_kl(generated: List[str], reference: List[str],
                        number_samples: Optional[int] = None) -> Dict:
    """GuacaMol's KL-divergence benchmark, via the official implementation.

    Nine physchem descriptors (BertzCT, MolLogP, MolWt, TPSA continuous;
    NumHAcceptors, NumHDonors, NumRotatableBonds, NumAliphaticRings,
    NumAromaticRings discrete) plus internal pairwise similarity, each mapped
    through ``exp(-KL)`` and averaged. Higher is better.

    This is NOT the four-property KDE proxy that
    ``experiments/training__*_uncond.py`` computes and labels a "GuacaMol
    normalized score" -- that number shares the name but not the definition.
    """
    from guacamol.distribution_learning_benchmark import KLDivBenchmark

    n = number_samples or min(len(generated), len(reference))
    benchmark = KLDivBenchmark(number_samples=n, training_set=reference)
    result = benchmark.assess_model(_FixedGenerator(generated))
    return {
        "kl_score": float(result.score),
        "kl_score_direction": "higher_is_better",
        "kl_per_descriptor": {k: float(v) for k, v in
                              result.metadata.get("kl_divs", {}).items()},
        "kl_number_samples": n,
    }


# ===========================================================================
# NSPDK
# ===========================================================================
def _mol_to_nx(smiles: str):
    """Atom-labelled, bond-labelled graph, as the GDSS lineage builds them."""
    import networkx as nx
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    g = nx.Graph()
    for atom in mol.GetAtoms():
        g.add_node(atom.GetIdx(), label=atom.GetSymbol())
    for bond in mol.GetBonds():
        g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                   label=str(int(bond.GetBondTypeAsDouble())))
    if g.number_of_nodes() == 0:
        return None
    return g


def compute_nspdk(generated: List[str], reference: List[str],
                  subsample: int = 1000, seed: int = 42) -> Dict:
    """NSPDK MMD between generated and reference molecular graphs.

    Follows GDSS's ``evaluation/stats.py``: EDeN vectorisation at complexity 4
    with discrete labels, a linear kernel, and the standard MMD

        E[k(X,X)] + E[k(Y,Y)] - 2 E[k(X,Y)]

    Lower is better. This is the ZINC250k metric with no official package, so
    the reference implementation is GDSS's and this follows it directly.

    The pairwise kernel is O(n*m), so both sides are subsampled. The cap is
    reported in the output rather than applied silently -- a truncated
    comparison that looks like a full one is exactly how these numbers stop
    being comparable.
    """
    from eden.graph import vectorize
    from sklearn.metrics.pairwise import pairwise_kernels

    rng = random.Random(seed)
    gen = generated if len(generated) <= subsample else rng.sample(generated, subsample)
    ref = reference if len(reference) <= subsample else rng.sample(reference, subsample)

    g_graphs = [g for g in (_mol_to_nx(s) for s in gen) if g is not None]
    r_graphs = [g for g in (_mol_to_nx(s) for s in ref) if g is not None]
    if not g_graphs or not r_graphs:
        return {"nspdk_mmd": float("nan"), "nspdk_note": "no valid graphs"}

    X = vectorize(g_graphs, complexity=4, discrete=True)
    Y = vectorize(r_graphs, complexity=4, discrete=True)

    kxx = pairwise_kernels(X, X, metric="linear").mean()
    kyy = pairwise_kernels(Y, Y, metric="linear").mean()
    kxy = pairwise_kernels(X, Y, metric="linear").mean()
    return {
        "nspdk_mmd": float(kxx + kyy - 2 * kxy),
        "nspdk_mmd_direction": "lower_is_better",
        "nspdk_n_generated": len(g_graphs),
        "nspdk_n_reference": len(r_graphs),
        "nspdk_subsample_cap": subsample,
        "nspdk_truncated": len(generated) > subsample or len(reference) > subsample,
    }


# ===========================================================================
# Scaffold similarity
# ===========================================================================
def compute_scaffold_similarity(generated: List[str], reference: List[str]) -> Dict:
    """Bemis-Murcko scaffold similarity, via MOSES's own ``ScafMetric``.

    Previously a local reimplementation, on the belief that ``molsets`` could not
    be installed. That was wrong -- see :func:`_install_molsets_shims` -- so this
    now calls the official implementation, which is what the protocol asks for.

    Higher is better.
    """
    _install_molsets_shims()
    from moses.metrics.metrics import ScafMetric

    value = float(ScafMetric()(gen=list(generated), ref=list(reference)))
    return {
        "scaffold_similarity": value,
        "scaffold_similarity_direction": "higher_is_better",
        "scaffold_implementation": "moses.metrics.metrics.ScafMetric (official)",
    }


def compute_moses_suite(generated: List[str], reference: List[str],
                        reference_scaffolds: Optional[List[str]] = None,
                        device: str = "cpu") -> Dict:
    """MOSES's own metric suite: Filters, SNN, Frag, Scaf, and FCD.

    FCD is reported against BOTH held-out sets when ``reference_scaffolds`` is
    given, because the protocol requires both and they are different numbers.
    That is precisely the distinction ``src/datasets/moses_dataset.py`` destroys
    by renaming ``test`` to "val" and ``test_scaffolds`` to "test".
    """
    _install_molsets_shims()
    from moses.metrics.metrics import (FCDMetric, FragMetric, ScafMetric, SNNMetric,
                                       fraction_passes_filters, fraction_unique,
                                       fraction_valid)

    gen, ref = list(generated), list(reference)
    out = {
        "moses_validity": float(fraction_valid(gen)),
        "moses_unique": float(fraction_unique(gen)),
        "moses_filters": float(fraction_passes_filters(gen)),
        "moses_snn_test": float(SNNMetric(device=device)(gen=gen, ref=ref)),
        "moses_frag_test": float(FragMetric(device=device)(gen=gen, ref=ref)),
        "moses_scaf_test": float(ScafMetric(device=device)(gen=gen, ref=ref)),
        "moses_fcd_test": float(FCDMetric(device=device)(gen=gen, ref=ref)),
        "moses_fcd_direction": "lower_is_better",
        "moses_implementation": "molsets 0.3.1 (official)",
        "moses_n_reference": len(ref),
    }

    # MOSES computes EVERY one of these against both held-out sets and reports
    # them as SNN/Test + SNN/TestSF, Scaf/Test + Scaf/TestSF, and so on. They
    # are far apart -- Scaf/Test runs ~0.87 on a decent model while Scaf/TestSF
    # is ~0.1, because test_scaffolds is built from scaffolds the training set
    # never contained. Emitting only one and labelling it "Scaf" is therefore a
    # 6x error waiting to happen, so both are always produced.
    #
    # WHICH ONE DEFOG REPORTS: the TestSF variants. Scoring their own published
    # MOSES samples against their paper row (Table 9, 500 steps):
    #
    #     metric   published   ours/Test   ours/TestSF
    #     SNN         0.55       0.590        0.555
    #     Scaf        0.144      0.868        0.107
    #     FCD         1.95       0.897        1.562
    #
    # SNN pins it to within 0.005 while Scaf/Test is off by 6x. Scaf and FCD do
    # not match exactly because the archive samples are not the paper's run --
    # the archive's own report gives validity 0.9166 against the paper's 0.928 --
    # but the SCALE is unambiguous. Compare like with like: an E1 MOSES row must
    # use the TestSF columns.
    if reference_scaffolds:
        sf = list(reference_scaffolds)
        out.update({
            "moses_snn_test_scaffolds": float(SNNMetric(device=device)(gen=gen, ref=sf)),
            "moses_frag_test_scaffolds": float(FragMetric(device=device)(gen=gen, ref=sf)),
            "moses_scaf_test_scaffolds": float(ScafMetric(device=device)(gen=gen, ref=sf)),
            "moses_fcd_test_scaffolds": float(FCDMetric(device=device)(gen=gen, ref=sf)),
            "moses_n_reference_scaffolds": len(sf),
        })

    # Reference size is load-bearing, not a convenience knob. Measured on
    # DeFoG's own published samples, truncating the reference from 176,074 to
    # 10,000 moved SNN 0.590 -> 0.454, Frag 0.997 -> 0.967, Scaf 0.868 -> 0.744
    # and FCD 0.897 -> 4.155. Always score against the full split.
    if len(ref) < 100000:
        out["moses_reference_truncated_warning"] = (
            f"reference has only {len(ref)} molecules; MOSES metrics are strongly "
            f"reference-size dependent and published numbers use the full split")
    return out


# ===========================================================================
# CLI
# ===========================================================================
DATASET_METRICS = {
    # Per docs/unconditional-protocol.md section 3.
    "zinc": ["fcd", "nspdk", "scaffold"],
    "guacamol": ["kl", "fcd"],
    # MOSES gets its own suite (Filters/SNN/Frag/Scaf + FCD against both
    # held-out sets) rather than the generic metrics.
    "moses": ["moses"],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generated", help="SMILES file or .json list of generated molecules")
    ap.add_argument("--reference", required=True, help="Reference split SMILES")
    ap.add_argument("--reference-scaffolds", default=None,
                    help="MOSES test_scaffolds SMILES; enables the second FCD the "
                         "protocol requires alongside FCD-vs-test")
    ap.add_argument("--dataset", choices=sorted(DATASET_METRICS), default=None)
    ap.add_argument("--metrics", default=None,
                    help="Comma-separated subset of fcd,kl,nspdk,scaffold")
    ap.add_argument("--limit", type=int, default=None, help="Subsample both sides")
    ap.add_argument("--nspdk-subsample", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-check", action="store_true",
                    help="Score the reference against itself. Every metric then has a "
                         "known answer (FCD~0, NSPDK~0, scaffold~1, KL~1), which "
                         "validates the harness without needing a model.")
    args = ap.parse_args()

    if not args.self_check and not args.generated:
        ap.error("--generated is required unless --self-check is given")

    reference = canonical(read_smiles(args.reference, args.limit))
    if args.self_check:
        # Split in half so the two sides are disjoint samples of one
        # distribution: comparing an identical list to itself would make some
        # metrics trivially exact and hide real bugs.
        mid = len(reference) // 2
        generated, reference = reference[:mid], reference[mid:]
        print(f"self-check: {len(generated)} vs {len(reference)} "
              f"disjoint halves of {args.reference}")
    else:
        generated = canonical(read_smiles(args.generated, args.limit))

    ref_scaffolds = (canonical(read_smiles(args.reference_scaffolds, args.limit))
                     if args.reference_scaffolds else None)

    which = (args.metrics.split(",") if args.metrics
             else DATASET_METRICS.get(args.dataset, ["fcd", "kl", "nspdk", "scaffold"]))

    results: Dict = {
        "reference_file": args.reference,
        "generated_file": args.generated if not args.self_check else "<self-check>",
        "dataset": args.dataset,
        "n_generated": len(generated),
        "n_reference": len(reference),
        "metrics_requested": which,
    }

    for name in which:
        print(f"computing {name} ...", flush=True)
        try:
            if name == "fcd":
                results.update(compute_fcd(generated, reference, device=args.device))
            elif name == "kl":
                results.update(compute_guacamol_kl(generated, reference))
            elif name == "nspdk":
                results.update(compute_nspdk(generated, reference,
                                             subsample=args.nspdk_subsample))
            elif name == "scaffold":
                results.update(compute_scaffold_similarity(generated, reference))
            elif name == "moses":
                results.update(compute_moses_suite(
                    generated, reference, reference_scaffolds=ref_scaffolds,
                    device=args.device))
            else:
                print(f"  unknown metric {name!r}, skipped")
        except Exception as exc:  # surface, don't swallow: a missing metric must
            # be visible in the output rather than silently absent from a table
            results[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {type(exc).__name__}: {exc}")

    print("\n" + json.dumps({k: v for k, v in results.items()
                             if not isinstance(v, dict)}, indent=2))
    if "kl_per_descriptor" in results:
        print("kl_per_descriptor:", json.dumps(results["kl_per_descriptor"], indent=2))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.out}")

    if args.self_check:
        return _report_self_check(results)
    return 0


def _report_self_check(r: Dict) -> int:
    """Two disjoint halves of one distribution should score near-perfectly.

    The FCD bound scales with n rather than being a constant, because the
    Frechet estimator is strongly biased at small samples: it fits a 512-dim
    ChemNet covariance from n molecules, and the bias falls roughly as 1/n.
    Measured on disjoint halves of the ZINC test split -- i.e. a *perfect*
    generator by construction:

        n per side      500     2000     6000    12443
        FCD            5.183    1.398    0.455    0.218

    A fixed "FCD < 0.5" bound would therefore fail a flawless harness at any
    n below ~6000. The 4000/n form tracks the observed curve with margin.

    This is protocol trap 4 quantified: at n=500 a perfect model scores 5.2, so
    an FCD compared against a published number computed at a different sample
    count is meaningless. Match the published n exactly.
    """
    n = min(r.get("n_generated", 0), r.get("n_reference", 0)) or 1
    fcd_bound = 4000.0 / n
    checks = [
        ("fcd_raw", r.get("fcd_raw"), lambda v: v < fcd_bound, f"< {fcd_bound:.3f}"),
        ("fcd_guacamol", r.get("fcd_guacamol"),
         lambda v: v > float(np.exp(-0.2 * fcd_bound)),
         f"> {float(np.exp(-0.2 * fcd_bound)):.3f}"),
        ("kl_score", r.get("kl_score"), lambda v: v > 0.95, "> 0.95"),
        ("nspdk_mmd", r.get("nspdk_mmd"), lambda v: abs(v) < 0.05, "|v| < 0.05"),
        ("scaffold_similarity", r.get("scaffold_similarity"),
         lambda v: v > 0.5, "> 0.5"),
    ]
    print("\n" + "=" * 62)
    print(f"{'metric':<24}{'value':>14}{'expected':>14}{'':>10}")
    print("=" * 62)
    failed = 0
    for name, value, ok, desc in checks:
        if value is None or (isinstance(value, float) and value != value):
            continue
        good = ok(value)
        failed += not good
        print(f"{name:<24}{value:>14.5f}{desc:>14}{'  OK' if good else '  <-- FAIL':>10}")
    print("=" * 62)
    print("SELF-CHECK PASSED" if not failed else f"SELF-CHECK FAILED ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
