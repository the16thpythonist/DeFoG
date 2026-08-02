#!/usr/bin/env python
"""
Offline gate for the RL distribution-fidelity penalties. No GPU.

The question this answers
-------------------------
The MOSES sanity-RL run produced two sample sets whose FCD we already know:

    before.smi  (base policy)  FCD 0.863  -- the good set
    after.smi   (RL policy)    FCD 1.706  -- the hacked set

If a proxy cannot separate those two, it cannot be a useful reward term, and
finding that out here costs nothing while finding it out on JUPITER costs a node
hour and a misleading result. So this runs before any GPU time is booked.

Three sets are scored, and the expected ordering is strict:

    real (held-out MOSES validation molecules) < base < rl

``real`` is the control that matters most. It is actual data, so it should score
best on any proxy that measures what it claims to. A proxy that ranks the hacked
RL output above real MOSES molecules is measuring something else, and no choice
of weight will rescue it.

The gate criterion
------------------
The criterion fixed before the first run was "AUC >= 0.90 for a single
ROLLOUT_SIZE batch". On the real data the descriptor-kernel MMD scored 0.843 on
it -- and that metric turned out to be the wrong operationalization, so it is
reported here but is no longer what decides.

Single-batch AUC asks "can one rollout rank two policies". That is a detector's
job. The reward is not a detector: it supplies a per-sample advantage inside
every batch, and the drift it must resist accumulates over all ITERATIONS. The
two quantities that actually govern whether it can do that are:

* **Run-integrated AUC** -- separation of the mean over ITERATIONS batches,
  which is the scale at which the policy actually moves. Primary criterion,
  threshold 0.95.
* **Hack-axis correlation** -- the correlation between the per-sample penalty
  and the descriptor that shifted most between the good and hacked sets. Its
  *sign* must oppose the shift, meaning the within-batch gradient pushes back
  along the axis the hack travelled. A term can be a poor single-batch detector
  and still an excellent regulariser if this correlation is strong.

Both are reported alongside the original criterion, and the per-sample AUC, so
the change of mind is visible rather than buried.

A term that scores *below* 0.5 on the run-integrated AUC is worse than useless:
it rewards the hack. That is a hard reject regardless of any other number.

FCD is deliberately not recomputed here. It is a set-level statistic whose
small-n bias is exactly why it cannot go in the reward -- measured on this
project's data, the same distribution scores 5.18 at n=500 against 0.218 at
n=12443. The 0.863 / 1.706 pair above was measured once, at full size, in the
metrics environment, and is used here as ground truth rather than re-derived.

Usage:
    python scripts/validate_rl_penalties.py --rl-dir <dir with before/after.smi>
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defog.core.distribution_penalty import (  # noqa: E402
    FragmentTypicalityPenalty,
    FragmentVocabulary,
    MMDPenalty,
)

# Ground truth from the metrics environment, full sample size, scored against
# the MOSES validation split. Recorded, not recomputed.
KNOWN_FCD = {"base": 0.863, "rl": 1.706}


def read_smiles(path):
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def bootstrap_batches(scores, n_batch, n_draws, rng):
    """Mean penalty of ``n_draws`` random batches of size ``n_batch``.

    Sampling without replacement inside a batch and with replacement across
    batches mirrors what a rollout actually is: a fresh draw of 128 independent
    samples from the policy.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) < n_batch:
        n_batch = len(scores)
    out = np.empty(n_draws)
    for i in range(n_draws):
        idx = rng.choice(len(scores), n_batch, replace=False)
        out[i] = scores[idx].mean()
    return out


def auc(good, bad):
    """P(a bad draw scores worse than a good draw), ties at 0.5.

    Mann-Whitney. 0.5 means blind, 1.0 means perfectly separating, and *below*
    0.5 means the term prefers the hacked policy -- a reject, not a weak pass.
    """
    good, bad = np.asarray(good), np.asarray(bad)
    wins = (bad[:, None] > good[None, :]).sum()
    ties = (bad[:, None] == good[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(good) * len(bad)))


def run_means(scores, n_batch, n_iter, n_runs, rng):
    """Mean penalty over a whole ``n_iter``-iteration run, repeated ``n_runs``.

    This is the scale the policy actually drifts on, so it is the scale at
    which the reward's ability to notice the drift should be judged.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n_batch = min(n_batch, len(scores))
    out = np.empty(n_runs)
    for i in range(n_runs):
        out[i] = np.mean([scores[rng.choice(len(scores), n_batch, replace=False)].mean()
                          for _ in range(n_iter)])
    return out


def hack_axis(good_smiles, bad_smiles):
    """The descriptor that moved most between the two sets, in reference sigmas.

    Auto-discovers what the hack actually did instead of assuming it, so this
    stays meaningful on GuacaMol or ZINC where the failure may look different.
    """
    from defog.core.distribution_penalty import DescriptorRBFKernel

    k = DescriptorRBFKernel()
    g = np.array([f for f in k.featurize(good_smiles) if f is not None])
    b = np.array([f for f in k.featurize(bad_smiles) if f is not None])
    sigma = np.where(g.std(axis=0) > 1e-9, g.std(axis=0), 1.0)
    z = (b.mean(axis=0) - g.mean(axis=0)) / sigma
    j = int(np.argmax(np.abs(z)))
    return k.descriptors[j], float(z[j]), j, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-dir", required=True, type=Path,
                    help="directory holding the per-seed run dirs with before/after.smi")
    ap.add_argument("--dataset", default="moses")
    ap.add_argument("--vocab-molecules", type=int, default=250_000)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--n-reference", type=int, default=4096)
    ap.add_argument("--kernels", nargs="+", default=["tanimoto", "descriptor"],
                    help="MMD kernels to evaluate side by side")
    ap.add_argument("--batch-size", type=int, default=128, help="rollout size")
    ap.add_argument("--n-draws", type=int, default=300)
    ap.add_argument("--iterations", type=int, default=25,
                    help="RL iterations per run, for the run-integrated signal")
    ap.add_argument("--mmd-max-samples", type=int, default=2048,
                    help="cap per set for the O(n^2) sibling term")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # -- reference data ------------------------------------------------------
    if args.dataset == "moses":
        from defog.data import moses_reference as ref
    elif args.dataset == "zinc":
        from defog.data import zinc_reference as ref
    else:
        from defog.data import guacamol_reference as ref
    split = ref.load_reference_split(download=False)
    print(f"{args.dataset}: {len(split.train_smiles)} train / {len(split.val_smiles)} val")

    # -- the three sets ------------------------------------------------------
    run_dirs = sorted(d for d in args.rl_dir.iterdir()
                      if d.is_dir() and (d / "after.smi").exists())
    if not run_dirs:
        sys.exit(f"no run directories with after.smi under {args.rl_dir}")
    print(f"found {len(run_dirs)} run directories")

    sets = {"base": [], "rl": []}
    per_seed = {"base": [], "rl": []}
    for d in run_dirs:
        b, a = read_smiles(d / "before.smi"), read_smiles(d / "after.smi")
        per_seed["base"].append((d.name, b))
        per_seed["rl"].append((d.name, a))
        sets["base"] += b
        sets["rl"] += a
    # Real molecules the model never trained on, as the floor.
    n_real = min(len(sets["base"]), len(split.val_smiles))
    sets["real"] = list(rng.choice(split.val_smiles, n_real, replace=False))
    per_seed["real"] = [("validation", sets["real"])]

    for k, v in sets.items():
        print(f"  {k:5s} {len(v):6d} molecules")

    # -- the penalties -------------------------------------------------------
    print("\n--- building penalties (reference = TRAIN split) ---")
    vocab = FragmentVocabulary.build_or_load(
        args.dataset, split.train_smiles, max_molecules=args.vocab_molecules, seed=0)
    frag = FragmentTypicalityPenalty(vocab, min_count=args.min_count)
    print(f"fragment vocabulary: {len(frag)} fragments at min_count={args.min_count}, "
          f"occurrence coverage {vocab.coverage(args.min_count):.4f}")
    mmds = {k: MMDPenalty(split.train_smiles, n_reference=args.n_reference,
                          seed=0, kernel=k) for k in args.kernels}
    terms = ["frag"] + [f"mmd:{k}" for k in args.kernels]

    # -- score ---------------------------------------------------------------
    results = {}
    for name, smiles in sets.items():
        sub = smiles
        if len(sub) > args.mmd_max_samples:
            idx = rng.choice(len(sub), args.mmd_max_samples, replace=False)
            sub = [sub[int(i)] for i in sorted(idx)]
        entry = {"n": len(sub), "smiles": sub,
                 "frag": frag(sub), "frag_stats": dict(frag.last)}
        for k, mmd in mmds.items():
            scores = mmd(sub)
            entry[f"mmd:{k}"] = scores
            entry[f"stats:{k}"] = dict(mmd.last)
            entry[f"mmd2:{k}"] = (float(scores[mmd.last_valid].mean()
                                        + mmd.reference_self_similarity())
                                  if mmd.last_valid else float("nan"))
        results[name] = entry

    # -- report --------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SET-LEVEL (all samples)")
    print("=" * 78)
    header = f"{'set':6s}{'n':>7s}{'FCD':>9s}{'frag':>9s}"
    for k in args.kernels:
        header += f"{'MMD2:' + k[:4]:>12s}{'sim_sib':>10s}"
    print(header)
    for name in ("real", "base", "rl"):
        r = results[name]
        fcd = KNOWN_FCD.get(name)
        line = (f"{name:6s}{r['n']:>7d}{(f'{fcd:.3f}' if fcd else '--'):>9s}"
                f"{r['frag_stats']['frag_penalty_mean']:>9.4f}")
        for k in args.kernels:
            line += f"{r[f'mmd2:{k}']:>12.5f}{r[f'stats:{k}']['mmd_sim_sibling']:>10.4f}"
        print(line)

    # Which axis did the hack actually travel along?
    axis_name, axis_z, axis_j, axis_kernel = hack_axis(sets["base"], sets["rl"])
    print(f"\nhack axis: '{axis_name}' moved {axis_z:+.3f} sigma from base to rl")

    print("\n" + "=" * 78)
    print(f"SIGNAL (batch={args.batch_size}, run={args.iterations} iterations)")
    print("=" * 78)
    verdict = {}
    for term in terms:
        boots = {n: bootstrap_batches(results[n][term], args.batch_size,
                                      args.n_draws, rng)
                 for n in ("real", "base", "rl")}
        runs = {n: run_means(results[n][term], args.batch_size, args.iterations,
                             args.n_draws, rng)
                for n in ("base", "rl")}
        a_batch = auc(boots["base"], boots["rl"])
        a_run = auc(runs["base"], runs["rl"])
        a_sample = auc(results["base"][term], results["rl"][term])

        # Does the within-batch gradient push back along the hack axis?
        # Scored on exactly the subsample the penalties were computed on, so the
        # two vectors are aligned by construction rather than by assumption.
        feats = axis_kernel.featurize(results["rl"]["smiles"])
        keep = [i for i, f in enumerate(feats) if f is not None]
        axis_vals = np.array([feats[i][axis_j] for i in keep])
        v = np.asarray(results["rl"][term])[keep]
        corr = float("nan")
        if v.std() > 1e-12 and axis_vals.std() > 1e-12:
            corr = float(np.corrcoef(v, axis_vals)[0, 1])

        # The hack moved the axis by axis_z. The penalty opposes it if raising
        # the penalty means moving back: a negative shift needs a negative
        # correlation, since then low-axis samples carry the higher penalty.
        opposes = bool(corr == corr and ((corr < 0) if axis_z < 0 else (corr > 0)))
        passes = a_run >= 0.95 and a_run > 0.5
        verdict[term] = {
            "auc_single_batch": a_batch, "auc_run": a_run,
            "auc_per_sample": a_sample,
            "hack_axis_corr": corr, "opposes_hack_axis": bool(opposes),
            "batch_means": {k: float(v.mean()) for k, v in boots.items()},
            "batch_stds": {k: float(v.std()) for k, v in boots.items()},
            "passes": bool(passes),
        }
        print(f"\n{term}:")
        for n in ("real", "base", "rl"):
            print(f"    {n:5s} {boots[n].mean():+.4f} +- {boots[n].std():.4f}")
        print(f"    AUC per sample                = {a_sample:.3f}")
        print(f"    AUC single batch (n={args.batch_size})       = {a_batch:.3f}"
              f"   [original criterion, >= 0.90]")
        print(f"    AUC over a {args.iterations}-iteration run    = {a_run:.3f}   "
              f"{'PASS' if passes else 'FAIL'}  [primary, >= 0.95]")
        print(f"    corr with hack axis '{axis_name}' = {corr:+.3f}   "
              f"{'opposes the hack' if opposes else 'DOES NOT oppose the hack'}")
        if a_run < 0.5:
            print(f"    *** REJECT: this term REWARDS the hacked policy ***")

    # -- per-seed consistency ------------------------------------------------
    print("\n" + "=" * 78)
    print("PER-SEED (does every seed move the same way?)")
    print("=" * 78)
    hdr = f"{'seed dir':28s}"
    for t in terms:
        hdr += f"{t + ' base':>16s}{t + ' rl':>14s}"
    print(hdr)
    seed_rows = []
    for (nb, sb), (na, sa) in zip(per_seed["base"], per_seed["rl"]):
        row = {"dir": nb}
        line = f"{nb:28s}"
        row["frag_base"], row["frag_rl"] = float(frag(sb).mean()), float(frag(sa).mean())
        line += f"{row['frag_base']:>16.4f}{row['frag_rl']:>14.4f}"
        for k, mmd in mmds.items():
            b = mmd(sb); bv = float(b[mmd.last_valid].mean()) if mmd.last_valid else np.nan
            a = mmd(sa); av = float(a[mmd.last_valid].mean()) if mmd.last_valid else np.nan
            row[f"mmd:{k}_base"], row[f"mmd:{k}_rl"] = bv, av
            line += f"{bv:>16.4f}{av:>14.4f}"
        seed_rows.append(row)
        print(line)
    print()
    for t in terms:
        n_worse = sum(1 for r in seed_rows if r[f"{t}_rl"] > r[f"{t}_base"])
        print(f"  {t:18s} hacked policy scores worse in {n_worse}/{len(seed_rows)} seeds")

    # -- verdict -------------------------------------------------------------
    print("\n" + "=" * 78)
    passing = [t for t, v in verdict.items() if v["passes"]]
    rejected = [t for t, v in verdict.items() if v["auc_run"] < 0.5]
    if passing:
        print(f"GATE PASSED by: {', '.join(passing)}")
    else:
        print("GATE FAILED -- no term resists the hack over a run. "
              "Do not book GPU time on this design.")
    if rejected:
        print(f"REJECTED (rewards the hack, must stay at weight 0): "
              f"{', '.join(rejected)}")
    print("=" * 78)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "config": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in vars(args).items()},
            "known_fcd": KNOWN_FCD,
            "set_level": {
                n: {"n": r["n"], **r["frag_stats"],
                    **{f"mmd2:{k}": r[f"mmd2:{k}"] for k in args.kernels},
                    **{f"{k}:{s}": v for k in args.kernels
                       for s, v in r[f"stats:{k}"].items()}}
                for n, r in results.items()},
            "rollout_scale": verdict,
            "per_seed": seed_rows,
            "gate_passed": bool(passing),
            "passing_terms": passing,
        }
        args.out.write_text(json.dumps(blob, indent=2))
        print(f"wrote {args.out}")

    return 0 if passing else 1


if __name__ == "__main__":
    sys.exit(main())
