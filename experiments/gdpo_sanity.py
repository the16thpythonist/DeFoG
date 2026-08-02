"""
GDPO RL fine-tuning of an E1 base against the three structural failure modes.

Targets, measured on the E1 final test pass (4 seeds, n=10,000 each):

    validity (relaxed)  0.9854 +- 0.0049
    disconnected        0.0615 +- 0.0219   <- worst deficit
    wonky rings         0.0356 +- 0.0129   (any ring outside [3,8])
    combined 'sanity'   0.9504

The E1 checkpoints are NOT touched. This writes to ``ckpts/zinc_rl_seed<N>/``
and leaves ``ckpts/zinc_e1_seed<N>/`` frozen, so the E1 table row -- produced
from those exact weights under a frozen sampling configuration -- stays
reproducible. The RL model is a separate artifact and a separate claim.

**Reward: graded, not a single AND.** ``r = valid + connected + rings_ok``, each
0/1, so r in {0,1,2,3}. A single sanity indicator would be the exact target, but
95% of samples already satisfy it, leaving a near-constant reward and almost no
group-relative advantage to learn from. The graded form separates the failing
5% *and* distinguishes one-fault from two-fault samples. Invalid scores 0 rather
than being decomposed, since connectivity and ring size are not assessable on a
molecule that does not parse -- which also makes invalid strictly worse than any
valid molecule, so there is no incentive to trade validity for either term.

**The failure mode this guards against** is reward hacking: sanity can be raised
by collapsing onto a narrow, safe region of chemical space, which would wreck
FCD while every targeted metric improves. Two defences: a KL penalty to the
frozen base (``KL_COEF``), and SMILES dumped before and after so FCD/NSPDK can
be scored externally. A run whose sanity improves while FCD degrades is a
FAILURE, and the exported artifacts are what make that judgeable.

Note ``ITERATIONS``: ``gdpo_connectivity.py`` records the over-optimisation
cliff at roughly 60-80 iterations for ``kl_coef=0.3``. The default here stays
below it.

Usage:
    python experiments/gdpo_sanity__zinc.py --__TESTING__ True
    python experiments/gdpo_sanity__zinc.py --SEED 42 \
        --BASE_CKPT "'ckpts/zinc_e1_seed42/best_model'" \
        --OUT_CKPT_DIR "'ckpts/zinc_rl_seed42'"
"""
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from defog.core import DeFoGModel, GDPOTrainer  # noqa: E402
from defog.core.distribution_penalty import (  # noqa: E402
    FragmentTypicalityPenalty,
    FragmentVocabulary,
    MMDPenalty,
)
from defog.data import guacamol_reference as gmref  # noqa: E402
from defog.data import moses_reference as mref  # noqa: E402
from defog.data import zinc_reference as zref  # noqa: E402

REFERENCES = {"zinc": zref, "guacamol": gmref, "moses": mref}
from defog.domains import MoleculeDomain  # noqa: E402
from defog.domains.molecule import build_encoders, ring_sizes_ok, validity_report  # noqa: E402

RDLogger.DisableLog("rdApp.*")
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Parameters
# ============================================================================
# :param DATASET: Selects the reference module, and therefore the frozen
#     vocabulary and bond set. Getting this wrong mis-decodes silently rather
#     than erroring -- ZINC is kekulized with 9 elements, GuacaMol aromatic with
#     12, MOSES aromatic with 8.
DATASET: str = "zinc"
BASE_CKPT: str = os.path.join(_PROJECT_DIR, "ckpts", "zinc_e1_seed42", "best_model")
OUT_CKPT_DIR: str = os.path.join(_PROJECT_DIR, "ckpts", "zinc_rl_seed42")
SEED: int = 42

RING_LO: int = 3
RING_HI: int = 8

# --- Distribution-fidelity penalties (anti reward-hacking) ------------------
# :param ALPHA_FRAG / BETA_MMD: Weights on the two penalty terms in
#     ``r = sanity - ALPHA_FRAG * frag - BETA_MMD * mmd``. **Both default to 0.0,
#     which makes this file behave exactly as it did before the penalties
#     existed** -- the control arm of the sweep is the original code, not an
#     approximation of it, and neither penalty object is even constructed.
#
#     Motivation: the MOSES run raised validity 0.885 -> 0.939 on 4/4 seeds
#     while FCD went 0.863 -> 1.706 on 4/4 seeds. The reward could see validity
#     and could not see the distribution, so it optimised the first by
#     destroying the second.
#
#     **BETA_MMD belongs well above 1.** The two terms live on very different
#     scales. Measured on the four MOSES runs, the hack bought +0.157 of mean
#     sanity reward while costing only +0.011 of MMD penalty, so the weight at
#     which the trade becomes reward-neutral is
#
#         beta* = 0.157 / 0.011 ~= 14      (12.5 to 20.2 across the four seeds)
#
#     An earlier version of this comment said to keep ``ALPHA_FRAG + BETA_MMD <
#     1``, reasoning that a penalty of 1 could push a valid molecule below an
#     invalid one. That bound is wrong here, and following it would have
#     guaranteed a null result. The MMD penalty is ``sim_sib - 2 * sim_ref``,
#     and with sim_sib ~= 0.385 against 2 * sim_ref ~= 0.76 it is reliably
#     *negative* -- subtracting it raises the reward, so the valid-above-invalid
#     ordering holds for any non-negative beta. It could only turn positive
#     under near-total collapse (sim_sib > 2 * sim_ref), which is the regime
#     where a large penalty is the correct response anyway.
#
#     ALPHA_FRAG is different: the fragment penalty is a fraction in [0, 1] and
#     is always subtracted, so the original bound does apply to it.
#
#     **ALPHA_FRAG should stay 0 on MOSES.** The offline gate
#     (scripts/validate_rl_penalties.py, results in
#     experiments/results/penalty_gate_moses.json) scored the fragment term at
#     run-level AUC 0.048 against the known hacked policy -- far below 0.5,
#     meaning it *prefers* the hacked samples and would accelerate the very
#     failure it was added to prevent. The reason is structural: the hack
#     narrowed toward simpler, more aliphatic molecules assembled from very
#     common fragments, and a term that only punishes unusual fragments is
#     blind to that by construction. It still separates real MOSES data from
#     model output cleanly (0.032 vs 0.073), so it measures something real --
#     just not this.
ALPHA_FRAG: float = 0.0
BETA_MMD: float = 0.0
# :param MMD_KERNEL: 'descriptor' (RBF on standardised physicochemical
#     descriptors) or 'tanimoto' (Morgan/ECFP4). The gate ranks descriptor
#     first: run-level AUC 1.000 vs 0.998, and correlation with the hack axis
#     (aromatic ring count) -0.49 vs -0.34, so it applies a markedly stronger
#     per-sample push back along the axis the hack travelled. Binary
#     fingerprints record which substructures are present, not how many, and
#     this hack was a change in how many.
MMD_KERNEL: str = "descriptor"
# :param FRAG_MIN_COUNT: Occurrences in the train vocabulary below which a BRICS
#     fragment counts as atypical. Trades off against how often ordinary
#     molecules get penalised -- check ``coverage`` in the log, not this number.
FRAG_MIN_COUNT: int = 5
# :param FRAG_VOCAB_MOLECULES: Train molecules to decompose when building the
#     vocabulary. BRICS runs at ~1.75 ms/molecule, so the full 1.58M MOSES rows
#     would cost 45 min for a vocabulary whose common entries are settled long
#     before that. Cached on disk and keyed by reference content.
FRAG_VOCAB_MOLECULES: int = 250_000
# :param MMD_N_REFERENCE: Reference molecules held as fingerprints for the MMD
#     kernel. Fixed across iterations so the reward is comparable over the run;
#     4096 keeps the per-iteration cost around 0.05 s against a ~30 s rollout.
MMD_N_REFERENCE: int = 4096

# --- RL (defaults from gdpo_connectivity.py, which tuned them on this model) --
# :param ITERATIONS: Run 1139128 peaked at ~iteration 20 (sanity 0.961,
#     disconnected 0.008) and then collapsed to sanity 0.14 by iteration 59. The
#     "~60-80 cliff" quoted in gdpo_connectivity.py was measured at ETA=0; at
#     ETA=25 the cliff arrived far earlier. 25 brackets the observed optimum
#     without entering the degrading region -- and CKPT_EVERY/SELECT_BEST below
#     mean the peak is captured even if it moves again.
ITERATIONS: int = 25
ROLLOUT_SIZE: int = 128
SUBSAMPLE_STEPS: int = 12
MINIBATCH_SIZE: int = 16
LAMBDA_EDGE: float = 1.0
REDUCTION: str = "sum"
ADVANTAGE_MODE: str = "mean"   # Dr. GRPO: mean baseline, no per-group std
POSITIVE_ONLY: bool = False
LR: float = 2e-5
WEIGHT_DECAY: float = 1e-5
GRAD_CLIP: float = 1.0
EMA_DECAY: float = 0.9
# :param KL_COEF: Strength of the pull toward the frozen base. Non-zero on
#     purpose -- this is the primary defence against optimising sanity by
#     collapsing the distribution.
KL_COEF: float = 0.3

# --- Checkpointing (never rely on the final iteration) ----------------------
# :param CKPT_EVERY / SELECT_BEST / SELECT_WINDOW:
#     Run 1139128 saved only the final model, so a genuinely good policy at
#     iteration 20 was unrecoverable and the run had to be redone. EMA does not
#     rescue that: at ema_decay=0.9 its memory is ~10 iterations, so the final
#     EMA reflected the collapsed region. Periodic checkpoints plus a best-so-far
#     make the peak survivable.
#
#     Selection uses a ROLLING MEAN of rollout sanity, not a single batch: at
#     rollout_size 128 a single batch carries ~±2% noise, so unsmoothed selection
#     would reward a lucky iteration rather than a genuinely better policy.
CKPT_EVERY: int = 5
SELECT_BEST: bool = True
SELECT_WINDOW: int = 3

# --- Rollout sampling -------------------------------------------------------
# :param ROLLOUT_SAMPLE_STEPS / ROLLOUT_ETA / ROLLOUT_OMEGA:
#     eta matches the DEPLOYED configuration (the E1 frozen eta=25) so the
#     policy is optimised under the stochasticity it will actually be run with.
#     Steps are reduced to 100 for rollout cost; eval uses the full deployed 500.
#     That step mismatch is a known approximation, which is why evaluation is
#     done at the deploy config rather than the rollout config.
#     Run 1139128 used ROLLOUT_ETA=25 to match deployment and diverged; the
#     cliff estimate borrowed from gdpo_connectivity.py was measured at eta=0,
#     and noisier rollouts raise gradient variance. Back to 0 -- one variable at
#     a time.
ROLLOUT_SAMPLE_STEPS: int = 100
ROLLOUT_ETA: float = 0.0
ROLLOUT_OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"

# --- Evaluation (at the E1 FROZEN deploy config) ----------------------------
EVAL_SAMPLES: int = 2048
EVAL_STEPS: int = 500
# :param EVAL_ETA: The DEPLOYED eta for this dataset's frozen config. ZINC and
#     MOSES froze eta=25; GuacaMol froze eta=75. Evaluating at the wrong one
#     measures a policy nobody will run.
EVAL_ETA: float = 25.0
EVAL_OMEGA: float = 0.0

__DEBUG__: bool = False
__TESTING__: bool = False


class SanityReward:
    """r = valid + connected + rings_ok, each 0/1 -> r in {0,1,2,3}.

    Invalid short-circuits to 0: neither connectivity nor ring size can be
    assessed on a molecule that does not parse, and scoring it below every valid
    molecule removes any incentive to trade validity away for the other terms.
    Records the last batch's category rates in ``self.last`` for the curve.

    Optionally subtracts distribution-fidelity penalties::

        r = sanity - alpha * frag_penalty - beta * mmd_penalty

    Both weights default to 0, in which case no penalty object is constructed
    and the reward is bit-for-bit the original. The penalties are applied only
    to molecules that decode: an invalid sample stays at exactly 0.0, so the
    "invalid is strictly worst" ordering that the graded reward depends on
    survives as long as ``alpha + beta < 1``.

    The penalties score the whole decoded SMILES, disconnected components
    included, rather than the largest fragment that the relaxed validity
    convention reports. The sanity term already drives disconnection toward
    zero, so the two representations converge on exactly the samples that end up
    mattering.
    """

    invalid = 0.0

    def __init__(self, domain, ring_lo=3, ring_hi=8, *,
                 frag_penalty=None, alpha=0.0, mmd_penalty=None, beta=0.0):
        self.domain = domain
        self.ring_lo, self.ring_hi = ring_lo, ring_hi
        self.frag_penalty, self.alpha = frag_penalty, float(alpha)
        self.mmd_penalty, self.beta = mmd_penalty, float(beta)
        self.last = {}

    def __call__(self, X1, E1, node_mask):
        from defog.core.data import dense_to_pyg

        n = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, n)
        out = torch.empty(len(datas))
        smiles = []
        n_valid = n_conn = n_rings = n_full = 0
        for i, d in enumerate(datas):
            smi = self.domain.identity(d)   # canonical SMILES iff genuinely valid
            smiles.append(smi)
            if smi is None:
                out[i] = 0.0
                continue
            n_valid += 1
            connected = "." not in smi
            mol = Chem.MolFromSmiles(smi)
            rings_ok = mol is not None and ring_sizes_ok(mol, self.ring_lo, self.ring_hi)
            n_conn += connected
            n_rings += rings_ok
            n_full += connected and rings_ok
            out[i] = 1.0 + float(connected) + float(rings_ok)
        k = max(1, len(datas))
        self.last = {
            "valid_frac": n_valid / k,
            "connected_frac": n_conn / k,
            "rings_ok_frac": n_rings / k,
            "sanity_frac": n_full / k,
            "disconnected_frac": (n_valid - n_conn) / k,
            "wonky_ring_frac": (n_valid - n_rings) / k,
            "sanity_reward_mean": float(out.mean()),
        }

        # Penalties last, so ``sanity_frac`` above always means the raw targeted
        # rate whatever the weights are -- the curve stays comparable across arms.
        if self.frag_penalty is not None and self.alpha:
            p = torch.as_tensor(self.frag_penalty(smiles), dtype=out.dtype)
            out -= self.alpha * p
            self.last.update(self.frag_penalty.last)
        if self.mmd_penalty is not None and self.beta:
            p = torch.as_tensor(self.mmd_penalty(smiles), dtype=out.dtype)
            out -= self.beta * p
            self.last.update(self.mmd_penalty.last)
        if self.alpha or self.beta:
            self.last["shaped_reward_mean"] = float(out.mean())
            self.last["shaped_reward_std"] = float(out.std())
            # The valid-above-invalid ordering is now an empirical property
            # rather than one a weight bound guarantees, so measure it instead
            # of asserting it. If this ever reaches 0 the gradient has started
            # preferring molecules that do not parse.
            valid_idx = [i for i, s in enumerate(smiles) if s is not None]
            self.last["min_valid_reward"] = (
                float(out[valid_idx].min()) if valid_idx else float("nan"))
        return out


@torch.no_grad()
def evaluate(model, atom_decoder, bond_decoder, n_samples, steps, eta, omega,
             device, train_ref, chunk=256):
    """Sample at the DEPLOY config and report every targeted metric plus SMILES.

    The SMILES matter as much as the numbers: FCD and NSPDK are computed
    externally in the metrics env, and they are what decide whether an
    improvement in sanity came at the cost of distribution match.
    """
    samples, remaining = [], n_samples
    while remaining > 0:
        cur = min(chunk, remaining)
        samples += model.sample(num_samples=cur, sample_steps=steps, eta=eta,
                                omega=omega, device=device, show_progress=False)
        remaining -= cur

    rep = validity_report(samples, atom_decoder, bond_decoder,
                          reference_smiles=train_ref)
    smiles = rep.pop("smiles")
    n_wonky = sum(1 for s in smiles
                  if (m := Chem.MolFromSmiles(s)) is not None and not ring_sizes_ok(m))
    rep["wonky_ring_frac"] = n_wonky / max(1, len(smiles))
    rep["sanity"] = sum(
        1 for s in smiles
        if "." not in s and (m := Chem.MolFromSmiles(s)) is not None and ring_sizes_ok(m)
    ) / max(1, n_samples)
    return rep, smiles


@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    e.log("GDPO sanity RL on a ZINC E1 base")
    torch.manual_seed(e.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.abspath(e.OUT_CKPT_DIR).rstrip("/").endswith(
            os.path.basename(os.path.dirname(os.path.abspath(e.BASE_CKPT)))):
        raise RuntimeError("OUT_CKPT_DIR would overwrite the E1 base; refusing.")
    os.makedirs(e.OUT_CKPT_DIR, exist_ok=True)

    mod = REFERENCES[e.DATASET]
    e.log(f"dataset={e.DATASET}: {len(mod.ATOM_TYPES)} atom types, bonds={mod.BOND_TYPES}")
    _, atom_decoder, _, bond_decoder = build_encoders(mod.ATOM_TYPES, mod.BOND_TYPES)
    split = mod.load_reference_split()
    train_ref = set(split.train_smiles)
    # Validation only. The test split is not read here -- the E1 test pass
    # already happened and this model is a separate artifact.
    val_path = os.path.join(e.path, "validation_reference.smi")
    with open(val_path, "w") as fh:
        fh.write("\n".join(split.val_smiles) + "\n")

    model = DeFoGModel.load(e.BASE_CKPT).to(device)
    e.log(f"loaded base {e.BASE_CKPT}")
    domain = MoleculeDomain(atom_decoder, bond_decoder, reference_smiles=train_ref)

    # -- BEFORE --------------------------------------------------------------
    model.eval()
    before, before_smiles = evaluate(model, atom_decoder, bond_decoder,
                                     e.EVAL_SAMPLES, e.EVAL_STEPS, e.EVAL_ETA,
                                     e.EVAL_OMEGA, device, train_ref)
    with open(os.path.join(e.path, "before.smi"), "w") as fh:
        fh.write("\n".join(before_smiles) + "\n")
    e.log(f"BEFORE  validity={before['validity_relaxed_largest_frag']:.4f} "
          f"disconnected={before['disconnected']:.4f} "
          f"wonky_rings={before['wonky_ring_frac']:.4f} "
          f"sanity={before['sanity']:.4f} uniq={before['uniqueness']:.4f}")
    e["before"] = before

    # -- TRAIN ---------------------------------------------------------------
    # Penalty references come from TRAIN. Evaluation FCD is scored against
    # VALIDATION, and that separation is the point: it leaves FCD an independent
    # verdict rather than a quantity the policy has been steered toward.
    frag_penalty = mmd_penalty = None
    if e.ALPHA_FRAG:
        vocab = FragmentVocabulary.build_or_load(
            e.DATASET, split.train_smiles,
            max_molecules=e.FRAG_VOCAB_MOLECULES, seed=0, log=e.log)
        frag_penalty = FragmentTypicalityPenalty(vocab, min_count=e.FRAG_MIN_COUNT)
        e.log(f"fragment penalty: {len(frag_penalty)} fragments at "
              f"min_count={e.FRAG_MIN_COUNT}, occurrence coverage "
              f"{vocab.coverage(e.FRAG_MIN_COUNT):.4f}, alpha={e.ALPHA_FRAG}")
    if e.BETA_MMD:
        mmd_penalty = MMDPenalty(split.train_smiles, n_reference=e.MMD_N_REFERENCE,
                                 seed=0, kernel=e.MMD_KERNEL, log=e.log)
        e.log(f"MMD penalty: beta={e.BETA_MMD}, kernel={e.MMD_KERNEL}")
    if e.ALPHA_FRAG >= 1.0:
        e.log(f"WARNING: alpha={e.ALPHA_FRAG} >= 1. The fragment penalty is a "
              f"fraction in [0,1] and is always subtracted, so a valid molecule "
              f"can now score below an invalid one and the gradient points at "
              f"invalidity.")

    reward = SanityReward(domain, e.RING_LO, e.RING_HI,
                          frag_penalty=frag_penalty, alpha=e.ALPHA_FRAG,
                          mmd_penalty=mmd_penalty, beta=e.BETA_MMD)
    trainer = GDPOTrainer(
        model, reward, rollout_size=e.ROLLOUT_SIZE,
        sample_steps=e.ROLLOUT_SAMPLE_STEPS, eta=e.ROLLOUT_ETA,
        omega=e.ROLLOUT_OMEGA, time_distortion=e.TIME_DISTORTION,
        subsample_steps=e.SUBSAMPLE_STEPS, minibatch_size=e.MINIBATCH_SIZE,
        lambda_edge=e.LAMBDA_EDGE, reduction=e.REDUCTION,
        advantage_mode=e.ADVANTAGE_MODE, positive_only=e.POSITIVE_ONLY,
        kl_coef=e.KL_COEF, lr=e.LR, weight_decay=e.WEIGHT_DECAY,
        grad_clip=e.GRAD_CLIP, ema_decay=e.EMA_DECAY, device=device, seed=e.SEED,
    )
    e.log(f"training {e.ITERATIONS} iterations (kl_coef={e.KL_COEF}, lr={e.LR})")

    history = []
    best = {"score": -float("inf"), "iter": None, "path": None}

    # Selection must follow the objective. Selecting on raw sanity while
    # training on the shaped reward would hand-pick the most reward-hacked
    # checkpoint of the run -- the exact failure the penalties exist to prevent.
    # With both weights at 0 this falls back to sanity_frac, so the control arm
    # selects identically to every previous run.
    shaped = bool(e.ALPHA_FRAG or e.BETA_MMD)
    select_key = "shaped_reward_mean" if shaped else "sanity_frac"
    e.log(f"checkpoint selection on rolling mean of '{select_key}' "
          f"(window {e.SELECT_WINDOW})")

    def on_iter(i, stats):
        row = {"iter": i, **{k: v for k, v in stats.items() if isinstance(v, (int, float))},
               **reward.last}
        history.append(row)

        if e.CKPT_EVERY and i % e.CKPT_EVERY == 0:
            trainer.save(os.path.join(e.OUT_CKPT_DIR, f"iter{i:03d}"), use_ema=True)

        if e.SELECT_BEST:
            window = [h.get(select_key, 0.0) for h in history[-e.SELECT_WINDOW:]]
            score = sum(window) / max(1, len(window))
            # Require a full window so the first iteration cannot win on a
            # one-batch fluke.
            if len(window) >= e.SELECT_WINDOW and score > best["score"]:
                best.update(score=score, iter=i,
                            path=trainer.save(os.path.join(e.OUT_CKPT_DIR, "best_model"),
                                              use_ema=True))
                e.log(f"  it {i:3d} new best smoothed {select_key}={score:.3f} -> best_model")

        if i % 5 == 0 or i == e.ITERATIONS - 1:
            extra = ""
            if shaped:
                extra = (f" mmd={reward.last.get('mmd_penalty_mean', float('nan')):+.3f}"
                         f" simsib={reward.last.get('mmd_sim_sibling', float('nan')):.3f}"
                         f" minvalid={reward.last.get('min_valid_reward', float('nan')):+.2f}")
            e.log(f"  it {i:3d} reward={row.get('reward_mean', float('nan')):.3f} "
                  f"sanity={reward.last.get('sanity_frac', float('nan')):.3f} "
                  f"disc={reward.last.get('disconnected_frac', float('nan')):.3f} "
                  f"wonky={reward.last.get('wonky_ring_frac', float('nan')):.3f}{extra}")

    trainer.fit(e.ITERATIONS, on_iter=on_iter)
    final_path = trainer.save(os.path.join(e.OUT_CKPT_DIR, "rl_model_final"), use_ema=True)
    e.log(f"saved final-iteration model -> {final_path}")
    e.commit_json("history.json", history)

    # Evaluate the SELECTED model, not the last one. The last iteration is only
    # the best policy if the run never degraded, which is not the common case.
    if e.SELECT_BEST and best["path"]:
        e.log(f"loading best checkpoint (iter {best['iter']}, "
              f"smoothed {select_key} {best['score']:.3f})")
        model = DeFoGModel.load(os.path.join(e.OUT_CKPT_DIR, "best_model")).to(device)
        out_path = best["path"]
    else:
        if trainer.ema is not None:
            trainer.ema.copy_to(model)
        out_path = final_path
    e["selected"] = {"iter": best["iter"], "smoothed_score": best["score"],
                     "select_key": select_key, "path": out_path}

    # -- AFTER ---------------------------------------------------------------
    model.eval()
    after, after_smiles = evaluate(model, atom_decoder, bond_decoder,
                                   e.EVAL_SAMPLES, e.EVAL_STEPS, e.EVAL_ETA,
                                   e.EVAL_OMEGA, device, train_ref)
    with open(os.path.join(e.path, "after.smi"), "w") as fh:
        fh.write("\n".join(after_smiles) + "\n")
    e["after"] = after

    def delta(k):
        return after[k] - before[k]

    e.log("=" * 62)
    e.log(f"{'metric':<28}{'before':>10}{'after':>10}{'delta':>12}")
    for k, lab in [("validity_relaxed_largest_frag", "validity (relaxed)"),
                   ("disconnected", "disconnected"),
                   ("wonky_ring_frac", "wonky rings"),
                   ("sanity", "sanity (all three)"),
                   ("uniqueness", "uniqueness")]:
        e.log(f"{lab:<28}{before[k]:>10.4f}{after[k]:>10.4f}{delta(k):>+12.4f}")
    e.log("=" * 62)
    e.log("FCD / NSPDK are NOT computed here. Score before.smi and after.smi "
          "against validation_reference.smi with scripts/e1_metrics.py: a sanity "
          "gain with degraded FCD is a FAILED run, not a successful one.")

    e.commit_json("summary.json", {
        "dataset": e.DATASET,
        "base_ckpt": e.BASE_CKPT, "out_ckpt": out_path, "seed": e.SEED,
        "selected_iter": best["iter"], "selected_smoothed_score": best["score"],
        "select_key": select_key,
        "final_iteration_ckpt": final_path,
        "reward": ("graded: valid + connected + rings_ok in [0,3]"
                   + (f" - {e.ALPHA_FRAG}*frag - {e.BETA_MMD}*mmd" if shaped else "")),
        "penalties": {
            "alpha_frag": e.ALPHA_FRAG, "beta_mmd": e.BETA_MMD,
            "mmd_kernel": e.MMD_KERNEL,
            "frag_min_count": e.FRAG_MIN_COUNT,
            "frag_vocab_molecules": e.FRAG_VOCAB_MOLECULES,
            "mmd_n_reference": e.MMD_N_REFERENCE,
            "reference_split": "train",
            "note": ("penalty reference is TRAIN; evaluation FCD is scored against "
                     "VALIDATION, so FCD stays an independent verdict"),
        },
        "kl_coef": e.KL_COEF, "iterations": e.ITERATIONS, "lr": e.LR,
        "rollout": {"steps": e.ROLLOUT_SAMPLE_STEPS, "eta": e.ROLLOUT_ETA,
                    "omega": e.ROLLOUT_OMEGA},
        "eval": {"steps": e.EVAL_STEPS, "eta": e.EVAL_ETA, "omega": e.EVAL_OMEGA,
                 "n": e.EVAL_SAMPLES, "scored_against": "validation"},
        "before": before, "after": after,
        "delta": {k: delta(k) for k in
                  ("validity_relaxed_largest_frag", "disconnected",
                   "wonky_ring_frac", "sanity", "uniqueness")},
        "e1_base_untouched": True,
    })

    # -- curve ---------------------------------------------------------------
    if history:
        it = [h["iter"] for h in history]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        a1.plot(it, [h.get("reward_mean", np.nan) for h in history], lw=1.5)
        a1.set_xlabel("iteration"); a1.set_ylabel("mean reward (0-3)")
        a1.set_title("reward"); a1.grid(alpha=0.3)
        for key, lab in [("sanity_frac", "sanity"), ("disconnected_frac", "disconnected"),
                         ("wonky_ring_frac", "wonky rings")]:
            a2.plot(it, [h.get(key, np.nan) for h in history], lw=1.5, label=lab)
        a2.set_xlabel("iteration"); a2.set_ylabel("fraction of rollout")
        a2.set_title("targeted failure modes"); a2.legend(fontsize=8); a2.grid(alpha=0.3)
        fig.tight_layout()
        e.commit_fig("rl_progress.png", fig)


@experiment.testing
def testing(e: Experiment):
    e.ITERATIONS = 4
    e.CKPT_EVERY = 2
    e.SELECT_WINDOW = 2
    e.ROLLOUT_SIZE = 8
    e.SUBSAMPLE_STEPS = 2
    e.MINIBATCH_SIZE = 4
    e.ROLLOUT_SAMPLE_STEPS = 5
    e.EVAL_SAMPLES = 16
    e.EVAL_STEPS = 5


experiment.run_if_main()
