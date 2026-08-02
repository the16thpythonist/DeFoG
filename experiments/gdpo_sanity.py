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
    """

    invalid = 0.0

    def __init__(self, domain, ring_lo=3, ring_hi=8):
        self.domain = domain
        self.ring_lo, self.ring_hi = ring_lo, ring_hi
        self.last = {}

    def __call__(self, X1, E1, node_mask):
        from defog.core.data import dense_to_pyg

        n = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, n)
        out = torch.empty(len(datas))
        n_valid = n_conn = n_rings = n_full = 0
        for i, d in enumerate(datas):
            smi = self.domain.identity(d)   # canonical SMILES iff genuinely valid
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
        }
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
    reward = SanityReward(domain, e.RING_LO, e.RING_HI)
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
    best = {"score": -1.0, "iter": None, "path": None}

    def on_iter(i, stats):
        row = {"iter": i, **{k: v for k, v in stats.items() if isinstance(v, (int, float))},
               **reward.last}
        history.append(row)

        if e.CKPT_EVERY and i % e.CKPT_EVERY == 0:
            trainer.save(os.path.join(e.OUT_CKPT_DIR, f"iter{i:03d}"), use_ema=True)

        if e.SELECT_BEST:
            window = [h.get("sanity_frac", 0.0) for h in history[-e.SELECT_WINDOW:]]
            score = sum(window) / max(1, len(window))
            # Require a full window so the first iteration cannot win on a
            # one-batch fluke.
            if len(window) >= e.SELECT_WINDOW and score > best["score"]:
                best.update(score=score, iter=i,
                            path=trainer.save(os.path.join(e.OUT_CKPT_DIR, "best_model"),
                                              use_ema=True))
                e.log(f"  it {i:3d} new best smoothed sanity={score:.3f} -> best_model")

        if i % 5 == 0 or i == e.ITERATIONS - 1:
            e.log(f"  it {i:3d} reward={row.get('reward_mean', float('nan')):.3f} "
                  f"sanity={reward.last.get('sanity_frac', float('nan')):.3f} "
                  f"disc={reward.last.get('disconnected_frac', float('nan')):.3f} "
                  f"wonky={reward.last.get('wonky_ring_frac', float('nan')):.3f}")

    trainer.fit(e.ITERATIONS, on_iter=on_iter)
    final_path = trainer.save(os.path.join(e.OUT_CKPT_DIR, "rl_model_final"), use_ema=True)
    e.log(f"saved final-iteration model -> {final_path}")
    e.commit_json("history.json", history)

    # Evaluate the SELECTED model, not the last one. The last iteration is only
    # the best policy if the run never degraded, which is not the common case.
    if e.SELECT_BEST and best["path"]:
        e.log(f"loading best checkpoint (iter {best['iter']}, "
              f"smoothed sanity {best['score']:.3f})")
        model = DeFoGModel.load(os.path.join(e.OUT_CKPT_DIR, "best_model")).to(device)
        out_path = best["path"]
    else:
        if trainer.ema is not None:
            trainer.ema.copy_to(model)
        out_path = final_path
    e["selected"] = {"iter": best["iter"], "smoothed_sanity": best["score"],
                     "path": out_path}

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
        "selected_iter": best["iter"], "selected_smoothed_sanity": best["score"],
        "final_iteration_ckpt": final_path,
        "reward": "graded: valid + connected + rings_ok in [0,3]",
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
