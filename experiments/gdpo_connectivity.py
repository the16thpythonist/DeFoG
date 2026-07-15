"""
GDPO connectivity fine-tune -- a SINGLE configuration, as a pycomex experiment.

Trains the disconnected-fragment failure mode out of a pretrained DeFoG molecular
model via GDPO's eager policy gradient. Reward: connected & valid molecule = 1.0,
everything else (disconnected -- a '.' in the SMILES -- OR invalid) = 0.0. Model-
agnostic: the atom vocabulary is reconstructed from the checkpoint's atom_weights,
so the same experiment runs on AqSolDB / ZINC / GuacaMol by only changing CKPT_PATH.

One run = one config. It measures the disconnected / valid / unique fractions of
freshly generated molecules BEFORE vs AFTER fine-tuning (matched eta/distortion so
train and eval are the same policy), and exports the reward+disconnected trace,
before/after grids, per-iter history, periodic pre-collapse snapshots, and metrics.

Optional stabilizers (all off by default -> plain GDPO):
  * ADVANTAGE_MODE="mean"  -- fixed baseline so the gradient fades as reward saturates
  * POSITIVE_ONLY=True     -- RAFT-style: never push DOWN bad endpoints (no atom-soup)
  * KL_ANCHOR="moving"     -- EMA-of-policy trust region (drift past the pretrained floor)
  * KL_TARGET=<float>      -- adaptively controls KL_COEF toward a target KL

The SWEEP is NOT done here: submit MANY of these runs (one per grid point) and
aggregate the archives (see run_gdpo_positive_sweep_kcist.sh).

Usage:
    python experiments/gdpo_connectivity.py --CKPT_PATH ~/Downloads/zinc_uncond_4e-4_best_model.ckpt
    python experiments/gdpo_connectivity.py --__TESTING__ True
"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import RDLogger
from rdkit.Chem import Draw, GetPeriodicTable
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from defog.core import DeFoGModel, GDPOTrainer, RolloutSampler
from defog.domains import MoleculeDomain
from defog.domains.molecule import build_encoders, pyg_data_to_mol, mol_to_smiles
from experiments.guided_logp_demo import BOND_TYPES

RDLogger.DisableLog("rdApp.*")

# == configuration ==========================================================
# :param CKPT_PATH: pretrained DeFoG molecular checkpoint to fine-tune (any dataset;
#     the atom vocabulary is read back from its atom_weights).
CKPT_PATH: str = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_best_model.ckpt")
SEED: int = 0

# --- training budget / rollout policy ---
# :param ROUNDS: re-anchoring rounds. Each round fine-tunes ITERATIONS updates with a
#     FIXED strong anchor to the round's starting weights, then the (EMA) result
#     becomes the next round's base AND anchor. A fixed anchor per round prevents
#     collapse; ratcheting it between rounds relaxes the floor -- the stable way to
#     push below it (a continuous moving anchor collapses instead). ROUNDS=1 = a
#     plain single fine-tune.
ROUNDS: int = 1
# :param ITERATIONS: GDPO updates PER round (keep short enough to stop before the
#     over-optimization cliff, ~60-80 for kl_coef=0.3).
ITERATIONS: int = 100
# :param ROLLOUT_SIZE: K rollout molecules per iteration (bigger = smoother gradient).
ROLLOUT_SIZE: int = 128
# :param SAMPLE_STEPS / ETA / OMEGA / TIME_DISTORTION: rollout (and matched eval) policy.
#     ETA=0 samples at the deployment point; the CTMC is still stochastic enough for signal.
SAMPLE_STEPS: int = 100
ETA: float = 0.0
OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"
# :param SUBSAMPLE_STEPS: noisy states per trajectory that enter the gradient.
SUBSAMPLE_STEPS: int = 12
# :param MINIBATCH_SIZE: trajectories per grad forward (bounds autograd memory).
MINIBATCH_SIZE: int = 16

# --- eager gradient / advantage ---
# :param REDUCTION: "sum" (true joint LL, keeps bond-gradient weight) | "mean".
REDUCTION: str = "sum"
# :param ADVANTAGE_MODE: "grpo" | "mean" (fades as reward saturates) | "none".
ADVANTAGE_MODE: str = "grpo"
# :param POSITIVE_ONLY: RAFT-style -- clamp advantage>=0, never push down bad endpoints.
POSITIVE_ONLY: bool = False
# :param LR: AdamW learning rate.
LR: float = 2e-5

# --- KL to reference (over-optimization guard) ---
# :param KL_COEF: KL-to-reference strength (0 -> no reference built, no KL).
KL_COEF: float = 0.3
# :param KL_ANCHOR: "fixed" (frozen initial weights) | "moving" (EMA-of-policy trust region).
KL_ANCHOR: str = "fixed"
# :param ANCHOR_DECAY: EMA decay of the moving anchor.
ANCHOR_DECAY: float = 0.99
# :param KL_TARGET: if set, adaptively nudge KL_COEF toward this target KL (else fixed).
KL_TARGET: float = None
# :param EMA_DECAY: deployment-weights EMA (evaluated / saved). ~0.9 for short runs.
EMA_DECAY: float = 0.9

# --- evaluation / checkpointing ---
# :param EVAL_SAMPLES / EVAL_STEPS: fresh molecules for the BEFORE / final measurement.
EVAL_SAMPLES: int = 2048
EVAL_STEPS: int = 100
# :param ROUND_EVAL_SAMPLES: cheaper eval after each intermediate round (the ratchet).
ROUND_EVAL_SAMPLES: int = 512
# :param CKPT_EVERY: save a pre-collapse policy snapshot every N iters (0=off).
CKPT_EVERY: int = 20

# NOTE: __DEBUG__ must be False when submitting a sweep -- debug mode writes to a
# single overwriteable folder, so parallel runs would clobber each other.
__DEBUG__: bool = False
__TESTING__: bool = False


def atom_decoder_from_ckpt(model):
    """Reconstruct the exact atom_decoder (class idx -> symbol) the model was trained
    with, from its molecular-features atom_weights -- so decoding matches the
    checkpoint regardless of dataset. Model-agnostic (AqSolDB/ZINC/GuacaMol/...)."""
    weights = model.hparams.get("atom_weights")
    if not weights:
        raise ValueError("checkpoint has no atom_weights (molecular_features off)")
    pt = GetPeriodicTable()
    cand = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Na", "Si", "B", "Se", "K"]
    tab = [(pt.GetAtomicWeight(pt.GetAtomicNumber(s)), s) for s in cand]
    return [min(tab, key=lambda t: abs(t[0] - w))[1] for w in weights]


class ConnectivityReward:
    """connected&valid = 1.0; else (disconnected '.' OR invalid) = 0.0. The two
    failure modes are EQUAL so there is no perverse invalid->disconnected lateral
    move. Tracks the last batch's category fractions in ``self.last``."""
    invalid = 0.0

    def __init__(self, domain):
        self.domain = domain
        self.last = {}

    def __call__(self, X1, E1, node_mask):
        from defog.core.data import dense_to_pyg
        n = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, n)
        out = torch.empty(len(datas))
        nc = nd = ni = 0
        for i, d in enumerate(datas):
            smi = self.domain.identity(d)  # canonical SMILES iff genuinely valid, else None
            if smi is None:
                out[i] = 0.0; ni += 1
            elif "." in smi:
                out[i] = 0.0; nd += 1
            else:
                out[i] = 1.0; nc += 1
        k = max(1, len(datas))
        self.last = {"connected_frac": nc / k, "disconnected_frac": nd / k, "invalid_frac": ni / k}
        return out


@torch.no_grad()
def evaluate(model, domain, n_samples, sample_steps, size_dist, device,
             eta, omega, time_distortion, seed=0):
    """Sample n_samples fresh molecules under the SAME policy the rollouts use and
    report valid / disconnected / connected / unique fractions + example mols."""
    torch.manual_seed(seed)
    mols, smis, all_smis = [], [], []
    n_valid = n_disc = 0
    remaining, chunk = n_samples, 64
    while remaining > 0:
        k = min(chunk, remaining)
        samples = model.sample(k, size_dist=size_dist, eta=eta, omega=omega,
                               sample_steps=sample_steps, time_distortion=time_distortion,
                               device=device, show_progress=False)
        for d in samples:
            mol = pyg_data_to_mol(d, domain.atom_decoder, domain.bond_decoder)
            smi = mol_to_smiles(mol) if mol is not None else None
            if smi is not None:
                n_valid += 1
                all_smis.append(smi)
                if "." in smi:
                    n_disc += 1
                if len(mols) < 25:
                    mols.append(mol); smis.append(smi)
        remaining -= k
    f = lambda x: x / n_samples
    return {
        "n": n_samples, "valid_frac": f(n_valid), "disconnected_frac_all": f(n_disc),
        "disconnected_frac_of_valid": (n_disc / n_valid) if n_valid else 0.0,
        "unique_frac_of_valid": (len(set(all_smis)) / n_valid) if n_valid else 0.0,
    }, mols, smis


def save_grid(mols, smis, path):
    if not mols:
        return
    legends = [("disc: " if "." in s else "") + s[:24] for s in smis]
    Draw.MolsToGridImage(mols[:25], molsPerRow=5, subImgSize=(230, 230),
                         legends=legends).save(path)


def save_curves(history, before_disc, path):
    def smooth(x, w=7):
        x = np.asarray(x, float)
        return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else x
    rm = [h["reward_mean"] for h in history]
    dc = [h.get("disconnected_frac", np.nan) for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(rm, color="#bbb", lw=1, alpha=0.6); a1.plot(range(len(smooth(rm))), smooth(rm), color="#2c7fb8", lw=2)
    a1.set_xlabel("iteration"); a1.set_ylabel("rollout mean reward (1=connected)")
    a1.set_title("reward"); a1.grid(alpha=0.3)
    a2.plot(dc, color="#f4a582", lw=1, alpha=0.6); a2.plot(range(len(smooth(dc))), smooth(dc), color="#d95f0e", lw=2)
    a2.axhline(before_disc, color="k", ls="--", lw=1, label=f"before eval ({before_disc:.1%})")
    a2.set_xlabel("iteration"); a2.set_ylabel("rollout disconnected fraction")
    a2.set_title("disconnected fraction (rollout)"); a2.legend(fontsize=8); a2.grid(alpha=0.3)
    fig.suptitle("GDPO connectivity fine-tune")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment) -> None:
    e.log(f"GDPO connectivity: ckpt={os.path.basename(e.CKPT_PATH)} K={e.ROLLOUT_SIZE} "
          f"iters={e.ITERATIONS} adv={e.ADVANTAGE_MODE} pos_only={e.POSITIVE_ONLY} "
          f"kl_coef={e.KL_COEF} kl_anchor={e.KL_ANCHOR} kl_target={e.KL_TARGET}")
    from pytorch_lightning import seed_everything
    seed_everything(e.SEED, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeFoGModel.load(e.CKPT_PATH, device="cpu").to(device)
    atom_types = atom_decoder_from_ckpt(model)
    ae, ad, be, bd = build_encoders(atom_types, BOND_TYPES)
    assert len(ad) == model.num_node_classes, \
        f"atom decoder {len(ad)} != model node classes {model.num_node_classes}"
    domain = MoleculeDomain(ad, bd)
    size_dist = model.default_size_dist
    e.log(f"atoms ({len(ad)}): {ad}")

    eval_kw = dict(eta=e.ETA, omega=e.OMEGA, time_distortion=e.TIME_DISTORTION)
    before, mols_b, smis_b = evaluate(model, domain, e.EVAL_SAMPLES, e.EVAL_STEPS,
                                      size_dist, device, seed=e.SEED, **eval_kw)
    e["results/before"] = before
    e.log(f"BEFORE: valid={before['valid_frac']:.1%} disc(all)={before['disconnected_frac_all']:.1%} "
          f"disc(of valid)={before['disconnected_frac_of_valid']:.1%} unique={before['unique_frac_of_valid']:.1%}")
    save_grid(mols_b, smis_b, os.path.join(e.path, "grid_before.png"))

    reward = ConnectivityReward(domain)
    ckpt_dir = os.path.join(e.path, "ckpts")
    if e.CKPT_EVERY > 0:
        os.makedirs(ckpt_dir, exist_ok=True)
    history = []
    round_results = []
    mols_a, smis_a = mols_b, smis_b

    for r in range(e.ROUNDS):
        # Fresh trainer each round: its KL reference is a frozen copy of the CURRENT
        # weights, so round r is anchored to round (r-1)'s result. A fixed strong
        # anchor per round keeps it stable; ratcheting the anchor between rounds
        # relaxes the floor -- the stable way down (a continuous moving anchor
        # collapses). Fresh optimizer + EMA per round; the old trainer is GC'd.
        trainer = GDPOTrainer(
            model, reward, rollout_size=e.ROLLOUT_SIZE, sample_steps=e.SAMPLE_STEPS,
            subsample_steps=e.SUBSAMPLE_STEPS, minibatch_size=e.MINIBATCH_SIZE,
            eta=e.ETA, omega=e.OMEGA, time_distortion=e.TIME_DISTORTION, size_dist=size_dist,
            advantage_mode=e.ADVANTAGE_MODE, reduction=e.REDUCTION, positive_only=e.POSITIVE_ONLY,
            kl_coef=e.KL_COEF, kl_anchor=e.KL_ANCHOR, anchor_decay=e.ANCHOR_DECAY, kl_target=e.KL_TARGET,
            lr=e.LR, ema_decay=e.EMA_DECAY, device=device, seed=e.SEED + r,
        )

        def on_iter(it, m, _r=r, _tr=trainer):
            m = {**m, "round": _r, "iter": it, **reward.last}
            history.append(m)
            for key in ("reward_mean", "connected_frac", "disconnected_frac", "invalid_frac",
                        "kl", "kl_coef", "grad_norm"):
                if key in m:
                    e.track(key, float(m[key]))
            if it % 5 == 0 or it == e.ITERATIONS - 1:
                e.log(f"  r{_r} iter {it:3d} reward={m['reward_mean']:+.3f} conn={m.get('connected_frac',0):.2f} "
                      f"disc={m.get('disconnected_frac',0):.2f} inval={m.get('invalid_frac',0):.2f} "
                      f"gnorm={m['grad_norm']:.1f} klc={m.get('kl_coef',0):.3f}")
            if e.CKPT_EVERY > 0 and (it + 1) % e.CKPT_EVERY == 0:
                _tr.save(os.path.join(ckpt_dir, f"round{_r}_iter{it + 1:04d}.ckpt"))
                e.commit_json("history.json", history)

        trainer.fit(e.ITERATIONS, on_iter=on_iter)
        if trainer.ema is not None:
            trainer.ema.copy_to(model)  # model = round output -> next round's base + anchor

        last = (r == e.ROUNDS - 1)
        n_eval = e.EVAL_SAMPLES if last else e.ROUND_EVAL_SAMPLES
        rev, mols_a, smis_a = evaluate(model, domain, n_eval, e.EVAL_STEPS,
                                       size_dist, device, seed=e.SEED + 100 + r, **eval_kw)
        round_results.append(rev)
        e[f"results/round_{r}"] = rev
        e.track("round_disc_of_valid", float(rev["disconnected_frac_of_valid"]))
        e.track("round_valid", float(rev["valid_frac"]))
        e.log(f"ROUND {r}: disc(of valid)={rev['disconnected_frac_of_valid']:.1%} "
              f"valid={rev['valid_frac']:.1%} unique={rev['unique_frac_of_valid']:.1%} (n={n_eval})")

    after = round_results[-1]
    e["results/after"] = after
    save_grid(mols_a, smis_a, os.path.join(e.path, "grid_after.png"))
    save_curves(history, before["disconnected_frac_all"], os.path.join(e.path, "reward_curve.png"))

    summary = {
        "disc_of_valid_before": before["disconnected_frac_of_valid"],
        "disc_of_valid_after": after["disconnected_frac_of_valid"],
        "valid_before": before["valid_frac"], "valid_after": after["valid_frac"],
        "unique_after": after["unique_frac_of_valid"],
        "ratchet": [round(rr["disconnected_frac_of_valid"], 4) for rr in round_results],
        "ROUNDS": e.ROUNDS, "ITERATIONS": e.ITERATIONS, "KL_COEF": e.KL_COEF,
        "POSITIVE_ONLY": e.POSITIVE_ONLY, "ADVANTAGE_MODE": e.ADVANTAGE_MODE, "ROLLOUT_SIZE": e.ROLLOUT_SIZE,
    }
    e["results/summary"] = summary
    e.commit_json("summary.json", summary)
    e.commit_json("history.json", history)
    model.save(os.path.join(e.path, "gdpo_connected.ckpt"))
    e.log(f"SUMMARY disc(of valid) {summary['disc_of_valid_before']:.1%} -> "
          f"{summary['disc_of_valid_after']:.1%} | ratchet={summary['ratchet']} | "
          f"valid {summary['valid_before']:.1%} -> {summary['valid_after']:.1%} | unique {summary['unique_after']:.1%}")


@experiment.testing
def testing(e: Experiment) -> None:
    e.ROUNDS = 2
    e.ITERATIONS = 3
    e.ROLLOUT_SIZE = 8
    e.SAMPLE_STEPS = 20
    e.SUBSAMPLE_STEPS = 2
    e.MINIBATCH_SIZE = 4
    e.EVAL_SAMPLES = 16
    e.ROUND_EVAL_SAMPLES = 16
    e.EVAL_STEPS = 20
    e.CKPT_EVERY = 0


experiment.run_if_main()
