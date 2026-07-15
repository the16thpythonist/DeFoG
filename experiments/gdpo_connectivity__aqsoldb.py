"""
GDPO fine-tune: train the disconnected-fragment failure mode out of AqSolDB DeFoG.

The base generator sometimes emits two disconnected molecule fragments (a '.' in the
canonical SMILES). This is a cheap, non-differentiable, whole-molecule reward -- a
perfect first GDPO target: reward a valid, single-connected molecule; penalize a
disconnected one; floor the invalid ones. We measure the disconnected fraction (and
validity) of freshly generated molecules BEFORE vs AFTER fine-tuning.

    PYTHONPATH=. .venv/bin/python experiments/gdpo_connectivity__aqsoldb.py --iterations 60

Outputs (to --outdir): reward_curve.png, grid_before.png, grid_after.png, metrics.json.
"""
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import RDLogger
from rdkit.Chem import Draw, GetPeriodicTable

from defog.core import DeFoGModel, GDPOTrainer, RolloutSampler
from defog.domains import MoleculeDomain
from defog.domains.molecule import build_encoders, pyg_data_to_mol, mol_to_smiles
from experiments.guided_logp_demo import BOND_TYPES

RDLogger.DisableLog("rdApp.*")


def atom_decoder_from_ckpt(model):
    """Reconstruct the exact atom_decoder (class idx -> symbol) the model was
    trained with, from its molecular-features ``atom_weights`` -- so decoding
    matches the checkpoint regardless of which dataset is around. Model-agnostic:
    works for AqSolDB, ZINC, GuacaMol, etc."""
    weights = model.hparams.get("atom_weights")
    if not weights:
        raise ValueError("checkpoint has no atom_weights (molecular_features off); "
                         "cannot reconstruct the atom vocabulary")
    pt = GetPeriodicTable()
    cand = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Na", "Si", "B", "Se", "K"]
    tab = [(pt.GetAtomicWeight(pt.GetAtomicNumber(s)), s) for s in cand]
    return [min(tab, key=lambda t: abs(t[0] - w))[1] for w in weights]


class ConnectivityReward:
    """Reward on the generated molecule, detecting disconnection via a '.' in the
    canonical SMILES (as the user requested).

    The ONLY good outcome is a valid, single-connected molecule (reward 1.0). Both
    failure modes -- a valid-but-disconnected molecule (a '.' in the SMILES) AND an
    undecodable/invalid graph -- get the SAME reward (default 0.0). Making them equal
    is deliberate: an earlier reward that scored disconnected (0.0) above invalid
    (-0.5) let the optimizer raise reward by turning invalid graphs into valid-but-
    disconnected ones, which lifted validity but INCREASED disconnection. With both
    failures equal there is no such lateral move -- the only way up is a connected,
    valid molecule.

    Tracks the last batch's category fractions in ``self.last`` for logging.
    """
    invalid = 0.0  # finite floor for reward_from_energy-style consumers

    def __init__(self, domain, connected=1.0, disconnected=0.0, invalid=0.0):
        self.domain = domain
        self.connected = connected
        self.disconnected = disconnected
        self.invalid = invalid
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
                out[i] = self.invalid; ni += 1
            elif "." in smi:
                out[i] = self.disconnected; nd += 1
            else:
                out[i] = self.connected; nc += 1
        k = max(1, len(datas))
        self.last = {"connected_frac": nc / k, "disconnected_frac": nd / k, "invalid_frac": ni / k}
        return out


@torch.no_grad()
def evaluate(model, domain, n_samples, sample_steps, size_dist, device,
             eta, omega, time_distortion, seed=0):
    """Sample n_samples fresh molecules and report validity / disconnected /
    connected / UNIQUE fractions + a few example mols for the grid.

    Samples under the SAME (eta, omega, time_distortion, sample_steps) policy the
    rollouts use, so BEFORE/AFTER measure the exact distribution being optimized
    (else the reward curve and the eval metric are different quantities)."""
    torch.manual_seed(seed)
    mols, smis = [], []
    all_smis = []
    n_valid = n_disc = 0
    remaining = n_samples
    chunk = 64  # bound memory on the 6 GB GPU
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
                    mols.append(mol)
                    smis.append(smi)
        remaining -= k
    frac = lambda x: x / n_samples
    return {
        "n": n_samples,
        "valid_frac": frac(n_valid),
        "disconnected_frac_all": frac(n_disc),
        "disconnected_frac_of_valid": (n_disc / n_valid) if n_valid else 0.0,
        # uniqueness guards against mode collapse (finding 4): a headline win on
        # connectivity while unique_frac craters means the generator collapsed.
        "unique_frac_of_valid": (len(set(all_smis)) / n_valid) if n_valid else 0.0,
    }, mols, smis


def save_grid(mols, smis, path, title):
    if not mols:
        return
    legends = [("disc: " if "." in s else "") + (s[:24]) for s in smis]
    img = Draw.MolsToGridImage(mols[:25], molsPerRow=5, subImgSize=(230, 230), legends=legends)
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/Downloads/aqsoldb_4e-4_best_model.ckpt"))
    ap.add_argument("--data", default="data/aqsoldb_conditional.csv")
    ap.add_argument("--outdir", default="experiments/_gdpo_connectivity")
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--rollout-size", type=int, default=48)
    ap.add_argument("--sample-steps", type=int, default=100)
    ap.add_argument("--subsample-steps", type=int, default=12)
    ap.add_argument("--minibatch-size", type=int, default=8,
                    help="trajectories per grad forward (bounds autograd memory)")
    ap.add_argument("--eta", type=float, default=2.0, help="stochasticity for BOTH rollout and eval")
    ap.add_argument("--omega", type=float, default=0.0)
    ap.add_argument("--time-distortion", default="polydec", help="for BOTH rollout and eval")
    ap.add_argument("--reduction", default="sum", choices=["sum", "mean"],
                    help="eager-logprob reduction; 'sum' = true joint LL, keeps bond-gradient weight")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-coef", type=float, default=0.0)
    ap.add_argument("--ema-decay", type=float, default=0.9)
    ap.add_argument("--eval-samples", type=int, default=256)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=20,
                    help="save a policy snapshot every N iters (0=off) so a pre-collapse "
                         "checkpoint is always recoverable; pick via history.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeFoGModel.load(args.ckpt, device="cpu").to(device)
    atom_types = atom_decoder_from_ckpt(model)  # exact vocab from the checkpoint
    ae, ad, be, bd = build_encoders(atom_types, BOND_TYPES)
    assert len(ad) == model.num_node_classes, \
        f"atom decoder {len(ad)} != model node classes {model.num_node_classes}"
    domain = MoleculeDomain(ad, bd)
    size_dist = model.default_size_dist  # the model's own size prior (honest test)

    print(f"[gdpo-conn] ckpt={os.path.basename(args.ckpt)} device={device} atoms={ad} "
          f"steps={args.sample_steps} K={args.rollout_size} iters={args.iterations}", flush=True)

    eval_kw = dict(eta=args.eta, omega=args.omega, time_distortion=args.time_distortion)
    before, mols_b, smis_b = evaluate(model, domain, args.eval_samples, args.eval_steps,
                                      size_dist, device, seed=args.seed, **eval_kw)
    print(f"[gdpo-conn] BEFORE: valid={before['valid_frac']:.1%} "
          f"disconnected(all)={before['disconnected_frac_all']:.1%} "
          f"disconnected(of valid)={before['disconnected_frac_of_valid']:.1%} "
          f"unique(of valid)={before['unique_frac_of_valid']:.1%}", flush=True)
    save_grid(mols_b, smis_b, os.path.join(args.outdir, "grid_before.png"), "before")

    reward = ConnectivityReward(domain)
    trainer = GDPOTrainer(
        model, reward, rollout_size=args.rollout_size, sample_steps=args.sample_steps,
        subsample_steps=args.subsample_steps, minibatch_size=args.minibatch_size,
        eta=args.eta, omega=args.omega,
        time_distortion=args.time_distortion, size_dist=size_dist,
        advantage_mode="grpo", reduction=args.reduction, kl_coef=args.kl_coef,
        lr=args.lr, ema_decay=args.ema_decay, device=device, seed=args.seed,
    )

    ckpt_dir = os.path.join(args.outdir, "ckpts")
    if args.ckpt_every > 0:
        os.makedirs(ckpt_dir, exist_ok=True)

    history = []
    def on_iter(it, m):
        m = {**m, "iter": it, **reward.last}  # attach this rollout's category fractions
        history.append(m)
        if it % 5 == 0 or it == args.iterations - 1:
            print(f"[gdpo-conn] iter {it:3d}  reward={m['reward_mean']:+.3f}  "
                  f"conn={m.get('connected_frac',0):.2f} disc={m.get('disconnected_frac',0):.2f} "
                  f"inval={m.get('invalid_frac',0):.2f}  gnorm={m['grad_norm']:.2f}", flush=True)
        # periodic snapshot (EMA/deployment weights) so a run that later collapses is
        # still recoverable -- inspect history.json for the last good iter, then load
        # ckpts/iter{N}.ckpt. Also flush history each time so it survives a kill.
        if args.ckpt_every > 0 and ((it + 1) % args.ckpt_every == 0):
            trainer.save(os.path.join(ckpt_dir, f"iter{it + 1:04d}.ckpt"))
            with open(os.path.join(args.outdir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)
            print(f"[gdpo-conn]   snapshot ckpts/iter{it + 1:04d}.ckpt "
                  f"(disc={m.get('disconnected_frac',0):.2f} inval={m.get('invalid_frac',0):.2f} "
                  f"valid_conn_reward={m['reward_mean']:.3f})", flush=True)
    trainer.fit(args.iterations, on_iter=on_iter)

    # evaluate the EMA weights (what save() would write)
    if trainer.ema is not None:
        trainer.ema.copy_to(model)
    after, mols_a, smis_a = evaluate(model, domain, args.eval_samples, args.eval_steps,
                                     size_dist, device, seed=args.seed + 1, **eval_kw)
    print(f"[gdpo-conn] AFTER:  valid={after['valid_frac']:.1%} "
          f"disconnected(all)={after['disconnected_frac_all']:.1%} "
          f"disconnected(of valid)={after['disconnected_frac_of_valid']:.1%} "
          f"unique(of valid)={after['unique_frac_of_valid']:.1%}", flush=True)
    save_grid(mols_a, smis_a, os.path.join(args.outdir, "grid_after.png"), "after")

    # reward + disconnected-fraction curves (rollout batches; smoothed)
    def smooth(x, w=7):
        x = np.asarray(x, float)
        if len(x) < w:
            return x
        k = np.ones(w) / w
        return np.convolve(x, k, mode="valid")
    rm = [h["reward_mean"] for h in history]
    dc = [h.get("disconnected_frac", np.nan) for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(rm, color="#bbb", lw=1, alpha=0.6); a1.plot(range(len(smooth(rm)) ), smooth(rm), color="#2c7fb8", lw=2)
    a1.set_xlabel("iteration"); a1.set_ylabel("rollout mean reward (1=connected)")
    a1.set_title("reward"); a1.grid(alpha=0.3)
    a2.plot(dc, color="#f4a582", lw=1, alpha=0.6); a2.plot(range(len(smooth(dc))), smooth(dc), color="#d95f0e", lw=2)
    a2.axhline(before["disconnected_frac_all"], color="k", ls="--", lw=1,
               label=f"before eval ({before['disconnected_frac_all']:.1%})")
    a2.set_xlabel("iteration"); a2.set_ylabel("rollout disconnected fraction")
    a2.set_title("disconnected fraction (rollout)"); a2.legend(fontsize=8); a2.grid(alpha=0.3)
    fig.suptitle("GDPO connectivity fine-tune")
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "reward_curve.png"), dpi=140)
    plt.close(fig)

    metrics = {"before": before, "after": after,
               "config": {k: getattr(args, k) for k in
                          ("iterations", "rollout_size", "sample_steps", "subsample_steps",
                           "eta", "lr", "kl_coef", "eval_samples")}}
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.outdir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)  # full per-iter trace for picking a snapshot
    trainer.save(os.path.join(args.outdir, "gdpo_connected.ckpt"))
    print(f"[gdpo-conn] DONE -> {args.outdir}  (snapshots in ckpts/, per-iter trace in history.json)", flush=True)


if __name__ == "__main__":
    main()
