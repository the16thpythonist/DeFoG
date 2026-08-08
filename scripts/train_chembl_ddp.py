#!/usr/bin/env python
"""
Lean multi-GPU (DDP) training entrypoint for the ChEMBL foundation model.

No pycomex -- deliberately, because the pycomex @Experiment archive is created
per-process and DDP re-runs the script on every rank (4 archives / a race). This
script does just: train the 12L/384 DeFoG on ChEMBL under DDP with FULL resumable
checkpointing (chain across 12h JUPITER windows), with all logging / figures /
checkpoints guarded to rank 0. The rich extended eval (validity / sanity /
connected / KL) is a SEPARATE single-GPU pass: --eval-only on a checkpoint.

Train (4-GPU DDP, one 12h chain link, auto-resumes from CKPT_DIR/last.ckpt):
    srun python scripts/train_chembl_ddp.py --devices 4 --lr 3e-4 --epochs 60 \
        --max-time-hours 9.5 --ckpt-dir ckpts/chembl_foundation_lr3e-4

Eval (single GPU, extended metrics on the best checkpoint):
    python scripts/train_chembl_ddp.py --eval-only \
        --eval-ckpt ckpts/chembl_foundation_lr3e-4/best_model.ckpt

Local CPU-DDP smoke test:
    CUDA_VISIBLE_DEVICES="" python scripts/train_chembl_ddp.py --devices 2 \
        --accelerator cpu --max-train 200 --max-val 60 --epochs 1 \
        --ckpt-dir /tmp/ddp_smoke --num-workers 0
"""
import argparse
import json
import os

import numpy as np
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, Timer


class PerLinkTimer(Timer):
    """A wall-clock cap that is PER-LINK, not cumulative across resumes.

    Lightning's Timer checkpoints its elapsed time, so a chained link restores the
    previous link's elapsed time and trips 'Time limit reached' immediately (once
    the total exceeds max_time) -- i.e. it does zero training. Ignoring the restored
    state gives each 12h-window link a fresh clock.
    """

    def load_state_dict(self, state_dict):
        pass  # do not restore cumulative elapsed time -> per-link cap

from experiments.utils import (
    build_encoders, smiles_to_pyg_data, make_generation_metrics_fn,
    molecular_metrics, property_distributions,
)
from defog.core import (
    DeFoGModel, TrainingMonitorCallback, SampleVisualizationCallback, EMACallback,
)
from defog.domains import MoleculeDomain

# --- Frozen schema ----------------------------------------------------------
# Sourced from defog.data.chembl_reference rather than restated here, so the
# vocabulary cannot drift from prepare_chembl.py and diagnose_validity.py.
from defog.data import chembl_reference as chembl_ref  # noqa: E402
from defog.data import vocabulary  # noqa: E402

ATOM_DECODER = list(chembl_ref.ATOM_TYPES)
BOND_TYPES = list(chembl_ref.BOND_TYPES)
ATOM_VALENCY = dict(chembl_ref.ATOM_VALENCY)
ATOM_WEIGHT = dict(chembl_ref.ATOM_WEIGHT)
MAX_ATOM_WEIGHT = chembl_ref.MAX_ATOM_WEIGHT


def resolve_vocab(args):
    """(representation, atom_encoder, atom_decoder, bond_encoder, bond_decoder).

    Every entry point goes through this so the encoders, the kekulize flag and
    the declared representation can never disagree -- the failure mode being
    that a 3-bond encoder without kekulize=True rejects every aromatic molecule,
    which SmilesGraphDataset would paper over by silently substituting the next
    one (see check_encodable below).
    """
    rep = chembl_ref.get_representation(args.representation)
    ae, ad, be, bd = build_encoders(list(rep.atom_types), list(rep.bond_types))
    return rep, ae, ad, be, bd


def stats_path(args, rep):
    """Explicit --stats wins; otherwise the default representation keeps the
    historical '{prefix}_stats.json' and any other gets its own file, because
    the marginals ARE the noise prior and differ per bond vocabulary."""
    if args.stats:
        return args.stats
    if rep.name == chembl_ref.DEFAULT_REPRESENTATION:
        return os.path.join(args.data_dir, f"{args.prefix}_stats.json")
    return os.path.join(args.data_dir, f"{args.prefix}_kek_stats.json")


def check_encodable(smiles, ae, be, rep, sample=2000, max_skip=0.01):
    """Fail loudly if the representation cannot encode the data.

    SmilesGraphDataset falls through to the next molecule when conversion
    returns None. That is a reasonable guard against a stray bad row, but it
    also means a mis-wired representation does not crash: it silently trains on
    whatever minority of the dataset happens to encode (for ChEMBL under a
    3-bond vocabulary without kekulize, the ~7% with no aromatic ring, cycled
    over and over). Measured skip rate for both declared representations is 0.
    """
    probe = smiles[:sample]
    skipped = sum(1 for s in probe
                  if smiles_to_pyg_data(s, ae, be, kekulize=rep.kekulize) is None)
    frac = skipped / max(1, len(probe))
    if frac > max_skip:
        raise SystemExit(
            f"representation {rep.name!r} cannot encode {frac:.1%} of a "
            f"{len(probe)}-molecule probe (bonds={rep.bond_types}, "
            f"kekulize={rep.kekulize}). Training would silently proceed on the "
            f"remainder. Check the representation matches the data.")
    return frac

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def is_rank0() -> bool:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))) == 0


def rprint(*a):
    if is_rank0():
        print("[rank0]", *a, flush=True)


def read_smiles(path, limit=None):
    out = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            s = line.strip()
            if s:
                out.append(s)
    return out


class SmilesGraphDataset(torch.utils.data.Dataset):
    """Lazy SMILES -> PyG Data (keeps 2.44M graphs off the heap)."""

    def __init__(self, smiles, atom_encoder, bond_encoder, kekulize=False):
        self.smiles = smiles
        self.atom_encoder = atom_encoder
        self.bond_encoder = bond_encoder
        # Must agree with the bond vocabulary: a bond set without AROMATIC needs
        # kekulize=True or every aromatic molecule converts to None.
        self.kekulize = kekulize

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        n = len(self.smiles)
        for off in range(n):
            d = smiles_to_pyg_data(self.smiles[(idx + off) % n],
                                   self.atom_encoder, self.bond_encoder,
                                   kekulize=self.kekulize)
            if d is not None:
                return d
        raise RuntimeError("no convertible SMILES")


def build_model(args, stats, rep=None):
    rep = rep or chembl_ref.get_representation(args.representation)
    # The stats file carries the marginals the model denoises from, so a stats
    # file built for another bond vocabulary is not a mismatch to discover at
    # the first backward pass.
    want_e = len(rep.bond_types) + 1
    if int(stats["num_edge_classes"]) != want_e:
        raise SystemExit(
            f"stats file has {stats['num_edge_classes']} edge classes but "
            f"representation {rep.name!r} implies {want_e}. The marginals are "
            f"the noise prior -- regenerate with scripts/compute_graph_stats.py "
            f"--representation {rep.name}.")
    node_marginals = torch.tensor(stats["node_marginals"], dtype=torch.float)
    edge_marginals = torch.tensor(stats["edge_marginals"], dtype=torch.float)
    max_nodes = int(stats["max_nodes"])
    node_counts = torch.zeros(max_nodes + 1)
    for k, v in stats["size_histogram"].items():
        node_counts[int(k)] = float(v)
    return DeFoGModel(
        num_node_classes=int(stats["num_node_classes"]),
        num_edge_classes=int(stats["num_edge_classes"]),
        n_layers=args.n_layers, hidden_dim=args.hidden_dim,
        hidden_mlp_dim=args.hidden_mlp_dim, n_heads=args.n_heads, dropout=0.1,
        noise_type="marginal", node_marginals=node_marginals,
        edge_marginals=edge_marginals, node_counts=node_counts, max_nodes=max_nodes,
        extra_features_type="rrwp", rrwp_steps=args.rrwp_steps,
        molecular_features=True,
        atom_valencies=[ATOM_VALENCY[a] for a in rep.atom_types],
        atom_weights=[ATOM_WEIGHT[a] for a in rep.atom_types],
        max_atom_weight=MAX_ATOM_WEIGHT,
        lr=args.lr, weight_decay=1e-5, lambda_edge=5.0,
        train_time_distortion="polydec", lr_scheduler="cosine", lr_min=1e-6,
        sample_steps=100, eta=0.0, omega=0.0, sample_time_distortion="polydec",
    )


def train(args):
    pl.seed_everything(args.seed, workers=True)
    rep, ae, ad, be, bd = resolve_vocab(args)

    train_smiles = read_smiles(os.path.join(args.data_dir, f"{args.prefix}_train.smiles"), args.max_train)
    val_smiles = read_smiles(os.path.join(args.data_dir, f"{args.prefix}_val.smiles"), args.max_val)
    rprint(f"train {len(train_smiles):,}  val {len(val_smiles):,}")
    skip_frac = check_encodable(train_smiles, ae, be, rep)
    rprint(f"representation={rep.name} atoms={len(rep.atom_types)} "
           f"edges={len(rep.bond_types) + 1} kekulize={rep.kekulize} "
           f"probe_skip={skip_frac:.5f}")

    train_loader = DataLoader(
        SmilesGraphDataset(train_smiles, ae, be, kekulize=rep.kekulize),
        batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        SmilesGraphDataset(val_smiles, ae, be, kekulize=rep.kekulize),
        batch_size=args.batch_size,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    ) if val_smiles else None

    with open(stats_path(args, rep)) as fh:
        stats = json.load(fh)
    rprint(f"stats: {stats_path(args, rep)}")
    model = build_model(args, stats, rep)
    rprint(f"params {sum(p.numel() for p in model.parameters()):,}  max_nodes {stats['max_nodes']}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    trackdir = os.path.join(args.ckpt_dir, "track")
    os.makedirs(trackdir, exist_ok=True)

    # rank-0-only figure saving (the callbacks already guard to global_zero)
    def save_progress(fig):
        fig.savefig(os.path.join(trackdir, "training_progress.png"), dpi=110)

    def save_samples(fig):
        fig.savefig(os.path.join(trackdir, "samples.png"), dpi=110)

    gen_fn = make_generation_metrics_fn(ad, bd, train_smiles)
    monitor = TrainingMonitorCallback(
        smoothing_window=5, generation_metrics_fn=gen_fn, gen_every_k=args.gen_every_k,
        gen_num_samples=64, gen_sample_steps=args.gen_sample_steps, gen_eta=5.0,
        checkpoint_dir=args.ckpt_dir, figure_callback=save_progress,
    )
    sampler = SampleVisualizationCallback(
        num_samples=8, every_k_epochs=args.sample_vis_every_k,
        sample_steps=args.gen_sample_steps, eta=5.0,
        domain=MoleculeDomain(ad, bd, reference_smiles=train_smiles),
        figure_callback=save_samples,
    )
    ckpt_cb = ModelCheckpoint(dirpath=args.ckpt_dir, save_last=True, save_top_k=0,
                              every_n_train_steps=args.ckpt_every_n_steps)
    callbacks = [EMACallback(decay=0.9999), monitor, sampler, ckpt_cb]

    # Per-link wall-clock cap via PerLinkTimer (NOT Trainer max_time, which is
    # cumulative across resumes and would make every chained link stop instantly).
    if args.max_time_hours:
        h = int(args.max_time_hours)
        m = int(round((args.max_time_hours - h) * 60))
        callbacks.append(PerLinkTimer(duration={"hours": h, "minutes": m}, interval="step"))

    strategy = ("ddp_find_unused_parameters_true" if args.devices != 1 else "auto")
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator=args.accelerator,
        devices=args.devices, num_nodes=args.num_nodes, strategy=strategy,
        enable_progress_bar=False, enable_checkpointing=True, logger=False,
        num_sanity_val_steps=0, callbacks=callbacks,
    )

    resume = None
    last = os.path.join(args.ckpt_dir, "last.ckpt")
    if os.path.exists(last):
        resume = last
    rprint(f"strategy={strategy} devices={args.devices}; "
           + (f"RESUMING {last}" if resume else "fresh start"))

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=resume)

    if trainer.is_global_zero:
        # on_fit_end baked EMA weights into the model -> save the inference model
        path = model.save(os.path.join(args.ckpt_dir, "foundation_model"))
        rprint(f"saved final (EMA) model -> {path}; best_validity={monitor.best_validity:.3f}")


def evaluate(args):
    """Single-GPU extended eval on a checkpoint (no DDP)."""
    rep, ae, ad, be, bd = resolve_vocab(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeFoGModel.load(args.eval_ckpt.replace(".ckpt", "")).to(device).eval()
    # Decoding with the wrong vocabulary yields plausible molecules made of the
    # wrong elements rather than an error, so every metric below would be a
    # number describing nothing.
    rprint(vocabulary.check_model(model, rep.atom_types, rep.bond_types,
                                  what=args.eval_ckpt))
    rprint(f"loaded {args.eval_ckpt} on {device} as {rep.name}")

    samples = []
    remaining = args.num_eval_samples
    while remaining > 0:
        cur = min(args.eval_chunk, remaining)
        samples += model.sample(num_samples=cur, sample_steps=args.eval_sample_steps,
                                device=device, show_progress=False)
        remaining -= cur

    ref_desc = None
    ref_path = os.path.join(args.data_dir, f"{args.prefix}_ref_descriptors.npz")
    if os.path.exists(ref_path):
        with np.load(ref_path) as z:
            ref_desc = {k: z[k] for k in z.files}
    train_smiles = set(read_smiles(os.path.join(args.data_dir, f"{args.prefix}_train.smiles")))
    metrics = molecular_metrics(samples, ad, bd, reference_smiles=train_smiles,
                                reference_descriptors=ref_desc, compute_kl=True)
    out = os.path.join(os.path.dirname(args.eval_ckpt) or ".", "eval_metrics.json")
    with open(out, "w") as fh:
        json.dump(metrics, fh, indent=2)
    for k in ("validity", "uniqueness", "novelty", "connected", "disconnected",
              "sanity", "wonky_ring_frac", "kl_logp", "kl_tpsa", "kl_qed", "kl_score"):
        if k in metrics:
            rprint(f"  {k:16s} = {metrics[k]:.4f}")
    rprint(f"wrote {out}")


def sweep(args):
    """Single-GPU eta/omega sampling sweep on a checkpoint -> best sampling config.

    The training probe used eta=5; the eta=0 eval is the harsh floor. This grids
    the CTMC sampling knobs (eta = error-correction stochasticity, omega = target
    guidance) at fixed time-distortion, scoring each with the full metric suite,
    so we can pick the sampling config the released model ships with.
    """
    rep, ae, ad, be, bd = resolve_vocab(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeFoGModel.load(args.eval_ckpt.replace(".ckpt", "")).to(device).eval()
    rprint(vocabulary.check_model(model, rep.atom_types, rep.bond_types,
                                  what=args.eval_ckpt))
    rprint(f"sweep on {args.eval_ckpt} ({device}); distortion={args.sweep_distortion} "
           f"steps={args.eval_sample_steps} samples/config={args.sweep_samples}")

    ref_desc = None
    ref_path = os.path.join(args.data_dir, f"{args.prefix}_ref_descriptors.npz")
    if os.path.exists(ref_path):
        with np.load(ref_path) as z:
            ref_desc = {k: z[k] for k in z.files}
    train_smiles = set(read_smiles(os.path.join(args.data_dir, f"{args.prefix}_train.smiles")))

    etas = [float(x) for x in args.sweep_etas.split(",")]
    omegas = [float(x) for x in args.sweep_omegas.split(",")]
    results = []
    for eta in etas:
        for omega in omegas:
            samples, rem = [], args.sweep_samples
            while rem > 0:
                cur = min(args.eval_chunk, rem)
                samples += model.sample(num_samples=cur, sample_steps=args.eval_sample_steps,
                                        eta=eta, omega=omega, time_distortion=args.sweep_distortion,
                                        device=device, show_progress=False)
                rem -= cur
            m = molecular_metrics(samples, ad, bd, reference_smiles=train_smiles,
                                  reference_descriptors=ref_desc, compute_kl=True)
            m = {"eta": eta, "omega": omega, **m}
            results.append(m)
            rprint(f"eta={eta:<6g} omega={omega:<6g} | val={m['validity']:.3f} san={m['sanity']:.3f} "
                   f"conn={m['connected']:.3f} kl={m['kl_score']:.3f} nov={m['novelty']:.3f}")

    out = os.path.join(os.path.dirname(args.eval_ckpt) or ".", "sweep_results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    ranked = sorted(results, key=lambda r: (r["sanity"], r["validity"]), reverse=True)
    rprint("=== top 5 by sanity (check kl_score/novelty aren't sacrificed) ===")
    for r in ranked[:5]:
        rprint(f"  eta={r['eta']:<6g} omega={r['omega']:<6g} san={r['sanity']:.3f} "
               f"val={r['validity']:.3f} conn={r['connected']:.3f} kl={r['kl_score']:.3f} nov={r['novelty']:.3f}")
    rprint(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=os.path.join(_HERE, "data", "chembl"))
    p.add_argument("--prefix", default="chembl",
                   help="data-file prefix: reads {prefix}_train.smiles / {prefix}_stats.json "
                        "/ {prefix}_ref_descriptors.npz (use 'union' for the ZINC∪ChEMBL set)")
    p.add_argument("--representation", default=chembl_ref.DEFAULT_REPRESENTATION,
                   choices=sorted(chembl_ref.REPRESENTATIONS),
                   help="graph vocabulary. 'aromatic_v1' is what v1/v2 shipped "
                        "(12 atom / 5 edge); 'kekulized_v2' drops the AROMATIC "
                        "class (12 / 4). Not interchangeable -- a checkpoint and "
                        "its representation must travel together.")
    p.add_argument("--stats", default=None,
                   help="explicit stats JSON (default: {prefix}_stats.json for "
                        "aromatic_v1, {prefix}_kek_stats.json otherwise). The "
                        "marginals are the noise prior, so they are per-vocabulary.")
    p.add_argument("--ckpt-dir", default=os.path.join(_HERE, "ckpts", "chembl_foundation"))
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=60)          # cosine horizon (fixed across links)
    p.add_argument("--max-time-hours", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--devices", type=int, default=4)
    p.add_argument("--num-nodes", type=int, default=1)
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--hidden-mlp-dim", type=int, default=768)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--rrwp-steps", type=int, default=20)
    p.add_argument("--gen-every-k", type=int, default=2)
    p.add_argument("--gen-sample-steps", type=int, default=250)
    p.add_argument("--sample-vis-every-k", type=int, default=5)
    p.add_argument("--ckpt-every-n-steps", type=int, default=2000)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    # eval-only
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-ckpt", default=None)
    p.add_argument("--num-eval-samples", type=int, default=1000)
    p.add_argument("--eval-sample-steps", type=int, default=500)
    p.add_argument("--eval-chunk", type=int, default=64)
    # eta/omega sampling sweep
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--sweep-etas", default="0,5,25,50,100")
    p.add_argument("--sweep-omegas", default="0,0.05,0.1")
    p.add_argument("--sweep-samples", type=int, default=500)
    p.add_argument("--sweep-distortion", default="polydec")
    args = p.parse_args()

    if args.sweep:
        assert args.eval_ckpt, "--sweep needs --eval-ckpt"
        sweep(args)
    elif args.eval_only:
        assert args.eval_ckpt, "--eval-only needs --eval-ckpt"
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
