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
from pytorch_lightning.callbacks import ModelCheckpoint

from experiments.utils import (
    build_encoders, smiles_to_pyg_data, make_generation_metrics_fn,
    molecular_metrics, property_distributions,
)
from defog.core import (
    DeFoGModel, TrainingMonitorCallback, SampleVisualizationCallback, EMACallback,
)
from defog.domains import MoleculeDomain

# --- Frozen schema (must match scripts/prepare_chembl.py) -------------------
ATOM_DECODER = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
ATOM_VALENCY = {"C": 4, "N": 3, "O": 2, "F": 1, "B": 3, "Br": 1, "Cl": 1, "I": 1,
                "P": 5, "S": 6, "Se": 2, "Si": 4}
ATOM_WEIGHT = {"C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "B": 10.81,
               "Br": 79.904, "Cl": 35.45, "I": 126.904, "P": 30.974, "S": 32.06,
               "Se": 78.971, "Si": 28.085}
MAX_ATOM_WEIGHT = 700.0

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

    def __init__(self, smiles, atom_encoder, bond_encoder):
        self.smiles = smiles
        self.atom_encoder = atom_encoder
        self.bond_encoder = bond_encoder

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        n = len(self.smiles)
        for off in range(n):
            d = smiles_to_pyg_data(self.smiles[(idx + off) % n],
                                   self.atom_encoder, self.bond_encoder)
            if d is not None:
                return d
        raise RuntimeError("no convertible SMILES")


def build_model(args, stats):
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
        atom_valencies=[ATOM_VALENCY[a] for a in ATOM_DECODER],
        atom_weights=[ATOM_WEIGHT[a] for a in ATOM_DECODER],
        max_atom_weight=MAX_ATOM_WEIGHT,
        lr=args.lr, weight_decay=1e-5, lambda_edge=5.0,
        train_time_distortion="polydec", lr_scheduler="cosine", lr_min=1e-6,
        sample_steps=100, eta=0.0, omega=0.0, sample_time_distortion="polydec",
    )


def train(args):
    pl.seed_everything(args.seed, workers=True)
    ae, ad, be, bd = build_encoders(ATOM_DECODER, BOND_TYPES)

    train_smiles = read_smiles(os.path.join(args.data_dir, "chembl_train.smiles"), args.max_train)
    val_smiles = read_smiles(os.path.join(args.data_dir, "chembl_val.smiles"), args.max_val)
    rprint(f"train {len(train_smiles):,}  val {len(val_smiles):,}")

    train_loader = DataLoader(
        SmilesGraphDataset(train_smiles, ae, be), batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        SmilesGraphDataset(val_smiles, ae, be), batch_size=args.batch_size,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    ) if val_smiles else None

    with open(os.path.join(args.data_dir, "chembl_stats.json")) as fh:
        stats = json.load(fh)
    model = build_model(args, stats)
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

    max_time = None
    if args.max_time_hours:
        h = int(args.max_time_hours)
        max_time = {"hours": h, "minutes": int(round((args.max_time_hours - h) * 60))}

    strategy = ("ddp_find_unused_parameters_true" if args.devices != 1 else "auto")
    trainer = pl.Trainer(
        max_epochs=args.epochs, max_time=max_time, accelerator=args.accelerator,
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
    ae, ad, be, bd = build_encoders(ATOM_DECODER, BOND_TYPES)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeFoGModel.load(args.eval_ckpt.replace(".ckpt", "")).to(device).eval()
    rprint(f"loaded {args.eval_ckpt} on {device}")

    samples = []
    remaining = args.num_eval_samples
    while remaining > 0:
        cur = min(args.eval_chunk, remaining)
        samples += model.sample(num_samples=cur, sample_steps=args.eval_sample_steps,
                                device=device, show_progress=False)
        remaining -= cur

    ref_desc = None
    ref_path = os.path.join(args.data_dir, "chembl_ref_descriptors.npz")
    if os.path.exists(ref_path):
        with np.load(ref_path) as z:
            ref_desc = {k: z[k] for k in z.files}
    train_smiles = set(read_smiles(os.path.join(args.data_dir, "chembl_train.smiles")))
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=os.path.join(_HERE, "data", "chembl"))
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
    args = p.parse_args()

    if args.eval_only:
        assert args.eval_ckpt, "--eval-only needs --eval-ckpt"
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
