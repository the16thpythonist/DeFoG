"""
Measure training throughput and peak memory for one E1 dataset at one batch size.

Sizing a recipe by extrapolating from another dataset does not work. Dense
batches pad to the largest molecule present and the edge tensor goes as n^2, so
the three E1 datasets sit at very different operating points:

    zinc      <=38 heavy atoms    219,568 train
    guacamol  <=72 heavy atoms  1,118,630 train (after the round-trip filter)
    moses     <=27 heavy atoms  1,579,663 train

Whether a given batch fits, and at what step rate, is empirical each time. The
GuacaMol probe (job 1116289) is the worked example: batch 256 turned out to be
no faster than 128 while sitting at 96% of card memory.

Deliberately NOT the full pipeline:

* **A subset, not the whole training set.** Peak memory is set by the largest
  molecule in a batch, so what matters is the size *tail*, which a random draw
  of ~50k contains with near-certainty. The observed max node count is reported
  so the assumption stays checkable.
* **No round-trip filter** (GuacaMol only). It changes dataset *size*, not step
  throughput, and its effect is already measured -- so steps/epoch is projected
  arithmetically instead.

Usage:
    python scripts/probe_throughput.py --dataset moses --batch-size 512
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

import torch

from defog.core import DeFoGModel
from defog.data import guacamol_reference as gmref
from defog.data import moses_reference as mref
from defog.data import zinc_reference as zref

# GuacaMol's round-trip filter keeps this fraction of train; measured on 50k and
# confirmed at full scale (0.8787 over 1,273,104). ZINC and MOSES apply no filter.
_GUACAMOL_FILTER_KEEP = 0.8787

_ZINC_VALENCY = {"C": 4, "N": 3, "O": 2, "F": 1, "P": 5, "S": 6, "Cl": 1, "Br": 1, "I": 1}
_ZINC_WEIGHT = {"C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "P": 30.974,
                "S": 32.06, "Cl": 35.45, "Br": 79.904, "I": 126.904}
_GM_VALENCY = {"C": 4, "N": 3, "O": 2, "F": 1, "B": 3, "Br": 1,
               "Cl": 1, "I": 1, "P": 5, "S": 6, "Se": 2, "Si": 4}
_GM_WEIGHT = {"C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "B": 10.81,
              "Br": 79.904, "Cl": 35.45, "I": 126.904, "P": 30.974, "S": 32.06,
              "Se": 78.971, "Si": 28.085}

DATASETS = {
    "zinc": dict(mod=zref, valency=_ZINC_VALENCY, weight=_ZINC_WEIGHT,
                 max_weight=500.0, keep=1.0),
    "guacamol": dict(mod=gmref, valency=_GM_VALENCY, weight=_GM_WEIGHT,
                     max_weight=1000.0, keep=_GUACAMOL_FILTER_KEEP),
    "moses": dict(mod=mref, valency=mref.ATOM_VALENCY, weight=mref.ATOM_WEIGHT,
                  max_weight=mref.MAX_ATOM_WEIGHT, keep=1.0),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--subset", type=int, default=50000)
    ap.add_argument("--warmup-steps", type=int, default=30)
    ap.add_argument("--timed-steps", type=int, default=150)
    ap.add_argument("--n-layers", type=int, default=9)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--hidden-mlp-dim", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--rrwp-steps", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link-hours", type=float, default=10.0,
                    help="Used only to project epochs per chained link.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    mod = cfg["mod"]
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dataset={args.dataset} device={device} batch_size={args.batch_size}", flush=True)

    split = mod.load_reference_split()
    train = split.train_smiles
    n_train_full = len(train)
    if args.subset and len(train) > args.subset:
        train = random.Random(args.seed).sample(train, args.subset)
    print(f"subset {len(train)} of {n_train_full} train molecules", flush=True)

    t0 = time.time()
    built = mod.build_graphs(train)          # zinc/moses -> 3-tuple, guacamol -> 4
    graphs = built[0]
    encode_s = time.time() - t0
    sizes = sorted(g.x.size(0) for g in graphs)
    print(f"encoded {len(graphs)} graphs in {encode_s:.0f}s "
          f"(nodes min={sizes[0]} p50={sizes[len(sizes)//2]} max={sizes[-1]})", flush=True)

    from torch_geometric.loader import DataLoader

    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, persistent_workers=False,
                        drop_last=True)

    atom_types = list(mod.ATOM_TYPES)
    model = DeFoGModel.from_dataloader(
        loader,
        n_layers=args.n_layers, hidden_dim=args.hidden_dim,
        hidden_mlp_dim=args.hidden_mlp_dim, n_heads=args.n_heads,
        dropout=0.1, noise_type="marginal", extra_features_type="rrwp",
        rrwp_steps=args.rrwp_steps, molecular_features=True,
        atom_valencies=[cfg["valency"][a] for a in atom_types],
        atom_weights=[cfg["weight"][a] for a in atom_types],
        max_atom_weight=cfg["max_weight"],
        lr=2e-4, weight_decay=1e-5, lambda_edge=5.0,
        train_time_distortion="polydec",
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    done, timed_start, oom = 0, None, None
    batch_max_nodes = 0
    try:
        while done < args.warmup_steps + args.timed_steps:
            for batch in loader:
                batch = batch.to(device)
                batch_max_nodes = max(batch_max_nodes,
                                      int(torch.bincount(batch.batch).max()))
                loss = model.training_step(batch, done)
                if isinstance(loss, dict):
                    loss = loss["loss"]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                done += 1
                if done == args.warmup_steps:
                    if device == "cuda":
                        torch.cuda.synchronize()
                    timed_start = time.time()
                if done >= args.warmup_steps + args.timed_steps:
                    break
    except torch.cuda.OutOfMemoryError as exc:
        oom = str(exc)[:200]
        print(f"OOM at batch_size={args.batch_size}: {oom}", flush=True)

    result = {
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "oom": oom is not None,
        "n_params": n_params,
        "subset": len(graphs),
        "max_nodes_in_subset": sizes[-1],
        "max_nodes_in_a_batch": batch_max_nodes,
        "encode_seconds": round(encode_s, 1),
    }

    if oom is None and timed_start is not None:
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - timed_start
        it_s = args.timed_steps / elapsed
        train_effective = int(n_train_full * cfg["keep"])
        steps_per_epoch = train_effective // args.batch_size
        result.update({
            "timed_steps": args.timed_steps,
            "elapsed_s": round(elapsed, 1),
            "it_per_s": round(it_s, 3),
            "s_per_step": round(1 / it_s, 4),
            # BOTH numbers, because they differ a lot and only one predicts OOM.
            # max_memory_allocated counts live tensors; the caching allocator's
            # RESERVED pool (plus CUDA context) is what nvidia-smi shows and what
            # exhausts the card. On GuacaMol leg 1 allocated peaked at 38.6 GB
            # while reserved transiently hit 93.7 of 95.6.
            "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                                     if device == "cuda" else None,
            "peak_gpu_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)
                                    if device == "cuda" else None,
            "projected": {
                "train_effective": train_effective,
                "steps_per_epoch": steps_per_epoch,
                "epoch_hours": round(steps_per_epoch / it_s / 3600, 3),
                "epochs_per_link": round(args.link_hours * 3600 * it_s / steps_per_epoch, 1),
                "link_hours": args.link_hours,
            },
        })

    print("\n" + json.dumps(result, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
