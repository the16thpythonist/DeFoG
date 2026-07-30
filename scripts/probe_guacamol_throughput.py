"""
Measure GuacaMol training throughput and peak memory at one batch size.

Sizing the GuacaMol recipe by extrapolating from ZINC does not work. Dense
batches pad to the largest molecule present, and the edge tensor goes as n^2:
ZINC caps at 38 heavy atoms, GuacaMol reaches 72, so the edge cost per graph is
~3.6x on ~5.8x as many molecules. Whether that fits at batch 256 on a 96 GB
GH200, and at what step rate, is an empirical question.

Deliberately NOT the full pipeline:

* **A subset, not all 1.27M molecules.** Peak memory is set by the largest
  molecule in a batch, so what the probe needs is the size *tail*, which a
  random draw of ~50k from 1.27M contains with near-certainty. Encoding the full
  set would cost 30-50 min of CPU and measure nothing extra. The observed max
  node count is reported so this assumption is checkable rather than implied.
* **No round-trip filter.** It costs an extra decode per molecule and changes
  only dataset *size*, not step throughput. Its effect is already measured
  (12.2% dropped), so steps/epoch is computed arithmetically below.

Usage (one GPU, one batch size):
    python scripts/probe_guacamol_throughput.py --batch-size 128 --subset 50000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

import torch

from defog.core import DeFoGModel
from defog.data import guacamol_reference as gm

# 12 elements, matching the frozen GuacaMol vocabulary.
ATOM_VALENCY = {"C": 4, "N": 3, "O": 2, "F": 1, "B": 3, "Br": 1,
                "Cl": 1, "I": 1, "P": 5, "S": 6, "Se": 2, "Si": 4}
ATOM_WEIGHT = {"C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "B": 10.81,
               "Br": 79.904, "Cl": 35.45, "I": 126.904, "P": 30.974,
               "S": 32.06, "Se": 78.971, "Si": 28.085}
MAX_ATOM_WEIGHT = 1000.0  # GuacaMol reaches ~72 heavy atoms

#: Measured on 50,000 molecules of the official train split; see
#: defog/data/guacamol_reference.build_graphs.
FILTER_KEEP_FRACTION = 0.87778


def main() -> int:
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} batch_size={args.batch_size}", flush=True)

    split = gm.load_reference_split()
    train = split.train_smiles
    if args.subset and len(train) > args.subset:
        train = random.Random(args.seed).sample(train, args.subset)
    print(f"subset {len(train)} of {split.n_train} train molecules", flush=True)

    t0 = time.time()
    graphs, _, _, stats = gm.build_graphs(train, filter_roundtrip=False)
    encode_s = time.time() - t0
    sizes = [g.x.size(0) for g in graphs]
    max_nodes = max(sizes)
    print(f"encoded {len(graphs)} graphs in {encode_s:.0f}s "
          f"(nodes min={min(sizes)} p50={sorted(sizes)[len(sizes)//2]} max={max_nodes}) "
          f"skipped={stats['encode_failed']}", flush=True)

    from torch_geometric.loader import DataLoader

    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, persistent_workers=False,
                        drop_last=True)

    model = DeFoGModel.from_dataloader(
        loader,
        n_layers=args.n_layers, hidden_dim=args.hidden_dim,
        hidden_mlp_dim=args.hidden_mlp_dim, n_heads=args.n_heads,
        dropout=0.1, noise_type="marginal", extra_features_type="rrwp",
        rrwp_steps=args.rrwp_steps, molecular_features=True,
        atom_valencies=[ATOM_VALENCY[a] for a in gm.ATOM_TYPES],
        atom_weights=[ATOM_WEIGHT[a] for a in gm.ATOM_TYPES],
        max_atom_weight=MAX_ATOM_WEIGHT,
        lr=2e-4, weight_decay=1e-5, lambda_edge=5.0,
        train_time_distortion="polydec",
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Manual loop rather than pl.Trainer: the probe measures the forward/backward
    # cost, and a Trainer would fold in callbacks, logging and validation that
    # the projection below should not inherit.
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
        "batch_size": args.batch_size,
        "oom": oom is not None,
        "n_params": n_params,
        "subset": len(graphs),
        "max_nodes_in_subset": max_nodes,
        "max_nodes_in_a_batch": batch_max_nodes,
        "encode_seconds": round(encode_s, 1),
    }

    if oom is None and timed_start is not None:
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - timed_start
        it_s = args.timed_steps / elapsed
        # Project onto the REAL training set: filtered size, not the probe subset.
        train_filtered = int(split.n_train * FILTER_KEEP_FRACTION)
        steps_per_epoch = train_filtered // args.batch_size
        result.update({
            "timed_steps": args.timed_steps,
            "elapsed_s": round(elapsed, 1),
            "it_per_s": round(it_s, 3),
            "s_per_step": round(1 / it_s, 4),
            # BOTH numbers, because they differ by a lot and only one predicts
            # OOM. max_memory_allocated counts live tensors; the caching
            # allocator's RESERVED pool (plus CUDA context) is what nvidia-smi
            # shows and what actually exhausts the card. On the first GuacaMol
            # leg, allocated peaked at 38.6 GB while reserved transiently hit
            # 93.7 GB of 95.6 -- reporting only the former made batch 128 look
            # like it had 2.4x the headroom it really had.
            "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                                     if device == "cuda" else None,
            "peak_gpu_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)
                                    if device == "cuda" else None,
            "projected": {
                "train_after_filter": train_filtered,
                "steps_per_epoch": steps_per_epoch,
                "epoch_hours": round(steps_per_epoch / it_s / 3600, 2),
                "epochs_per_10h_link": round(10.0 * 3600 * it_s / steps_per_epoch, 1),
            },
        })

    print("\n" + json.dumps(result, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
