#!/usr/bin/env python
"""Export a molsmith adapter package to the ``AdaLNAdapter.load`` ckpt format.

The RL experiment takes ``--ADAPTER_CKPT`` and calls ``AdaLNAdapter.load`` on it, but the
adapters now live only as packages (safetensors + metadata.yml) -- the original ckpts were
retired. This turns one back into the other.

It goes through molsmith's ``load_adapter`` rather than reading the safetensors directly,
so the package's compatibility verdict against the base runs on the way: an adapter that
does not bind to the base it is about to be fine-tuned against is refused here rather than
producing plausible-looking nonsense 12 hours later.

The round-trip is verified bit-for-bit before the file is accepted, because a silently
mangled export is indistinguishable from a bad RL run.

Usage:
    python scripts/export_adapter_ckpt.py --adapter molsmith/qed@2.0.0 \
        --base molsmith/zinc-kek --out ckpts/qed_adapter_pre_rl.ckpt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for extra in ("/media/ssd2/Programming/defog-web", "/home/tm4030/Programming/defog-web"):
    if Path(extra).is_dir():
        sys.path.insert(0, extra)


def main() -> int:
    import torch

    from molsmith import store
    from molsmith.weights import load as weights_load

    from defog.core import AdaLNAdapter

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    base_pkg = store.resolve_package(args.base)
    adapter_pkg = store.resolve_package(args.adapter)
    print(f"base    {base_pkg.metadata.coordinates}")
    print(f"adapter {adapter_pkg.metadata.coordinates}  "
          f"(source_checkpoint={(adapter_pkg.metadata.model_extra or {}).get('training', {}).get('source_checkpoint')})")

    base_module = weights_load.load_base(base_pkg, device=args.device)
    adapter = weights_load.load_adapter(adapter_pkg, base_pkg, base_module, device=args.device)
    print(f"loaded adapter: cond_dim={adapter.cond_dim}  "
          f"cond_mean={adapter.cond_mean.tolist()}  cond_std={adapter.cond_std.tolist()}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    adapter.save(str(out))

    # Bit-for-bit round trip. A mangled export and a bad RL run look the same from the logs.
    back = AdaLNAdapter.load(str(out), device=args.device)
    a, b = adapter.state_dict(), back.state_dict()
    if a.keys() != b.keys():
        print(f"ROUND TRIP FAILED: key mismatch {set(a) ^ set(b)}")
        return 1
    for k in a:
        if not torch.equal(a[k].cpu(), b[k].cpu()):
            print(f"ROUND TRIP FAILED: tensor {k} differs")
            return 1
    n = sum(v.numel() for v in a.values())
    print(f"round trip OK: {len(a)} tensors, {n:,} parameters identical")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
