#!/usr/bin/env python
"""Package an RL'd AdaLNAdapter checkpoint as an installed molsmith adapter.

The RL experiment writes a bare .ckpt into its pycomex archive; the E2 harness takes a
store reference. `molsmith adapter migrate` does this conversion but no CLI is installed
here, so this calls the same library function.

The head is bundled deliberately: without it `--method fk` on the resulting package would
refuse, and every FK number to date has come from a bundled head. The property stats come
from the same reference split the adapter was conditioned on, so `spec.scale` fills to the
right std and the FK energy is z-scored (see molsmith.sample._energy).

Usage:
    python scripts/package_rl_adapter.py --ckpt <path> --id molsmith/qed --version 5.0.0 \
        --title "QED (head-RL r1, seed 21)" --head ckpts/heads/qed_head.ckpt
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
    from molsmith import migrate, properties, store
    from molsmith.spec import container

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--id", default="molsmith/qed")
    ap.add_argument("--version", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--base", default="molsmith/zinc-kek")
    ap.add_argument("--property", default="qed")
    ap.add_argument("--head", default=None)
    args = ap.parse_args()

    base = store.resolve_package(args.base)
    out = store.cache_dir(store.ensure()) / "migrate" / args.version
    res = migrate.migrate_adapter(
        args.ckpt, pkg_id=args.id, base_metadata=base.metadata,
        adapter_kind="scalar_property", title=args.title, out_dir=out,
        version=args.version, property_id=args.property, head_ckpt=args.head,
    )
    root = res.package_dir
    entry = store.install(str(root), expect_kind="adapter")
    pkg = store.resolve_package(f"{args.id}@{args.version}")
    md = pkg.metadata
    prop = md.property
    print(f"installed {md.coordinates}")
    print(f"  base      {md.base.id}")
    print(f"  head      present={bool(md.head and md.head.present)}")
    print(f"  property  {prop.name if prop else None} "
          f"prop_mean={getattr(prop, 'prop_mean', None)} "
          f"prop_std={getattr(prop, 'prop_std', None)}")
    if not (md.head and md.head.present):
        print("WARNING: no head bundled -- --method fk would refuse on this package")
    if not prop or not prop.prop_std:
        print("WARNING: no prop_std -- the FK energy would run un-normalised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
