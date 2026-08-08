#!/usr/bin/env python
"""
Why do this model's invalid samples fail? Categorise, rather than guess.

Motivation: the MOSES E1 base sits at ~0.90 validity while GuacaMol reaches
~0.98 -- despite MOSES being the easier dataset (smaller molecules, 8 atom
types against 12, curated and filtered). That ~8-point deficit is larger than
anything RL has been able to win back, so knowing its cause is worth more than
another reward-shaping round. RL is currently fighting for 3-5 points on top of
a base that may be leaving 8 on the table.

The evaluation path reports only valid/invalid, which cannot distinguish
"the graph has an impossible valence" from "the ring system will not kekulize".
Those have completely different fixes -- the first is a capacity or training
problem, the second is a representation problem (MOSES and GuacaMol train on
aromatic bonds; ZINC trains kekulized and reaches 0.99).

So this decodes to an *unsanitized* molecule, then sanitizes separately and
classifies the RDKit failure. Categories:

    kekulize        aromatic ring system cannot be kekulized
    valence         an atom exceeds its permitted valence
    other_sanitize  any other sanitization failure, message recorded
    decode_failed   graph -> RWMol construction failed outright
    disconnected    sanitizes, but is a multi-fragment molecule
    ok              sanitizes and is a single connected molecule

Usage:
    python scripts/diagnose_validity.py --ckpt <path> --dataset moses --n 1024
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

from defog.core import DeFoGModel  # noqa: E402
from defog.data import vocabulary  # noqa: E402
from defog.domains.molecule import pyg_data_to_mol  # noqa: E402

RDLogger.DisableLog("rdApp.*")

REFERENCES = {"zinc": "zinc_reference", "guacamol": "guacamol_reference",
              "moses": "moses_reference", "chembl": "chembl_reference"}


def classify(mol):
    """(category, detail) for one decoded, unsanitized RWMol."""
    if mol is None:
        return "decode_failed", ""
    try:
        probe = Chem.Mol(mol)
        Chem.SanitizeMol(probe)
    except Chem.rdchem.KekulizeException as exc:
        return "kekulize", str(exc)[:120]
    except Chem.rdchem.AtomValenceException as exc:
        return "valence", str(exc)[:120]
    except Exception as exc:                       # noqa: BLE001
        return "other_sanitize", f"{type(exc).__name__}: {exc}"[:120]
    smi = Chem.MolToSmiles(probe)
    if not smi:
        return "other_sanitize", "empty SMILES"
    if "." in smi:
        return "disconnected", ""
    return "ok", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True, choices=sorted(REFERENCES))
    ap.add_argument("--representation", default=None,
                    help="MOSES only: 'aromatic_v1' (default) or 'kekulized_v2'. "
                         "Must match what the checkpoint was trained with -- the "
                         "wrong one mis-decodes silently rather than erroring.")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eta", type=float, default=25.0)
    ap.add_argument("--omega", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--seed", type=int, default=None,
                    help="Torch seed for sampling. Set it to make a run "
                         "reproducible, or to draw a controlled REPLICATE of the "
                         "same model -- two seeds on one checkpoint measure the "
                         "sampling noise directly, which is the only honest way "
                         "to know whether a difference between two models means "
                         "anything.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-smiles", default=None,
                    help="write the valid canonical SMILES here, for external "
                         "FCD/NSPDK scoring in the metrics environment")
    args = ap.parse_args()

    import importlib
    mod = importlib.import_module(f"defog.data.{REFERENCES[args.dataset]}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"sampling seed {args.seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeFoGModel.load(args.ckpt).to(device)
    model.eval()

    # Decoding with the wrong vocabulary produces plausible-looking garbage
    # rather than an error, so this refuses rather than reporting numbers that
    # describe nothing.
    try:
        (atom_types, bond_types, atom_decoder, bond_decoder,
         rep, msg) = vocabulary.resolve_and_check(mod, model, args.representation)
    except vocabulary.VocabularyMismatch as exc:
        sys.exit(f"VOCABULARY MISMATCH: {exc}")
    print(f"{args.dataset}: atoms={atom_types} bonds={bond_types}"
          + (f" [representation={rep.name}]" if rep else ""))
    print(msg)
    print(f"loaded {args.ckpt} on {device}; sampling {args.n} at "
          f"steps={args.steps} eta={args.eta}")

    samples, remaining = [], args.n
    while remaining > 0:
        cur = min(args.chunk, remaining)
        samples += model.sample(num_samples=cur, sample_steps=args.steps,
                                eta=args.eta, omega=args.omega, device=device,
                                show_progress=False)
        remaining -= cur

    counts = collections.Counter()
    details = collections.Counter()
    ring_sizes = collections.Counter()
    smiles_out = []
    for data in samples:
        mol = pyg_data_to_mol(data, atom_decoder, bond_decoder,
                              charge_correction=True)
        cat, detail = classify(mol)
        counts[cat] += 1
        if cat in ("ok", "disconnected"):
            probe = Chem.Mol(mol)
            Chem.SanitizeMol(probe)
            smiles_out.append(Chem.MolToSmiles(probe))
        if detail:
            details[detail] += 1
        # Ring census on whatever RDKit could read, sanitized or not -- a
        # kekulization failure is usually about ring size or fusion, so the
        # distribution is the follow-up question.
        if mol is not None and cat in ("kekulize", "valence"):
            try:
                # FastFindRings must run BEFORE GetRingInfo: on an unsanitized
                # molecule ring perception has not happened, and reading
                # RingInfo first returns uninitialised garbage (it reported a
                # "177404-ring" the first time this ran).
                probe = Chem.Mol(mol)
                Chem.FastFindRings(probe)
                for r in probe.GetRingInfo().AtomRings():
                    ring_sizes[len(r)] += 1
            except Exception:                       # noqa: BLE001
                pass

    total = sum(counts.values())
    print("\n" + "=" * 60)
    print(f"{'category':18s}{'count':>8s}{'share':>10s}")
    print("=" * 60)
    for cat, c in counts.most_common():
        print(f"{cat:18s}{c:>8d}{c / total:>10.4f}")
    invalid = total - counts["ok"] - counts["disconnected"]
    print("-" * 60)
    print(f"{'valid+connected':18s}{counts['ok']:>8d}{counts['ok'] / total:>10.4f}")
    print(f"{'hard-invalid':18s}{invalid:>8d}{invalid / total:>10.4f}")

    if details:
        print("\ntop failure messages:")
        for msg, c in details.most_common(6):
            print(f"  {c:5d}  {msg}")
    if ring_sizes:
        print("\nring sizes among failing molecules:")
        for size, c in sorted(ring_sizes.items()):
            print(f"  {size}-ring: {c}")

    if smiles_out:
        uniq = len(set(smiles_out)) / len(smiles_out)
        print(f"\nuniqueness among valid: {uniq:.4f}  (n={len(smiles_out)})")

    if args.dump_smiles:
        Path(args.dump_smiles).write_text("\n".join(smiles_out) + "\n")
        print(f"wrote {len(smiles_out)} SMILES -> {args.dump_smiles}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "ckpt": args.ckpt, "dataset": args.dataset, "n": total,
            "representation": (rep.name if rep else None),
            "atom_types": atom_types, "bond_types": bond_types,
            "sampling_seed": args.seed,
            "config": {"steps": args.steps, "eta": args.eta, "omega": args.omega},
            "counts": dict(counts), "top_messages": details.most_common(20),
            "ring_sizes_in_failures": dict(ring_sizes),
        }, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
