"""
Can Feynman-Kac / SMC steering beat the single-trajectory guidance ceiling?

For the HIGH logP target (dataset p90), with the same tight conditional-size
distribution (bw=0.5) throughout, compare:
  (b) guidance-only        : ExactGuidance reweight (our best, ~4.25)
  (c) FK-SMC (base)        : frozen base proposal + reward resampling
  (d) FK-SMC (guided)      : guided proposal + reward resampling

Reward energy on the predicted clean molecule: (logP - target)^2.
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from rdkit.Chem import Crippen

from defog.core import (
    DeFoGModel, GuidedSampler, ExactGuidance, FeynmanKacSampler,
    MoleculePropertyEnergy, ConditionalSizeDistribution,
)
from defog.domains import MoleculeDomain
from defog.domains.molecule import build_encoders, pyg_data_to_mol, mol_to_smiles
from experiments.guided_logp_demo import derive_atom_types, build_dataset, BOND_TYPES

RDLogger.DisableLog("rdApp.*")


@torch.no_grad()
def collect(sampler, n, chunk, size_dist, target, device, ad, bd):
    logps = []
    remaining = n
    while remaining > 0:
        cur = min(chunk, remaining)
        cond = torch.full((cur, 1), float(target))
        for s in sampler.sample(cur, size_dist=size_dist, condition=cond,
                                device=device, show_progress=False):
            mol = pyg_data_to_mol(s, ad, bd)
            if mol is not None and mol_to_smiles(mol) is not None:
                try:
                    logps.append(float(Crippen.MolLogP(mol)))
                except Exception:
                    pass
        remaining -= cur
    return np.array(logps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/Downloads/aqsoldb_4e-4_best_model.ckpt"))
    ap.add_argument("--guidance", default="experiments/_guided_logp_out/guided_logp_amortized.ckpt")
    ap.add_argument("--data", default="data/aqsoldb_conditional.csv")
    ap.add_argument("--level", default="high", choices=["low", "med", "high"])
    ap.add_argument("--num-eval", type=int, default=90)
    ap.add_argument("--particles", type=int, default=30, help="K per SMC run (= chunk)")
    ap.add_argument("--sample-steps", type=int, default=250)
    ap.add_argument("--eta", type=float, default=2.0)
    ap.add_argument("--omega", type=float, default=0.3)
    ap.add_argument("--weight", type=float, default=2.0, help="guided-proposal guidance weight")
    ap.add_argument("--beta", type=float, default=1.5, help="FK reward tilt strength")
    ap.add_argument("--resample-interval", type=int, default=25)
    ap.add_argument("--size-bandwidth", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.data)
    atom_types = derive_atom_types(df["smiles"])
    ae, ad, be, bd = build_encoders(atom_types, BOND_TYPES)
    logp_all = df["logp"].values.astype(float)
    prop_mean, prop_std = float(logp_all.mean()), float(logp_all.std())
    target = float(np.percentile(logp_all, {"low": 10, "med": 50, "high": 90}[args.level]))

    graphs, _, _ = build_dataset(df, ae, be)
    conds = torch.tensor([[float(g.prop_val)] for g in graphs])
    sizes = torch.tensor([int(g.x.size(0)) for g in graphs])
    size_dist = ConditionalSizeDistribution(conds, sizes, method="kernel", bandwidth=args.size_bandwidth)

    base = DeFoGModel.load(args.ckpt, device="cpu").to(device).eval()
    h = DeFoGModel.load(args.guidance, device="cpu").to(device).eval()
    domain = MoleculeDomain(ad, bd)
    energy_fn = MoleculePropertyEnergy(domain, Crippen.MolLogP, target)

    guidance = ExactGuidance(h, prop_mean=prop_mean, prop_std=prop_std,
                             weight=args.weight).set_target(target)

    print(f"[smc] level={args.level} target={target:.2f} K={args.particles} beta={args.beta} "
          f"resample_every={args.resample_interval} steps={args.sample_steps}", flush=True)

    sk = dict(eta=args.eta, omega=args.omega, sample_steps=args.sample_steps, time_distortion="polydec")
    fk = dict(beta=args.beta, resample_interval=args.resample_interval, **sk)

    runs = {
        "b_guidance_only": GuidedSampler(base, guidance, **sk),
        "c_fk_base":       FeynmanKacSampler(base, energy_fn, proposal_transform=None, **fk),
        "d_fk_guided":     FeynmanKacSampler(base, energy_fn, proposal_transform=guidance.reweight, **fk),
    }
    for name, sampler in runs.items():
        lps = collect(sampler, args.num_eval, args.particles, size_dist, target, device, ad, bd)
        if len(lps):
            print(f"[smc] {name:16s} achieved_mean={lps.mean():+.2f}  median={np.median(lps):+.2f}  "
                  f"MAE={np.mean(np.abs(lps-target)):.2f}  valid={len(lps)}/{args.num_eval}  "
                  f"frac>=4={np.mean(lps>=4.0):.0%}", flush=True)
        else:
            print(f"[smc] {name:16s} no valid samples", flush=True)
    print("[smc] DONE", flush=True)


if __name__ == "__main__":
    main()
