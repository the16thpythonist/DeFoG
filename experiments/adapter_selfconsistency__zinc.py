"""PROTOTYPE: train a ClogP interior adapter WITH a learned property head H and a
SELF-CONSISTENCY loss, to (a) tighten conditioning passively and (b) yield a learned
property predictor usable inside FK (no RDKit).

Losses (base frozen; adapter + H trained):
  L_denoise : the usual conditional denoising CE (adapter).
  L_ground  : H(true clean graph) -> true property  (grounds H as a real structure->property
              predictor; H never sees the condition, so no leakage).
  L_sc      : H(adapter's SOFT predicted-clean) -> condition, with H's PARAMS DETACHED so the
              gradient flows only into the ADAPTER (self-consistency: generate on-target).
  total = L_denoise + lambda_ground*L_ground + lambda_sc*L_sc

H is a small message-passing head (soft-graph capable, sum-pooled) -- NOT base-feature
pooling, because L_sc must backprop through soft graphs (base RRWP features can't).

Eval: adapter steering MAE at w=1 (does self-consistency tighten it?) + H-vs-RDKit MAE
(is H an accurate predictor, i.e. FK-usable?).

Usage:
    python experiments/adapter_selfconsistency__zinc.py --__TESTING__ True
    python experiments/adapter_selfconsistency__zinc.py --LAMBDA_SC 1.0
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
import pytorch_lightning as pl
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from experiments.utils import build_encoders, smiles_to_pyg_data, pyg_data_to_mol, mol_to_smiles
from defog.core import (DeFoGModel, AdaLNAdapter, AdapterComposition, ConditionBranch,
                        AdaptedSampler)
from defog.core.data import to_dense
from defog.core.callbacks import EMACallback

RDLogger.DisableLog("rdApp.*")
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROP_FNS = {"logp": lambda m: float(Crippen.MolLogP(m)), "tpsa": lambda m: float(Descriptors.TPSA(m))}

# ============================================================================
CSV_PATH: str = os.path.join(_PROJECT_DIR, "data", "zinc_250k_rdkit.csv")
SMILES_COLUMN: str = "smiles"
BOND_TYPES: list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
ATOM_TYPES: list = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
BASE_CKPT: str = os.path.expanduser("~/Downloads/zinc_uncond_4e-4_connectivity.ckpt")
PROPERTY: str = "logp"

# adapter
H_HIDDEN: int = 256
STREAMS: list = ["X", "E", "y"]
INTERIOR_FF: bool = True
INTERIOR_ATTN: bool = True
L10_LR_SCALE: float = 0.3

# property head H
HEAD_HIDDEN: int = 128
HEAD_LAYERS: int = 3
LR_HEAD: float = 1e-3

# loss weights
LAMBDA_GROUND: float = 1.0
LAMBDA_SC: float = 1.0          # swept per-arm: the self-consistency strength

# training
EPOCHS: int = 20
BATCH_SIZE: int = 24
LEARNING_RATE: float = 2e-4
MAX_TIME_HOURS: float = 5.0

# eval
EVAL_STEPS: int = 500
ETA: float = 5.0
OMEGA: float = 0.0
TIME_DISTORTION: str = "polydec"
TARGET_PERCENTILES: list = [5, 95]
LEVEL_NAMES: list = ["low", "high"]
N_PER_TARGET: int = 128
EVAL_CHUNK: int = 32
HEAD_EVAL_N: int = 1000         # held-out mols for H-vs-RDKit accuracy
SEED: int = 42
__DEBUG__: bool = False
__TESTING__: bool = False


# ---------------------------------------------------------------------------
class PropertyHead(nn.Module):
    """Soft-graph message-passing head: (X,E,node_mask) -> normalized property (bs,).
    SUM pooling suits ~additive properties (logP/TPSA)."""

    def __init__(self, na, nb, hid=128, layers=3):
        super().__init__()
        self.xin = nn.Linear(na, hid)
        self.ein = nn.Linear(nb, hid)
        self.msg = nn.ModuleList([nn.Linear(2 * hid, hid) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Linear(hid, hid) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.act = nn.SiLU()
        self.out = nn.Sequential(nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, 1))

    def forward(self, X, E, node_mask):
        bs, n, _ = X.shape
        m = node_mask.float().unsqueeze(-1)                     # (bs,n,1)
        em = m.unsqueeze(2) * m.unsqueeze(1)                    # (bs,n,n,1)
        h = self.act(self.xin(X)) * m
        e = self.act(self.ein(E)) * em
        for msg, upd, norm in zip(self.msg, self.upd, self.norm):
            hj = h.unsqueeze(1).expand(bs, n, n, h.size(-1))    # h_j
            mij = self.act(msg(torch.cat([e, hj], -1))) * em    # message per (i,j)
            agg = mij.sum(2)                                    # sum over neighbors j
            h = norm(h + upd(agg)) * m
        return self.out(h.sum(1)).squeeze(-1)                   # SUM pool -> (bs,)


def head_frozen(head, X, E, mask):
    """H(...) with params DETACHED -> gradient flows only to the inputs (the adapter)."""
    pd_ = {k: v.detach() for k, v in head.named_parameters()}
    bd_ = {k: v.detach() for k, v in head.named_buffers()}
    return functional_call(head, {**pd_, **bd_}, (X, E, mask))


class SelfConsistModule(pl.LightningModule):
    def __init__(self, base, adapter, head, prop_mean, prop_std, lr=2e-4, lr_head=1e-3,
                 l10_lr_scale=0.3, lambda_ground=1.0, lambda_sc=1.0):
        super().__init__()
        self.base = base.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.adapter, self.head = adapter, head
        self.pm, self.ps = float(prop_mean), float(prop_std)
        self.lr, self.lr_head, self.l10s = lr, lr_head, l10_lr_scale
        self.lg, self.lsc = float(lambda_ground), float(lambda_sc)

    def _dense(self, batch):
        dense, mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        dense = dense.mask(mask)
        return dense.X, dense.E, mask

    def training_step(self, batch, _):
        self.base.eval()
        X1, E1, node_mask = self._dense(batch)
        bs, device = X1.size(0), X1.device
        c = batch.cond.to(device).view(bs, -1).float()                   # raw (bs,1)
        c_norm = ((c.view(bs) - self.pm) / self.ps)                       # (bs,)
        y0 = torch.zeros(bs, 0, device=device)
        with torch.no_grad():
            noisy = self.base._apply_noise(X1, E1, y0, node_mask)
            extra = self.base._compute_extra_data(noisy)
        mod = self.adapter(c, t=noisy["t"])
        pred = self.base.forward(noisy, extra, node_mask, cond_modulation=mod)
        L_denoise = self.base.train_loss(pred_X=pred.X, pred_E=pred.E, pred_y=pred.y,
                                         true_X=X1, true_E=E1, true_y=y0, node_mask=node_mask)
        # grounding: H on the TRUE clean graph -> true (normalized) property
        L_ground = F.mse_loss(self.head(X1, E1, node_mask), c_norm)
        # self-consistency: H (params detached) on the adapter's SOFT predicted-clean -> condition
        pX, pE = F.softmax(pred.X, -1), F.softmax(pred.E, -1)
        L_sc = F.mse_loss(head_frozen(self.head, pX, pE, node_mask), c_norm)
        loss = L_denoise + self.lg * L_ground + self.lsc * L_sc
        self.log_dict({"loss": loss, "denoise": L_denoise, "ground": L_ground, "sc": L_sc},
                      prog_bar=True, on_epoch=True, batch_size=bs)
        return loss

    def configure_optimizers(self):
        a = self.adapter
        groups = [{"params": self.head.parameters(), "lr": self.lr_head}]
        if a.interior_attn and self.l10s != 1.0:
            l10 = {id(p) for p in a.interior_attn_parameters()}
            groups += [{"params": [p for p in a.parameters() if id(p) not in l10], "lr": self.lr},
                       {"params": [p for p in a.parameters() if id(p) in l10], "lr": self.lr * self.l10s}]
        else:
            groups += [{"params": a.parameters(), "lr": self.lr}]
        return torch.optim.AdamW(groups, weight_decay=1e-5)


# ---------------------------------------------------------------------------
def props_of(samples, ad, bd, fn):
    vals = []
    for s in samples:
        mol = pyg_data_to_mol(s, ad, bd)
        smi = mol_to_smiles(mol) if mol is not None else None
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            continue
        try:
            vals.append(fn(m))
        except Exception:
            pass
    return np.asarray(vals, dtype=float)


@Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())
def experiment(e: Experiment) -> None:
    e.log(f"self-consistency adapter+head: property={e.PROPERTY} lambda_ground={e.LAMBDA_GROUND} "
          f"lambda_sc={e.LAMBDA_SC}")
    pl.seed_everything(e.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fn = PROP_FNS[e.PROPERTY]
    atom_enc, atom_dec, bond_enc, bond_dec = build_encoders(e.ATOM_TYPES, e.BOND_TYPES)

    df = pd.read_csv(e.CSV_PATH)
    graphs, vals = [], []
    for smi in df[e.SMILES_COLUMN]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        d = smiles_to_pyg_data(smi, atom_enc, bond_enc)
        if d is None:
            continue
        try:
            v = fn(mol)
        except Exception:
            continue
        d.cond = torch.tensor([[v]], dtype=torch.float)
        graphs.append(d); vals.append(v)
    vals = np.asarray(vals)
    cond_mean, cond_std = float(vals.mean()), float(vals.std() or 1.0)
    e.log(f"{len(graphs)} graphs; {e.PROPERTY} mean={cond_mean:.2f} std={cond_std:.2f}")
    n_hold = min(e.HEAD_EVAL_N, len(graphs) // 10)
    hold_graphs = graphs[-n_hold:]
    train_graphs = graphs[:-n_hold]

    base = DeFoGModel.load(e.BASE_CKPT, device="cpu").to(device).eval()
    assert base.cond_dim == 0
    adapter = AdaLNAdapter.for_base(base, cond_dim=1, hidden=e.H_HIDDEN, streams=tuple(e.STREAMS),
                                    cond_mean=[cond_mean], cond_std=[cond_std],
                                    interior_ff=e.INTERIOR_FF, interior_attn=e.INTERIOR_ATTN,
                                    name=f"{e.PROPERTY}_adapter_sc", cond_type=e.PROPERTY).to(device)
    na = len(e.ATOM_TYPES); nb = len(e.BOND_TYPES) + 1
    head = PropertyHead(na, nb, hid=e.HEAD_HIDDEN, layers=e.HEAD_LAYERS).to(device)
    e.log(f"adapter {sum(p.numel() for p in adapter.parameters()):,} params | "
          f"head {sum(p.numel() for p in head.parameters()):,} params (na={na} nb={nb})")

    module = SelfConsistModule(base, adapter, head, cond_mean, cond_std, lr=e.LEARNING_RATE,
                               lr_head=e.LR_HEAD, l10_lr_scale=e.L10_LR_SCALE,
                               lambda_ground=e.LAMBDA_GROUND, lambda_sc=e.LAMBDA_SC)

    from torch_geometric.loader import DataLoader
    loader = DataLoader(train_graphs, batch_size=e.BATCH_SIZE, shuffle=True)
    e.log(f"Training: epochs<={e.EPOCHS} batch={e.BATCH_SIZE} lr={e.LEARNING_RATE} "
          f"lr_head={e.LR_HEAD} max_time={e.MAX_TIME_HOURS}h")
    trainer = pl.Trainer(max_epochs=e.EPOCHS, accelerator=("gpu" if device == "cuda" else "cpu"),
                         devices=1, logger=False, enable_checkpointing=False,
                         max_time={"hours": e.MAX_TIME_HOURS}, enable_progress_bar=False,
                         callbacks=[EMACallback(decay=0.999)])
    trainer.fit(module, loader)

    adapter.eval(); head.eval()
    a_path = adapter.save(os.path.join(e.path, f"{e.PROPERTY}_adapter_sc"))
    torch.save({"state_dict": head.state_dict(), "na": na, "nb": nb, "hid": e.HEAD_HIDDEN,
                "layers": e.HEAD_LAYERS, "prop_mean": cond_mean, "prop_std": cond_std},
               os.path.join(e.path, "property_head.ckpt"))
    e.log(f"saved adapter -> {a_path} + property_head.ckpt")

    # ---- eval 1: H-vs-RDKit accuracy on held-out molecules ----
    from torch_geometric.loader import DataLoader as DL
    h_pred, h_true = [], []
    with torch.no_grad():
        for batch in DL(hold_graphs, batch_size=64):
            batch = batch.to(device)
            dense, mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            dense = dense.mask(mask)
            p = head(dense.X, dense.E, mask) * cond_std + cond_mean       # un-normalize
            h_pred += p.cpu().tolist()
            h_true += batch.cond.view(-1).cpu().tolist()
    h_pred, h_true = np.asarray(h_pred), np.asarray(h_true)
    head_mae = float(np.mean(np.abs(h_pred - h_true)))
    head_r = float(np.corrcoef(h_pred, h_true)[0, 1])
    e["eval/head_mae"] = head_mae
    e.log(f"[H accuracy] MAE(H vs RDKit {e.PROPERTY}) = {head_mae:.3f}  r={head_r:.3f}  (n={len(h_true)})")

    # ---- eval 2: adapter steering MAE at w=1 ----
    targets = dict(zip(e.LEVEL_NAMES, [float(x) for x in np.percentile(vals, e.TARGET_PERCENTILES)]))
    e.log("[steering] w=1  (does self-consistency tighten conditioning?)")
    steer = {}
    for lvl, tgt in targets.items():
        comp = AdapterComposition([ConditionBranch(adapter, torch.tensor([tgt]), 1.0)], base=base, mode="product")
        samp = AdaptedSampler(base, comp, eta=e.ETA, omega=e.OMEGA, sample_steps=e.EVAL_STEPS,
                              time_distortion=e.TIME_DISTORTION)
        out, rem = [], e.N_PER_TARGET
        while rem > 0:
            cur = min(e.EVAL_CHUNK, rem)
            out += samp.sample(cur, device=device, show_progress=False)
            rem -= cur
        gv = props_of(out, atom_dec, bond_dec, fn)
        mae = float(np.mean(np.abs(gv - tgt))) if gv.size else float("nan")
        steer[lvl] = {"target": tgt, "mae": mae, "mean": float(gv.mean()) if gv.size else None, "n": int(gv.size)}
        e.log(f"  {lvl}->{tgt:.2f}: MAE={mae:.3f} mean={steer[lvl]['mean']} n={gv.size}")

    e.commit_json("sc_metrics.json", {"lambda_ground": e.LAMBDA_GROUND, "lambda_sc": e.LAMBDA_SC,
                                      "head_mae": head_mae, "head_r": head_r, "steer": steer})
    e.log("Done.")


@experiment.testing
def testing(e: Experiment):
    e.EPOCHS = 2
    e.BATCH_SIZE = 16
    e.MAX_TIME_HOURS = 0.1
    e.H_HIDDEN = 32
    e.HEAD_HIDDEN = 32
    e.EVAL_STEPS = 5
    e.N_PER_TARGET = 8
    e.EVAL_CHUNK = 8
    e.HEAD_EVAL_N = 100
    import pandas as _pd
    df = _pd.read_csv(e.CSV_PATH).head(400)
    smoke = os.path.join(folder_path(__file__), "_sc_smoke.csv")
    df.to_csv(smoke, index=False)
    e.CSV_PATH = smoke


experiment.run_if_main()
