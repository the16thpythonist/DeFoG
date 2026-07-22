"""Grounded property head + its Feynman-Kac energy.

``PropertyHead`` is a light message-passing GNN that maps a (discrete, one-hot) graph to a
scalar property. It is trained by GROUNDING only -- a regression against the TRUE property
of real molecules (RDKit or measured) -- so it never sees the conditioning target and cannot
leak it. It is deliberately kept independent of any adapter.

``LearnedPropertyEnergy`` turns a trained head into an FK ``energy_fn``: it scores each
predicted-clean particle by the squared error of the head's (un-normalized) prediction to a
target. Feynman-Kac feeds the head the DISCRETE one-hot predicted-clean graph it already
argmaxes internally (``FeynmanKacSampler._predict_clean``), so the head stays in-distribution
-- which is exactly why this works where the (soft-input) self-consistency coupling did not.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PropertyHead(nn.Module):
    """structure -> scalar property (predicts the NORMALIZED property; ``predict`` un-normalizes).

    Args:
        na: node-feature dim (== base model's node classes).
        nb: edge-feature dim (== base model's edge classes).
        hid, layers: GNN width / depth.
        prop_mean, prop_std: normalization stats stored as buffers (for ``predict``).
    """

    def __init__(self, na, nb, hid=128, layers=3, prop_mean=0.0, prop_std=1.0):
        super().__init__()
        self.xin = nn.Linear(na, hid)
        self.ein = nn.Linear(nb, hid)
        self.msg = nn.ModuleList([nn.Linear(2 * hid, hid) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Linear(hid, hid) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.act = nn.SiLU()
        self.out = nn.Sequential(nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, 1))
        self.register_buffer("prop_mean", torch.tensor(float(prop_mean)))
        self.register_buffer("prop_std", torch.tensor(float(prop_std)))

    def forward(self, X, E, node_mask):
        """Normalized property (bs,). X:(bs,n,na) E:(bs,n,n,nb) node_mask:(bs,n)."""
        bs, n, _ = X.shape
        m = node_mask.float().unsqueeze(-1)                    # (bs,n,1)
        em = m.unsqueeze(2) * m.unsqueeze(1)                   # (bs,n,n,1)
        h = self.act(self.xin(X)) * m
        e = self.act(self.ein(E)) * em
        for msg, upd, norm in zip(self.msg, self.upd, self.norm):
            hj = h.unsqueeze(1).expand(bs, n, n, h.size(-1))   # h_j
            mij = self.act(msg(torch.cat([e, hj], -1))) * em   # message per (i,j)
            agg = mij.sum(2)                                   # sum over neighbors j
            h = norm(h + upd(agg)) * m
        return self.out(h.sum(1)).squeeze(-1)                  # SUM pool -> (bs,)

    @torch.no_grad()
    def predict(self, X, E, node_mask):
        """Un-normalized property prediction (bs,)."""
        return self.forward(X.float(), E.float(), node_mask) * self.prop_std + self.prop_mean

    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "na": self.xin.in_features, "nb": self.ein.in_features,
            "hid": self.xin.out_features, "layers": len(self.msg),
            "prop_mean": float(self.prop_mean), "prop_std": float(self.prop_std),
        }, path)
        return path

    @classmethod
    def load(cls, path, device="cpu"):
        """Load a head saved by this class OR by the training experiment (same key schema)."""
        ck = torch.load(path, map_location=device, weights_only=False)
        head = cls(ck["na"], ck["nb"], hid=ck.get("hid", 128), layers=ck.get("layers", 3),
                   prop_mean=ck.get("prop_mean", 0.0), prop_std=ck.get("prop_std", 1.0))
        # strict=False: the experiment's state_dict predates the prop_mean/std buffers,
        # which we (re)build from the ckpt scalars above.
        head.load_state_dict(ck["state_dict"], strict=False)
        return head.to(device).eval()


class LearnedPropertyEnergy:
    """FK energy from a trained :class:`PropertyHead`.

    ``energy_fn(X1, E1, node_mask) -> (K,)`` = ``(head.predict(mol) - target) ** 2`` (lower is
    better). Drop-in replacement for ``MoleculePropertyEnergy`` where the property has no
    closed-form RDKit function (or, uniformly, for any property).

    Each predicted-clean graph is DECODED to a molecule (validity gate) and RE-ENCODED in the
    head's native ``to_dense(smiles_to_pyg_data(...))`` format before scoring. This matters:
    the head is trained on that encoding, and the model's raw argmax predicted-clean graph
    (spurious diagonal edges, off-support classes) does NOT match it -- feeding the head the
    raw graph makes it mispredict and FK steer the wrong way. Invalid / undecodable graphs get
    ``invalid_energy`` (their FK weight -> 0), keeping the search on-manifold.

    Args:
        head: trained :class:`PropertyHead`.
        target: desired (un-normalized) property value.
        domain: object with ``.decode(pyg_data) -> Optional[Mol]`` (e.g. MoleculeDomain).
        atom_encoder, bond_encoder: the domain's encoders (for the native re-encoding).
    """

    def __init__(self, head: PropertyHead, target: float, domain, atom_encoder, bond_encoder,
                 invalid_energy: float = 1e3):
        self.head = head.eval()
        self.target = float(target)
        self.domain = domain
        self.ae, self.be = atom_encoder, bond_encoder
        self.invalid = float(invalid_energy)

    def _desc(self):
        return f"LearnedPropertyEnergy(target={self.target})"

    @torch.no_grad()
    def __call__(self, X1, E1, node_mask):
        from rdkit import Chem
        from torch_geometric.data import Batch

        from .data import dense_to_pyg, to_dense
        from ..domains.molecule import smiles_to_pyg_data

        n = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, n)
        out = X1.new_full((len(datas),), self.invalid)
        reenc, idx = [], []
        for i, d in enumerate(datas):
            mol = self.domain.decode(d)
            if mol is None:
                continue
            try:
                rd = smiles_to_pyg_data(Chem.MolToSmiles(mol), self.ae, self.be)
            except Exception:
                rd = None
            if rd is not None and getattr(rd, "x", None) is not None:
                reenc.append(rd)
                idx.append(i)
        if reenc:
            dev = self.head.prop_mean.device
            batch = Batch.from_data_list(reenc).to(dev)
            dense, mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            dense = dense.mask(mask)
            preds = self.head.predict(dense.X, dense.E, mask).reshape(-1)
            for j, i in enumerate(idx):
                out[i] = (preds[j] - self.target) ** 2
        return out
