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

    ``energy_fn(X1, E1, node_mask) -> (K,)`` = ``(head.predict(graph) - target) ** 2`` (lower
    is better). Drop-in replacement for ``MoleculePropertyEnergy`` where the property has no
    closed-form RDKit function (or, uniformly, for any property).
    """

    def __init__(self, head: PropertyHead, target: float):
        self.head = head.eval()
        self.target = float(target)

    def _desc(self):
        return f"LearnedPropertyEnergy(target={self.target})"

    @torch.no_grad()
    def __call__(self, X1, E1, node_mask):
        dev = self.head.prop_mean.device
        pred = self.head.predict(X1.to(dev), E1.to(dev), node_mask.to(dev))
        return (pred.reshape(-1) - self.target) ** 2
