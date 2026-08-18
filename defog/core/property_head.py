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

    def config(self) -> dict:
        """The architecture config needed to rebuild this head.

        Exposed for out-of-band serialization: a container that stores tensors
        separately from declarations cannot rely on ``save``/``load`` round-tripping a
        pickled dict.
        """
        return {
            "na": self.xin.in_features, "nb": self.ein.in_features,
            "hid": self.xin.out_features, "layers": len(self.msg),
            "prop_mean": float(self.prop_mean), "prop_std": float(self.prop_std),
        }

    @classmethod
    def from_config(cls, config: dict, state_dict: dict, device="cpu") -> "PropertyHead":
        """Rebuild a head from a ``config()`` dict and a separately-stored state dict.

        ``strict=False`` matches :meth:`load`: state dicts written before the
        prop_mean/prop_std buffers existed are still loadable, with those buffers
        rebuilt from the config scalars.
        """
        head = cls(config["na"], config["nb"], hid=config.get("hid", 128),
                   layers=config.get("layers", 3), prop_mean=config.get("prop_mean", 0.0),
                   prop_std=config.get("prop_std", 1.0))
        head.load_state_dict(state_dict, strict=False)
        return head.to(device).eval()

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
        from ..domains.molecule import needs_kekulize, smiles_to_pyg_data

        n = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, n)
        out = X1.new_full((len(datas),), self.invalid)
        reenc, idx = [], []
        for i, d in enumerate(datas):
            mol = self.domain.decode(d)
            if mol is None:
                continue
            try:
                rd = smiles_to_pyg_data(Chem.MolToSmiles(mol), self.ae, self.be,
                                        kekulize=needs_kekulize(self.be))
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


def fit_property_head(head, graphs, *, epochs: int = 60, lr: float = 1e-3, batch_size: int = 32,
                      seed: int = 0, device: str = "cpu", grad_clip: float = 1.0,
                      progress=None):
    """Fit a :class:`PropertyHead` on its own, by grounding regression.

    The head is normally trained jointly with an adapter
    (:class:`defog.core.adapter.GroundedAdapterModule`), which is fine when the head is
    a by-product of that run. It is NOT fine when the head's whole purpose is to be an
    INDEPENDENT ruler — as in `molsmith adapter refine`, which fits a second head to score a
    policy that was optimised against the first. That use needs a fit path with no adapter
    anywhere near it, which is this.

    ``graphs`` are PyG ``Data`` objects carrying the RAW target in ``.cond``, exactly as the
    training pipeline builds them. Targets are normalised with the head's own buffers, so the
    fitted head and its ``predict`` agree by construction.

    ``seed`` seeds BOTH the parameter re-initialisation and the shuffling. Two heads fit on
    the same data with different seeds differ in where they are wrong, which is the entire
    point when one is checking the other.
    """
    import torch
    from torch_geometric.loader import DataLoader

    from .data import to_dense

    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    for module in head.modules():                 # re-init: an independent ruler must not
        if hasattr(module, "reset_parameters"):   # inherit the head it is checking
            module.reset_parameters()

    head = head.to(device).train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True, generator=generator)
    history = []
    for epoch in range(epochs):
        total, seen = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            dense, mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            dense = dense.mask(mask)
            target = (batch.cond.view(-1).float() - head.prop_mean) / head.prop_std
            predicted = head(dense.X.float(), dense.E.float(), mask)
            loss = torch.nn.functional.mse_loss(predicted, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            opt.step()
            total += float(loss.detach()) * target.numel()
            seen += target.numel()
        mean_loss = total / max(1, seen)
        history.append(mean_loss)
        if progress is not None and (epoch + 1) % 10 == 0:
            progress(f"  head fit epoch {epoch + 1}/{epochs}  loss {mean_loss:.4f}")
    return head.eval(), history


def fit_size_model(conditions, sizes, *, min_size=None, max_size=None, hidden: int = 512,
                   layers: int = 2, epochs: int = 200, lr: float = 1e-3,
                   batch_size: int = 256, seed: int = 0, device: str = "cpu",
                   val_frac: float = 0.1, grad_clip: float = 1.0,
                   property_name: str = "", property_from: str = "",
                   progress=None):
    """Fit a :class:`~defog.core.size_distribution.LearnedSizeDistribution` on
    ``(raw condition, node count)`` pairs.

    Takes plain tensors. **No graphs, no PyG, no adapter anywhere in the loop** -- this
    model never sees a molecule, only a property value and an atom count, so it is a few
    seconds of CPU work. The practical consequence is that it retrofits to an
    already-shipped adapter without retraining it.

    Normalization statistics and the marginal ``P(n)`` are derived from the data here, so
    a fitted model and its ``log_pmf`` agree by construction -- the same guarantee
    :func:`fit_property_head` gives for ``predict``.

    Returns ``(model, metrics)``. ``metrics`` carries the number that decides whether the
    model is worth shipping:

    ``gain_nats``
        ``nll_marginal - nll_val``. This is how much better than doing nothing the model
        predicts held-out sizes. **At ~0 the property carries no size information and the
        conditional draw should not be used for it** -- report this rather than assuming
        a fitted model is an improved one. For reference, a quantile-bucketed estimate on
        the full ZINC train split gives ~0.10 nats for logP, ~0.17 for QED, ~0.14 for TPSA.

    ``shrink``
        Held-out ``std(n - E[n|c]) / std(n)``. Complements ``gain_nats`` by saying what
        KIND of improvement it is. On ZINC this stays near 0.9: conditioning moves the
        size distribution's centre (E[n|logP] spans 20.2-26.7 heavy atoms across deciles)
        far more than it narrows its width. A bias correction, not a variance reduction.
    """
    import torch

    from .size_distribution import LearnedSizeDistribution

    conditions = torch.as_tensor(conditions, dtype=torch.float32)
    if conditions.dim() == 1:
        conditions = conditions.unsqueeze(-1)
    sizes = torch.as_tensor(sizes, dtype=torch.long).reshape(-1)
    assert conditions.size(0) == sizes.size(0), "conditions and sizes must align"
    assert conditions.size(0) > 1, "need more than one (condition, size) pair"

    lo = int(sizes.min()) if min_size is None else int(min_size)
    hi = int(sizes.max()) if max_size is None else int(max_size)
    assert (sizes >= lo).all() and (sizes <= hi).all(), \
        f"sizes fall outside the requested grid {lo}..{hi}"

    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    perm = torch.randperm(sizes.numel(), generator=generator)
    n_val = max(1, int(round(val_frac * sizes.numel()))) if val_frac > 0 else 0
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    assert train_idx.numel() > 0, "val_frac left no training data"

    # The marginal and the normalisation come from the TRAINING rows only: a marginal
    # that has seen the validation sizes makes `gain_nats` flatter than it is.
    #
    # Add-one smoothing across the declared grid, which is a real decision and not a
    # numerical nicety. LearnedSizeDistribution treats a zero-mass bin as structurally
    # unreachable, and that invariant is right for sizes OUTSIDE [lo, hi] -- the caller
    # declaring the grid is saying the model must never emit them. It is wrong for a size
    # inside the grid that this particular finite sample happened to miss: absence from a
    # subsample is not impossibility, and hard-zeroing it sends every held-out molecule of
    # that size to an infinite NLL, which is how `gain_nats` came back as `inf` rather than
    # as a number. Callers who genuinely want hard zeros can build the marginal themselves
    # and pass it to the constructor, which honours them.
    counts = torch.bincount(sizes[train_idx] - lo, minlength=hi - lo + 1).float() + 1.0
    model = LearnedSizeDistribution(
        conditions.size(1), lo, hi, hidden=hidden, layers=layers,
        cond_mean=conditions[train_idx].mean(0), cond_std=conditions[train_idx].std(0),
        marginal=counts, property_name=property_name, property_from=property_from,
    ).to(device)

    ct, st = conditions[train_idx].to(device), sizes[train_idx].to(device)
    cv, sv = conditions[val_idx].to(device), sizes[val_idx].to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(st.numel(), generator=generator).to(device)
        total, seen = 0.0, 0
        for i in range(0, order.numel(), batch_size):
            idx = order[i:i + batch_size]
            loss = -model.log_pmf(ct[idx]).gather(
                1, (st[idx] - lo).unsqueeze(1)).squeeze(1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total += float(loss.detach()) * idx.numel()
            seen += idx.numel()
        history.append(total / max(1, seen))
        if progress is not None and (epoch + 1) % 20 == 0:
            progress(f"  size fit epoch {epoch + 1}/{epochs}  nll {history[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        log_marg = model.log_marginal
        if n_val:
            lp = model.log_pmf(cv)
            bins = (sv - lo).unsqueeze(1)
            nll_val = float(-lp.gather(1, bins).squeeze(1).mean())
            nll_marg = float(-log_marg[sv - lo].mean())
            grid = model.sizes().float()
            expected = (lp.exp() * grid).sum(-1)
            resid = (sv.float() - expected).std()
            shrink = float(resid / sv.float().std()) if sv.numel() > 1 else float("nan")
        else:
            nll_val = nll_marg = shrink = float("nan")

    metrics = {
        "nll_train": history[-1] if history else float("nan"),
        "nll_val": nll_val,
        "nll_marginal": nll_marg,
        "gain_nats": nll_marg - nll_val,
        "shrink": shrink,
        "n_train": int(train_idx.numel()), "n_val": int(n_val),
        "min_size": lo, "max_size": hi,
        "history": history,
    }
    return model, metrics
