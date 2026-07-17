"""
Frozen-base AdaLN/FiLM CFG-adapters for DeFoG.

Localize classifier-free guidance to a small, swappable, stackable adapter on a
FROZEN unconditional base (ControlNet / T2I-Adapter / IP-Adapter family), instead
of retraining the base per condition. See ``docs/ADAPTER_PLAN.md``.

Mechanism
---------
* The base ``DeFoGModel`` (``cond_dim=0``) is frozen. An :class:`AdaLNAdapter`
  reads a condition ``c`` and emits, per transformer layer, a gated-FiLM
  modulation ``{scale, shift, gate}`` for the node (X), edge (E) and global (y)
  hidden streams. The modulation is a **gated residual** applied to each frozen
  layer's *output*: ``h' = h + mask ⊙ (gate ⊙ (scale·h + shift))``.
* The **gate** head is zero-initialized, so at init (and for the unconditional
  branch, which applies no modulation at all) the base is reproduced **exactly**
  -- the property N-branch product-of-experts composition relies on.
* Training (:class:`AdapterModule`) optimizes ONLY the adapter with the base's own
  denoising cross-entropy loss -> a conditional denoiser ``p(x1|x_t,c)`` (NOT the
  Bregman/positive-pairing objective that flattened the earlier guidance adapter).
* Composition (:class:`AdapterComposition`) stacks N conditions as product-of-
  experts on the rate matrices (generalizing the shipped 2-branch CFG blend to
  N+1 branches, run as one batched ``(N+1)·B`` forward). Consumed by
  :class:`~defog.core.sampler.AdaptedSampler`.

Adapters are EXTERNAL objects (never attached to ``DeFoGModel``) so the base's
``save``/``load`` and every existing checkpoint are untouched.
"""

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .layers import timestep_embedding
from .guidance import _GuidanceModuleBase

_STREAMS = ("X", "E", "y")
_DIMKEY = {"X": "dx", "E": "de", "y": "dy"}


# ===========================================================================
# Modulation: per-layer FiLM params for a (possibly stacked) batch
# ===========================================================================
class Modulation:
    """Per-layer gated-FiLM parameters for one forward pass.

    ``layers`` is a list (len == n_layers) of dicts; each holds, for the streams
    present, keys ``scale{X,E,y}``, ``shift{X,E,y}``, ``gate{X,E,y}`` each of shape
    ``(B, channel)``. A zero ``gate`` makes that stream's delta exactly 0 (bypass).
    """

    def __init__(self, layers: List[Dict[str, torch.Tensor]]):
        self.layers = layers

    def apply(self, i, X, E, y, x_mask, e_mask):
        """Apply layer ``i``'s modulation to the block outputs (gated residual).
        ``x_mask`` (B,n,1) / ``e_mask`` (B,n,n,1) mask the *delta* (never the
        hidden state) so gate=0 reproduces the base's own padding behavior."""
        m = self.layers[i]
        if "gateX" in m:
            X = X + x_mask * (m["gateX"][:, None] * (m["scaleX"][:, None] * X + m["shiftX"][:, None]))
        if "gateE" in m:
            E = E + e_mask * (m["gateE"][:, None, None] * (m["scaleE"][:, None, None] * E + m["shiftE"][:, None, None]))
        if "gatey" in m:
            y = y + m["gatey"] * (m["scaley"] * y + m["shifty"])
        return X, E, y

    def bypass_rows(self, mask: torch.Tensor) -> "Modulation":
        """Zero every gate for rows where ``mask`` is True (-> those rows bypass to
        the frozen base). Used for optional condition-dropout during training."""
        out = []
        for m in self.layers:
            d = dict(m)
            for k in list(d.keys()):
                if k.startswith("gate"):
                    d[k] = torch.where(mask[:, None], torch.zeros_like(d[k]), d[k])
            out.append(d)
        return Modulation(out)

    @staticmethod
    def stack_groups(mods: Sequence["Modulation"], bs: int, device) -> "Modulation":
        """Build a ``(N+1)·bs`` modulation: group 0 = uncond (all-zero => bypass),
        groups 1..N = each adapter's modulation, concatenated along the batch dim."""
        n_layers = len(mods[0].layers)
        combined = []
        for L in range(n_layers):
            keys = mods[0].layers[L].keys()
            d = {}
            for k in keys:
                ch = mods[0].layers[L][k].shape[-1]
                zero = torch.zeros(bs, ch, device=device)
                d[k] = torch.cat([zero] + [m.layers[L][k] for m in mods], dim=0)
            combined.append(d)
        return Modulation(combined)


# ===========================================================================
# AdaLNAdapter: c -> per-layer modulation
# ===========================================================================
def _base_token(base) -> float:
    """A cheap, stable identity token for a frozen base (to catch hot-swapping an
    adapter onto a different base with matching dims). Sum of a fixed weight."""
    with torch.no_grad():
        return float(base.model.mlp_in_X[0].weight.detach().double().sum().cpu())


class AdaLNAdapter(nn.Module):
    """Zero-init gated-FiLM adapter over a FROZEN base's transformer stack.

    Maps a condition ``c`` (+ optionally the flow-time ``t``) to per-layer
    ``{scale, shift, gate}`` for the enabled streams. Exact no-op at init (gate
    heads zero-initialized) -> the base is reproduced bit-for-bit.
    """

    def __init__(self, cond_dim: int, n_layers: int, dims: Dict[str, int],
                 hidden: int = 256, time_conditioned: bool = True,
                 streams: Sequence[str] = _STREAMS, time_emb_dim: int = 64,
                 cond_mean=None, cond_std=None, name: str = "", cond_type: str = "",
                 base_token: Optional[float] = None):
        super().__init__()
        self.cond_dim = cond_dim
        self.n_layers = n_layers
        self.dims = dict(dims)
        self.hidden = hidden
        self.time_conditioned = time_conditioned
        self.time_emb_dim = time_emb_dim
        self.streams = tuple(streams)
        self.name, self.cond_type = name, cond_type
        self.base_token = base_token

        cond_in = cond_dim + (time_emb_dim if time_conditioned else 0)
        self.trunk = nn.Sequential(
            nn.Linear(cond_in, hidden), nn.SiLU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(),
        )
        # per (layer, stream): a scale/shift head (normal init) + a zero-init gate.
        self.ss = nn.ModuleList()
        self.gate = nn.ModuleList()
        for _ in range(n_layers):
            ss_l, gate_l = nn.ModuleDict(), nn.ModuleDict()
            for s in self.streams:
                ch = dims[_DIMKEY[s]]
                ss_l[s] = nn.Linear(hidden, 2 * ch)
                g = nn.Linear(hidden, ch)
                nn.init.zeros_(g.weight); nn.init.zeros_(g.bias)   # zero-init gate => exact no-op
                gate_l[s] = g
            self.ss.append(ss_l); self.gate.append(gate_l)

        m = torch.zeros(cond_dim) if cond_mean is None else torch.as_tensor(cond_mean, dtype=torch.float32).reshape(-1)
        s = torch.ones(cond_dim) if cond_std is None else torch.as_tensor(cond_std, dtype=torch.float32).reshape(-1).clamp_min(1e-6)
        self.register_buffer("cond_mean", m)   # buffer -> follows .to(device) and is in state_dict
        self.register_buffer("cond_std", s)

    # --- construction helper ------------------------------------------------
    @classmethod
    def for_base(cls, base, cond_dim: int, **kw) -> "AdaLNAdapter":
        """Build an adapter matching ``base``'s transformer dims (read from the live
        module) and layer count."""
        attn = base.model.tf_layers[0].self_attn
        dims = {"dx": attn.dx, "de": attn.de, "dy": attn.dy}
        n_layers = len(base.model.tf_layers)
        kw.setdefault("base_token", _base_token(base))
        return cls(cond_dim, n_layers, dims, **kw)

    # --- forward ------------------------------------------------------------
    def normalize(self, c: torch.Tensor) -> torch.Tensor:
        return (c - self.cond_mean) / self.cond_std

    def forward(self, c: torch.Tensor, t: Optional[torch.Tensor] = None) -> Modulation:
        c = self.normalize(c.float())
        if self.time_conditioned:
            assert t is not None, "time_conditioned adapter requires t"
            temb = timestep_embedding(t.reshape(-1, 1), self.time_emb_dim)
            h = self.trunk(torch.cat([c, temb], dim=-1))
        else:
            h = self.trunk(c)
        layers = []
        for L in range(self.n_layers):
            d = {}
            for s in self.streams:
                scale, shift = self.ss[L][s](h).chunk(2, dim=-1)
                gate = self.gate[L][s](h)
                d[f"scale{s}"], d[f"shift{s}"], d[f"gate{s}"] = scale, shift, gate
            layers.append(d)
        return Modulation(layers)

    # --- compatibility / io -------------------------------------------------
    def check_compatible(self, base):
        attn = base.model.tf_layers[0].self_attn
        assert self.dims["dx"] == attn.dx and self.dims["de"] == attn.de and self.dims["dy"] == attn.dy, \
            f"adapter dims {self.dims} != base ({attn.dx},{attn.de},{attn.dy})"
        assert self.n_layers == len(base.model.tf_layers), \
            f"adapter n_layers {self.n_layers} != base {len(base.model.tf_layers)}"
        if self.base_token is not None:
            tok = _base_token(base)
            if abs(tok - self.base_token) > 1e-3 * (1 + abs(self.base_token)):
                warnings.warn(
                    f"adapter '{self.name}' was trained on a different base "
                    f"(token {self.base_token:.4g} != {tok:.4g}); steering may be meaningless.",
                    RuntimeWarning)

    def _config(self):
        return dict(cond_dim=self.cond_dim, n_layers=self.n_layers, dims=self.dims,
                    hidden=self.hidden, time_conditioned=self.time_conditioned,
                    streams=list(self.streams), time_emb_dim=self.time_emb_dim,
                    name=self.name, cond_type=self.cond_type, base_token=self.base_token)

    def save(self, path):
        if not path.endswith(".ckpt"):
            path = path + ".ckpt"
        torch.save({"state_dict": self.state_dict(), "config": self._config()}, path)
        return path

    @classmethod
    def load(cls, path, device="cpu") -> "AdaLNAdapter":
        if not path.endswith(".ckpt"):
            path = path + ".ckpt"
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = dict(ck["config"]); cfg["streams"] = tuple(cfg["streams"])
        a = cls(**cfg)
        a.load_state_dict(ck["state_dict"])   # includes cond_mean/cond_std buffers
        return a.to(device)


# ===========================================================================
# Composition
# ===========================================================================
@dataclass
class ConditionBranch:
    """One condition in a composition: an adapter, a RAW target (the adapter
    normalizes internally), and its CFG weight ``w``."""
    adapter: AdaLNAdapter
    condition: torch.Tensor
    weight: float = 1.0


class AdapterComposition:
    """N-branch product-of-experts spec consumed by ``denoise_step`` /
    ``AdaptedSampler``. ``mode='product'`` sums the log-ratios; ``'mean'`` averages
    (recommended for N>1 to keep the effective uncond coefficient bounded)."""

    def __init__(self, branches: Sequence[ConditionBranch], base=None, mode: str = "product"):
        assert mode in ("product", "mean")
        self.branches = list(branches)
        self.mode = mode
        if base is not None:
            for b in self.branches:
                b.adapter.check_compatible(base)

    def __len__(self):
        return len(self.branches)

    @torch.no_grad()
    def build_modulation(self, bs: int, t: torch.Tensor) -> Modulation:
        """Combined ``(N+1)·bs`` modulation: group 0 uncond bypass, group i = adapter_i."""
        device = t.device
        mods = []
        for br in self.branches:
            c = torch.as_tensor(br.condition, dtype=torch.float32, device=device)
            if c.dim() == 1:
                c = c.unsqueeze(0)
            if c.size(0) == 1 and bs > 1:
                c = c.expand(bs, -1)
            assert c.size(0) == bs, f"branch condition batch {c.size(0)} != {bs}"
            mods.append(br.adapter(c, t=t))
        return Modulation.stack_groups(mods, bs, device)

    def weights(self, device, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor([b.weight for b in self.branches], device=device, dtype=dtype)

    def set_weights(self, ws):
        for b, w in zip(self.branches, ws):
            b.weight = float(w)
        return self


class AdapterRegistry:
    """Name -> adapter map, for hot-swapping adapters at inference."""

    def __init__(self):
        self._d: Dict[str, AdaLNAdapter] = {}

    def register(self, name, adapter):
        self._d[name] = adapter
        return self

    def get(self, name) -> AdaLNAdapter:
        return self._d[name]

    def names(self):
        return list(self._d.keys())

    @classmethod
    def load_dir(cls, path, device="cpu") -> "AdapterRegistry":
        reg = cls()
        for fn in sorted(os.listdir(path)):
            if fn.endswith(".ckpt"):
                a = AdaLNAdapter.load(os.path.join(path, fn), device=device)
                reg.register(a.name or os.path.splitext(fn)[0], a)
        return reg


# ===========================================================================
# Training module
# ===========================================================================
class AdapterModule(_GuidanceModuleBase):
    """Train ONLY the adapter (frozen base) with the base's own denoising CE loss:
    a direct conditional denoiser ``p(x1|x_t,c)``. Reuses ``_GuidanceModuleBase``
    for freeze + ``base.*``-stripping checkpoint plumbing; overrides
    ``configure_optimizers`` (the base hardcodes ``self.h``)."""

    def __init__(self, base, adapter: AdaLNAdapter, cond_attr: str = "cond",
                 cond_drop_prob: float = 0.0, lr: float = 2e-4):
        super().__init__()
        self._freeze_base(base)
        self.adapter = adapter
        self.cond_attr = cond_attr
        self.cond_drop_prob = float(cond_drop_prob)
        self.lr = float(lr)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.adapter.parameters(), lr=self.lr, weight_decay=1e-5)

    def training_step(self, batch, batch_idx):
        self.base.eval()
        X1, E1, node_mask = self._dense(batch)
        bs, device = X1.size(0), X1.device
        c = getattr(batch, self.cond_attr).to(device).view(bs, -1).float()   # RAW; adapter normalizes
        y0 = torch.zeros(bs, 0, device=device)
        with torch.no_grad():
            noisy = self.base._apply_noise(X1, E1, y0, node_mask)
            extra = self.base._compute_extra_data(noisy)
        mod = self.adapter(c, t=noisy["t"])
        if self.cond_drop_prob:
            drop = torch.rand(bs, device=device) < self.cond_drop_prob
            mod = mod.bypass_rows(drop)
        pred = self.base.forward(noisy, extra, node_mask, cond_modulation=mod)
        loss = self.base.train_loss(pred_X=pred.X, pred_E=pred.E, pred_y=pred.y,
                                    true_X=X1, true_E=E1, true_y=y0, node_mask=node_mask)
        self.log("adapter/loss", loss, prog_bar=True, on_epoch=True, batch_size=bs)
        return loss
