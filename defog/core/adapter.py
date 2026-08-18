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

# Accepted keys of AdaLNAdapter.__init__, used by from_config to ignore extras rather
# than raise on a config written by a newer version.
_CONFIG_KEYS = frozenset({
    "cond_dim", "n_layers", "dims", "hidden", "time_conditioned", "streams",
    "time_emb_dim", "cond_mean", "cond_std", "name", "cond_type",
    "interior_ff", "interior_attn", "base_token", "cond_encoder",
})


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
        groups 1..N = each adapter's modulation, concatenated along the batch dim.

        Robust to HETEROGENEOUS key sets across branches: an adapter that lacks an
        interior key (e.g. an output-only adapter composed with an interior-enabled
        one) is treated as gate=0 -> exact bypass at that site. Takes the UNION of
        keys over all branches and zero-fills any key a branch does not define."""
        n_layers = len(mods[0].layers)
        combined = []
        for L in range(n_layers):
            keys = set()
            for m in mods:
                keys |= set(m.layers[L].keys())
            d = {}
            for k in sorted(keys):
                ch = next(m.layers[L][k].shape[-1] for m in mods if k in m.layers[L])
                zero = torch.zeros(bs, ch, device=device)
                rows = [zero]                                    # group-0 uncond bypass
                for m in mods:
                    t = m.layers[L].get(k)
                    rows.append(t if t is not None else torch.zeros(bs, ch, device=device))
                d[k] = torch.cat(rows, dim=0)
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


class FingerprintEncoder(nn.Module):
    """Residual MLP encoder for a wide, flat condition such as a Morgan fingerprint.

    Why this exists: without an encoder the trunk's first layer is the *only* thing
    between the condition and every FiLM head, and it is ``Linear(cond_dim + t, hidden)``
    with ``hidden`` fixed at 256. Widening the fingerprint therefore does not widen the
    path — 512 and 1024 bits are both compressed to 256 in one step, and 2048 would be an
    8:1 squeeze. Measured: 512 -> 1024 bits bought +0.015 Tanimoto lift, which is what
    pouring more information into an unchanged bottleneck looks like. This encoder moves
    the narrowest point off the first layer so that "more bits" and "wider path" become
    separable choices rather than one confounded one.

    Unlike :class:`SpectrumEncoder` there is no spatial structure to exploit: Morgan bits
    are hash buckets, so adjacency is meaningless and a convolution would be modelling an
    artifact of the hash. Depth with residual connections is the right prior instead —
    it lets the encoder learn bit *co-occurrence* (substructures that appear together)
    without assuming anything about bit ordering.

    The adapter's exact-no-op-at-init property is unaffected: it comes from the zero-init
    gate heads, which sit downstream of this module.
    """

    def __init__(self, in_dim: int, out_dim: int = 512, hidden: int = 1024,
                 n_blocks: int = 2, dropout: float = 0.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = int(hidden)
        self.n_blocks = int(n_blocks)
        self.dropout = float(dropout)
        self.proj_in = nn.Linear(self.in_dim, self.hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.hidden), nn.SiLU(),
                nn.Linear(self.hidden, self.hidden),
                *( [nn.Dropout(self.dropout)] if self.dropout > 0 else [] ),
            )
            for _ in range(self.n_blocks)
        ])
        self.proj_out = nn.Sequential(nn.LayerNorm(self.hidden), nn.SiLU(),
                                      nn.Linear(self.hidden, self.out_dim))

    def _config(self) -> dict:
        """Enough to rebuild this encoder — see :meth:`SpectrumEncoder._config` for why
        omitting it would silently corrupt the GDPO KL reference."""
        return dict(kind="mlp", in_dim=self.in_dim, out_dim=self.out_dim,
                    hidden=self.hidden, n_blocks=self.n_blocks, dropout=self.dropout)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        if c.size(-1) != self.in_dim:
            raise ValueError(f"FingerprintEncoder expects {self.in_dim} inputs, got {c.size(-1)}")
        h = self.proj_in(c)
        for blk in self.blocks:
            h = h + blk(h)          # pre-norm residual: identity path stays clean
        return self.proj_out(h)


class SpectrumEncoder(nn.Module):
    """1-D convolutional encoder for a ``[values(n_bins), mask(n_bins)]`` condition.

    An adaLN trunk fed a raw 3602-dimensional vector has to learn from scratch that adjacent
    bins are related, that a band is a local shape a few dozen bins wide, and that the mask
    channel gates the value channel bin-by-bin. A spectrum has all of that structure by
    construction, and a convolution gets it for free.

    Position is preserved deliberately. Global pooling would answer "is there a carbonyl
    somewhere" while discarding *where* — and where is the entire content of a spectrum. The
    final adaptive pool keeps 16 coarse positions so the trunk still sees roughly which region
    of the spectrum carries the signal.

    The two channels are the value and the mask, so the convolution can learn "specified and
    absent" as a distinct pattern from "unspecified" — the distinction the mask exists to make
    and the one a flat vector buries.
    """

    def __init__(self, n_bins: int, out_dim: int = 256, channels: Sequence[int] = (32, 64, 128, 128),
                 kernel: int = 9, pooled: int = 16):
        super().__init__()
        self.n_bins = int(n_bins)
        self.out_dim = int(out_dim)
        self.pooled = int(pooled)
        self.channels = tuple(channels)
        self.kernel = int(kernel)
        layers, in_ch = [], 2
        for ch in channels:
            layers += [nn.Conv1d(in_ch, ch, kernel_size=kernel, stride=2, padding=kernel // 2),
                       nn.GroupNorm(num_groups=min(8, ch), num_channels=ch), nn.SiLU()]
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(in_ch * self.pooled, self.out_dim)

    def _config(self) -> dict:
        """Enough to rebuild this encoder. Without it, anything that reconstructs an adapter
        from ``_config()`` — notably ``AdapterGDPOTrainer``'s frozen KL reference — would
        rebuild it *without* the encoder and silently compare against a different model."""
        return dict(kind="spectrum_cnn", n_bins=self.n_bins, out_dim=self.out_dim,
                    channels=list(self.channels), kernel=self.kernel, pooled=self.pooled)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        bs = c.size(0)
        if c.size(-1) != 2 * self.n_bins:
            raise ValueError(f"SpectrumEncoder expects {2 * self.n_bins} inputs, got {c.size(-1)}")
        x = c.view(bs, 2, self.n_bins)          # channel 0 = values, 1 = mask
        x = self.conv(x)
        x = nn.functional.adaptive_avg_pool1d(x, self.pooled)
        return self.proj(x.reshape(bs, -1))


#: Condition encoders addressable from a saved config's ``kind`` field. Keeping the
#: registry next to the classes means adding one is a single edit, and an unknown kind
#: fails loudly at load rather than silently rebuilding the wrong architecture.
_COND_ENCODERS = {"spectrum_cnn": SpectrumEncoder, "mlp": FingerprintEncoder}


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
                 interior_ff: bool = False, interior_attn: bool = False,
                 base_token: Optional[float] = None,
                 cond_encoder: Optional[nn.Module] = None):
        super().__init__()
        self.cond_dim = cond_dim
        self.n_layers = n_layers
        self.dims = dict(dims)
        self.hidden = hidden
        self.time_conditioned = time_conditioned
        self.time_emb_dim = time_emb_dim
        self.streams = tuple(streams)
        self.name, self.cond_type = name, cond_type
        self.interior_ff = bool(interior_ff)      # L4: pre-FFN FiLM on X,E
        self.interior_attn = bool(interior_attn)  # L10: condition e_mul (edge->attn logits)
        self.base_token = base_token

        # An encoder, when present, sits between the (already normalised) condition and the
        # trunk, so everything downstream — state_dict, ConditionBranch, the package format —
        # still sees exactly one module with one cond_dim.
        if isinstance(cond_encoder, dict):
            spec = dict(cond_encoder)
            # Configs written before `kind` existed are all spectrum encoders, so that
            # stays the default -- changing it would silently rebuild old adapters wrong.
            kind = spec.pop("kind", "spectrum_cnn")
            if kind not in _COND_ENCODERS:
                raise ValueError(f"unknown cond_encoder kind {kind!r} in {cond_encoder!r}; "
                                 f"known: {sorted(_COND_ENCODERS)}")
            cond_encoder = _COND_ENCODERS[kind](**spec)
        self.cond_encoder = cond_encoder
        encoded_dim = cond_dim if cond_encoder is None else int(cond_encoder.out_dim)
        cond_in = encoded_dim + (time_emb_dim if time_conditioned else 0)
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

        # --- L4 (pre-FFN adaLN-Zero): gated FiLM heads on X(dx), E(de) ---
        if self.interior_ff:
            self.ff = nn.ModuleList()
            for _ in range(n_layers):
                ff_l = nn.ModuleDict()
                for s in ("X", "E"):
                    ch = dims[_DIMKEY[s]]
                    ff_l[f"ss_{s}"] = nn.Linear(hidden, 2 * ch)
                    g = nn.Linear(hidden, ch)
                    nn.init.zeros_(g.weight); nn.init.zeros_(g.bias)   # zero-init gate => no-op
                    ff_l[f"gate_{s}"] = g
                self.ff.append(ff_l)

        # --- L10 (edge->attention-logit): gated FiLM head on e_mul (dx) ---
        if self.interior_attn:
            self.attn = nn.ModuleList()
            ch = dims["dx"]
            for _ in range(n_layers):
                a_l = nn.ModuleDict()
                a_l["ss"] = nn.Linear(hidden, 2 * ch)
                g = nn.Linear(hidden, ch)
                nn.init.zeros_(g.weight); nn.init.zeros_(g.bias)       # zero-init gate => no-op
                a_l["gate"] = g
                self.attn.append(a_l)

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
        if self.cond_encoder is not None:
            c = self.cond_encoder(c)
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
            if self.interior_ff:                     # L4 pre-FFN keys
                for s in ("X", "E"):
                    sc, sh = self.ff[L][f"ss_{s}"](h).chunk(2, dim=-1)
                    d[f"scale_ff{s}"], d[f"shift_ff{s}"], d[f"gate_ff{s}"] = sc, sh, self.ff[L][f"gate_{s}"](h)
            if self.interior_attn:                   # L10 edge->attn keys
                sc, sh = self.attn[L]["ss"](h).chunk(2, dim=-1)
                d["scale_emul"], d["shift_emul"], d["gate_emul"] = sc, sh, self.attn[L]["gate"](h)
            layers.append(d)
        return Modulation(layers)

    def interior_attn_parameters(self):
        """L10 head params (for an optional smaller-LR optimizer group)."""
        return list(self.attn.parameters()) if self.interior_attn else []

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
        encoder = None if self.cond_encoder is None else self.cond_encoder._config()
        return dict(cond_encoder=encoder,
                    cond_dim=self.cond_dim, n_layers=self.n_layers, dims=self.dims,
                    hidden=self.hidden, time_conditioned=self.time_conditioned,
                    streams=list(self.streams), time_emb_dim=self.time_emb_dim,
                    name=self.name, cond_type=self.cond_type,
                    interior_ff=self.interior_ff, interior_attn=self.interior_attn,
                    base_token=self.base_token)

    def config(self) -> dict:
        """The architecture config needed to rebuild this adapter (public alias).

        Exposed for out-of-band serialization -- e.g. a container that stores tensors
        separately from declarations (safetensors + a metadata file) and therefore
        cannot rely on ``save``/``load`` round-tripping a pickled dict.
        """
        return self._config()

    @classmethod
    def from_config(cls, config: dict, state_dict: dict, device="cpu") -> "AdaLNAdapter":
        """Rebuild an adapter from a ``config()`` dict and a separately-stored state dict.

        The counterpart to :meth:`config`, for callers that keep tensors and
        declarations in different files. ``load`` remains the path for ``.ckpt``
        checkpoints written by :meth:`save`.

        Unknown config keys are ignored rather than raising, so a config written by a
        newer version stays loadable as long as the tensors match.
        """
        cfg = {k: v for k, v in config.items() if k in _CONFIG_KEYS}
        if "streams" in cfg:
            cfg["streams"] = tuple(cfg["streams"])
        a = cls(**cfg)
        a.load_state_dict(state_dict)   # includes the cond_mean/cond_std buffers
        return a.to(device)

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
    normalizes internally), and its CFG weight ``w``.

    The default is 2.0, which is the measured optimum for prob-space blending on
    both logP and QED. It was 1.0 for as long as blending happened in RATE space,
    where w>1 is not merely suboptimal but broken: the blend extrapolates past the
    conditional and ``_stabilize``'s clamp silently drops rates, collapsing logP
    validity to 0.526 and MAE to 5.59 at w=2. That is a property of the old blend
    space, not of the task -- see :class:`AdapterComposition`. With ``blend_space
    ="prob"`` the same w=2 gives the best MAE we have measured.
    """
    adapter: AdaLNAdapter
    condition: torch.Tensor
    weight: float = 2.0


class AdapterComposition:
    """N-branch product-of-experts spec consumed by ``denoise_step`` /
    ``AdaptedSampler``. ``mode='product'`` sums the log-ratios; ``'mean'`` averages
    (recommended for N>1 to keep the effective uncond coefficient bounded)."""

    def __init__(self, branches: Sequence[ConditionBranch], base=None, mode: str = "product",
                 blend_space: str = "prob"):
        """
        ``mode`` is the FORM of the blend (geometric product vs its per-branch mean).
        ``blend_space`` is WHERE it is applied, and the two are independent axes:

        * ``"prob"`` (default) -- blend the predicted clean-graph marginals and derive
          ONE rate matrix from the result. This is where FreeGress applies guidance
          (Eq. 10/11), it spends a single ``X_1`` draw, and it makes the trajectory
          consistent with the terminal step, which has always blended clean log-probs.
        * ``"rate"`` -- build a rate matrix per branch and blend those.
          ``compute_rate_matrices`` draws its own ``X_1`` sample inside each call, so
          an N-branch blend mixes N+1 independent draws. Retained so that every number
          reported before 2026-08-17 reproduces exactly.

        The two are equivalent at w=1 (measured: paired mean -0.0034 over 100 targets,
        51/100, Wilcoxon p=0.72) and diverge sharply above it: at w=2 the rate path
        collapses to logP MAE 5.59 / validity 0.526 while prob gives 0.5420 / 0.982.
        "w=1 always wins" was an artifact of the blend space, which is why the default
        moved here and why :class:`ConditionBranch` now defaults to w=2.
        """
        assert mode in ("product", "mean")
        assert blend_space in ("rate", "prob"), f"unknown blend_space {blend_space!r}"
        self.branches = list(branches)
        self.mode = mode
        self.blend_space = blend_space
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
        """``(N,)`` for scalar branch weights, ``(N, bs)`` if any branch carries a
        per-molecule weight.

        The per-molecule form exists for closed-loop control, where each molecule has its
        own error signal and therefore earns its own guidance strength. Mixing the two is
        allowed: a scalar branch is broadcast across the batch."""
        ws = [b.weight for b in self.branches]
        if not any(torch.is_tensor(w) and w.numel() > 1 for w in ws):
            return torch.tensor([float(w) for w in ws], device=device, dtype=dtype)
        bs = max(w.numel() for w in ws if torch.is_tensor(w))
        rows = []
        for w in ws:
            if torch.is_tensor(w) and w.numel() > 1:
                rows.append(w.to(device=device, dtype=dtype).reshape(-1))
            else:
                rows.append(torch.full((bs,), float(w), device=device, dtype=dtype))
        return torch.stack(rows, dim=0)

    def set_weights(self, ws):
        for b, w in zip(self.branches, ws):
            # A per-molecule weight vector must survive as a tensor; float() on it would
            # raise, and coercing to a scalar would silently discard the per-molecule
            # signal that closed-loop control exists to carry.
            b.weight = w if (torch.is_tensor(w) and w.numel() > 1) else float(w)
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
                 cond_drop_prob: float = 0.0, lr: float = 2e-4, l10_lr_scale: float = 1.0):
        super().__init__()
        self._freeze_base(base)
        self.adapter = adapter
        self.cond_attr = cond_attr
        self.cond_drop_prob = float(cond_drop_prob)
        self.lr = float(lr)
        self.l10_lr_scale = float(l10_lr_scale)   # smaller LR on the L10 (attention) heads

    def configure_optimizers(self):
        a = self.adapter
        if a.interior_attn and self.l10_lr_scale != 1.0:
            l10_ids = {id(p) for p in a.interior_attn_parameters()}
            l10 = [p for p in a.parameters() if id(p) in l10_ids]
            rest = [p for p in a.parameters() if id(p) not in l10_ids]
            return torch.optim.AdamW(
                [{"params": rest, "lr": self.lr},
                 {"params": l10, "lr": self.lr * self.l10_lr_scale}], weight_decay=1e-5)
        return torch.optim.AdamW(a.parameters(), lr=self.lr, weight_decay=1e-5)

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


class GroundedAdapterModule(AdapterModule):
    """Train an adapter AND a :class:`~defog.core.property_head.PropertyHead` in one run,
    with the two objectives deliberately UNCOUPLED.

    Two losses, one per module:

    * ``L_denoise`` -- the base's own denoising CE, inherited verbatim from
      :class:`AdapterModule`: the adapter becomes a conditional denoiser ``p(x1|x_t,c)``.
    * ``L_ground`` -- ``MSE(head(true clean graph), normalized true property)``. A plain
      supervised regression over REAL molecules.

    **The head never sees the condition as an input, and no loss pushes the adapter through
    the head.** That is the point, not an omission. A head fitted against the condition on
    the adapter's own output can satisfy it by learning to detect *what the adapter was
    aiming at* rather than *what the molecule is* -- and such a head is worthless as the
    Feynman-Kac energy, because it would reward intent instead of achievement
    (see :mod:`defog.core.property_head`). The soft-input self-consistency coupling that
    couples them was tried and did not hold up; ``LearnedPropertyEnergy`` works because FK
    feeds the head DISCRETE re-encoded graphs, which is a different input distribution from
    the softmax tensors such a coupling would train on.

    The normalization is read from the head's OWN ``prop_mean``/``prop_std`` buffers rather
    than passed in separately, so the scale the head is trained at and the scale
    ``PropertyHead.predict`` un-normalizes with cannot drift apart.

    Args:
        base: frozen unconditional ``DeFoGModel``.
        adapter: the :class:`AdaLNAdapter` to train.
        head: a ``PropertyHead`` constructed with ``prop_mean``/``prop_std`` of the data.
        lr / lr_head: learning rates for the adapter and the head.
        l10_lr_scale: smaller LR on the adapter's L10 (interior-attention) heads.
        lambda_ground: weight on the grounding term.
    """

    def __init__(self, base, adapter, head, cond_attr: str = "cond",
                 cond_drop_prob: float = 0.0, lr: float = 2e-4, lr_head: float = 1e-3,
                 l10_lr_scale: float = 1.0, lambda_ground: float = 1.0):
        super().__init__(base, adapter, cond_attr=cond_attr, cond_drop_prob=cond_drop_prob,
                         lr=lr, l10_lr_scale=l10_lr_scale)
        self.head = head
        self.lr_head = float(lr_head)
        self.lambda_ground = float(lambda_ground)

    def configure_optimizers(self):
        # Reuse the parent's adapter param groups (including the L10 validity guard) exactly,
        # then attach the head as one more group. Rebuilding them here would risk the two
        # drifting apart.
        opt = super().configure_optimizers()
        opt.add_param_group({"params": list(self.head.parameters()), "lr": self.lr_head})
        return opt

    def training_step(self, batch, batch_idx):
        denoise = super().training_step(batch, batch_idx)

        # _dense() is recomputed rather than threaded out of the parent's step: it is a
        # to_dense() over one batch, negligible beside a 9-layer transformer forward, and the
        # alternative is changing AdapterModule's contract for this subclass's benefit.
        X1, E1, node_mask = self._dense(batch)
        bs = X1.size(0)
        c = getattr(batch, self.cond_attr).to(X1.device).view(bs, -1).float()
        c_norm = (c.view(bs) - self.head.prop_mean) / self.head.prop_std

        ground = torch.nn.functional.mse_loss(self.head(X1, E1, node_mask), c_norm)
        loss = denoise + self.lambda_ground * ground
        self.log_dict({"head/ground": ground, "train/loss": loss},
                      prog_bar=True, on_epoch=True, batch_size=bs)
        return loss
