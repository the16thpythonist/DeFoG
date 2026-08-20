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

import math
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
    "cond_fourier", "xattn_tokens", "xattn_dim", "xattn_heads",
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

    def __init__(self, layers: List[Dict[str, torch.Tensor]], xattn=None):
        self.layers = layers
        #: Optional per-layer node cross-attention, as a list (len == n_layers) of
        #: ``[(row_slice, fn), ...]``. ``fn(X_slice, x_mask_slice) -> delta`` is a
        #: closure over the adapter's own layer module and its condition tokens, which
        #: is why this cannot be a plain tensor like the FiLM parameters: the module has
        #: to run INSIDE the frozen base's stack, on that group's rows only.
        #:
        #: The (slice, fn) list-per-layer form is what "keeps the door open" for
        #: N-branch stacking: a second branch is one more entry at a different offset.
        #: It is refused today (untested, see AdapterComposition) rather than absent.
        self.xattn = xattn

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
        if self.xattn and self.xattn[i]:
            # Built as a full-size delta and added, rather than assigned into X in
            # place: X is needed for the backward pass, and an in-place write on it
            # would either error or silently corrupt the gradient.
            delta = torch.zeros_like(X)
            for sl, fn in self.xattn[i]:
                delta[sl] = fn(X[sl], x_mask[sl])
            X = X + delta
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
        xa = None
        if self.xattn:
            # Zeroing the FiLM gates alone would NOT make a dropped row bypass to the
            # base once cross-attention exists -- its delta does not pass through any
            # gate. Scale it per row instead, or condition dropout silently trains a
            # model whose "unconditional" branch is still conditioned.
            keep = (~mask).to(torch.float32)
            xa = [[(sl, _row_scaled(fn, keep[sl])) for sl, fn in entries]
                  for entries in self.xattn]
        return Modulation(out, xattn=xa)

    @staticmethod
    def stack_groups(mods: Sequence["Modulation"], bs: int, device,
                     guide: Optional["Modulation"] = None) -> "Modulation":
        """Build a ``(N+1)·bs`` modulation: group 0 = the negative branch,
        groups 1..N = each adapter's modulation, concatenated along the batch dim.

        Robust to HETEROGENEOUS key sets across branches: an adapter that lacks an
        interior key (e.g. an output-only adapter composed with an interior-enabled
        one) is treated as gate=0 -> exact bypass at that site. Takes the UNION of
        keys over all branches and zero-fills any key a branch does not define.

        ``guide`` is the AUTOGUIDANCE hook. Without it group 0 is all-zero, i.e. an
        exact bypass to the frozen base -- the unconditional branch of ordinary CFG,
        and the historical behaviour. With it, group 0 carries a DEGRADED CONDITIONAL's
        modulation instead, so the blend pushes away from a weak version of the same
        conditional model rather than away from the unconditional one (Karras et al.'s
        autoguidance; MolGuidance reports it improving structural validity where CFG
        costs it).

        Note the invariant this preserves: a guide whose gates are all zero produces
        exactly the zero row, so autoguidance with an *untrained* guide reduces to
        standard CFG bit-for-bit. That is what ``test_guide_at_init_is_plain_cfg``
        pins down, and it is why the zero-init gate convention had to be kept."""
        all_mods = list(mods) if guide is None else [guide, *mods]
        n_layers = len(all_mods[0].layers)
        for m in all_mods:
            assert len(m.layers) == n_layers, (
                f"modulation layer-count mismatch: {len(m.layers)} != {n_layers}; "
                f"a guide adapter must have the same n_layers as the branches")
        combined = []
        for L in range(n_layers):
            keys = set()
            for m in all_mods:
                keys |= set(m.layers[L].keys())
            d = {}
            for k in sorted(keys):
                ch = next(m.layers[L][k].shape[-1] for m in all_mods if k in m.layers[L])
                # group 0 first, then one row per branch. A missing key (or no guide at
                # all) becomes zeros, which is an exact bypass at that site.
                rows = []
                for m in (guide, *mods):
                    t = None if m is None else m.layers[L].get(k)
                    rows.append(t if t is not None else torch.zeros(bs, ch, device=device))
                d[k] = torch.cat(rows, dim=0)
            combined.append(d)

        # Cross-attention closures are re-homed from "all rows" onto this group's
        # rows. Group g occupies [g*bs, (g+1)*bs); group 0 is the negative branch.
        xa = None
        if any(m.xattn for m in all_mods):
            xa = [[] for _ in range(n_layers)]
            for g, m in enumerate(all_mods if guide is not None else [None, *mods]):
                if m is None or not m.xattn:
                    continue
                off = g * bs
                for L in range(n_layers):
                    for sl, fn in m.xattn[L]:
                        assert sl == slice(None), (
                            "stack_groups expects each source modulation to carry "
                            "whole-batch cross-attention entries; got a pre-sliced one")
                        xa[L].append((slice(off, off + bs), fn))
        return Modulation(combined, xattn=xa)


def _bind_xattn(module, tokens: torch.Tensor):
    """Bind one layer's cross-attention module to this call's condition tokens.

    The result is what ``Modulation`` carries: ``fn(X, x_mask) -> masked delta``. The
    mask is applied to the DELTA, never to X, so a padded row is left exactly as the
    frozen base left it -- the same contract the FiLM path uses."""
    def fn(X, x_mask):
        return x_mask * module(X, tokens)
    return fn


def _row_scaled(fn, keep: torch.Tensor):
    """Wrap a cross-attention closure so its delta is scaled per row (0 = bypass)."""
    def wrapped(X, x_mask):
        return fn(X, x_mask) * keep.view(-1, *([1] * (X.dim() - 1)))
    return wrapped


class NodeConditionCrossAttention(nn.Module):
    """One frozen-base layer's cross-attention from NODES to condition tokens.

    Why this exists. The FiLM path computes every scale/shift/gate from ``c`` alone and
    broadcasts the same numbers to every atom: ``RESEARCH.md`` §2.2 calls this an
    open-loop controller and argues it is why property targeting plateaus -- logP is a
    sum of per-atom environment contributions, so the right action at a given atom
    depends on that atom, and a uniform push cannot express it.

    Here each atom forms a QUERY from its own current representation and pulls a weighted
    mixture of condition tokens.

    BE PRECISE ABOUT WHAT THIS BUYS, because the obvious claim is wrong. "Two atoms get
    different deltas" is ALREADY true of FiLM -- its delta at node i is
    ``g(c) * (s(c) * X_i + b(c))``, which varies with X_i. Both mechanisms are node-local
    functions of ``(X_i, c)``. The difference is the FORM of that function: FiLM applies
    one DIAGONAL AFFINE map, identical for every atom, whereas cross-attention is a
    content-addressed nonlinear lookup -- the atom's own representation selects which
    condition tokens it reads, so different kinds of atom can receive qualitatively
    different corrections rather than the same correction scaled per channel.

    Because those representations come out of the frozen base's own message passing, by
    the middle layers a query already reflects the atom's neighbourhood, not just its
    element. It is still NOT the global running total that a closed-loop readout
    (PLAN.md Wave 5) would provide: nothing here tells the adapter what the partial
    molecule's logP currently is.

    Structure follows IP-Adapter: a SEPARATE attention path added to the frozen block's
    output, with a ZERO-INITIALISED output projection, so at init the delta is exactly
    zero and the base is reproduced bit-for-bit. That is the same invariant the zero-init
    FiLM gate provides, and the product-of-experts composition depends on it.

    Applied at the layer OUTPUT (via :meth:`Modulation.apply`) rather than spliced inside
    the block: the frozen base's own attention is untouched, so nothing about its
    computation changes and the only new tensor is an additive residual.
    """

    def __init__(self, dx: int, d_tok: int, n_heads: int = 8):
        super().__init__()
        if dx % n_heads:
            raise ValueError(f"dx={dx} not divisible by n_heads={n_heads}")
        self.dx, self.d_tok, self.n_heads = int(dx), int(d_tok), int(n_heads)
        # Pre-norm on the query side only. The frozen base's activation scale is not
        # ours to assume, and an unnormalised query would make the attention logits
        # depend on it; the keys/values come from our own tokens and are already tame.
        self.norm = nn.LayerNorm(dx)
        self.q = nn.Linear(dx, dx)
        self.k = nn.Linear(d_tok, dx)
        self.v = nn.Linear(d_tok, dx)
        self.out = nn.Linear(dx, dx)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)          # zero-init => exact no-op at init

    def forward(self, X: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """``X`` (B,n,dx), ``tokens`` (B,m,d_tok) -> delta (B,n,dx)."""
        B, n, _ = X.shape
        h, dh = self.n_heads, self.dx // self.n_heads
        q = self.q(self.norm(X)).view(B, n, h, dh).transpose(1, 2)          # B,h,n,dh
        k = self.k(tokens).view(B, -1, h, dh).transpose(1, 2)               # B,h,m,dh
        v = self.v(tokens).view(B, -1, h, dh).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, n, self.dx)
        return self.out(o)
        # NOTE: no mask over the KEYS -- condition tokens are always present. Padded
        # QUERY rows are zeroed by the caller's x_mask, same as the FiLM delta.

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
                 cond_encoder: Optional[nn.Module] = None,
                 cond_fourier: int = 0, xattn_tokens: int = 0,
                 xattn_dim: int = 128, xattn_heads: int = 8):
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
        self.cond_fourier = int(cond_fourier)     # Fourier bands on the condition (0 = off)
        self.xattn_tokens = int(xattn_tokens)     # node->condition cross-attention (0 = off)
        self.xattn_dim = int(xattn_dim)
        self.xattn_heads = int(xattn_heads)

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

        # --- Fourier features on the (normalised) condition ---------------------
        # The trunk otherwise sees the property as ONE raw float while the flow-time
        # gets a 64-dim sinusoidal embedding. Tancik et al. (arXiv:2006.10739): an MLP
        # over a low-dimensional input converges to the high-frequency part of its
        # target impractically slowly, so the learned map c -> modulation gets the
        # smooth part long before the part that distinguishes 3.5 from 4.2. That is
        # the measured failure signature -- bias without tracking.
        #
        # LOW-DIMENSIONAL ONLY, which is the regime the result is about. A fingerprint
        # or spectrum condition goes through its own encoder instead; expanding 512 hash
        # bits into bands would be both meaningless and enormous.
        if self.cond_fourier:
            if cond_encoder is not None:
                raise ValueError("cond_fourier is for a raw low-dimensional condition; "
                                 "this adapter already has a cond_encoder")
            if cond_dim > 8:
                raise ValueError(f"cond_fourier with cond_dim={cond_dim}: Fourier features "
                                 f"are for LOW-dimensional conditions (<= 8)")
            # Frequencies 0.5, 1, 2, ... cycles per z-unit on the z-scored condition.
            #
            # THE BANDWIDTH IS A CEILING, NOT A KNOB TO RAISE. The band vector must stay
            # SIMILAR for nearby targets (so the adapter interpolates to targets it never
            # saw, which is most of them) while being clearly DIFFERENT for far ones.
            # Measured cosine similarity of the feature vector on ZINC logP
            # (cond_std 1.158, targets spanning ~9.3 units, i.e. ~8 z):
            #
            #     n   f_max   cycles/range   cos(0.1 logP)   cos(2.5 logP)
            #     2     1.0        8            0.910            0.722     under-resolved
            #     3     2.0       16            0.763            0.358     <- default
            #     4     4.0       32            0.432            0.087
            #     6    16.0      128            0.105           -0.099     aliased
            #     8    64.0      512           -0.039            0.162     noise
            #
            # n=3 keeps neighbours correlated while separating the ends; by n=6 a
            # 0.1-logP step has already decorrelated the features and the mapping is
            # memorisation, not interpolation. The raw normalised scalar rides alongside
            # and carries the smooth component, so the bands only have to add resolution.
            # `test_fourier_bandwidth_keeps_neighbours_correlated` fails if this aliases.
            self.register_buffer("fourier_freqs",
                                 0.5 * (2.0 ** torch.arange(self.cond_fourier, dtype=torch.float32)))
        fourier_dim = 2 * cond_dim * self.cond_fourier
        cond_in = encoded_dim + fourier_dim + (time_emb_dim if time_conditioned else 0)
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

        # --- node -> condition cross-attention (one module per layer) -----------
        if self.xattn_tokens:
            self.tok = nn.Linear(hidden, self.xattn_tokens * self.xattn_dim)
            self.xattn = nn.ModuleList([
                NodeConditionCrossAttention(dims["dx"], self.xattn_dim, self.xattn_heads)
                for _ in range(n_layers)
            ])

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

    def _fourier(self, c_norm: torch.Tensor) -> torch.Tensor:
        """(B, D) normalised condition -> (B, 2*D*n_bands) sin/cos features."""
        ang = 2.0 * math.pi * c_norm.unsqueeze(-1) * self.fourier_freqs   # (B, D, F)
        return torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(1)

    def forward(self, c: torch.Tensor, t: Optional[torch.Tensor] = None) -> Modulation:
        c_norm = self.normalize(c.float())
        # The encoder replaces the raw condition; Fourier features ACCOMPANY it. The raw
        # normalised scalar is kept alongside the bands so nothing the trunk could learn
        # before is taken away -- turning the feature on cannot lose information.
        parts = [self.cond_encoder(c_norm) if self.cond_encoder is not None else c_norm]
        if self.cond_fourier:
            parts.append(self._fourier(c_norm))
        if self.time_conditioned:
            assert t is not None, "time_conditioned adapter requires t"
            parts.append(timestep_embedding(t.reshape(-1, 1), self.time_emb_dim))
        h = self.trunk(torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0])

        xattn = None
        if self.xattn_tokens:
            tokens = self.tok(h).view(h.size(0), self.xattn_tokens, self.xattn_dim)
            # slice(None) here; Modulation.stack_groups re-homes each closure onto its
            # group's rows when the branches are batched together.
            xattn = [[(slice(None), _bind_xattn(mod, tokens))] for mod in self.xattn]

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
        return Modulation(layers, xattn=xattn)

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
                    base_token=self.base_token,
                    cond_fourier=self.cond_fourier, xattn_tokens=self.xattn_tokens,
                    xattn_dim=self.xattn_dim, xattn_heads=self.xattn_heads)

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


@dataclass
class GuideBranch:
    """The negative branch for autoguidance: a DEGRADED conditional model.

    Ordinary CFG blends away from the frozen unconditional base. Autoguidance blends
    away from a deliberately worse version of the *conditional* model -- undertrained,
    or lower-capacity -- so that the flaws the two share cancel and only the quality
    difference is amplified. The guide is normally conditioned on the SAME target as
    the branch it negates.

    Deliberately NOT a :class:`ConditionBranch`. That type carries a ``weight``, and a
    weight on group 0 does nothing: ``_blend_logp`` reads group 0 as the baseline the
    other branches are measured against, so a ``weight`` field here would be silently
    ignored. Having no such field is the difference between a knob that does nothing
    and a knob that *looks* like it does something.
    """

    adapter: AdaLNAdapter
    condition: torch.Tensor


class AdapterComposition:
    """N-branch product-of-experts spec consumed by ``denoise_step`` /
    ``AdaptedSampler``. ``mode='product'`` sums the log-ratios; ``'mean'`` averages
    (recommended for N>1 to keep the effective uncond coefficient bounded)."""

    def __init__(self, branches: Sequence[ConditionBranch], base=None, mode: str = "product",
                 blend_space: str = "prob", guide: Optional[GuideBranch] = None):
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
        self.guide = guide
        if guide is not None:
            # One guide negates one branch. With N>1 there is no defined answer to
            # "which target is the guide conditioned on", and silently reusing one
            # branch's target would make the other branches' guidance mean something
            # nobody chose.
            if len(self.branches) != 1:
                raise ValueError(
                    f"autoguidance is defined for a single branch; got {len(self.branches)}. "
                    f"Compose without a guide, or run one property at a time.")
            # blend_space="rate" reads group 0 as the base in a way autoguidance breaks:
            # `_blend_rates` zeroes any transition where R[0] == 0, on the reasoning that
            # a structurally-forbidden transition stays forbidden. With a guide, R[0] is
            # the GUIDE's rate matrix, so that guard would silently start deriving the
            # forbidden set from a degraded model. Untested, and the combination has no
            # use: the w>1 sweep autoguidance exists to enable is exactly what rate space
            # cannot do.
            if blend_space != "prob":
                raise ValueError(
                    f"autoguidance requires blend_space='prob', got {blend_space!r}. "
                    f"In rate space the forbidden-transition guard would be derived from "
                    f"the guide instead of the base, and w>1 is broken there anyway.")
        # Cross-attention branches: single-branch only, for now, ON PURPOSE.
        # Modulation carries them as (row_slice, closure) per layer, so a second branch
        # is structurally just one more entry at a different offset -- the door is open.
        # It is refused rather than allowed because nothing tests it, and an untested
        # composition path that "should work" is how a wrong number gets shipped. Delete
        # this guard together with a test that stacks two xattn adapters.
        n_xattn = sum(1 for b in self.branches if getattr(b.adapter, "xattn_tokens", 0))
        if n_xattn > 1:
            raise NotImplementedError(
                f"{n_xattn} branches use node cross-attention; stacking them is untested. "
                f"Compose at most one cross-attention adapter (FiLM-only adapters stack "
                f"as before), or add a test and remove this guard.")
        if base is not None:
            for b in self.branches:
                b.adapter.check_compatible(base)
            if guide is not None:
                guide.adapter.check_compatible(base)

    def __len__(self):
        return len(self.branches)

    @staticmethod
    def _broadcast_condition(condition, bs: int, device) -> torch.Tensor:
        c = torch.as_tensor(condition, dtype=torch.float32, device=device)
        if c.dim() == 1:
            c = c.unsqueeze(0)
        if c.size(0) == 1 and bs > 1:
            c = c.expand(bs, -1)
        assert c.size(0) == bs, f"branch condition batch {c.size(0)} != {bs}"
        return c

    @torch.no_grad()
    def build_modulation(self, bs: int, t: torch.Tensor) -> Modulation:
        """Combined ``(N+1)·bs`` modulation: group 0 = the negative branch (the frozen
        base by default, the guide under autoguidance), group i = adapter_i."""
        device = t.device
        mods = [br.adapter(self._broadcast_condition(br.condition, bs, device), t=t)
                for br in self.branches]
        guide_mod = None
        if self.guide is not None:
            guide_mod = self.guide.adapter(
                self._broadcast_condition(self.guide.condition, bs, device), t=t)
        return Modulation.stack_groups(mods, bs, device, guide=guide_mod)

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
