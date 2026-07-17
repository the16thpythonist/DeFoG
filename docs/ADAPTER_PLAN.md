> **Revision 1 — MAJOR verifier fixes folded in + implemented (tests pass).**
> The 3 convergent MAJOR findings from the final verifiers are now reflected in
> this plan and in the shipped code:
> 1. **Post-blend rate re-stabilization** — `_blend_rates` applies `nan_to_num` +
>    `>1e5 → 0` after the geometric blend (`model.py`), so `w>1` with a small
>    `R_uncond` can't explode into `multinomial`.
> 2. **`composition=None`/empty is the true off-switch** — `denoise_step` only
>    enters the composed branch when `composition is not None and len>0`; empty/None
>    falls through byte-identically to the legacy body (verified: empty-composition
>    `AdaptedSampler` == plain `Sampler` under a fixed seed).
> 3. **N=1 conformance is scoped** — the post-blend forbidden-mask makes N=1 an
>    intentional (correctness-improving) divergence near t→1; conformance to the
>    shipped 2-branch formula is tested only where the mask provably can't fire.
>
> Implemented in `defog/core/adapter.py` (+ `transformer.py`/`model.py`/`sampler.py`
> wiring) and covered by `tests/test_adapter.py` (6/6 passing: null=base exact,
> live-modulation, batched (N+1)·B bypass, empty-composition fall-through,
> 2-branch sampling, save/load).

# Frozen-Base AdaLN/FiLM CFG-Adapter — Final Implementation Plan

## 0. Grounding (verified against source)

- **Base denoiser** `DeFoGModel` (`defog/core/model.py`) wraps a `GraphTransformer` at `self.model` (`model.py:235`). Legacy conditioning is *input-concat CFG*: `cond_encoder`+`cond_norm`+`null_embedding` (`model.py:274-281`) produce a `cond_emb_dim` vector concatenated into the global-`y` **input** (`input_dims["y"] = cond_emb_dim + 1 + …`, `model.py:222-227`). This requires `cond_dim>0` baked in at construction; a frozen unconditional base (`cond_dim=0`) yields `y = zeros(bs,0)` (`model.py:355,564`).
- **Transformer is post-LN** (`layers.py:341,345,349,354,359,364`): `X = normX2(X + ffn)` etc. The global `y` already FiLM-modulates X and E inside every `NodeEdgeBlock` (`layers.py:218-220,237-239`), so `y` is a high-leverage conditioning channel (it re-modulates X/E in the *next* layer's attention). `GraphTransformer.forward` masks X/E once before the layer loop (`transformer.py:191-192`) and masks the result once at the end (`:217`) — **never between layers**; the per-layer loop is `transformer.py:195-196`.
- **CFG lives entirely in `denoise_step`** (`model.py:780-936`). Crucially:
  - `pred_X`/`pred_E` are computed **once** from a single always-executed conditional forward (`model.py:850-855`), optionally rectified by `posterior_transform` in place (`:860-861`).
  - The `guidance_scale` block (`:871-908`) runs a **second** uncond forward via `_embed_condition(None,…)` and blends only the **rate matrices** geometrically in log-space (`:899-908`). **It never touches `pred_X`/`pred_E`.**
  - The final-step MAP decode (`:920-922`) argmaxes `pred_X`/`pred_E` — i.e. **CFG has zero effect on the terminal decode today** (only `posterior_transform` does, because it reassigns `pred_X`/`pred_E`).
- **`_compute_extra_data`** (`model.py:985-1017`) depends only on `X_t,E_t,t` — never on `y`. So it is identical across all conditioning branches and can be computed once and tiled.
- **`Sampler._advance`** gates legacy CFG on `use_cfg` (`sampler.py:121`), and `use_cfg` is **always False when `cond_dim==0`** (`model.py:550-565`) — the exact model class this feature targets.
- **`RateMatrixDesigner.compute_rate_matrices`** (`rate_matrix.py:74-129`) internally draws a stochastic `X_1` sample (`:104`) and `_stabilize` (`:390-423`) zeros structurally-forbidden rates and rates `>1e5`.
- **Frozen-adapter precedent**: `_GuidanceModuleBase` freezes the base (`guidance.py:388-389`) and strips `base.*` from checkpoints (`:391-396`); `configure_optimizers` is hardcoded to `self.h` (`:398-399`). **PoE stacking** already exists at the posterior level in `CompositeGuidance` (`guidance.py:333-372`).
- **`DeFoGModel.save()`** dumps `self.state_dict()` **unfiltered** (`model.py:1102`); `load()` calls `load_state_dict(...)` with default `strict=True` (`:1155`). Any `nn.Module` submodule attached to the base *will* serialize into the base `.ckpt` and break loading old checkpoints — decisive for the ownership decision below.
- Default ZINC dims (`model.py:239-246`, `hidden_dim=256`): `dx=256, de=64, dy=64`, `n_layers=9`.
- **Experiment reality** (`experiments/fingerprint_guidance__zinc.py`): imports `LatentGuidanceModule, ExactGuidance, GuidedSampler, Sampler` (`:63`); `IntermediateReporter._probe` builds `ExactGuidance(pl_module.h,…)` → `GuidedSampler(pl_module.base,…)` → `guidance.save(…)` (`:272-298`); `pl.Trainer(..., callbacks=[reporter])` (`:441`). It does **not** use `MoleculeDomain`, `ConditionalSizeDistribution`, or `EMACallback` — those live in `conditional_training__zinc.py` (`:57-60,273,281,303`).

---

## 1. Design decisions on forks (a)–(g)

### (a) Retrofit zero-init gated-FiLM onto the existing frozen base ✅ (vs bake-in + retrain)
**RETROFIT.** Add a gated-FiLM side-branch on each frozen layer's **output**; frozen weights are never touched or retrained. This satisfies HARD REQ 1 (base is train-once), lets `zinc_uncond_*.ckpt` be reused verbatim (the same ckpt the fingerprint experiment loads), and gives an **exactly** shared, bit-identical `R_uncond` across arbitrarily many independently-trained adapters — the precise property PoE stacking (REQ 3) needs.

*Why not bake-in + retrain:* true DiT AdaLN-zero assumes **pre-LN** blocks whose sublayers are identity at init; our base is **post-LN and already trained**, so we cannot drop/replace sublayers — only add a gated branch. Bake-in would cost the retrain we are explicitly avoiding, invalidate every checkpoint, and buy little (the gated-FiLM-on-output form is an exact no-op at init regardless of norm placement). Keep bake-in as a **documented Phase-8 fallback** only if the retrofit demonstrably underfits.

*Consequence:* injection is a gated residual FiLM on the block output, `h' = h + mask ⊙ (gate ⊙ (scale ⊙ h + shift))`, not the pre-LN `x + gate·attn(modulate(norm(x)))`. Same goal (per-block conditional affine tilt), exact bypass at init.

### (b) Injection sites + adapter architecture
**Inject one gated FiLM per transformer layer, on all three block-output streams (X, E, y), in the `GraphTransformer.forward` loop** — `layers.py` is left untouched. This resolves the internal inconsistency both plans carried (block-output vs per-sublayer): we commit to **block-output, one FiLM per layer**, which is the minimal, cleanest wiring and keeps `NodeEdgeBlock`/`XEyTransformerLayer` frozen and unedited. (Per-sublayer is a later capacity dial, §9.)

- Modulate **all three streams**: X (`dx=256`) for atom steering; E (`de=64`, cheap, broadcast over `(n,n)`) for bond steering; y (`dy=64`) as the highest-leverage per-param channel (it re-modulates X/E in every subsequent layer). `streams` is configurable so E can be dropped if it hurts validity.
- **Permutation equivariance is preserved**: `c` is graph-level and the modulation is *shared across all nodes/edges* (broadcast), never per-node — verified against `NodeEdgeBlock`'s attention and edge-symmetrization.
- **Adapter architecture** (small, ~2–3M params ≪ base):
  - Shared trunk: `Linear(cond_in → H) → SiLU → LayerNorm → Linear(H → H) → SiLU`, `H≈256` (mirrors `cond_encoder`+`cond_norm`).
  - **Time-conditioned (default on, ablatable):** `cond_in = cond_dim + 64`, concatenating `timestep_embedding(t, 64)` (reuse `layers.py:369`) so the tilt can differ early vs late in denoising (the codebase repeatedly finds late-time behavior decisive). Ablate in the go/no-go phase.
  - Per-layer, per-stream heads. **Zero-init resolution:** each `(stream)` head is *two* `nn.Linear(H → channel)` — a `scale/shift` linear (normal init) and a **separate `gate` linear with weight=0 and bias=0**. Because the delta is `gate ⊙ (…)`, gate=0 makes the whole delta exactly 0 at init regardless of scale/shift — no fragile row-slice zeroing of a fused head.

### (c) How null = base is guaranteed (two independent guarantees, both used)
1. **Structural bypass (exact, load-bearing for composition):** the unconditional branch applies **no modulation at all** — the frozen path runs unchanged, bit-identical to the base. This is what every stacked adapter's `R_uncond` shares. Strictly stronger than the base's learned `null_embedding` (which only *approximates* the unconditional manifold and would drift per-adapter).
2. **Gate zero-init (stable start):** at training step 0 the *conditional* branch also equals the base for every `c`; training moves gates off 0.
- **Delta-masking (correctness fix, resolves a Plan-2 blocker):** the FiLM delta is masked, **not** the hidden state: `h' = h + mask ⊙ film(h;c)`. At gate=0 the delta is 0 so `h' = h` **exactly** (no deviation from the frozen base's own padding behavior), and when gate≠0 no injected value ever lands on a padded position to poison the next layer's `Xtoy`/`Etoy` pools. Plan 2's `(h+film)*mask` was wrong precisely because it re-zeroes the base's own (nonzero, LayerNorm-bias) padding values and thus breaks the init==base guarantee. The `y` stream is global (per-graph) and takes no mask.
- **Condition-dropout is optional, default 0.** The uncond branch *is* the frozen base, so `p(x|null)=p(x)` holds by construction and CFG dropout is not needed for correctness. Keep a `cond_drop_prob` knob (default 0, a small 0.05 available as a regularizer); dropped rows route to the structural bypass.

### (d) Wiring + N-branch composition — detailed in §4.

### (e) Adapters are **external, never stored on the frozen base** (resolves the save/load contradiction in *both* plans). `AdaLNAdapter`, `AdapterComposition`, `AdapterRegistry` live outside `DeFoGModel` and are passed per-call, exactly as `ExactGuidance` is passed to `GuidedSampler`. **No `attach_adapter`/`self._adapters`/`nn.ModuleDict` on the base** — that API was dead (the sampling path reads adapters from the composition, not the model) and would leak adapter weights into `base.save()` and break `base.load()` on old checkpoints. `DeFoGModel.save/load` stay genuinely unchanged. Detailed in §2.

### (f) `AdapterModule(pl.LightningModule)` trains **only the adapter** with the **base's own denoising CE loss** (`base.train_loss`) — a direct teacher-forced conditional denoiser `p(x1|x_t,c)`, **not** the Bregman/positive-biased-pairing objective that `LatentGuidanceModule` needed (this sidesteps the vanishing-gradient pairing problem that hobbled the prior fingerprint attempt). Detailed in §3.

### (g) **v1 scope = `cond_dim=0` frozen bases.** All new arguments default to current behavior; the legacy `cond_dim>0` input-concat CFG, `ExactGuidance`/`CompositeGuidance`, `GuidedSampler`, `FeynmanKacSampler`, and `save/load` are untouched. **Jointly composing an adapter's N-branch blend with a `cond_dim>0` base's own 2-branch input-CFG in one `denoise_step` is explicitly out of scope for v1** (it would need a 2×(N+1) cross-product blend with two independent weight systems — neither plan specified it, and it is not needed for the fingerprint/cluster goals). An adapter can still be *trained* on a `cond_dim>0` base's frozen backbone (the modulation path is orthogonal to the y-input path) and *sampled* by passing `condition=None` so `y_t` is the base null — but combining both CFG mechanisms simultaneously is not supported in v1. This resolves the self-contradiction both plans carried.

---

## 2. Interface / API (`defog/core/adapter.py`, new)

```python
class Modulation:
    """Per-layer FiLM params for one (possibly stacked) batch.
    layers: list over n_layers of dict(scaleX,shiftX,gateX, scaleE,shiftE,gateE,
    scaley,shifty,gatey), each (B, dx|de|dy). Group-0 (uncond) rows carry
    all-zero gates => zero delta."""
    def apply(self, i, X, E, y, x_mask, e_mask):
        m = self.layers[i]
        X = X + x_mask * (m["gateX"][:, None] * (m["scaleX"][:, None] * X + m["shiftX"][:, None]))
        E = E + e_mask * (m["gateE"][:, None, None] * (m["scaleE"][:, None, None] * E + m["shiftE"][:, None, None]))
        y = y +           m["gatey"]            * (m["scaley"]            * y + m["shifty"])
        return X, E, y

class AdaLNAdapter(nn.Module):
    """Zero-init gated-FiLM adapter over a FROZEN base's transformer stack.
    Maps a condition c -> per-layer {scale,shift,gate} for X,E,y. Exact no-op at
    init and whenever bypassed -> base reproduced bit-for-bit."""
    def __init__(self, cond_dim, n_layers, dims,            # dims={"dx":256,"de":64,"dy":64}
                 hidden=256, time_conditioned=True, streams=("X","E","y"),
                 cond_mean=None, cond_std=None, name="", cond_type=""): ...
    @classmethod
    def for_base(cls, base, cond_dim, **kw):
        hp = dict(base.hparams)                              # NOT base.model.dx (doesn't exist)
        h = hp["hidden_dim"]; dims = {"dx": h, "de": h // 4, "dy": h // 4}
        return cls(cond_dim, hp["n_layers"], dims, **kw)
    def forward(self, c, t=None) -> Modulation:
        c = self.normalize(c)                                # normalize INTERNALLY (see below)
        ...                                                  # trunk(+time_emb) -> per-layer heads
    def normalize(self, c):                                  # (c - cond_mean)/cond_std, stored on adapter
    def check_compatible(self, base): ...                    # assert dims/n_layers match
    def save(self, path): ...                                # {state_dict, config, cond_mean, cond_std, name, cond_type}
    @classmethod
    def load(cls, path, device="cpu"): ...

@dataclass
class ConditionBranch:
    adapter: AdaLNAdapter
    condition: torch.Tensor      # (B, cond_dim) RAW target (adapter normalizes internally)
    weight: float                # w_i, the per-condition CFG scale

class AdapterComposition:
    """N-branch product-of-experts spec consumed by denoise_step / AdaptedSampler.
    mode='product' -> sum of log-ratios; 'mean' -> averaged (mirrors CompositeGuidance)."""
    def __init__(self, branches, base=None, mode="product"):
        self.branches, self.mode = list(branches), mode
        if base is not None:                                 # fail-fast cross-base check
            for b in self.branches: b.adapter.check_compatible(base)
    def build_batched(self, base, X_t, E_t, y_t, t, node_mask):
        """-> (Xb,Eb,yb,nmb, Modulation_b) for the (N+1)*B stacked forward:
        group 0 = uncond bypass (zero-gate rows); group i = adapter_i(c_i, t)."""

class AdapterRegistry:
    def register(self, name, adapter): ...
    def get(self, name) -> AdaLNAdapter: ...
    @classmethod
    def load_dir(cls, path) -> "AdapterRegistry": ...        # hot-swap by name at inference
```

**Normalization contract (resolves Plan-1 minor):** `AdaLNAdapter.forward` **always normalizes internally** from its stored `cond_mean/cond_std` (mirroring `ExactGuidance._normalized_target`). `ConditionBranch.condition` and the training `training_step` therefore both pass **raw** conditions — no external `.normalize()` in the call path, eliminating the train/inference mismatch risk. `.normalize()` stays public only for tests.

Modified signatures:
```python
# transformer.py
def GraphTransformer.forward(self, X, E, y, node_mask, modulation: Optional[Modulation]=None): ...
# model.py
def DeFoGModel.forward(self, noisy_data, extra_data, node_mask, cond_modulation=None): ...
def DeFoGModel.denoise_step(self, ..., composition: Optional[AdapterComposition]=None): ...
# sampler.py
class AdaptedSampler(Sampler):  # stores composition; passes it unconditionally
```

---

## 3. Training

`AdapterModule` in `defog/core/adapter.py`, subclassing `_GuidanceModuleBase` for the freeze/checkpoint plumbing, **overriding `configure_optimizers`** (the base hardcodes `self.h`; ours trains `self.adapter`):

```python
class AdapterModule(_GuidanceModuleBase):
    def __init__(self, base, adapter, cond_attr="cond",
                 cond_mean=None, cond_std=None, cond_drop_prob=0.0, lr=2e-4):
        super().__init__()
        self._freeze_base(base)                              # guidance.py:388 -> requires_grad_(False)
        self.adapter = adapter
        self.register_buffer("cond_mean", torch.as_tensor(cond_mean))
        self.register_buffer("cond_std",  torch.as_tensor(cond_std))
        self.cond_attr, self.cond_drop_prob, self.lr = cond_attr, cond_drop_prob, lr

    def configure_optimizers(self):                          # OVERRIDE (base uses self.h)
        return torch.optim.AdamW(self.adapter.parameters(), lr=self.lr, weight_decay=1e-5)
    # on_save_checkpoint inherited verbatim: strips base.* (guidance.py:391-396). Adapter ckpt stays small.

    def training_step(self, batch, _):
        self.base.eval()
        X1, E1, node_mask = self._dense(batch)               # guidance.py:401-405
        bs, device = X1.size(0), X1.device
        c = getattr(batch, self.cond_attr).to(device).view(bs, -1).float()   # RAW; adapter normalizes
        y0 = torch.zeros(bs, 0, device=device)               # base is cond_dim=0
        with torch.no_grad():
            noisy = self.base._apply_noise(X1, E1, y0, node_mask)            # same t schedule as base
            extra = self.base._compute_extra_data(noisy)
        mod = self.adapter(c, t=noisy["t"])                  # (time-conditioned)
        if self.cond_drop_prob:                              # optional: zero-gate dropped rows -> bypass
            mod = _bypass_rows(mod, torch.rand(bs, device=device) < self.cond_drop_prob)
        pred = self.base.forward(noisy, extra, node_mask, cond_modulation=mod)
        loss = self.base.train_loss(pred_X=pred.X, pred_E=pred.E, pred_y=pred.y,
                                    true_X=X1, true_E=E1, true_y=y0, node_mask=node_mask)
        self.log("adapter/loss", loss, prog_bar=True, on_epoch=True, batch_size=bs)
        return loss
```

- **Data plumbing** mirrors `fingerprint_guidance__zinc.py:379-395`: attach `g.cond = φ(x1)` per graph (the 128-bit Morgan fingerprint now; the soft cluster-membership vector later — same pipeline, different `φ`). `cond_mean/cond_std` = per-dim stats of the fingerprint matrix (`fp.mean(0)/fp.std(0)`, `:403-404`).
- **EMA is a non-issue** (resolves both plans' open question): `EMACallback` already filters `requires_grad` params (`callbacks.py:1272-1282`) and `_freeze_base` sets the base to `requires_grad_(False)`, so `EMACallback(decay=…)` automatically scopes to the adapter with zero extra wiring. Add it to the trainer callbacks.

**Experiment `experiments/adapter_training__zinc.py`** (clone of `fingerprint_guidance__zinc.py`, honest scope — resolves Plan-1 finding #3):
- **Reuse verbatim** the fingerprint infra that *actually* lives in that file: `morgan_matrix`, `topk_tanimoto_neighbors`, `tanimoto_to_target`, `decode_and_fp`, the fp/neighbor caches, target selection, `cond_mean/cond_std`, and the Tanimoto-lift aggregation/plots (`:536-544`).
- **Swap the training object**: load frozen base via `DeFoGModel.load` (assert `cond_dim==0`), `adapter = AdaLNAdapter.for_base(base, cond_dim=FP_BITS, cond_mean=…, cond_std=…)`, wrap in `AdapterModule`, fit. Replace `LatentGuidanceModule` (`:405`).
- **Rewrite the `IntermediateReporter._probe`** (it is a rewrite, not a one-line swap): replace `ExactGuidance(pl_module.h,…)` + `GuidedSampler` + `guidance.save` (`:272-298`) with `AdaptedSampler(pl_module.base, AdapterComposition([ConditionBranch(pl_module.adapter, target, w)]))` + `pl_module.adapter.save`.
- `EMACallback`, and optionally `MoleculeDomain`/`ConditionalSizeDistribution` if size-aware previews are wanted, are pulled from `conditional_training__zinc.py`'s pattern (imports `EMACallback, ConditionalSizeDistribution` from `defog.core` and `MoleculeDomain` from `defog.domains`), **not** from the fingerprint script.

---

## 4. Inference & composition

### Wiring
- **`GraphTransformer.forward(..., modulation=None)`**: after each layer in the loop (`transformer.py:195-196`), `if modulation is not None: X,E,y = modulation.apply(i, X,E,y, x_mask, e_mask)` using the `x_mask/e_mask` already computed at `:189-190`. `None` ⇒ byte-identical current behavior.
- **`DeFoGModel.forward(..., cond_modulation=None)`**: pass straight to `self.model(...)`.
- **`Sampler`**: `AdaptedSampler(Sampler)` stores `self.composition` and threads it into `denoise_step` **unconditionally** (NOT gated by `use_cfg` — critical, since `use_cfg==False` for every `cond_dim=0` base). `_advance` passes `composition=self.composition` (only `AdaptedSampler` sets it; base `Sampler` leaves it `None`). `AdaptedSampler` is thin sugar analogous to `GuidedSampler` (`sampler.py:360-379`); it composes with inpainting/refinement via subclassing, and `posterior_transform` still flows through.
- **`condition` vs composition**: the `condition` arg to `sample()` continues to drive size distribution (`ConditionalSizeDistribution`) via `_prepare_generation`; the per-branch *modulation* conditions live in the composition. For the (size-independent) fingerprint verification these are decoupled; for the cluster-membership goal, pass a representative condition to `sample()` for sizing if size correlates.

### `denoise_step` composition branch (early, self-contained — resolves the leading-forward + terminal-decode issues)
When `composition is not None`, take a **dedicated branch that does NOT run the base's leading conditional forward** (`model.py:849-866` is skipped entirely — no stale `pred_X` and no wasted `B`-sized forward). It reuses only the eta/omega set-up (`:835-838`) and the `_compute_step_probs`/`sample_from_probs`/re-mask tail (`:910-936`):

```python
N = len(composition.branches)
Xb, Eb, yb, nmb, mod_b = composition.build_batched(self, X_t, E_t, y_t, t, node_mask)  # (N+1)*B
nd = {"X_t": Xb, "E_t": Eb, "y_t": yb, "t": t.repeat(N+1, 1), "node_mask": nmb}
extra_b = _tile(self._compute_extra_data({"X_t":X_t,"E_t":E_t,"y_t":y_t,"t":t,"node_mask":node_mask}), N+1)  # compute ONCE, tile
pred = self.forward(nd, extra_b, nmb, cond_modulation=mod_b)         # ONE frozen forward, all N+1 branches
pX = F.softmax(pred.X, -1).view(N+1, bs, *pred.X.shape[1:])
pE = F.softmax(pred.E, -1).view(N+1, bs, *pred.E.shape[1:])

# Optional per-branch posterior_transform (keeps exact-guidance stacking; matches model.py:860-861,890-893)
if posterior_transform is not None:
    for g in range(N+1): pX[g],pE[g] = posterior_transform(pX[g],pE[g], _slice(nd,g), node_mask)

# Per-branch rate matrices (each branch draws its own X_1 sample, as the shipped 2-branch code already does)
RX = []; RE = []
for g in range(N+1):
    rx, re = self.rate_matrix_designer.compute_rate_matrices(t, node_mask, X_t, E_t, pX[g], pE[g])
    RX.append(rx); RE.append(re)
RX = torch.stack(RX); RE = torch.stack(RE)                          # (N+1, bs, ...)

w = torch.tensor([b.weight for b in composition.branches], device=X_t.device)
def blend(R):
    lu, lc = torch.log(R[0] + 1e-6), torch.log(R[1:] + 1e-6)       # uncond, conds
    dev = torch.einsum("i,i...->...", w, lc - lu)                  # Σ w_i (logRc_i - logRu)
    if composition.mode == "mean": dev = dev / N                   # symmetric with CompositeGuidance
    Rb = (lu + dev).exp()
    Rb = torch.where(R[0] == 0, torch.zeros_like(Rb), Rb)          # forbidden transitions stay forbidden
    return Rb
R_t_X, R_t_E = blend(RX), blend(RE)
```

- **N=1, product, w=w** reduces to the shipped 2-branch formula (`model.py:899-908`) bit-for-bit (verified algebraically by all four verifiers).
- **`mode="mean"` is the recommended default for N>1** (product lets `1−Σw_i` go strongly negative and, via `_stabilize`'s `R>1e5 → 0` clamp, silently drops rates as N grows). Mean keeps the effective uncond coefficient bounded. Both modes are exposed.
- **Structural-zero safety:** re-apply the **uncond branch's own post-`_stabilize` zero mask** (`torch.where(R[0]==0,…)`) after the blend — a well-defined, single-branch mask representing physically forbidden transitions (`p(x_t|x_1)=0`), avoiding the ill-posed "intersection of independently-sampled masks." Independent per-branch `X_1` sampling (and its variance compounding) is inherited from the shipped 2-branch code and tolerated for small N; **common-random-numbers** (a shared `X_1` sample across branches) is a documented Phase-5 hardening if large-N variance bites, requiring one optional arg on `compute_rate_matrices`.

### Terminal-step decode (intentional, adapter-path-only new behavior — resolves the blocker)
The legacy path decodes the terminal step from the single conditional `pred_X`/`pred_E`, so CFG has no terminal effect (documented in §0). For the adapter path there is **no single conditional branch**, so we define the terminal marginal by the **same PoE blend applied to the clean-graph log-probabilities**:

```python
if s[0].item() >= 1.0 - 1e-6:
    def blend_logp(p):                                # p: (N+1, bs, ...) softmax marginals
        lu, lc = torch.log(p[0]+1e-8), torch.log(p[1:]+1e-8)
        dev = torch.einsum("i,i...->...", w, lc - lu)
        if composition.mode == "mean": dev = dev / N
        return (lu + dev)                              # unnormalized log q; argmax is normalization-invariant
    prob_X = F.one_hot(blend_logp(pX).argmax(-1), pX.shape[-1]).float()
    prob_E = F.one_hot(blend_logp(pE).argmax(-1), pE.shape[-1]).float()
```

This is an **intentional, documented behavior change scoped to the adapter path only**; the legacy `guidance_scale` terminal decode is left exactly as shipped (backward compatibility). It ensures the last step — the codebase's own identified dominant driver of validity — actually reflects the composed conditioning.

### Heterogeneous adapters (different condition **types**)
`build_batched` runs each `adapter_i` on its own condition sub-batch and scatters its `Modulation` into group `i`'s rows; group 0 is zero-gate. Adapters meet **only in log-rate space**, never co-activated inside a single forward row — satisfying the OOD-avoidance requirement while keeping the single `(N+1)·B` forward (REQ 3). Cost: one backbone forward of size `(N+1)·B` + N+1 designer calls per step; for large N, an optional `chunk` fallback loops groups (trading the single-forward property for memory).

---

## 5. File-by-file changes

**New**
- `defog/core/adapter.py` — `Modulation`, `AdaLNAdapter`, `ConditionBranch`, `AdapterComposition`, `AdapterRegistry`, `AdapterModule`.
- `experiments/adapter_training__zinc.py` — clone of `fingerprint_guidance__zinc.py`; trains the adapter, evals Tanimoto-lift via `AdaptedSampler` (§3).
- `tests/test_adapter.py` — null=base, zero-init, N=1-reduces-to-2-branch, stacking, save/load/registry (§7).

**Modified**
- `defog/core/transformer.py` — `GraphTransformer.forward(..., modulation=None)` (`:137`); apply `modulation.apply(i,…)` in the layer loop (`:195-196`) using `x_mask/e_mask` (`:189-190`). Store `n_layers` (already present) for `for_base`. **No `layers.py` change.**
- `defog/core/model.py` — `forward(..., cond_modulation=None)` (`:283`) passthrough; `denoise_step(..., composition=None)` (`:780`) early composition branch per §4, legacy path untouched. **`save`/`load` unchanged.**
- `defog/core/sampler.py` — `_advance` passes `composition` (default `None`) into `denoise_step`; `class AdaptedSampler(Sampler)` after `GuidedSampler` (`:360-379`) storing the composition and threading it unconditionally.
- `defog/core/__init__.py` — export `AdaLNAdapter, Modulation, ConditionBranch, AdapterComposition, AdapterRegistry, AdapterModule, AdaptedSampler`.

**Untouched:** `rate_matrix.py`, `guidance.py`, `feynman_kac.py`, `data.py`, `noise.py`, `layers.py`, `callbacks.py`.

---

## 6. Backward compatibility

- Every new parameter defaults to `None`/current behavior; `GraphTransformer.forward(modulation=None)` and `DeFoGModel.forward(cond_modulation=None)` are byte-identical to today (regression test asserts this). `_advance` sets `composition=None` unless an `AdaptedSampler` is used.
- The legacy `cond_dim>0` input-concat CFG (`cond_encoder/null_embedding/cond_norm`, `_embed_condition`, the 2-branch `guidance_scale` blend and its terminal decode) is entirely untouched. Adapter feature is **scoped to `cond_dim=0` bases in v1** (§1g); combining both CFG mechanisms in one step is explicitly unsupported v1.
- **Adapters are external and never enter `base.state_dict()`** → `DeFoGModel.save/load` are genuinely unchanged and every existing `.ckpt` loads and behaves identically. Adapter checkpoints are tiny (`AdaLNAdapter.save`, no `base.*`).
- `ExactGuidance`/`CompositeGuidance`/`GuidedSampler`/`FeynmanKacSampler` unaffected and still compose (per-branch `posterior_transform` in §4).

---

## 7. Testing / verification

1. **null = base (exact, deterministic — the strongest test).** At the *forward* level (no sampling RNG): fresh `AdaLNAdapter` (gate zero-init), arbitrary `c`, assert `base.forward(nd, extra, mask)` and `base.forward(nd, extra, mask, cond_modulation=adapter(c))` are `allclose`. Assert `GraphTransformer.forward(modulation=None)` is byte-identical to a pre-change golden tensor.
2. **Full-loop equivalence (resolves the RNG-fragility note).** Compare a plain `Sampler` against an `AdaptedSampler` with an **empty composition** (N=0 → the branch skips all adapter/rate work and falls through to the legacy path) under a fixed seed — bit-identical. Do **not** rely on a zero-weight branch (which would draw extra RNG in `compute_rate_matrices` and desync).
3. **N=1 reduces to 2-branch (non-terminal steps).** A single-branch `product` composition reproduces the committed geometric-blend rates (`model.py:899-908`) to float tolerance for `s < 1−1e-6`. The terminal step is checked separately against the *specified* blended-marginal decode (§4), since the legacy terminal decode intentionally differs.
4. **Fingerprint Tanimoto-lift (headline go/no-go).** Reuse `fingerprint_guidance__zinc.py`'s eval verbatim: hold out targets, sample `AdaptedSampler` vs unconditional `Sampler`, assert `mean/median/max Tanimoto(gen, target)` lift over baseline, monotone in a weight sweep `w∈{1,3}`, with validity/uniqueness ≥ the `LatentGuidanceModule` result at matched compute.
5. **Stacking.** Two adapters via `AdapterComposition`: (i) one weight→0 recovers the single-adapter result; (ii) each condition's metric moves toward its target; (iii) assert exactly **one** `(N+1)·B` forward per step (call-count).
6. **Save/load/registry round-trip + hot-swap.** `AdaLNAdapter.save→load` reproduces modulations; `AdapterRegistry.load_dir` swaps adapters between two `sample()` calls on the same frozen base with no base reload; `check_compatible` fails fast on dim/n_layers mismatch.
7. **Backward-compat smoke.** Run the existing `conditional_training__zinc` / sampler `--__TESTING__` paths to confirm legacy CFG and save/load are unchanged.

---

## 8. Phased roadmap

- **M1 — Plumbing (no behavior change).** `modulation`/`cond_modulation`/`composition` args through `transformer.py → model.py → sampler.py`, default `None`. Ship tests 1–2 first (golden-identical, empty-composition equivalence).
- **M2 — Adapter module.** `AdaLNAdapter` (trunk + separate zero-init gate heads, time-conditioned), `Modulation.apply` (delta-masked), `for_base` (dims from `base.hparams`), save/load. Tests 1, 3.
- **M3 — Single-adapter training + inference.** `AdapterModule` (frozen base, CE loss, override `configure_optimizers`, EMA), `AdaptedSampler`, single-branch composition branch in `denoise_step` (incl. terminal blended decode).
- **M4 — Fingerprint verification (GO/NO-GO).** `experiments/adapter_training__zinc.py` on frozen `zinc_uncond_*.ckpt`; Tanimoto-lift (test 4) + time-conditioning ablation.
- **M5 — N-branch composition.** `build_batched`, generalized blend, `mode="mean"` default for N>1, `AdapterRegistry`, structural-zero safety, `__init__` exports. Tests 5–6. (Optional CRN hardening if large-N variance bites.)
- **M6 — Real goal.** Swap the 128-bit fingerprint for the soft cluster-membership `φ` (same pipeline).
- **M7 — Polish.** Docs, per-sublayer capacity dial if needed, chunked-forward fallback.
- **M8 (contingent) — Bake-in fallback.** Only if M4 underfits: pre-LN AdaLN-zero variant behind a base-config flag + one unconditional retrain.

---

## 9. Risks / open questions

- **Post-LN retrofit expressivity (central bet).** A gated-FiLM side-branch on post-norm outputs is less parameter-efficient than from-scratch pre-LN AdaLN-zero. Levers before M8: modulate all three streams (default), larger trunk `H`, per-sublayer injection (2× points), time-conditioning. The direct CE objective is a stronger signal than the Bregman ratio that struggled before, which de-risks capacity.
- **High-dim `c`.** Folded 128-bit fingerprints collide (many near-duplicate `φ`); the adapter may average over collisions. Soft cluster-membership (the real goal) is smoother/lower-rank and should behave better. Steering strength still hinges on `φ` being informative.
- **Composition is a heuristic** (same status `CompositeGuidance` documents). Independently-trained adapters blended at the rate/score level are not guaranteed jointly correct; `Σw_i>1` extrapolates (negative uncond coefficient, and `_stabilize`'s `>1e5→0` clamp can silently drop rates as N grows). Mitigation: `mode="mean"` default for N>1, per-branch weight sweeps, verify per test 5.
- **Per-branch sampling variance.** Each branch draws its own `X_1` in `compute_rate_matrices`; blending N+1 independent single-sample estimates compounds variance. Inherited from shipped 2-branch CFG, tolerable for small N; CRN is the Phase-5 fix.
- **Memory of `(N+1)·B` forward** grows linearly in N; chunked fallback available.
- **Base competence caps quality.** The blend targets the frozen `R_uncond`; the connectivity-improved ZINC base is the intended, adequate target.
- **Open (lean yes): time-conditioning** — default on, ablate in M4; fall back to condition-only if it destabilizes (the base already carries `t` through `y`).
- **Non-issues (do not spend roadmap time):** EMA scope is auto-resolved by `requires_grad` filtering; `save/load` need no changes given external adapter ownership.
---

## 10. Residual risks (synthesis of the 3 final Sonnet verifiers)

**No blockers were found.** All three verified the plan line-by-line against the source and confirmed the load-bearing mechanics (see "Confirmed sound" below). The items below are consistency/robustness fixes, each with a one-line resolution; fold them into the milestones noted.

### Convergent MAJOR findings — RESOLVED (folded into §4/§7 + code; see Revision 1)

1. **Post-blend rate stabilization is missing (→ M5, highest priority).** `Rb = (lu + dev).exp()` is not re-clamped after the blend. With `w>1` (the eval sweep already uses `w=3`) and a small-but-nonzero `R_uncond`, `dev = Σ wᵢ(logR_condᵢ − logR_uncond)` can blow up and feed unnormalized/outlier rates into `multinomial`. This rate-blend path was effectively **dead code** for `cond_dim=0` bases (legacy CFG never fires there), so this plan is the first to actually exercise it. **Fix:** apply a `_stabilize`-equivalent (`nan_to_num` + `Rb[Rb>1e5]=0`) to `Rb` right after the blend.

2. **The "N=1 reduces bit-for-bit" claim conflicts with the post-blend zero-mask (→ M5 / Test 3).** The added `torch.where(R[0]==0, 0, Rb)` safety mask makes N=1 differ from the shipped 2-branch formula exactly where the two branches' forbidden-transition sets differ (most likely near t→1). It's a correctness *improvement*, but **Test 3 must be re-scoped** — test in a configuration where the mask provably can't fire (uniform noise, mid-range t), or test the with-mask behavior against its own defined spec — rather than asserting universal bit-for-bit equality.

3. **Empty-composition (N=0) is not truly a no-op (→ M1 / Test 2).** An empty-but-non-`None` `AdapterComposition` still routes through the `log→+ε→exp` round-trip the legacy path never performs, so it's numerically equal but **not bit-identical**. **Fix:** make `composition=None` the real "off" sentinel and short-circuit N=0 to fall through to the untouched legacy body; downgrade "bit-identical" test language to `allclose` at float32 tolerance wherever the composition path is actually exercised (GPU reduction-order differences preclude exact equality).

### MINOR findings worth folding in (defensive; not correctness-critical for v1)

- **`composition` + `guidance_scale` co-set:** add `assert composition is None or guidance_scale is None` at the top of `denoise_step` (unreachable via samplers in v1 scope, but a silent footgun for direct callers).
- **`ConditionBranch` shape/broadcast:** validate `condition.shape[0] == num_samples` and support a 1-D `(cond_dim,)` target via broadcast (reuse `ExactGuidance._normalized_target`'s convenience — the `IntermediateReporter` probe passes single `(d,)` targets).
- **Sampler wiring:** base `Sampler.__init__` must set `self.composition = None` (else `_advance` `AttributeError`s); `AdaptedSampler.__init__` should call `check_compatible(self.model)` on every branch's adapter.
- **Base-identity footgun (registry):** `check_compatible` only checks shapes (`dx/de/dy`, `n_layers`), so an adapter trained on base A silently applies to any base B with the same `hidden_dim`/`n_layers`, producing plausible-but-meaningless steering. Persist a **base-identity token** (checkpoint hash) in the adapter config and check it — important because the registry is built for hot-swapping.
- **Normalization stats:** `register_buffer` `cond_mean`/`cond_std` on `AdaLNAdapter` (so they follow `.to(device)`); drop the duplicated copies on `AdapterModule` (single source of truth is the adapter).
- **Spec gaps:** define the `_bypass_rows(mod, mask)` and `_slice(nd, g)` helpers used in pseudocode; optionally zero-init scale/shift too (DiT-style, smoother "uncorking" as gates leave 0); short-circuit the terminal rate-matrix loop; prefer deriving `dx/de/dy` from live module attrs (`base.model.tf_layers[0].self_attn.{dx,de,dy}`) over recomputing `hidden_dim//4`.
- **Cosmetic:** §0 mislabels `n_layers=9` as a class default (the class default is 6; 9 is the ZINC experiment's choice — `for_base` reads it dynamically, so no impact).

### Confirmed sound by all three (no changes needed)
null=base (structural bypass + separate zero-init gate); delta-masking discipline (matches the base's own between-layer masking); `_compute_extra_data` branch-invariance → compute-once-and-tile; N-branch batchability (attention/pooling never cross the batch dim); EMA auto-scoping via `requires_grad` filtering; `save`/`load` left genuinely untouched via external adapters; backward-compatibility of the appended `modulation`/`cond_modulation`/`composition` kwargs (every call-site uses ≤3 positional args); and the N=1 *algebraic* reduction to the shipped 2-branch formula.
