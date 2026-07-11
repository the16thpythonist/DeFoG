# Exact Discrete Guidance for DeFoG — Design & Implementation Plan

> Integrating the posterior-based guidance of **"Discrete Guidance Matching: Exact
> Guidance for Discrete Flow Matching"** (ICLR 2026, arXiv:2509.21912v2) into the
> `defog/` library.
>
> Status: **design frozen, validated (2× approve-with-changes), not yet implemented.**
> Produced by a multi-agent design workflow (2 Opus drafts → 4 Sonnet audits →
> Opus consolidation → 2 Sonnet validations). This doc folds the validators'
> required changes into the final design so it is ready to build from.

---

## 1. The idea in one paragraph

The paper's Theorem 1 says: the guided (target) posterior `q_{1|t}` is just the
pretrained posterior `p_{1|t}` **reweighted coordinate-wise by a learned
density-ratio expectation `h_t` and renormalized** — `q_{1|t} ∝ h_t · p_{1|t}`.
DeFoG already builds its entire CTMC from the predicted clean-graph marginals
`pred_X`/`pred_E` (the softmax at `model.py:822-823`) before handing them to
`RateMatrixDesigner`. So the whole method is a **pure transform of those two
tensors, inserted at exactly that one boundary.** Everything downstream (R\*,
`η·R^DB`, `ω·R^TG`, `_stabilize`, `_compute_step_probs`, the final-step MAP
argmax, edge symmetrization, `node_mask` re-masking) is reused verbatim.

## 2. Meta-approach (what the two drafts contributed)

- **From the "first-class subsystem" draft:** the base model stays
  guidance-agnostic and gains **one generic optional `posterior_transform` hook**
  in `denoise_step` — the same factoring that made `Sampler`/`Constraint` clean.
  The model never imports or names "guidance", exactly as it never names
  "constraint". A named `GuidedSampler(Sampler)` is the discoverable entry point,
  parallel to `InpaintingSampler`.
- **From the "minimal-surface" draft:** the reweight operator is the numerically
  ideal `softmax(g + log p)`, where the guidance network's raw logits `g` **are**
  `log h` (i.e. `h = exp(g)`). This equals `h·p / E_p[r]` exactly, computed in log
  space. And the guidance network is **just a `DeFoGModel`** reused wholesale.
- **The key unifying decision (resolves both drafts' critical audit bugs):** the
  guidance network `h_t` is a `DeFoGModel` that mirrors the base's categorical
  dims / features / limit distribution but is **always unconditional
  (`cond_dim=0`)** and fed its own **0-width `y`**. Because it is a full
  `DeFoGModel`, it computes its own RRWP + molecular extra-features internally
  (dissolving the "extra_data wiring" bug), and because it is unconditional and
  never sees the base's conditional `y_t`, the "train-vs-sample `cond_dim`/`y_t`
  mismatch" bug simply cannot occur.

## 3. Architecture

### 3.1 The single hook (only model touch)

`denoise_step` gains one optional kwarg. Immediately after the marginals are
produced by softmax, and before `compute_rate_matrices`, a guarded line runs the
transform. Guidance-off (`posterior_transform=None`) is a **bitwise no-op** — full
backward compatibility.

```python
# defog/core/model.py :: denoise_step(...)
def denoise_step(self, t, s, X_t, E_t, y_t, node_mask,
                 guidance_scale=None, eta=None, omega=None,
                 posterior_transform=None):          # <-- NEW
    ...
    pred_X = F.softmax(pred.X, dim=-1)               # model.py:822-823 (unchanged)
    pred_E = F.softmax(pred.E, dim=-1)
    if posterior_transform is not None:              # <-- NEW (conditional branch)
        pred_X, pred_E = posterior_transform(pred_X, pred_E, noisy_data, node_mask)
    R_t_X, R_t_E = self.rate_matrix_designer.compute_rate_matrices(   # unchanged
        t, node_mask, X_t, E_t, pred_X, pred_E)
    if guidance_scale is not None:                   # CFG block (unchanged structure)
        ...
        pred_X_uncond = F.softmax(pred_uncond.X, dim=-1)   # model.py:845-846
        pred_E_uncond = F.softmax(pred_uncond.E, dim=-1)
        if posterior_transform is not None:          # <-- NEW (uncond branch; heuristic)
            pred_X_uncond, pred_E_uncond = posterior_transform(
                pred_X_uncond, pred_E_uncond, noisy_data_uncond, node_mask)
        ... # log-space rate blend unchanged
```

Because `pred_X`/`pred_E` are reassigned in place, **both** the rate build and the
final-step MAP argmax (`model.py:873-875`) automatically use the rectified `q`.

`model.sample(...)` also gains a pass-through `posterior_transform=None` so the
facade can drive guidance directly.

### 3.2 The reweight math (Theorem 1, exact)

Per coordinate `d` (each node, each edge), with the guidance net's logits `g`:

```
q(x1^d | x_t) = softmax_d( g^d + log p(x1^d | x_t) )
             = exp(g^d)·p^d / Σ_s exp(g_s)·p_s
             = h^d · p^d / E_{x1~p}[r]          # Theorem 1 / Eq. 6, softmax denom = normalizer
```

Always-valid: `p > 0` (softmax) and `exp(g) > 0`, so `q` is a proper distribution —
no NaN. Edge symmetry is preserved (symmetric `g.E` + symmetric `log pred_E` →
per-class softmax stays symmetric; `sample_from_probs` still mirrors the upper
triangle). `node_mask` padding is re-masked at `model.py:886-888` and excluded
from `R_t` by `_stabilize`.

**Cost: exactly one extra `h` forward pass per step** (contrast the rate-based
Theorem 2 path: `D+1` passes — a documented non-goal here).

### 3.3 The density-ratio abstraction

One contract unifies all three paper cases; each returns a nonnegative per-graph
scalar `r(x1) = q1(x1)/p1(x1)` of shape `(bs,)`, known only up to a positive
constant (renormalization + `r_scale` cancel it):

| Case (paper) | Class | `r(x1)` |
|---|---|---|
| Energy-guided (Sec 4.1) | `EnergyRatio(energy_fn, gamma)` | `exp(-gamma · E(x1))` |
| Classifier / class-conditional (Eq. 7) | `ClassifierRatio(prob_fn)` | `p(y_target | x1)` |
| RLHF preference (Sec 3.5) | `RewardRatio(reward_fn, tau)` | `exp(R(x1)/tau)` |

For DeFoG's molecular logP/SAS steering the natural instance is the energy case
via **`MoleculePropertyEnergy`**: `E(x1) = (prop(mol) - target)²`, decoding dense
one-hot graphs through the existing `MoleculeDomain.decode` (`molecule.py:425`) and
reusing the `Crippen.MolLogP` / `sascorer` callbacks already wired into
`conditional_training__aqsoldb.py`. Invalid-to-decode molecules get
`invalid_energy` → `r ≈ 0` (they lie off the source support; guidance pushes mass
away). Joint (logP, SAS) steering = sum of energies (product of ratios). **`r`
depends only on `x1`**, so it is precomputed once per training molecule and cached
as `batch.r`.

### 3.4 Training the guidance network (Bregman objective)

`GuidanceModule(pl.LightningModule)`, driven the repo way:
`pl.Trainer(...).fit(GuidanceModule(base, r), source_loader)`. Each step:

1. `base.eval()` asserted (Lightning's per-epoch `.train()` must not re-enable
   dropout on the frozen base — primarily protects the optional `L_{h,q}` path,
   see §5).
2. Densify the clean batch → `(X1, E1, node_mask)`.
3. Read cached `r = batch.r` (or compute live), **detach**, divide by fixed
   `r_scale` (legit: `q` is exactly invariant to global rescaling of `r`).
4. Draw `t ~ U[0,1]` **explicitly** (Eq. 11, decoupled from the base's polydec
   train distortion).
5. Noise with the base's own `_apply_noise(X1, E1, y0=zeros(bs,0), node_mask, t=t)`.
6. Forward `h` (its own `_compute_extra_data`).
7. Bregman loss with `F(x)=⟨x, log x⟩` and `h = exp(g)`, gathered at each sample's
   **true clean class**: `exp(g) - r·g` (no log of a network output — an ℓ² loss is
   deliberately avoided; the paper notes it fails for positive density ratios). Its
   per-coordinate minimizer is provably `E[r | x1^d=z, x_t] = h_t^d`. Nodes masked
   by `node_mask`; edges over the **upper triangle only** (`_edge_upper_mask`),
   weighted by `base.train_loss.lambda_edge`.

Stability: clamp `g` at `g_clamp` before `exp`; normalize `r` to O(1) via `r_scale`.

## 4. Public interface (final, corrected)

New module `defog/core/guidance.py`. **The four validator-required corrections are
folded in** (see §5): `ExactGuidance` name, single-`g`-per-step reuse under CFG,
`if r is None` (not `or`), no dead stub, `L_{h,q}` scoped as a documented future
extension.

```python
# defog/core/guidance.py   (NEW — the whole subsystem)
from abc import ABC, abstractmethod
from typing import Callable, Optional
import torch, torch.nn.functional as F
import pytorch_lightning as pl
from .model import DeFoGModel
from .data import to_dense, dense_to_pyg

# ---- density ratio r(x1) = q1/p1 --------------------------------------------
class DensityRatio(ABC):
    @abstractmethod
    def __call__(self, X1, E1, node_mask) -> torch.Tensor: ...   # (bs,) >= 0

class EnergyRatio(DensityRatio):        # r = exp(-gamma * E(x1))
    def __init__(self, energy_fn, gamma=1.0): self.energy_fn, self.gamma = energy_fn, gamma
    def __call__(self, X1, E1, node_mask):
        return torch.exp(-self.gamma * self.energy_fn(X1, E1, node_mask))

class ClassifierRatio(DensityRatio):    # r = p(y_target | x1)
    def __init__(self, prob_fn): self.prob_fn = prob_fn
    def __call__(self, X1, E1, node_mask):
        return self.prob_fn(X1, E1, node_mask).clamp_min(1e-12)

class RewardRatio(DensityRatio):        # r = exp(R(x1)/tau)
    def __init__(self, reward_fn, tau=1.0): self.reward_fn, self.tau = reward_fn, tau
    def __call__(self, X1, E1, node_mask):
        return torch.exp(self.reward_fn(X1, E1, node_mask) / self.tau)

class MoleculePropertyEnergy:           # E(x1) = (prop(mol) - target)^2
    def __init__(self, domain, prop_callback, target, invalid_energy=1e3):
        self.domain, self.prop, self.target, self.invalid = domain, prop_callback, target, invalid_energy
    def __call__(self, X1, E1, node_mask):
        n = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, n)
        out = X1.new_full((len(datas),), self.invalid)
        for i, d in enumerate(datas):
            mol = self.domain.decode(d)
            if mol is not None:
                try: out[i] = (float(self.prop(mol)) - self.target) ** 2
                except Exception: pass
        return out

# ---- build the guidance network (a DeFoGModel, always unconditional) --------
def build_guidance_network(base: DeFoGModel, **overrides) -> DeFoGModel:
    hp = dict(base.hparams)
    hp.update(cond_dim=0, cond_drop_prob=0.0, guidance_scale=1.0)
    hp.update(overrides)                 # may shrink h: n_layers, hidden_dim, ...
    return DeFoGModel(**hp)

def _edge_upper_mask(node_mask):
    m = node_mask[:, :, None] & node_mask[:, None, :]
    return torch.triu(m, diagonal=1)     # each undirected edge once, no diag, no pad

# ---- sample-time reweighter (renamed from `Guidance` -> `ExactGuidance`) -----
class ExactGuidance:
    """Exact posterior-based discrete guidance (Theorem 1). Its `reweight` is the
    `posterior_transform` a Sampler runs each step. `g` is memoized within a step
    so composing with CFG costs ONE h forward pass, not two (the cond & uncond
    branches share X_t/E_t/t; only y_t differs and is ignored here)."""
    def __init__(self, h_model: DeFoGModel, density_ratio: Optional[DensityRatio] = None):
        assert h_model.cond_dim == 0, "guidance network must be unconditional"
        self.h = h_model.eval()
        self.density_ratio = density_ratio
        self._memo = None                # (key, gX, gE)

    @torch.no_grad()
    def _g(self, noisy_data, node_mask):
        nd = noisy_data
        key = (id(nd["X_t"]), id(nd["E_t"]), float(nd["t"].reshape(-1)[0]))
        if self._memo is not None and self._memo[0] == key:
            return self._memo[1], self._memo[2]
        bs = nd["X_t"].size(0)
        h_noisy = {"X_t": nd["X_t"], "E_t": nd["E_t"], "t": nd["t"],
                   "node_mask": node_mask,
                   "y_t": torch.zeros(bs, 0, device=node_mask.device)}
        g = self.h.forward(h_noisy, self.h._compute_extra_data(h_noisy), node_mask)
        gX, gE = g.X, 0.5 * (g.E + g.E.transpose(1, 2))   # symmetric edges
        self._memo = (key, gX, gE)
        return gX, gE

    @torch.no_grad()
    def reweight(self, pred_X, pred_E, noisy_data, node_mask, eps=1e-8):
        gX, gE = self._g(noisy_data, node_mask)
        q_X = F.softmax(gX + torch.log(pred_X.clamp_min(eps)), dim=-1)
        q_E = F.softmax(gE + torch.log(pred_E.clamp_min(eps)), dim=-1)
        return q_X, q_E

    def save(self, path): return self.h.save(path)
    @classmethod
    def load(cls, path, density_ratio=None): return cls(DeFoGModel.load(path), density_ratio)

# ---- training (Bregman, Eq. 11) ---------------------------------------------
class GuidanceModule(pl.LightningModule):
    def __init__(self, base: DeFoGModel, density_ratio: DensityRatio,
                 h_model: Optional[DeFoGModel] = None, lr: float = 2e-4,
                 lambda_edge: Optional[float] = None, g_clamp: float = 20.0,
                 r_scale: float = 1.0, **h_overrides):
        super().__init__()
        self.base = base.requires_grad_(False)          # frozen, eval-asserted per step
        self.h = h_model if h_model is not None else build_guidance_network(base, **h_overrides)
        self.density_ratio, self.lr = density_ratio, lr
        self.g_clamp, self.r_scale = g_clamp, r_scale
        self.lambda_edge = lambda_edge if lambda_edge is not None else base.train_loss.lambda_edge

    def training_step(self, batch, _):
        self.base.eval()
        d, node_mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        d = d.mask(node_mask); X1, E1 = d.X, d.E
        r = getattr(batch, "r", None)
        if r is None:                                   # NB: `if r is None`, NOT `r or ...`
            r = self.density_ratio(X1, E1, node_mask)
        r = (r / self.r_scale).detach()
        y0 = torch.zeros(X1.size(0), 0, device=X1.device)
        t = torch.rand(X1.size(0), 1, device=X1.device)          # Eq. 11: t ~ U[0,1]
        with torch.no_grad():
            noisy = self.base._apply_noise(X1, E1, y0, node_mask, t=t)
        g = self.h.forward(noisy, self.h._compute_extra_data(noisy), node_mask)
        gX = g.X.gather(-1, X1.argmax(-1, keepdim=True)).squeeze(-1).clamp(max=self.g_clamp)
        gE = g.E.gather(-1, E1.argmax(-1, keepdim=True)).squeeze(-1).clamp(max=self.g_clamp)
        emask = _edge_upper_mask(node_mask)
        loss_X = (gX.exp() - r[:, None]       * gX)[node_mask].mean()
        loss_E = (gE.exp() - r[:, None, None] * gE)[emask].mean()
        loss = loss_X + self.lambda_edge * loss_E
        self.log("guid/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.h.parameters(), lr=self.lr, weight_decay=1e-5)

    # Keep guidance checkpoints small: don't serialize the frozen base.
    def on_save_checkpoint(self, ckpt):
        ckpt["state_dict"] = {k: v for k, v in ckpt["state_dict"].items()
                              if not k.startswith("base.")}

# defog/core/sampler.py :  base Sampler gains `posterior_transform=None` (stored,
# forwarded in _advance -> denoise_step). Then:
class GuidedSampler(Sampler):
    def __init__(self, model, guidance: ExactGuidance, **kwargs):
        super().__init__(model, posterior_transform=guidance.reweight, **kwargs)
        self.guidance = guidance
# Guided inpainting for free:
#   InpaintingSampler(model, constraint, posterior_transform=guidance.reweight)
```

### End-to-end usage (AqSolDB: steer a frozen base toward logP = 2.0)

```python
from defog.core import DeFoGModel, GuidanceModule, ExactGuidance, GuidedSampler
from defog.core.guidance import EnergyRatio, MoleculePropertyEnergy
from defog.domains import MoleculeDomain
from rdkit.Chem import Crippen
import pytorch_lightning as pl

base = DeFoGModel.load("aqsoldb_uncond.ckpt")          # frozen source p1
domain = MoleculeDomain(atom_decoder, bond_decoder)
r = EnergyRatio(MoleculePropertyEnergy(domain, Crippen.MolLogP, target=2.0), gamma=4.0)

module = GuidanceModule(base, r, n_layers=5, hidden_dim=128)   # h smaller than base
pl.Trainer(max_epochs=50, enable_checkpointing=False).fit(module, train_loader)
guidance = ExactGuidance(module.h, r); guidance.save("logp2_guidance")

sampler = GuidedSampler(base, guidance, eta=100.0, omega=0.3, time_distortion="polydec")
mols = sampler.sample(num_samples=500)                 # ONE h forward pass per step
```

## 5. Corrections folded in from validation

Both validators returned **approve-with-changes**. All required changes are already
reflected in §4 above:

1. **No double `h` pass under CFG (major).** `ExactGuidance` memoizes `g` within a
   step keyed on `(id(X_t), id(E_t), t)`; the cond and uncond CFG branches share
   `X_t/E_t/t`, so `g` is computed once. Restores the paper's "one extra forward
   pass per step" claim. (The primary path — a frozen *unconditional* base with **no
   CFG** — never hits the CFG block at all.)
2. **`if r is None`, not `getattr(batch,'r',None) or ...` (major).** The `or` form
   raises `RuntimeError: Boolean value of Tensor ... is ambiguous` for `bs>1` when
   `batch.r` is present (the common cached case). Fixed to the explicit `is None`.
3. **`L_{h,q}` regularizer scoped out of v1 (major).** The optional target-sample
   term (Eq. 12) is a **documented future extension**, not in the shipped interface
   (no `lam`, no `target_loader`, no `_reg_term`). v1 is `L_{h,p}` only, which needs
   source data alone and covers de-novo property steering (where real `q1` samples
   don't exist). See §8 for the follow-up plumbing sketch.
4. **Dead `_bregman` stub deleted (minor).** The Bregman loss lives inline in
   `training_step`.
5. **Naming collision resolved (minor).** The sample-time class is `ExactGuidance`
   (not bare `Guidance`), disambiguating from the existing `guidance_scale` (CFG)
   and `omega` / "target guidance" `R^TG` vocabulary. `GuidedSampler` reads
   unambiguously and is kept.
6. **Checkpoint hygiene (minor).** `GuidanceModule.on_save_checkpoint` strips
   `base.*` so guidance checkpoints don't embed the frozen base; the deployment
   artifact is `guidance.save()` (h only). Experiments use
   `enable_checkpointing=False`.
7. **`base.eval()` rationale clarified (minor).** On the default `L_{h,p}`-only path
   the base is touched only via `_apply_noise` (a closed-form noising formula, no
   transformer/dropout), so the guard chiefly matters once `L_{h,q}` (future) does a
   frozen-base forward.

The soundness validator additionally **verified against the real code** that: the
insertion point and file:line anchors are exact; guidance-off is a true no-op;
`build_guidance_network`'s `cond_dim=0` reuse is legitimate (`DeFoGModel` args are
pure hyperparameters via `save_hyperparameters()`); `softmax(g+log p) = h·p/E_p[r]`;
the Bregman minimizer is `E[r|z,x_t]`; and `q` is invariant to a global rescale of
`r` (justifying `r_scale`).

## 6. File-by-file change map

| File | Change |
|---|---|
| **`defog/core/guidance.py`** | **NEW.** The whole subsystem: `DensityRatio` ABC + `EnergyRatio`/`ClassifierRatio`/`RewardRatio`, `MoleculePropertyEnergy`, `build_guidance_network`, `_edge_upper_mask`, `ExactGuidance`, `GuidanceModule`. |
| `defog/core/model.py` | `denoise_step`: add `posterior_transform=None`; guarded reweight after softmax (L822-823) and in the CFG uncond branch (L845-846). `sample()`: add pass-through kwarg, forward into `Sampler(...)` (L600-607). **No other logic changes.** |
| `defog/core/sampler.py` | `Sampler.__init__`: add `posterior_transform=None` (store); `_advance`: forward it into `denoise_step`. Add `GuidedSampler(Sampler)`. |
| `defog/core/__init__.py` | Export `ExactGuidance`, `GuidanceModule`, `DensityRatio`, `EnergyRatio`, `ClassifierRatio`, `RewardRatio`, `MoleculePropertyEnergy`, `build_guidance_network`, `GuidedSampler`. |
| `experiments/guided_generation__aqsoldb.py` | **NEW** pycomex experiment mirroring `conditional_training__aqsoldb.py`. |
| `defog/domains/molecule.py`, `rate_matrix.py`, `constraint.py`, `loss.py`, `noise.py` | **No change** — the reweight rides entirely on the predicted-marginals interface these already consume. |

## 7. Implementation plan (ordered, with verification)

1. **Hook in the model** (`model.py`). Add `posterior_transform` to `denoise_step`
   (+ both guarded insertions) and `sample()`. *Verify:* `posterior_transform=None`
   → bitwise-identical output to pre-change; identity transform → identical to None;
   a fixed-distribution transform → sampled state changes; `model.py` imports no
   guidance symbol.
2. **Thread through the Sampler** (`sampler.py`). Add the stored attribute +
   forwarding; add `GuidedSampler`. *Verify:* default `Sampler` reproduces existing
   `sample()` output; `InpaintingSampler(..., posterior_transform=fn)` forwards `fn`
   (guided inpainting composes with zero subclass changes); `GuidedSampler` sets
   `posterior_transform = guidance.reweight`.
3. **Density ratios + helpers** (`guidance.py`). `DensityRatio` subclasses,
   `MoleculePropertyEnergy`, `build_guidance_network`, `_edge_upper_mask`. *Verify:*
   `build_guidance_network(base).cond_dim == 0` even when `base.cond_dim > 0`, and
   its class counts / limit dist match base; `MoleculePropertyEnergy` returns
   `invalid_energy` for a broken one-hot graph and the right squared error for a
   known molecule; `_edge_upper_mask` excludes diagonal + padded nodes.
4. **`ExactGuidance.reweight`** (`guidance.py`). *Verify (numerical):*
   `softmax(g + log p) == (exp(g)*p)/(exp(g)*p).sum(-1,keepdim)` within 1e-6;
   `q_E` symmetric when `pred_E` symmetric; `reweight` never reads
   `noisy_data['y_t']` (feed a wrong-width base `y_t` → no crash/use); the memo
   makes a cond+uncond pair cost exactly one `h` forward (count/mock).
5. **`GuidanceModule`** (Bregman training). *Verify (overfit sanity):* on a tiny
   synthetic set with a **known** `r` (e.g. `r` = f(count of node-class-0)), a few
   hundred steps make `exp(g at true class)` track the analytic `E[r|x_t]` and loss
   decrease; base params get no grad and stay in eval; loss stays finite under a
   large-range `r` (overflow guard).
6. **Exports** (`__init__.py`). *Verify:*
   `from defog.core import ExactGuidance, GuidanceModule, GuidedSampler, EnergyRatio, MoleculePropertyEnergy`
   succeeds; `__all__` updated.
7. **End-to-end smoke.** Small unconditional base → `EnergyRatio(MoleculePropertyEnergy(...))`
   → `GuidanceModule` fit 2 epochs on ~150 molecules → `GuidedSampler.sample(50)` →
   decode. *Verify:* runs without shape errors on a **conditional-hparams** base
   (proves the `cond_dim=0` decoupling); guided mean logP moves toward target
   vs unguided (directional).
8. **`experiments/guided_generation__aqsoldb.py`** (pycomex). Frozen unconditional
   base; precompute `r` per molecule (`batch.r`); fit `GuidanceModule`; evaluate the
   3×3 (logP, SAS) target grid with `GuidedSampler(eta=100, omega=0.3,
   time_distortion="polydec")`; reuse `build_encoders`, `MoleculeDomain`,
   `tag_generated_smiles`, `plot_target`; add an `@experiment.testing` smoke variant.
   *Verify:* `python experiments/guided_generation__aqsoldb.py --__TESTING__ True`
   produces the 3×3 figures + `grid_metrics.json`; per-target logP MAE reported
   against a no-guidance baseline.

## 8. Open questions & non-goals

- **η/ω re-tuning.** The reweight changes the marginals `R^DB`/`R^TG` consume, so
  DeFoG's tuned `eta=100`/`omega=0.3` operating point may shift; treat them as
  jointly re-tunable, not transferable unchanged.
- **`gamma` tuning** for `MoleculePropertyEnergy`: too large → peaked `r`, noisy
  Bregman estimates / mode collapse; too small → weak steering. No auto-tuning;
  `r_scale`/`g_clamp` fix numerics, not statistical variance.
- **`h` capacity.** Defaults to mirroring the base's features for correctness; a
  leaner `h` may suffice (it estimates a scalar field, not the full posterior).
  Empirical.
- **Coordinate-wise factorization.** Eq. 3 assumes tokens are conditionally
  independent given `x1`; DeFoG edges are coupled via symmetry. Per-edge reweight +
  symmetrize is invariant-preserving, but the effect on strongly correlated
  substructures is untested.
- **`L_{h,q}` (future extension).** Needs real `q1` graphs (usually absent in
  de-novo steering). When added: `GuidanceModule(..., lam>0, target_loader=...)`,
  materialize a persistent cycled iterator over `target_loader`, noise a target
  batch with `base._apply_noise`, forward the frozen base (unconditional `y`) to get
  `p`, form `q = softmax(g + log p)`, cross-entropy at the realized target class.
  Feed the second loader via `CombinedLoader` or a manual cycle — do **not** conflate
  it with condition amortization.
- **Non-goal: rate-based guidance (Theorem 2).** Needs source==target corruption and
  `D+1` forward passes and would post-multiply `R_t`; the posterior path is strictly
  cheaper and covers the molecular use case.
- **Non-goal (default): amortized target-conditioned `h`.** One `h` taking the
  target as input would serve the whole grid with one network, but that goes *beyond*
  the paper's fixed-`r` formulation and must not naively reuse the `cond_dim=0`
  decoupling; left as a flagged extension.

---

*Provenance: `defog/core` file:line anchors verified against the working tree on
branch `feat/conditional-aqsoldb`. Re-verify before editing if the tree has moved.*
