"""Feynman-Kac energy over a predicted vibrational spectrum.

Steering toward molecules whose IR spectrum has requested features: a band here, no band
there, or a whole measured spectrum to reproduce. Like the other energies in this package it
is *just a reward* -- it decodes each predicted-clean graph to a molecule, asks a forward
predictor what that molecule's spectrum looks like, and scores the answer. Nothing is
differentiated, so the predictor can be any black box, including a model trained by someone
else on data we do not have.

Why bands rather than a bare 1801-bin vector
--------------------------------------------
A spectrum is not a vector of free parameters. Peaks are strongly correlated -- every organic
molecule has C-H stretches near 2900, and the fingerprint region is a whole-molecule signature
rather than a sum of independent features -- so a UI that lets someone paint arbitrary
intensities lets them request spectra no molecule has.

Measurement backs this up. Against held-out NIST spectra a current forward predictor
reproduces a *whole* spectrum only moderately well, while placing a *functional group's band*
very well (carbonyl present/absent AUC 0.93; ester vs amide separated by +76 cm-1 against a
literature +75). So the requestable unit here is a band -- a window, and how much of the
molecule's absorption should fall inside it -- which is both what the predictor is reliably
good at and what a chemist means when they point at a peak.

:class:`SpectrumEnergy` still accepts a full ``reference`` spectrum, for the case where one was
actually measured and the whole shape is the request. The two combine: "look like this
reference, and definitely have a nitrile" is one energy.

Degrading gracefully matters here. An unreachable request under Feynman-Kac simply means the
particle weights never concentrate and the closest molecules found are returned -- there is no
regime where an impossible spectrum yields confident nonsense.
"""

from typing import Callable, Optional, Sequence

import numpy as np
import torch

from .data import dense_to_pyg


class Band:
    """One requested spectral feature.

    The penalty is **one-sided**, which is not a refinement but the difference between the
    energy working and not. Pointing at 1710 and saying "carbonyl" means *at least* this much
    absorption there; a molecule with a very strong carbonyl has satisfied the request, not
    overshot it. A two-sided squared error ranks such a molecule as badly as one with no
    carbonyl at all — which is what the first version of this class did, and what its tests
    caught. Two-sided matching is what ``reference`` is for.

    Args:
        centre: band centre in cm-1.
        half_width: half-window in cm-1; the band is ``[centre - half_width, centre + half_width]``.
        target: fraction of the molecule's total predicted absorption for the window.
        mode: ``"at_least"`` penalises falling short, ``"at_most"`` penalises exceeding.
            Defaults to ``"at_least"`` for a positive target and ``"at_most"`` for ``0.0``, so
            "give me a carbonyl" and "give me no free O-H" both read the obvious way.
        weight: relative importance among the requested bands.
    """

    __slots__ = ("centre", "half_width", "target", "mode", "weight")
    MODES = ("at_least", "at_most")

    def __init__(self, centre: float, half_width: float = 30.0,
                 target: float = 0.05, weight: float = 1.0,
                 mode: Optional[str] = None):
        if half_width <= 0:
            raise ValueError(f"band half_width must be positive, got {half_width}")
        if not 0.0 <= target <= 1.0:
            raise ValueError(f"band target is a fraction of total absorption, got {target}")
        if mode is None:
            mode = "at_least" if target > 0 else "at_most"
        if mode not in self.MODES:
            raise ValueError(f"band mode must be one of {self.MODES}, got {mode!r}")
        self.centre = float(centre)
        self.half_width = float(half_width)
        self.target = float(target)
        self.mode = mode
        self.weight = float(weight)

    def window(self, grid: np.ndarray) -> np.ndarray:
        return (grid >= self.centre - self.half_width) & (grid <= self.centre + self.half_width)

    def shortfall(self, mass: np.ndarray) -> np.ndarray:
        """How far each molecule is on the wrong side of the request; zero once it is met."""
        gap = self.target - mass if self.mode == "at_least" else mass - self.target
        return np.clip(gap, 0.0, None)

    def __repr__(self) -> str:
        return (f"Band({self.centre:g}±{self.half_width:g} cm-1, {self.mode} "
                f"{self.target:g}, weight={self.weight:g})")


class SpectrumEnergy:
    """``E(x1)`` = weighted squared error between a molecule's predicted spectrum and what
    was asked of it.

    The predictor is called ONCE per batch with every decodable molecule, not once per
    molecule: a neural forward model is one to two orders of magnitude cheaper batched, and
    this energy is evaluated repeatedly inside the sampling loop.

    Args:
        domain: object with ``.decode(pyg_data) -> Optional[Mol]`` (e.g. MoleculeDomain).
        predictor: ``list[Mol] -> array (n, len(grid))`` of nonnegative intensities. Any black
            box; it is never differentiated.
        grid: wavenumbers matching the predictor's output columns, shape ``(B,)``.
        bands: requested features. May be empty if ``reference`` is given.
        reference: a full spectrum to reproduce, shape ``(B,)``. NaN entries are treated as
            unmeasured and excluded -- an instrument that did not cover a region made no claim
            about it, and scoring zeros there would invent one.
        reference_weight: weight of the reference term relative to the bands.
        invalid_energy: energy for graphs that do not decode, so their FK weight goes to zero
            and the search stays on-manifold.
    """

    def __init__(self, domain, predictor: Callable, grid: Sequence[float],
                 bands: Optional[Sequence[Band]] = None,
                 reference: Optional[Sequence[float]] = None,
                 reference_weight: float = 1.0,
                 invalid_energy: float = 1e3):
        self.domain = domain
        self.predictor = predictor
        self.grid = np.asarray(grid, dtype=float)
        self.bands = list(bands or ())
        self.invalid = float(invalid_energy)
        self.reference_weight = float(reference_weight)

        if not self.bands and reference is None:
            raise ValueError("SpectrumEnergy needs at least one band or a reference spectrum")

        if reference is None:
            self.reference = None
            self.ref_mask = None
        else:
            ref = np.asarray(reference, dtype=float)
            if ref.shape != self.grid.shape:
                raise ValueError(f"reference has shape {ref.shape}, grid has {self.grid.shape}")
            self.ref_mask = np.isfinite(ref)
            if not self.ref_mask.any():
                raise ValueError("reference spectrum is entirely NaN")
            self.reference = _l1(np.where(self.ref_mask, ref, 0.0), self.ref_mask)

        # Precomputed once: the windows do not change between evaluations.
        self.windows = [b.window(self.grid) for b in self.bands]
        for band, window in zip(self.bands, self.windows):
            if not window.any():
                raise ValueError(f"{band} selects no point on the supplied grid")

    def score(self, spectra: np.ndarray) -> np.ndarray:
        """Energy per predicted spectrum, shape ``(n,)``. Separated from :meth:`__call__` so
        the scoring can be tested, and inspected in a UI, without a graph batch."""
        spectra = np.asarray(spectra, dtype=float)
        # NaN means "this phase cannot report here" -- a real predictor output, not a bug: a
        # nujol mull masks the paraffin's own bands and a CCl4 solution its solvent windows.
        # Treated as no absorption rather than propagated, because one NaN reaching the FK
        # weights turns every particle's weight into NaN and takes the whole run with it.
        spectra = np.clip(np.nan_to_num(spectra, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        totals = spectra.sum(axis=1, keepdims=True)
        # A spectrum with no absorption anywhere carries no information about any band; it
        # scores as maximally wrong rather than dividing by zero.
        dead = totals[:, 0] <= 0
        norm = np.divide(spectra, np.where(totals > 0, totals, 1.0))

        out = np.zeros(len(spectra))
        for band, window in zip(self.bands, self.windows):
            mass = norm[:, window].sum(axis=1)
            out += band.weight * band.shortfall(mass) ** 2

        if self.reference is not None:
            masked = _l1_rows(norm[:, self.ref_mask])
            out += self.reference_weight * ((masked - self.reference[self.ref_mask]) ** 2).sum(axis=1)

        out[dead] = self.invalid
        return out

    @torch.no_grad()
    def __call__(self, X1, E1, node_mask):
        n = node_mask.sum(-1)
        datas = dense_to_pyg(X1, E1, None, node_mask, n)
        out = X1.new_full((len(datas),), self.invalid)

        mols, index = [], []
        for i, d in enumerate(datas):
            mol = self.domain.decode(d)
            if mol is not None:
                mols.append(mol)
                index.append(i)
        if not mols:
            return out

        try:
            spectra = self.predictor(mols)
        except Exception:
            # A predictor that fails on a batch must not take the sampling run down with it;
            # every particle in it keeps the invalid energy and the search moves on.
            return out
        spectra = np.asarray(spectra, dtype=float)
        if spectra.shape != (len(mols), len(self.grid)):
            raise ValueError(
                f"predictor returned {spectra.shape}, expected {(len(mols), len(self.grid))}"
            )

        energies = self.score(spectra)
        for i, e in zip(index, energies):
            out[i] = float(e)
        return out


def _l1(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    total = v[mask].sum()
    return v / total if total > 0 else v


def _l1_rows(v: np.ndarray) -> np.ndarray:
    totals = v.sum(axis=1, keepdims=True)
    return np.divide(v, np.where(totals > 0, totals, 1.0))


def band_masses(spectra: np.ndarray, grid: Sequence[float],
                bands: Sequence[Band]) -> np.ndarray:
    """What each molecule actually achieved per band, shape ``(n, len(bands))``.

    The reporting counterpart of the energy: a request the sampler could not satisfy should be
    visible as a number beside the one that was asked for, not inferred from a molecule.
    """
    grid = np.asarray(grid, dtype=float)
    spectra = np.clip(np.asarray(spectra, dtype=float), 0.0, None)
    norm = _l1_rows(spectra)
    return np.stack([norm[:, b.window(grid)].sum(axis=1) for b in bands], axis=1)
