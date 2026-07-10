"""
Graph-domain adapters for DeFoG.

DeFoG core is agnostic to what its categorical node/edge classes *mean*: a
sampled graph is a PyG ``Data`` with one-hot ``x`` (node classes) and one-hot
``edge_attr`` (edge classes, class 0 = "no edge"). A :class:`GraphDomain`
interprets those classes for a specific kind of graph and bundles three
*decoupled* concerns behind one object:

* **decode**    - turn a sample into a domain object (e.g. an RDKit molecule);
* **visualize** - draw a sample onto a matplotlib ``Axes`` (plus an optional
                  per-sample caption and a per-batch summary line);
* **evaluate**  - validity / uniqueness / novelty over a batch of samples.

Only :meth:`GraphDomain.render` is required; every other method has a sensible
generic default. :class:`GenericGraphDomain` is the default node-link renderer
used whenever no domain is supplied, so existing behaviour is preserved.

Domain-specific implementations that pull in heavy optional dependencies (e.g.
the RDKit-backed molecule domain) live *outside* ``defog.core`` -- see
``defog.domains`` -- so importing the core stays lightweight and dependency-free
beyond what the model itself needs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def _num_nodes(data) -> int:
    x = getattr(data, "x", None)
    return int(x.shape[0]) if x is not None else 0


def _num_edges(data) -> int:
    """Undirected edge count (edge_index stores both directions)."""
    ei = getattr(data, "edge_index", None)
    if ei is None or ei.numel() == 0:
        return 0
    return int(ei.shape[1] // 2)


def _node_classes(data):
    """Per-node class index from one-hot (or already-index) node features."""
    x = getattr(data, "x", None)
    n = _num_nodes(data)
    if x is not None and x.dim() == 2 and x.shape[1] > 1:
        return x.argmax(dim=-1).tolist()
    return [0] * n


def _cmap(name: str):
    try:
        return matplotlib.colormaps[name]
    except Exception:  # older matplotlib
        return plt.get_cmap(name)


class GraphDomain(ABC):
    """Adapter that interprets DeFoG's categorical graphs for a concrete domain.

    Subclass and override what you need. The only required method is
    :meth:`render`. Sensible generic defaults are provided for decoding
    (none), validity (everything valid), identity (none), captions (none),
    the batch summary (node/edge averages), and metrics (computed from
    ``is_valid``/``identity``).
    """

    # ------------------------------------------------------------------ decode
    def decode(self, data) -> Optional[Any]:
        """Return a domain object for ``data`` (e.g. an RDKit Mol), or None."""
        return None

    def is_valid(self, data) -> bool:
        """Whether ``data`` is a valid instance of this domain. Default: True."""
        return True

    def identity(self, data) -> Optional[str]:
        """Canonical key for de-duplication / novelty, or None if not defined."""
        return None

    # --------------------------------------------------------------- visualize
    @abstractmethod
    def render(self, ax: "plt.Axes", data) -> None:
        """Draw a single sampled graph onto ``ax``."""

    def caption(self, data) -> Optional[str]:
        """Optional short caption shown under a sample's cell. Default: none."""
        return None

    def summarize(self, samples: Sequence[Any]) -> str:
        """One-line summary of a batch, shown in the figure title."""
        if not samples:
            return "no samples"
        avg_n = float(np.mean([_num_nodes(s) for s in samples]))
        avg_e = float(np.mean([_num_edges(s) for s in samples]))
        return f"avg nodes: {avg_n:.1f} | avg edges: {avg_e:.1f}"

    # ----------------------------------------------------------------- metrics
    def metrics(self, samples, reference: Optional[set] = None) -> Dict[str, float]:
        """Validity / uniqueness / novelty over ``samples`` (see module fn)."""
        return generation_metrics(self, samples, reference)


def generation_metrics(
    domain: GraphDomain,
    samples,
    reference: Optional[set] = None,
) -> Dict[str, float]:
    """Generic validity / uniqueness / novelty using a domain's primitives.

    - validity   = valid samples / total
    - uniqueness = distinct identities / valid samples
    - novelty    = unique identities absent from ``reference`` / unique
      (NaN when no reference set is given)
    """
    n = len(samples)
    valid_ids = [domain.identity(s) for s in samples if domain.is_valid(s)]
    n_valid = len(valid_ids)
    unique = {i for i in valid_ids if i is not None}
    n_unique = len(unique)
    if reference is not None:
        n_novel = sum(1 for i in unique if i not in reference)
        novelty = n_novel / n_unique if n_unique else 0.0
    else:
        novelty = float("nan")
    return {
        "validity": n_valid / n if n else 0.0,
        "uniqueness": n_unique / n_valid if n_valid else 0.0,
        "novelty": novelty,
    }


class GenericGraphDomain(GraphDomain):
    """Default domain: node-link rendering with node-class coloring.

    Renders with a networkx spring layout (matching DeFoG's previous default
    renderer) and derives a Weisfeiler-Lehman graph hash as the identity, so
    uniqueness/novelty work out of the box for arbitrary categorical graphs.
    """

    def __init__(self, layout_seed: int = 42, cmap: str = "Set3", node_size: int = 80):
        self.layout_seed = layout_seed
        self.cmap = cmap
        self.node_size = node_size

    def _to_nx(self, data):
        import networkx as nx

        n = _num_nodes(data)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        ei = getattr(data, "edge_index", None)
        if ei is not None and ei.numel() > 0:
            for k in range(ei.shape[1]):
                u, v = int(ei[0, k]), int(ei[1, k])
                if u < v:
                    g.add_edge(u, v)
        return g, n

    def render(self, ax, data) -> None:
        import networkx as nx

        g, n = self._to_nx(data)
        if n == 0:
            ax.text(0.5, 0.5, "empty", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            return
        classes = _node_classes(data)
        cmap = _cmap(self.cmap)
        num_classes = max(classes) + 1 if classes else 1
        colors = [cmap(c / max(num_classes, 1)) for c in classes]
        pos = nx.spring_layout(g, seed=self.layout_seed)
        nx.draw_networkx(
            g, pos, ax=ax,
            node_color=colors, node_size=self.node_size,
            width=0.8, edge_color="#888888", with_labels=False,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    def identity(self, data) -> Optional[str]:
        try:
            import networkx as nx
            from networkx.algorithms.graph_hashing import (
                weisfeiler_lehman_graph_hash as wl_hash,
            )

            g, n = self._to_nx(data)
            classes = _node_classes(data)
            for i in range(n):
                g.nodes[i]["c"] = str(classes[i]) if i < len(classes) else "0"
            return wl_hash(g, node_attr="c")
        except Exception:
            return None
