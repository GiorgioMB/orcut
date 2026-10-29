"""Python entry points of the solver."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Optional, Sequence

import numpy as np

from . import _kernels as K
from .graph import Graph

__all__ = ["exact", "as_graph", "threads"]


@contextmanager
def threads(n_jobs: Optional[int]):
    """Temporarily set the number of Numba threads (scikit-learn semantics:
    None = Numba default, -1 = all, k > 0 = k threads, k < -1 = all but |k| - 1)."""
    import numba

    prev = numba.get_num_threads()
    if n_jobs is None:
        k = prev
    else:
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        cpus = numba.config.NUMBA_NUM_THREADS
        k = cpus if n_jobs == -1 else (n_jobs if n_jobs > 0 else max(1, cpus + 1 + n_jobs))
        k = max(1, min(k, cpus))
    numba.set_num_threads(k)
    try:
        yield
    finally:
        numba.set_num_threads(prev)


def as_graph(graph) -> Graph:
    """A :class:`Graph` from a Graph, a networkx graph, a PyG ``Data``, or an edge array."""
    if isinstance(graph, Graph):
        return graph
    if hasattr(graph, "edge_index"):
        return Graph.from_edge_index(graph.edge_index, getattr(graph, "num_nodes", None))
    if hasattr(graph, "nodes") and hasattr(graph, "edges") and callable(getattr(graph, "edges")):
        g, _ = Graph.from_networkx(graph)
        return g
    return Graph.from_edges(graph)


def exact(graph, which: Sequence[str] = ("c_OR", "c_OR0", "c_LLY"), edges: Optional[np.ndarray] = None,
          n_jobs: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Exact curvatures of the undirected edges of ``graph`` (all, or ``edges``, an array of edge ids).

    ``which`` is any subset of {"c_OR" (lazy, idleness 1/(deg+1)), "c_OR0" (non-lazy), "c_LLY"
    (Lin--Lu--Yau)}. Values are aligned with ``graph.edges`` (or with ``edges``); for a networkx
    graph, the edge order is that of :meth:`Graph.from_networkx`.
    """
    which = tuple(which)
    bad = set(which) - {"c_OR", "c_OR0", "c_LLY"}
    if bad:
        raise ValueError(f"unknown curvature(s): {sorted(bad)}")
    g = as_graph(graph)
    import numba

    ids = np.arange(g.num_edges, dtype=np.int64) if edges is None else np.ascontiguousarray(edges, dtype=np.int64)
    flags = ("c_OR" in which, "c_OR0" in which, "c_LLY" in which)
    with threads(n_jobs):
        raw = K.exact_ids(g.indptr, g.indices, g.edges, ids, *flags, 8 * numba.get_num_threads())
    out = {}
    for k, name in enumerate(("c_OR", "c_OR0", "c_LLY")):
        if flags[k]:
            out[name] = raw[:, k]
    return out
