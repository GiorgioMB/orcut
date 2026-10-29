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


# ------------------------------------------------------------------ orcut >= 1.1

def curvature(graph, measure="lazy", edges: Optional[np.ndarray] = None, n_jobs: Optional[int] = None,
              exact_fraction: bool = False, diagnostics: bool = False, flags: Optional[int] = None):
    """Exact curvature 1 - W1(m_i, m_j) of the undirected edges of ``graph`` (all, or the edge ids
    ``edges``), aligned with ``graph.edges``.

    ``measure``: ``"lazy"`` (default, idleness 1/(deg+1)), ``"nonlazy"``, ``"lly"`` (Lin--Lu--Yau),
    a number or Fraction p in [0, 1) (uniform idleness p), or a :class:`orcut.measures.Walk`
    (weighted walk with rational idleness). With ``exact_fraction`` the result also contains the
    exact value of W1 as an integer fraction (numerator, denominator) per edge (for ``"lly"``,
    curvature = (1 - W1) (deg_max + 1) / deg_max); with ``diagnostics`` the per-edge diagnostics of
    :mod:`orcut._core` (``N_DIAG`` columns). ``flags`` switches off engineering steps (ablations;
    values do not change). Returns an array, or a dict when extra outputs are requested.
    """
    from . import _core as C
    from .measures import parse

    g = as_graph(graph)
    mode, pn, pd, tabs = parse(measure, g)
    import numba

    ids = np.arange(g.num_edges, dtype=np.int64) if edges is None else np.ascontiguousarray(edges, dtype=np.int64)
    with threads(n_jobs):
        val, num, diag = C.core_ids(g.indptr, g.indices, g.edges, ids, mode, pn, pd,
                                    C.F_ALL if flags is None else flags, 8 * numba.get_num_threads(), tabs)
    if not (exact_fraction or diagnostics):
        return val
    out = {"curvature": val}
    if exact_fraction:
        out["w1_num"] = num[:, 0]
        out["w1_den"] = num[:, 1]
    if diagnostics:
        out["diagnostics"] = diag
    return out


def ball_curvature(graph, radius: int = 2, edges: Optional[np.ndarray] = None, n_jobs: Optional[int] = None,
                   certify: bool = False):
    """Curvature for m_u uniform on the ball of radius ``radius`` around u (Ollivier's ball
    measures), exact, by the layered closure with D = 2 radius + 1 (reference implementation:
    explicit distances). With ``certify`` the diagnostics report, per edge, whether the plan and the
    potentials of the minimum cut were verified (column 5: 1 = verified)."""
    from . import _layered as LY

    g = as_graph(graph)
    ids = np.arange(g.num_edges, dtype=np.int64) if edges is None else np.ascontiguousarray(edges, dtype=np.int64)
    import numba

    with threads(n_jobs):
        val, diag = LY.layered_ids(g.indptr, g.indices, g.edges, ids, LY.K_BALL, radius, certify,
                                   8 * numba.get_num_threads())
    return (val, diag) if certify else val


def walk_curvature(graph, steps: int = 2, lazy: bool = True, edges: Optional[np.ndarray] = None,
                   n_jobs: Optional[int] = None, certify: bool = False):
    """Curvature for the ``steps``-step random walk (lazy: uniform on N[v] at every step), by the
    layered closure with D = 2 steps + 1; masses are rounded to 2^-40 (error of W1 at most
    D (|B_i| + |B_j|) 2^-40)."""
    from . import _layered as LY

    g = as_graph(graph)
    ids = np.arange(g.num_edges, dtype=np.int64) if edges is None else np.ascontiguousarray(edges, dtype=np.int64)
    import numba

    with threads(n_jobs):
        val, diag = LY.layered_ids(g.indptr, g.indices, g.edges, ids, LY.K_WALK_LAZY if lazy else LY.K_WALK,
                                   steps, certify, 8 * numba.get_num_threads())
    return (val, diag) if certify else val
