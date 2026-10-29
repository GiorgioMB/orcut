"""orcut: exact Ollivier--Ricci curvature with one minimum cut per edge.

>>> import orcut
>>> g = orcut.Graph.from_edges([(0, 1), (1, 2), (2, 0), (2, 3)])
>>> out = orcut.exact(g)                  # lazy, non-lazy, and Lin--Lu--Yau curvature
>>> out["c_OR"]                           # aligned with g.edges (u < v)

Core dependencies are numpy and numba.
"""
from .graph import Graph
from .api import as_graph, exact
from ._kernels import N_EXACT, exact_ids, exact_or_ids

__all__ = ["Graph", "as_graph", "exact", "exact_ids", "exact_or_ids", "N_EXACT"]
__version__ = "1.0.0"
