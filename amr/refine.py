"""refine.py — one-step and multi-iteration refinement drivers for 2D AMR.

This module connects a QuadTree mesh, a scalar field, and error indicators to
perform adaptive refinement. It supports absolute-threshold or percentile-based
selection, optional (lightweight) 2:1-like balancing, and simple growth limits.

Key APIs
--------
- sample_cell_corners(cell, field): ensure corner samples are cached on a cell.
- refine_once(mesh, field, ...): run a single refine iteration and return stats.
- run_adaptive(mesh, field, ...): run multiple iterations, collecting stats.

Notes on balancing
------------------
We implement a *very* simple neighbor-based pass to avoid large level jumps.
Neighbors are detected by shared edges with small floating tolerance. If two
neighboring leaves differ in depth by ≥ 2, we mark the coarser one to refine.
One balancing pass is applied after the main refinement step when
`balance=True`. This is intentionally minimal and O(N^2) in leaf count—fine
for didactic demos.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .mesh import Cell, QuadTreeMesh
from .domain import ScalarField2D
from .error_indicator import compute_errors


# --------------------------
# Utilities
# --------------------------

def sample_cell_corners(cell: Cell, field: ScalarField2D) -> None:
    """Ensure `cell.f_corners` is populated by sampling f at the 4 corners."""
    if cell.f_corners is not None:
        return
    x0, x1, y0, y1 = cell.x0, cell.x1, cell.y0, cell.y1
    f00 = field(x0, y0)
    f10 = field(x1, y0)
    f01 = field(x0, y1)
    f11 = field(x1, y1)
    cell.f_corners = np.array([[f00, f10], [f01, f11]], dtype=float)


def _select_threshold_by_fraction(values: np.ndarray, frac: float) -> float:
    """Return a cutoff so that approximately the top `frac` fraction are kept.

    If `frac <= 0`, returns +inf (select none); if `frac >= 1`, returns -inf
    (select all). For 0 < frac < 1, chooses the value at rank ceil(frac*N)-1
    in descending order.
    """
    N = int(values.size)
    if N == 0:
        return float("inf")
    if frac <= 0:
        return float("inf")
    if frac >= 1:
        return float("-inf")
    k = max(1, int(np.ceil(frac * N)))  # how many to keep at least
    # kth largest → index N-k in ascending order
    idx = np.argpartition(values, N - k)[N - k]
    cutoff = float(values[idx])
    return cutoff


def _shared_edge(a: Cell, b: Cell, *, eps: float = 1e-12) -> bool:
    """Return True if cells share an edge (not just a corner)."""
    # horizontal adjacency (share vertical edge)
    if abs(a.x1 - b.x0) <= eps or abs(a.x0 - b.x1) <= eps:
        # overlapping y-interval strictly (more than a point)
        return (a.y0 < b.y1 - eps) and (b.y0 < a.y1 - eps)
    # vertical adjacency (share horizontal edge)
    if abs(a.y1 - b.y0) <= eps or abs(a.y0 - b.y1) <= eps:
        return (a.x0 < b.x1 - eps) and (b.x0 < a.x1 - eps)
    return False


def _balance_once(mesh: QuadTreeMesh, *, growth_budget_splits: Optional[int]) -> int:
    """One pass of coarse neighbor refinement if depth difference ≥ 2.

    Returns number of additional refinements performed.
    """
    leaves = mesh.leaf_cells()
    n = len(leaves)
    to_refine: set[Cell] = set()
    # O(n^2) pairwise; sufficient for demo sizes
    for i in range(n):
        for j in range(i + 1, n):
            a, b = leaves[i], leaves[j]
            if not _shared_edge(a, b):
                continue
            if abs(a.depth - b.depth) >= 2:
                coarse = a if a.depth < b.depth else b
                to_refine.add(coarse)
    refined = 0
    if not to_refine:
        return 0
    # Respect remaining growth budget if provided
    candidates = sorted(to_refine, key=lambda c: (c.depth, c.id))  # deterministic
    for c in candidates:
        if growth_budget_splits is not None and growth_budget_splits <= 0:
            break
        mesh.subdivide(c)
        refined += 1
        if growth_budget_splits is not None:
            growth_budget_splits -= 1
    return refined


# --------------------------
# Refinement drivers
# --------------------------

def refine_once(
    mesh: QuadTreeMesh,
    field: ScalarField2D,
    *,
    threshold: Optional[float] = None,
    method: str = "corner_range",
    target_top_fraction: Optional[float] = None,
    max_depth: Optional[int] = None,
    balance: bool = False,
    growth_limit: Optional[int] = None,
) -> Dict[str, float | int]:
    """Run a single refinement iteration.

    Selection modes:
    - If `threshold` is provided: refine all eligible cells with error ≥ threshold.
    - Else: refine the top `target_top_fraction` fraction (default 0.2).

    Parameters
    ----------
    mesh : QuadTreeMesh
        Current mesh.
    field : ScalarField2D
        Scalar field f(x,y).
    threshold : float, optional
        Absolute error cutoff.
    method : str, optional
        Error indicator method (see `compute_errors`).
    target_top_fraction : float, optional
        If `threshold` is None, keep this top fraction. Default 0.2.
    max_depth : int, optional
        Do not refine leaves at or beyond this depth.
    balance : bool, optional
        Apply a light neighbor-based balancing pass after refinement.
    growth_limit : int, optional
        Upper bound on number of leaves allowed after this call. If None,
        unbounded. (Note: subdividing one leaf increases leaf count by +3.)

    Returns
    -------
    dict
        {"refined": int, "leaves": int, "max_depth": int, "threshold_used": float,
         "balanced": int}
    """
    # Determine leaf growth budget
    start_leaves = mesh.count_leaves()
    if growth_limit is not None:
        max_leaves_allowed = int(growth_limit)
        growth_budget_splits = max(0, (max_leaves_allowed - start_leaves) // 3)
    else:
        max_leaves_allowed = None
        growth_budget_splits = None

    # Eligible leaves (depth filter)
    leaves_all = mesh.leaf_cells()
    leaves: List[Cell] = [c for c in leaves_all if (max_depth is None or c.depth < max_depth)]

    # Early exit if nothing eligible or no budget
    if not leaves or (growth_budget_splits is not None and growth_budget_splits <= 0):
        return {
            "refined": 0,
            "leaves": start_leaves,
            "max_depth": mesh.max_depth(),
            "threshold_used": float("nan") if threshold is None else float(threshold),
            "balanced": 0,
        }

    # Compute errors
    errs = compute_errors(leaves, field, method=method)

    # Determine cutoff
    if threshold is None:
        frac = 0.2 if (target_top_fraction is None) else float(target_top_fraction)
        cutoff = _select_threshold_by_fraction(errs, frac)
    else:
        cutoff = float(threshold)

    # Pick candidates (errs >= cutoff). Deterministic ordering by (error desc, id)
    idx = np.where(errs >= cutoff)[0]
    # If nothing selected due to strict cutoff, keep the single max cell
    if idx.size == 0 and leaves:
        idx = np.array([int(np.argmax(errs))])
    order = sorted(idx.tolist(), key=lambda i: (-float(errs[i]), leaves[i].id))
    selected: List[Cell] = [leaves[i] for i in order]

    # Respect growth budget
    if growth_budget_splits is not None:
        selected = selected[:growth_budget_splits]

    # Subdivide
    refined_count = 0
    for cell in selected:
        mesh.subdivide(cell)
        refined_count += 1

    # Update budget after main step
    if growth_budget_splits is not None:
        growth_budget_splits = max(0, growth_budget_splits - refined_count)

    # Optional balancing pass (single pass)
    balanced_count = 0
    if balance and (growth_budget_splits is None or growth_budget_splits > 0):
        balanced_count = _balance_once(mesh, growth_budget_splits=growth_budget_splits)

    return {
        "refined": int(refined_count),
        "leaves": int(mesh.count_leaves()),
        "max_depth": int(mesh.max_depth()),
        "threshold_used": float(cutoff),
        "balanced": int(balanced_count),
    }


def run_adaptive(
    mesh: QuadTreeMesh,
    field: ScalarField2D,
    *,
    iterations: int = 3,
    threshold: Optional[float] = None,
    target_top_fraction: float = 0.2,
    method: str = "corner_range",
    max_depth: Optional[int] = 8,
    balance: bool = False,
    growth_limit: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict[str, float | int]]:
    """Run multiple refinement iterations, collecting per-iteration stats.

    Stops early if an iteration refines zero cells.
    """
    stats: List[Dict[str, float | int]] = []
    for k in range(int(iterations)):
        s = refine_once(
            mesh,
            field,
            threshold=threshold,
            method=method,
            target_top_fraction=target_top_fraction,
            max_depth=max_depth,
            balance=balance,
            growth_limit=growth_limit,
        )
        stats.append(s)
        if verbose:
            print(
                f"iter {k+1}: refined={s['refined']} leaves={s['leaves']} "
                f"max_depth={s['max_depth']} balance+={s['balanced']} cutoff={s['threshold_used']:.6g}"
            )
        if s["refined"] == 0 and s["balanced"] == 0:
            break
    return stats


__all__ = [
    "sample_cell_corners",
    "refine_once",
    "run_adaptive",
]


if __name__ == "__main__":  # tiny demo
    from .domain import make_field
    from .mesh import QuadTreeMesh

    f = make_field("gaussian", x0=0.55, y0=0.55, sigma=0.04, A=1.0)
    mesh = QuadTreeMesh(seed_splits=2)
    out = run_adaptive(mesh, f, iterations=3, target_top_fraction=0.2, balance=True, verbose=True)
    print(out[-1] if out else {})
