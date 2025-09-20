"""error_indicator.py — cell-wise error metrics for AMR decisions.

This module provides simple, robust error indicators to decide which quadtree
cells to refine. Two choices are included:

1) corner_range: max(corner_samples) - min(corner_samples)
   - Discrete and cheap. Closest to a 2-point 1D endpoint difference.
   - Works well for discontinuities (steps), sharp peaks, and edges.

2) center_gradient: finite-difference gradient magnitude at the cell center
   - Smooth and scale-aware. Useful when fields are continuous but steep.

`compute_errors(cells, field, method)` returns a NumPy array of error values
aligned with the input `cells` list.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .mesh import Cell
from .domain import ScalarField2D


# --------------------------
# Corner sampling helpers
# --------------------------

def _sample_corners_vectorized(cells: List[Cell], field: ScalarField2D) -> None:
    """Populate `cell.f_corners` for any cells that don't have it yet.

    Uses a single vectorized field evaluation for all corners across `cells`.
    After this call, every cell in `cells` is guaranteed to have `f_corners`.
    """
    # Indices of cells needing samples
    need_idx = [i for i, c in enumerate(cells) if c.f_corners is None]
    if not need_idx:
        return

    # Build arrays of corner coordinates for missing cells
    x0 = np.array([cells[i].x0 for i in need_idx], dtype=float)
    x1 = np.array([cells[i].x1 for i in need_idx], dtype=float)
    y0 = np.array([cells[i].y0 for i in need_idx], dtype=float)
    y1 = np.array([cells[i].y1 for i in need_idx], dtype=float)

    # Evaluate field at 4 corners in a vectorized way
    f00 = field(x0, y0)  # (x0, y0)
    f10 = field(x1, y0)  # (x1, y0)
    f01 = field(x0, y1)  # (x0, y1)
    f11 = field(x1, y1)  # (x1, y1)

    # Store back per cell (2x2 array)
    for k, i in enumerate(need_idx):
        cells[i].f_corners = np.array([[f00[k], f10[k]], [f01[k], f11[k]]], dtype=float)


# --------------------------
# Error indicators
# --------------------------

def corner_range(cell: Cell) -> float:
    """Return max-min of corner values for a cell.

    Requires `cell.f_corners` to be populated. For convenience, callers may use
    `compute_errors(..., method='corner_range')` which ensures sampling.
    """
    if cell.f_corners is None:
        raise ValueError("corner_range: cell.f_corners is None (not sampled)")
    fc = cell.f_corners.ravel()
    return float(fc.max() - fc.min())


def center_gradient(cell: Cell, field: ScalarField2D) -> float:
    """Finite-difference gradient magnitude at cell center.

    Uses symmetric differences at offsets h_x = width/4 and h_y = height/4
    confined to the cell, so no samples spill outside the cell.
    """
    xc, yc = cell.center()
    hx = 0.25 * cell.width()
    hy = 0.25 * cell.height()

    # Evaluate along x
    fxp = field(xc + hx, yc)
    fxm = field(xc - hx, yc)
    dfdx = (fxp - fxm) / (2.0 * hx)

    # Evaluate along y
    fyp = field(xc, yc + hy)
    fym = field(xc, yc - hy)
    dfdy = (fyp - fym) / (2.0 * hy)

    g2 = float(dfdx * dfdx + dfdy * dfdy)
    return float(np.sqrt(g2))


def compute_errors(
    cells: List[Cell],
    field: ScalarField2D,
    *,
    method: str = "corner_range",
) -> np.ndarray:
    """Compute an error value for each cell.

    Parameters
    ----------
    cells : list[Cell]
        Cells to evaluate (typically current leaves).
    field : ScalarField2D
        Callable scalar field f(x,y).
    method : str, optional (default: 'corner_range')
        One of {'corner_range', 'center_gradient'}; aliases accepted:
        - 'corner_range': {'corner_range', 'corner', 'corners', 'range', 'cr'}
        - 'center_gradient': {'center_gradient', 'gradient', 'grad', 'cg'}

    Returns
    -------
    np.ndarray
        Errors aligned with `cells` (dtype=float, shape (N,)).
    """
    key = method.strip().lower()

    if key in {"corner_range", "corner", "corners", "range", "cr"}:
        # Ensure all needed corner samples exist, then compute ranges
        _sample_corners_vectorized(cells, field)
        errs = np.empty(len(cells), dtype=float)
        for i, c in enumerate(cells):
            fc = c.f_corners.ravel()  # type: ignore[union-attr]
            errs[i] = float(fc.max() - fc.min())
        return errs

    if key in {"center_gradient", "gradient", "grad", "cg"}:
        errs = np.empty(len(cells), dtype=float)
        for i, c in enumerate(cells):
            errs[i] = center_gradient(c, field)
        return errs

    raise ValueError(
        "Unknown method '{method}'. Use 'corner_range' or 'center_gradient'.".format(method=method)
    )


__all__ = ["corner_range", "center_gradient", "compute_errors"]
