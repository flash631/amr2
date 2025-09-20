"""visualize.py — plotting helpers for AMR meshes and fields.

Provides light-weight plotting utilities built on matplotlib:

- plot_mesh(mesh, ax=None, linewidth=0.75)
    Draw leaf-cell boundaries as a wireframe grid.

- plot_points(mesh, ax=None)
    Scatter unique corner points of all leaf cells; useful to see clustering.

- plot_field_contours(mesh, field, ax=None, samples_per_cell=3, levels=15)
    Sample the scalar field on a per-cell lattice and draw tricontours.

- plot_error(mesh, cells, errors, ax=None)
    Color each leaf by its error value using a colormap.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

from .mesh import QuadTreeMesh, Cell
from .domain import ScalarField2D


# --------------------------
# Helpers
# --------------------------
def _ax(ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    return ax


def _set_equal_box(ax, bounds: tuple[float, float, float, float]):
    x0, x1, y0, y1 = bounds
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


# --------------------------
# Public plotting functions
# --------------------------
def plot_mesh(mesh: QuadTreeMesh, ax=None, *, linewidth: float = 0.75):
    """Plot the wireframe of all leaf cells."""
    ax = _ax(ax)
    lines = []
    for c in mesh.leaf_cells():
        x0, x1, y0, y1 = c.x0, c.x1, c.y0, c.y1
        lines.extend([
            [(x0, y0), (x1, y0)],  # bottom
            [(x0, y1), (x1, y1)],  # top
            [(x0, y0), (x0, y1)],  # left
            [(x1, y0), (x1, y1)],  # right
        ])
    lc = LineCollection(lines, linewidths=linewidth)
    ax.add_collection(lc)
    _set_equal_box(ax, mesh.bounds)
    ax.set_title("Leaf mesh (wireframe)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def plot_points(mesh: QuadTreeMesh, ax=None):
    """Scatter unique corner points from leaf cells."""
    ax = _ax(ax)
    pts = []
    for c in mesh.leaf_cells():
        pts.append((c.x0, c.y0))
        pts.append((c.x1, c.y0))
        pts.append((c.x0, c.y1))
        pts.append((c.x1, c.y1))
    if not pts:
        _set_equal_box(ax, mesh.bounds)
        return ax
    P = np.array(list(set(pts)), dtype=float)  # unique
    ax.scatter(P[:, 0], P[:, 1], s=8)
    _set_equal_box(ax, mesh.bounds)
    ax.set_title("Corner point distribution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def plot_field_contours(
    mesh: QuadTreeMesh,
    field: ScalarField2D,
    ax=None,
    *,
    samples_per_cell: int = 3,
    levels: int = 15,
):
    """Tri-contour plot of the scalar field sampled per leaf cell.

    We create a small Cartesian grid inside each leaf and sample the field on
    that lattice, then use a global triangulation for contouring. This avoids
    forcing a uniform grid and displays refinement nicely.
    """
    import matplotlib.tri as mtri

    ax = _ax(ax)
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    spc = max(2, int(samples_per_cell))

    for c in mesh.leaf_cells():
        # build a tiny grid inside the cell
        gx = np.linspace(c.x0, c.x1, spc)
        gy = np.linspace(c.y0, c.y1, spc)
        X, Y = np.meshgrid(gx, gy, indexing="xy")
        Z = field(X, Y)
        xs.extend(X.ravel().tolist())
        ys.extend(Y.ravel().tolist())
        zs.extend(np.asarray(Z).ravel().tolist())

    if len(xs) < 3:
        _set_equal_box(ax, mesh.bounds)
        return ax

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    tri = mtri.Triangulation(xs, ys)
    cntr = ax.tricontourf(tri, zs, levels=levels, alpha=0.9)
    plt.colorbar(cntr, ax=ax, label="f(x, y)")

    _set_equal_box(ax, mesh.bounds)
    ax.set_title("Field contours (tricontourf)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


ess_default_norm = Normalize(vmin=None, vmax=None)


def plot_error(
    mesh: QuadTreeMesh,
    cells,
    errors: np.ndarray,
    ax=None,
    *,
    cmap: str = "viridis",
    norm: Optional[Normalize] = None,
    draw_edges: bool = True,
):
    """Color each provided cell by its error value.

    Parameters
    ----------
    cells : iterable of Cell
        Cells to color (usually the current leaves).
    errors : ndarray
        Error values aligned with `cells` (same length).
    cmap : str
        Matplotlib colormap name.
    norm : matplotlib.colors.Normalize, optional
        If None, uses min/max from `errors`.
    draw_edges : bool
        If True, draw cell borders in a light outline.
    """
    ax = _ax(ax)
    patches: List[Rectangle] = []
    for c in cells:
        patches.append(Rectangle((c.x0, c.y0), c.width(), c.height()))

    if norm is None:
        vmin = float(np.nanmin(errors))
        vmax = float(np.nanmax(errors))
        if vmin == vmax:
            vmax = vmin + 1e-12
        norm = Normalize(vmin=vmin, vmax=vmax)

    pc = PatchCollection(patches, cmap=cmap, norm=norm, linewidths=0.6 if draw_edges else 0.0,
                         edgecolor="black" if draw_edges else None)
    pc.set_array(np.asarray(errors, dtype=float))
    ax.add_collection(pc)
    plt.colorbar(pc, ax=ax, label="error")

    _set_equal_box(ax, mesh.bounds)
    ax.set_title("Cell-wise error shading")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


__all__ = [
    "plot_mesh",
    "plot_points",
    "plot_field_contours",
    "plot_error",
]
