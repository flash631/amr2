"""run_demo.py — command-line driver for the 2D AMR demo.

Examples
--------
Run a Gaussian peak with three refinement iterations and show mesh + points:

    python run_demo.py --field gaussian --x0 0.55 --y0 0.55 --sigma 0.03         --seed-splits 2 --iters 3 --plot mesh points contours

Save a side-by-side figure to a PNG:

    python run_demo.py --plot mesh contours --save demo.png

Notes
-----
- If --threshold is omitted, the script refines top --top-fraction fraction
  each iteration (default 0.2).
- Use --method center_gradient to refine on |∇f| instead of corner range.
"""
from __future__ import annotations

import argparse
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from amr.domain import make_field, DEFAULT_BOUNDS
from amr.mesh import QuadTreeMesh
from amr.refine import run_adaptive
from amr.error_indicator import compute_errors
from amr.visualize import plot_mesh, plot_points, plot_field_contours, plot_error


PLOT_CHOICES = ["mesh", "points", "contours", "error"]


def _add_field_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("field")
    g.add_argument("--field", type=str, default="gaussian",
                   help="One of: gaussian, step_x, circle_step")
    # Common params (only some apply depending on field)
    g.add_argument("--x0", type=float, default=0.55)
    g.add_argument("--y0", type=float, default=0.55)
    g.add_argument("--sigma", type=float, default=0.04)
    g.add_argument("--A", type=float, default=1.0)
    g.add_argument("--cx", type=float, default=0.5)
    g.add_argument("--cy", type=float, default=0.5)
    g.add_argument("--r", type=float, default=0.2)
    g.add_argument("--low", type=float, default=0.0)
    g.add_argument("--high", type=float, default=1.0)


def _add_mesh_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("mesh")
    g.add_argument("--bounds", type=float, nargs=4, default=list(DEFAULT_BOUNDS),
                   metavar=("x0", "x1", "y0", "y1"), help="Domain bounds")
    g.add_argument("--seed-splits", type=int, default=2,
                   help="Uniform splits applied to root (2-> 4x4)")


def _add_refine_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("refine")
    g.add_argument("--iters", type=int, default=3, help="Iterations")
    g.add_argument("--method", type=str, default="corner_range",
                   help="Error method: corner_range|center_gradient")
    g.add_argument("--threshold", type=float, default=None,
                   help="Absolute error cutoff; if omitted uses top-fraction")
    g.add_argument("--top-fraction", type=float, default=0.2,
                   help="When threshold is None, refine this top fraction")
    g.add_argument("--max-depth", type=int, default=8, help="Max quadtree depth")
    g.add_argument("--balance", dest="balance", action="store_true",
                   help="Enable simple balancing pass")
    g.add_argument("--no-balance", dest="balance", action="store_false",
                   help="Disable balancing pass")
    g.set_defaults(balance=False)
    g.add_argument("--growth-limit", type=int, default=None,
                   help="Max number of leaves allowed after each iteration")


def _add_plot_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("plot")
    g.add_argument("--plot", type=str, nargs="*", default=["mesh", "points"],
                   choices=PLOT_CHOICES, help="What to plot")
    g.add_argument("--samples-per-cell", type=int, default=3,
                   help="For contours: sample grid density per cell")
    g.add_argument("--levels", type=int, default=15, help="Contour levels")
    g.add_argument("--save", type=str, default=None, help="Save figure to path")



def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="2D AMR demo")
    _add_field_args(parser)
    _add_mesh_args(parser)
    _add_refine_args(parser)
    _add_plot_args(parser)

    args = parser.parse_args(argv)

    # Build field (only relevant kwargs will be used by the chosen field)
    field_kwargs = dict(
        x0=args.x0, y0=args.y0, sigma=args.sigma, A=args.A,
        cx=args.cx, cy=args.cy, r=args.r, low=args.low, high=args.high,
    )
    field = make_field(args.field, **field_kwargs)

    # Mesh and refinement
    mesh = QuadTreeMesh(bounds=tuple(args.bounds), seed_splits=args.seed_splits)
    run_adaptive(
        mesh,
        field,
        iterations=args.iters,
        threshold=args.threshold,
        target_top_fraction=args.top_fraction,
        method=args.method,
        max_depth=args.max_depth,
        balance=args.balance,
        growth_limit=args.growth_limit,
        verbose=True,
    )

    # Prepare figure with subplots for selected plots
    nplots = len(args.plot)
    fig, axes = plt.subplots(1, nplots, figsize=(6 * nplots, 6), squeeze=False)

    for k, what in enumerate(args.plot):
        ax = axes[0, k]
        if what == "mesh":
            plot_mesh(mesh, ax=ax)
        elif what == "points":
            plot_points(mesh, ax=ax)
        elif what == "contours":
            plot_field_contours(mesh, field, ax=ax,
                                samples_per_cell=args.samples_per_cell,
                                levels=args.levels)
        elif what == "error":
            leaves = mesh.leaf_cells()
            errs = compute_errors(leaves, field, method=args.method)
            plot_error(mesh, leaves, errs, ax=ax)
        else:
            ax.text(0.5, 0.5, f"Unknown plot: {what}", ha="center", va="center")
            ax.axis("off")

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {args.save}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
