"""mesh.py — minimal QuadTree mesh for 2D AMR demos.

This module provides a tiny quadtree data structure suitable for adaptive
mesh refinement examples. It keeps a single root cell that can be uniformly
pre-split (seed) and selectively subdivided later.

Key concepts
------------
- Cell: axis-aligned rectangle [x0,x1] × [y0,y1] with optional children.
- QuadTreeMesh: holds the root cell and convenience methods for traversal.

Design notes
------------
- Children order is [SW, SE, NW, NE] (bottom-left, bottom-right,
  top-left, top-right). This is documented for reproducibility.
- "Leaf" means a cell with `children is None`.
- Corner samples `f_corners` (shape (2,2)) are cached per cell when computed
  by refinement routines; this module doesn't evaluate fields by itself.
- Seed splits: if `seed_splits=k`, we uniformly subdivide all leaves k times
  starting from the single root, yielding a coarse 2^k × 2^k grid.

This mesh purposefully omits neighbor tables and 2:1 balancing to stay small.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Iterable

import numpy as np


@dataclass(slots=True)
class Cell:
    """A rectangular cell in a quadtree.

    Attributes
    ----------
    x0, x1, y0, y1 : float
        Cell bounds with x0 < x1 and y0 < y1.
    depth : int
        0 for the root, increases by 1 with each subdivision.
    f_corners : Optional[np.ndarray]
        Cached samples of the scalar field at corners in a (2,2) array:
        [[f(x0,y0), f(x1,y0)],
         [f(x0,y1), f(x1,y1)]]
    children : Optional[List[Cell]]
        None if this is a leaf; otherwise 4 children in order [SW, SE, NW, NE].
    id : int
        A unique integer identifier, assigned by the mesh.
    """

    x0: float
    x1: float
    y0: float
    y1: float
    depth: int = 0
    f_corners: Optional[np.ndarray] = None
    children: Optional[List["Cell"]] = None
    id: int = -1

    # ----- convenience methods -----
    def is_leaf(self) -> bool:
        return self.children is None

    def width(self) -> float:
        return self.x1 - self.x0

    def height(self) -> float:
        return self.y1 - self.y0

    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))

    def corners_xy(self) -> np.ndarray:
        """Return the 4 corner coordinates as array of shape (4, 2).
        Order: [(x0,y0),(x1,y0),(x0,y1),(x1,y1)].
        """
        return np.array(
            [
                [self.x0, self.y0],
                [self.x1, self.y0],
                [self.x0, self.y1],
                [self.x1, self.y1],
            ],
            dtype=float,
        )

    def __post_init__(self) -> None:
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError("Invalid cell bounds: require x0<x1 and y0<y1")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"Cell(id={self.id}, depth={self.depth}, "
            f"[{self.x0:.3f},{self.x1:.3f}]x[{self.y0:.3f},{self.y1:.3f}], "
            f"leaf={self.is_leaf()})"
        )


class QuadTreeMesh:
    """Quadtree mesh over a rectangular domain.

    Parameters
    ----------
    bounds : tuple[float,float,float,float]
        (x_min, x_max, y_min, y_max) domain rectangle.
    seed_splits : int, optional (default: 0)
        Number of uniform refinement levels to apply to the root. A value of
        k yields an initial 2^k × 2^k grid.
    """

    __slots__ = ("root", "_next", "bounds")

    def __init__(self, bounds=(0.0, 1.0, 0.0, 1.0), seed_splits: int = 0) -> None:
        x0, x1, y0, y1 = bounds
        if not (x0 < x1 and y0 < y1):
            raise ValueError("Invalid bounds: require x0<x1 and y0<y1")
        self.bounds = (float(x0), float(x1), float(y0), float(y1))
        self._next = 0  # id counter
        self.root = Cell(x0, x1, y0, y1, depth=0, id=self._alloc_id())

        # Apply uniform seeding if requested
        if seed_splits > 0:
            for _ in range(int(seed_splits)):
                for cell in list(self.leaf_cells()):
                    self.subdivide(cell)

    # ----- id management -----
    def _alloc_id(self) -> int:
        i = self._next
        self._next += 1
        return i

    def recompute_ids(self) -> None:
        """Reassign sequential ids in a stable breadth-first order."""
        self._next = 0
        for cell in self._bfs_cells():
            cell.id = self._alloc_id()

    # ----- traversal -----
    def _bfs_cells(self) -> Iterable[Cell]:
        q: List[Cell] = [self.root]
        while q:
            c = q.pop(0)
            yield c
            if c.children is not None:
                q.extend(c.children)

    def leaf_cells(self) -> List[Cell]:
        """Return a list of current leaf cells."""
        leaves: List[Cell] = []
        stack: List[Cell] = [self.root]
        while stack:
            c = stack.pop()
            if c.children is None:
                leaves.append(c)
            else:
                stack.extend(c.children)
        return leaves

    # ----- modification -----
    def subdivide(self, cell: Cell) -> List[Cell]:
        """Split a leaf cell into four children [SW, SE, NW, NE].

        Returns the list of children.
        """
        if cell.children is not None:
            # already split
            return cell.children

        x0, x1, y0, y1 = cell.x0, cell.x1, cell.y0, cell.y1
        xm = 0.5 * (x0 + x1)
        ym = 0.5 * (y0 + y1)
        d = cell.depth + 1

        children = [
            Cell(x0, xm, y0, ym, depth=d, id=self._alloc_id()),  # SW
            Cell(xm, x1, y0, ym, depth=d, id=self._alloc_id()),  # SE
            Cell(x0, xm, ym, y1, depth=d, id=self._alloc_id()),  # NW
            Cell(xm, x1, ym, y1, depth=d, id=self._alloc_id()),  # NE
        ]
        cell.children = children
        return children

    # ----- stats -----
    def count_leaves(self) -> int:
        return len(self.leaf_cells())

    def max_depth(self) -> int:
        m = 0
        for c in self._bfs_cells():
            if c.depth > m:
                m = c.depth
        return m


__all__ = ["Cell", "QuadTreeMesh"]


if __name__ == "__main__":  # quick smoke test
    mesh = QuadTreeMesh(bounds=(0, 1, 0, 1), seed_splits=2)
    print("Initial leaves:", mesh.count_leaves(), "max_depth:", mesh.max_depth())
    # Refine the central cell roughly by picking the leaf containing (0.5,0.5)
    for leaf in mesh.leaf_cells():
        x0, x1, y0, y1 = leaf.x0, leaf.x1, leaf.y0, leaf.y1
        if x0 <= 0.5 <= x1 and y0 <= 0.5 <= y1:
            mesh.subdivide(leaf)
            break
    print("After one targeted split: leaves:", mesh.count_leaves(), "max_depth:", mesh.max_depth())
