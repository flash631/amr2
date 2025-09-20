"""domain.py — scalar fields for 2D AMR demos.

This module defines a small collection of 2D scalar fields f(x, y) that
exhibit sharp features (discontinuities, steep gradients) to drive simple
adaptive mesh refinement (AMR) examples. Functions are vectorized with NumPy
so you can pass floats, lists, or numpy arrays.

Main entry points
-----------------
- ScalarField2D: lightweight wrapper class for a callable f(x, y).
- make_field(name, **kwargs): factory returning a ScalarField2D instance.
- Built-in fields:
    * gaussian_peak(x, y, x0=0.5, y0=0.5, sigma=0.05, A=1.0)
    * step_x(x, y, x0=0.5, low=0.0, high=1.0)
    * circle_step(x, y, cx=0.5, cy=0.5, r=0.2, inside=1.0, outside=0.0)

All functions support broadcasting:
    f(np.array([0.1, 0.9]), 0.5)  # returns a vector

Default domain bounds are the unit square (0, 1) × (0, 1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional

import numpy as np
import inspect

# Public default bounds for demos: (x_min, x_max, y_min, y_max)
DEFAULT_BOUNDS: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)


def _asfloat(a):
    """Convert input to a NumPy float array without copying if already array-like."""
    return np.asarray(a, dtype=float)


@dataclass(slots=True)
class ScalarField2D:
    """Lightweight wrapper for a 2D scalar field f(x, y).

    Parameters
    ----------
    func : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Vectorized function that maps (x, y) → f.
    name : str, optional
        Human-readable name for the field (e.g., 'gaussian').
    params : dict, optional
        Parameters used to build the field; stored for reference/metadata.

    Notes
    -----
    - The callable must be **vectorized**: given arrays X and Y of compatible
      shapes, it should return an array of the broadcasted shape.
    - Instances are callable: `f(x, y)` forwards to the underlying function.
    """

    func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    name: str = "custom"
    params: Optional[Dict[str, Any]] = None

    def __call__(self, x, y):  # type: ignore[override]
        X = _asfloat(x)
        Y = _asfloat(y)
        return self.func(X, Y)

    def __repr__(self) -> str:  # pragma: no cover - simple metadata repr
        p = {} if self.params is None else dict(self.params)
        return f"ScalarField2D(name={self.name!r}, params={p})"


# --------------------------
# Built-in scalar fields
# --------------------------

def gaussian_peak(
    x: np.ndarray | float,
    y: np.ndarray | float,
    *,
    x0: float = 0.5,
    y0: float = 0.5,
    sigma: float = 0.05,
    A: float = 1.0,
    **_: dict,
) -> np.ndarray:
    """Narrow Gaussian bump.

    f(x, y) = A * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2))

    A small sigma (e.g., 0.03–0.06) creates a steep feature that triggers
    refinement around (x0, y0).
    """
    X = _asfloat(x)
    Y = _asfloat(y)
    # Guard against sigma→0
    s2 = max(float(sigma) ** 2, np.finfo(float).tiny)
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    return A * np.exp(-0.5 * r2 / s2)


def step_x(
    x: np.ndarray | float,
    y: np.ndarray | float,  # noqa: ARG001 (kept for uniform signature)
    *,
    x0: float = 0.5,
    low: float = 0.0,
    high: float = 1.0,
    **_: dict,
) -> np.ndarray:
    """Discontinuous step across the vertical line x = x0.

    Returns `low` for x < x0 and `high` for x ≥ x0.
    """
    X = _asfloat(x)
    return np.where(X < x0, low, high)


def circle_step(
    x: np.ndarray | float,
    y: np.ndarray | float,
    *,
    cx: float = 0.5,
    cy: float = 0.5,
    r: float = 0.2,
    inside: float = 1.0,
    outside: float = 0.0,
    **_: dict,
) -> np.ndarray:
    """Binary disc: step discontinuity across a circle of radius r.

    Returns `inside` for points with (x-cx)^2 + (y-cy)^2 < r^2 else `outside`.
    """
    X = _asfloat(x)
    Y = _asfloat(y)
    return np.where((X - cx) ** 2 + (Y - cy) ** 2 < r ** 2, inside, outside)


# --------------------------
# Factory
# --------------------------

def _filter_kwargs(func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return only kwargs accepted by `func` (drop extras)."""
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def make_field(name: str, **kwargs) -> ScalarField2D:
    """Create a ScalarField2D by name.

    Parameters
    ----------
    name : str
        One of: 'gaussian', 'step_x', 'circle_step'. Case-insensitive and
        accepts common aliases (e.g., 'gauss', 'disc').
    **kwargs
        Passed through to the underlying field function; **extra keys are
        safely ignored** if the chosen field doesn't accept them.

    Returns
    -------
    ScalarField2D
        Callable wrapper for the selected field.
    """
    key = name.strip().lower()

    if key in {"gaussian", "gauss", "peak", "gaussian_peak"}:
        params = _filter_kwargs(gaussian_peak, kwargs)
        func = lambda X, Y, _p=params: gaussian_peak(X, Y, **_p)  # noqa: E731
        return ScalarField2D(func, name="gaussian", params=params)

    if key in {"step_x", "step", "xstep"}:
        params = _filter_kwargs(step_x, kwargs)
        func = lambda X, Y, _p=params: step_x(X, Y, **_p)  # noqa: E731
        return ScalarField2D(func, name="step_x", params=params)

    if key in {"circle_step", "disc", "disk", "circle"}:
        params = _filter_kwargs(circle_step, kwargs)
        func = lambda X, Y, _p=params: circle_step(X, Y, **_p)  # noqa: E731
        return ScalarField2D(func, name="circle_step", params=params)

    raise ValueError(
        f"Unknown field '{name}'. Available: 'gaussian', 'step_x', 'circle_step'."
    )


__all__ = [
    "ScalarField2D",
    "DEFAULT_BOUNDS",
    "gaussian_peak",
    "step_x",
    "circle_step",
    "make_field",
]


if __name__ == "__main__":  # quick sanity check / demo
    # Vectorization smoke test
    f = make_field("gaussian", x0=0.55, y0=0.55, sigma=0.04, A=1.0)
    xs = np.linspace(0, 1, 5)
    ys = np.linspace(0, 1, 5)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = f(X, Y)
    print("demo: gaussian field shape:", Z.shape, "min/max:", float(Z.min()), float(Z.max()))
