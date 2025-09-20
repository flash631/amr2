# 2D Adaptive Mesh Refinement (AMR) Demo (QuadTree)

A tiny, pure-NumPy/Matplotlib demo of **2D adaptive refinement** on a scalar field.
It refines a quadtree mesh where the field varies rapidly, so you see more cells
clustered around sharp features (e.g., a narrow Gaussian peak or a step).

## Features
- Minimal **quadtree** mesh (`amr/mesh.py`)
- Field definitions with sharp features (`amr/domain.py`)
- Simple **error indicators**: corner range (max−min) and center gradient (`amr/error_indicator.py`)
- One-step or multi-iteration **refinement** driver (`amr/refine.py`)
- **Visualization** helpers for mesh, points, field contours, and error shading (`amr/visualize.py`)
- **Command-line** runner: `run_demo.py`

## Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run examples
Refine a narrow Gaussian peak and show mesh, point distribution, and contours:
```bash
python run_demo.py --field gaussian --x0 0.55 --y0 0.55 --sigma 0.03   --seed-splits 2 --iters 3 --plot mesh points contours
```

Refine using the center-gradient indicator and save a figure:
```bash
python run_demo.py --method center_gradient --plot mesh error --save demo.png
```

Try a discontinuous step in *x* or a circular disc:
```bash
python run_demo.py --field step_x --x0 0.4 --plot mesh error
python run_demo.py --field circle_step --cx 0.6 --cy 0.5 --r 0.2 --plot mesh contours
```

## Repo layout
```
amr2d/
├─ run_demo.py
├─ requirements.txt
├─ README.md
├─ amr/
│  ├─ __init__.py
│  ├─ domain.py
│  ├─ mesh.py
│  ├─ error_indicator.py
│  ├─ refine.py
│  └─ visualize.py
└─ .gitignore
```

## Notes
- If you omit `--threshold`, the top `--top-fraction` (default 0.2) of cells by error are refined each iteration.
- The demo’s 2:1 balancing pass is intentionally simple and O(N²); leave `--balance` off unless you need it.
