# Vision Path Planner

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-image%20pipeline-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/tests-5%20passing-22c55e)](tests/)
[![License](https://img.shields.io/badge/license-MIT-8b5cf6)](LICENSE)

Vision Path Planner converts a floorplan, maze, map, or rendered scene into an occupancy grid and compares classical search, stochastic planning, reinforcement learning, and a small from-scratch neural classifier on the same start/goal problem.

The project is designed to be inspectable: pathfinding is implemented locally, benchmarks are deterministic, and the headline numbers below come from the benchmark runner rather than hand-written claims.

![3D-render input](data/floorplans/fp5.jpg)

The repository includes a real rendered-room test image, a shared HSV/morphology segmentation path, and a machine-readable validation report. The PDF report is generated locally and intentionally excluded from GitHub.

## Resume-ready highlights

- Image-to-occupancy-grid pipeline with morphology cleanup, scale safety, coordinate-safe overlays, and automatic endpoint selection.
- BFS, DFS, A*, Weighted A*, Bidirectional A*, MDP value iteration, Q-Learning, and NN+A*.
- PyQt5 dashboard with non-blocking workers, heatmaps, training curves, live obstacle editing, and step-by-step animation.
- Seeded evaluation harness with success rate, path optimality, latency, expanded nodes, and p95 latency.
- Regression tests cover planner optimality, bidirectional search, reproducibility, and dynamic-obstacle isolation.

## Variable benchmark — no fixed or invented claims

```bash
python -m src.evaluation.benchmark --scenarios 8 --repeats 3 --size 48 --seed 17 \
  --density 0.16 \
  --output outputs/benchmarks/benchmark_latest.json
```

Every setting is configurable: `--scenarios`, `--repeats`, `--size`, `--seed`, `--density`, `--algorithms`, and `--no-route-carve`. The table below is one measured reference run from the command above; timings vary with hardware and should be regenerated before quoting them. The JSON report contains the raw run-level evidence.

| Algorithm | Success | Mean nodes | Reduction vs BFS | Mean optimality | Mean time |
|---|---:|---:|---:|---:|---:|
| BFS | 100% | 1,775.75 | baseline | 100% | 1.3953 ms |
| A* (Manhattan) | 100% | 1,432.25 | 19.34% | 100% | 2.7578 ms |
| Bidirectional A* | 100% | 848.38 | 52.22% | 100% | 2.3396 ms |
| Weighted A* (1.5) | 100% | 154.75 | 91.29% | 92.624% | 0.3394 ms |
| DFS | 100% | 838.88 | 52.76% | 51.516% | 0.7848 ms |

These are planning metrics from the current reference run, not a fabricated computer-vision accuracy score. The current image classifier is trained against threshold-derived labels; a real perception accuracy claim requires a human-labelled dataset, which is intentionally not claimed here.

## Real 3D-render validation

The bundled 3D-style images are tested through the shared HSV + morphology image pipeline:

```bash
python -m src.evaluation.image_validation \
  --image data/floorplans/fp5.jpg data/floorplans/fp7.jpeg \
  --mode 3d --max-dimension 160
```

Recorded validation run:

| Image | Grid | Walkable cells | BFS | A* | Weighted A* |
|---|---:|---:|---:|---:|---:|
| `fp5.jpg` | 160×158 | 70.293% | valid / 304 steps | valid / 304 | valid / 312 |
| `fp7.jpeg` | 133×160 | 91.908% | valid / 279 steps | valid / 279 | valid / 279 |

These values describe the generated occupancy grid and path validity, not ground-truth pixel segmentation accuracy. New images, modes, dimensions, and endpoints produce new results in `image_validation.json`.

## Run it

```bash
pip install -r requirements.txt
python dashboard.py
```

For reproducible/headless CLI runs:

```bash
python src/main.py --image data/maze/image1.jpg --max-dimension 160 --rl-episodes 200 --headless
python src/main.py --image data/floorplans/fp2.jpg --interactive
python src/main.py --image data/floorplans/fp5.jpg --mode 3d --headless
python src/main.py --image data/maze/image1.jpg --include-iddfs --headless
python -m src.evaluation.benchmark --scenarios 12 --repeats 5 --size 64 --seed 91 --density 0.22 --algorithms "BFS,A*,Weighted A*"
python -m src.evaluation.image_validation --image data/floorplans/fp5.jpg data/floorplans/fp7.jpeg --mode 3d
```

`--max-dimension` keeps large camera images from becoming unexpectedly expensive search graphs. The CLI downscales the rendered image and grid together so path overlays stay aligned. IDDFS is available for teaching/exhaustive comparison but is opt-in because repeated depth-limited search is intentionally expensive on winding maps.

## Architecture

Open [architecture.html](architecture.html) in a browser for the animated, perspective system map. It shows image upload and endpoint selection, grid construction, parallel planners, learned policies, dashboard workers, and persisted reports.

```text
Input image → ImageToGrid → occupancy grid
                         ├→ BFS / DFS / A* / Weighted A* / Bidirectional A*
                         ├→ MDP value iteration / Q-Learning
                         └→ 3×3 patch NN → A* validation
                                   ↓
              overlays + heatmaps + training curves + JSON benchmark
```

## Repository layout

- `src/algorithms/` — BFS, DFS/IDDFS, A*, bidirectional A*, dynamic obstacles
- `src/evaluation/` — seeded benchmark and machine-readable reports
- `src/image_processing/` — image-to-grid conversion
- `src/mdp/` — value iteration and Q-Learning
- `src/neural/` — manual forward/backprop classifier
- `src/utils/` — endpoint selection and visualization helpers
- `tests/` — fast regression tests for the planner core
- `data/` — example floorplans, mazes, and maps
- `dashboard.py` — PyQt5 application entry point
- `architecture.html` — animated browser architecture view

## Engineering notes

- No external pathfinding, reinforcement-learning, or deep-learning framework is used for the core algorithms.
- The benchmark records its seed, grid size, movement model, ground truth, raw runs, and aggregates in JSON.
- Dynamic obstacles are isolated from the caller's base grid and can be reset deterministically.
- Diagonal motion prevents corner cutting; all paths are returned in grid coordinates.

## Verification

```bash
python -m unittest discover -s tests -v
python -m compileall -q src dashboard.py
```

## Resume bullets

> Built an image-to-occupancy-grid navigation system in Python/OpenCV with BFS, A*, Weighted A*, Bidirectional A*, MDP value iteration, Q-Learning, and a manual backpropagation classifier; added a configurable seeded benchmark that reports route success, optimality, latency, and expanded nodes from raw runs.

> In a measured reference run, reduced expanded nodes by 91.29% with Weighted A* versus BFS at 92.624% mean shortest-path optimality; exact results vary with map size, obstacle density, seed, and hardware.

> Delivered a PyQt5 dashboard with live obstacle edits, stepwise search animation, RL/MDP heatmaps, JSON benchmark export, and regression tests for planner optimality and reproducibility.

## License

MIT.

## Authors

Dhruv Raj · Yashica Gupta
