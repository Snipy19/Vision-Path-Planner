# Vision Path Planner

> From any image to an optimal path — drop a floorplan, maze, or 3D render and watch 8 AI algorithms navigate through it in real time.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![PyQt5](https://img.shields.io/badge/PyQt5-GUI-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## What It Does

Upload any image — a hand-drawn maze, an architectural floorplan, a 3D rendered office, or a satellite map. The system automatically converts it into a navigable grid and solves it using 8 different AI paradigms side by side, letting you compare their paths, speed, and efficiency visually.

**No external AI or pathfinding libraries. Everything built from scratch.**

---

## Demo

| Simple Maze | 2D Floorplan | 3D Rendered Space |
|---|---|---|
| BFS finds shortest path | A* navigates rooms | HSV-based floor detection |

---

## Algorithms Implemented

### Uninformed Search
| Algorithm | Strategy | Optimal? |
|---|---|---|
| BFS | Level-by-level exploration | Yes |
| DFS | Depth-first with backtracking | No |
| IDDFS | Iterative deepening — BFS optimal + DFS memory | Yes |

### Informed Search — A* with 5 Heuristics
| Heuristic | Formula | Best For |
|---|---|---|
| Manhattan | `\|dr\| + \|dc\|` | 4-direction grids |
| Euclidean | `√(dr²+dc²)` | Free movement |
| Chebyshev | `max(\|dr\|,\|dc\|)` | 8-direction grids |
| Octile | `max + 0.414·min` | Most accurate 8-dir |
| Dijkstra (h=0) | `0` | Uniform cost search |

**Weighted A\* (W=2)** — deliberately trades optimality for speed

### Planning Under Uncertainty
**MDP + Bellman Value Iteration**
- Stochastic transitions — 80% intended, 10% slip left, 10% slip right
- Solves: `V(s) = max_a [R(s,a) + γ · Σ P(s'|s,a) · V(s')]`
- Outputs full policy for every walkable state

### Reinforcement Learning
**Q-Learning (Model-Free)**
- No environment model — learns purely by trial and error
- ε-greedy exploration: ε decays 1.0 → 0.05 as agent learns
- TD Bellman update: `Q(s,a) ← Q(s,a) + α[R + γ·max Q(s',a') - Q(s,a)]`

### Neural Network + A*
- **Architecture**: Input(9) → Hidden(16, ReLU) → Output(1, Sigmoid)
- **Features**: 3×3 pixel patch brightness values per cell
- **Training**: Binary Cross-Entropy loss, manual backpropagation, Xavier initialization
- Classifies every pixel as walkable or obstacle — replaces simple thresholding

---

## Smart Image Processing Pipeline

The system handles 4 types of input images automatically:

| Mode | How It Works |
|---|---|
| **Simple B&W** | Threshold + morphological cleanup |
| **Complex 3D Render** | HSV color space — detects colored equipment + dark walls as obstacles, light floor as walkable |
| **Satellite / Map** | Low-saturation road detection + adaptive threshold |
| **Occupancy Grid** | Smart crop, border removal, auto-invert |

Auto-detection runs by default — no manual configuration needed.

---

## Dashboard Features

- Upload any image via file dialog
- Click to place Start and Goal, or use Auto-Select
- Choose which algorithms to run individually
- View results across 7 tabs:
  - **Grid View** — original image, binary grid, overlay
  - **Algorithm Comparison** — all paths drawn on image
  - **MDP & RL Heatmaps** — V(s) plasma map, Q(s,a) viridis map
  - **Training Curves** — Q-Learning reward per episode, NN loss curve
  - **Heuristic Analysis** — nodes explored, time, path length per heuristic
  - **Performance Table** — full metrics for all algorithms
  - **Best Algorithm** — composite ranking with reasoning

---

## Project Structure

```
Vision-Path-Planner/
├── dashboard.py                   ← Main GUI — run this
├── requirements.txt
├── README.md
├── src/
│   ├── algorithms/
│   │   ├── bfs.py                 ← BFS (manual FIFO queue)
│   │   ├── dfs.py                 ← DFS + IDDFS
│   │   └── astar.py               ← A* with 5 heuristics + Weighted A*
│   ├── mdp/
│   │   ├── mdp.py                 ← MDP + Value Iteration
│   │   └── qlearning.py           ← Q-Learning agent
│   ├── neural/
│   │   └── nn_classifier.py       ← 2-layer NN, backprop from scratch
│   ├── image_processing/
│   │   └── grid.py                ← Image → Binary grid
│   └── utils/
│       ├── point_selector.py      ← Largest-component-aware point selection
│       └── visualization.py       ← Drawing and plotting helpers
├── data/
│   ├── floorplans/
│   ├── maze/
│   └── maps/
└── outputs/                       ← Generated result images saved here
```

---

## Setup & Run

```bash
pip install opencv-python numpy matplotlib PyQt5
```

```bash
python dashboard.py
```

### CLI Mode
```bash
cd src
python main.py --image data/floorplans/fp1.jpg
python main.py --image data/maze/image1.jpg --rl-episodes 300
python main.py --interactive
```

---

## Algorithm Comparison at a Glance

| Algorithm | Category | Optimal | Handles Uncertainty |
|---|---|---|---|
| BFS | Uninformed Search | Yes | No |
| DFS | Uninformed Search | No | No |
| IDDFS | Uninformed Search | Yes | No |
| A* | Informed Heuristic Search | Yes (admissible h) | No |
| Weighted A* | Informed Heuristic Search | No (bounded) | No |
| MDP | Planning Under Uncertainty | Yes | Yes |
| Q-Learning | Model-Free RL | Yes (asymptotic) | Yes |
| NN + A* | ML + Search Hybrid | Depends on NN | No |

---

## Built With

- Python 3.8+
- OpenCV — image processing and grid generation
- NumPy — grid operations and array math
- PyQt5 — interactive dashboard GUI
- Matplotlib — visualization and heatmaps

---

## Authors

**Dhruv Raj** · [GitHub](https://github.com/Snipy19)

**Yashica Gupta** · [GitHub](https://github.com/gyashica1hub)

---

## License

MIT License — free to use, modify, and distribute.
