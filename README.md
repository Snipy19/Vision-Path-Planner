# Vision Path Planner — AI Fundamentals Showcase
### From Pixels to Paths: A Complete AI Navigation System

---

## What This Project Does

This system takes a **real-world floor plan or maze image** and navigates through it using every major AI technique from the course. It converts the image into a grid, then solves the navigation problem using 7 different AI paradigms — comparing their speed, optimality, and intelligence.

---

## AI Techniques Implemented (ALL from scratch — no AI libraries)

### 1. BFS — Breadth-First Search (Uninformed)
- Explores nodes layer-by-layer (like water ripples)
- **Guarantees shortest path** in unweighted graphs
- Real use: Emergency evacuation routing, network packet routing

### 2. DFS — Depth-First Search (Uninformed)
- Dives deep before backtracking (like solving a maze by always turning left)
- Low memory (O(depth) vs O(V) for BFS)
- Real use: Game tree exploration (Chess/Go)

### 3. IDDFS — Iterative Deepening DFS (Uninformed)
- Combines BFS optimality + DFS memory efficiency
- Tries depth limits 1, 2, 3, ... until goal found
- Real use: Chess engine move search (Alpha-Beta base)

### 4. A* with 5 Heuristics (Informed Search)
Uses f(n) = g(n) + h(n) — the most important formula in AI pathfinding

| Heuristic | Formula | Best For |
|-----------|---------|----------|
| Manhattan | `|dr| + |dc|` | 4-direction grids (city blocks) |
| Euclidean | `√(dr²+dc²)` | Free-movement agents |
| Chebyshev | `max(|dr|,|dc|)` | 8-direction grids (king's move) |
| Octile | `max+0.414*min` | Most accurate 8-dir heuristic |
| Dijkstra (h=0) | `0` | Weighted graphs, no estimation |

### 5. Weighted A* (Speed-Optimality Tradeoff)
- `f = g + W*h` where W=2
- W=1 → Standard optimal A*; W→∞ → Greedy best-first
- Explores fewer nodes, slightly longer path (bounded sub-optimality)

### 6. MDP + Value Iteration (Stochastic Decision Making)
**MDP Components (SARP):**
- **S**tates: Every walkable cell
- **A**ctions: {UP, DOWN, LEFT, RIGHT}
- **R**eward: +100 at goal, -0.04 per step
- **P**robability: 80% intended move, 10% slip left, 10% slip right

**Bellman Equation:** `V(s) = max_a [R(s,a) + γ * Σ P(s'|s,a) * V(s')]`

Real use: Robot navigation under uncertainty, self-driving car decisions

### 7. Q-Learning (Model-Free Reinforcement Learning)
- No environment model needed — learns by trial and error
- **ε-greedy exploration**: explore randomly (ε) OR exploit best known action (1-ε)
- **Bellman TD Update**: `Q(s,a) ← Q(s,a) + α[R + γ·max Q(s',a') - Q(s,a)]`
- ε decays from 1.0 → 0.05 as agent learns
- Real use: DeepMind games, robotics locomotion, recommendation systems

### 8. Neural Network Obstacle Classifier
- **Architecture**: Input(9) → Hidden(16, ReLU) → Output(1, Sigmoid)
- **Features**: 3×3 pixel patch brightness values
- **Training**: Binary Cross-Entropy loss, manual backpropagation
- **Xavier initialization**, SGD optimizer — all from scratch
- Classifies each pixel as walkable/obstacle better than simple thresholding
- Real use: Satellite image road detection, autonomous vehicle perception

---

## How to Run

### Requirements
```
pip install opencv-python numpy matplotlib
```

### Basic Run (uses default floorplan)
```
python src/main.py
```

### Choose a specific map
```
python src/main.py --image data/floorplans/fp1.jpg
python src/main.py --image data/maze/image2.jpg
```

### Click to select start/goal points
```
python src/main.py --interactive
```

### More RL training (better Q-Learning policy)
```
python src/main.py --rl-episodes 500
```

---

## Output Files (saved in `outputs/`)

| File | Contents |
|------|----------|
| `01_algorithm_comparison.png` | All algorithms on the map side-by-side |
| `02_mdp_rl_heatmaps.png` | MDP value function + Q-value heatmaps |
| `03_qlearning_training.png` | RL training reward curve |
| `04_nn_training_loss.png` | Neural network loss curve |
| `05_heuristic_comparison.png` | A* heuristics nodes/time comparison |

---

## Project Structure

```
Vision_Path_Planner/
├── src/
│   ├── main.py                    ← Run this
│   ├── algorithms/
│   │   ├── bfs.py                 ← BFS (manual FIFO queue)
│   │   ├── dfs.py                 ← DFS + IDDFS
│   │   └── astar.py               ← A* + 5 heuristics + Weighted A*
│   ├── mdp/
│   │   ├── mdp.py                 ← MDP + Value Iteration
│   │   └── qlearning.py           ← Q-Learning agent
│   ├── neural/
│   │   └── nn_classifier.py       ← 2-layer NN, backprop from scratch
│   ├── image_processing/
│   │   └── grid.py                ← Image → Binary grid
│   └── utils/
│       ├── point_selector.py      ← Click-to-select or auto points
│       └── visualization.py       ← Drawing & plotting helpers
├── data/
│   ├── floorplans/               ← Building floor plan images
│   ├── maze/                     ← Maze images
│   └── maps/                     ← Street/outdoor maps
├── outputs/                      ← Generated result images
├── requirements.txt
└── README.md
```

---

## Problem Formulation (AI Terms — CLO-2)

**Problem Type**: Sequential decision-making + search

| Algorithm | AI Category | Optimal? | Handles Uncertainty? |
|-----------|-------------|----------|----------------------|
| BFS | Uninformed Search | ✓ | ✗ |
| DFS | Uninformed Search | ✗ | ✗ |
| IDDFS | Uninformed Search | ✓ | ✗ |
| A* | Informed Heuristic Search | ✓ (admissible h) | ✗ |
| Weighted A* | Informed Heuristic Search | ✗ (bounded) | ✗ |
| MDP | Planning under Uncertainty | ✓ | ✓ |
| Q-Learning | Model-Free RL | ✓ (asymptotic) | ✓ |
| NN+A* | ML + Search Hybrid | depends | ✗ |

---

*All core AI logic implemented from scratch. No external pathfinding, RL, or ML libraries used.*
