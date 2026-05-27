"""
A* (A-Star) Search — Informed Heuristic Search
-----------------------------------------------
Real-world use: Google Maps routing, Robot path planning (ROS),
                Video game NPC navigation (Warcraft, Starcraft AI),
                Logistics & delivery optimization

AI Category: Informed Search (Best-First)
Guarantee: Optimal if heuristic is ADMISSIBLE (never over-estimates)

f(n) = g(n) + h(n)
  g(n) = actual cost from start to n   (known)
  h(n) = estimated cost from n to goal (heuristic)

Weighted A* (WA*): f = g + W*h
  W=1   → Standard A* (optimal)
  W>1   → Faster but sub-optimal (trades quality for speed)
  W=∞   → Pure Greedy Best-First Search

No external pathfinding library used. Manual binary heap via heapq.
"""

import heapq
import math


# ─────────────────────── Heuristic Functions ──────────────────────────────

def manhattan(a, b):
    """
    Manhattan Distance — L1 norm
    Only valid for 4-directional movement (grid streets).
    Named after NYC's grid layout.
    h = |r1-r2| + |c1-c2|
    Always admissible for 4-dir grids.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    """
    Euclidean Distance — L2 norm (straight line)
    Valid for any-direction movement.
    h = sqrt((r1-r2)^2 + (c1-c2)^2)
    Admissible but looser than Chebyshev for 8-dir grids.
    """
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def chebyshev(a, b):
    """
    Chebyshev Distance — L∞ norm (King moves on chess board)
    Best heuristic for 8-directional movement grids.
    h = max(|r1-r2|, |c1-c2|)
    Admissible for 8-dir movement.
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def octile(a, b):
    """
    Octile Distance — Most accurate for 8-dir grids
    Accounts for diagonal cost = sqrt(2) ≈ 1.414
    h = max(dr,dc) + (sqrt(2)-1) * min(dr,dc)
    Best admissible heuristic for 8-directional movement.
    Used in professional game engine pathfinders.
    """
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)


def dijkstra_zero(a, b):
    """
    Zero Heuristic — h=0
    Reduces A* to Dijkstra's algorithm.
    Explores more nodes but guaranteed optimal on weighted graphs.
    """
    return 0


HEURISTICS = {
    "manhattan":  manhattan,
    "euclidean":  euclidean,
    "chebyshev":  chebyshev,
    "octile":     octile,
    "dijkstra":   dijkstra_zero,
}

HEURISTIC_INFO = {
    "manhattan": "L1 norm. Best for 4-dir grids. Like counting city blocks.",
    "euclidean": "L2 norm. Straight-line distance. Good for free-move agents.",
    "chebyshev": "L∞ norm. Best for 8-dir grids. King's move distance.",
    "octile":    "Most accurate 8-dir heuristic. Accounts for diagonal cost √2.",
    "dijkstra":  "h=0. Explores everything. Optimal but slow (Dijkstra mode).",
}


# ─────────────────────── A* Search Class ──────────────────────────────────

class AStar:
    """
    A* with pluggable heuristics and Weighted A* support.
    Can do 4 or 8-directional movement.
    """

    MOVES_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    MOVES_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def __init__(self, grid, heuristic="manhattan", weight=1.0, allow_diagonal=False):
        """
        heuristic: one of 'manhattan','euclidean','chebyshev','octile','dijkstra'
        weight:    W in f = g + W*h. Use 1.0 for standard optimal A*.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.heuristic_name = heuristic
        self.heuristic_fn = HEURISTICS.get(heuristic, manhattan)
        self.weight = weight
        self.moves = self.MOVES_8 if allow_diagonal else self.MOVES_4
        # Diagonal cost = sqrt(2), straight = 1
        self.allow_diagonal = allow_diagonal

    def _move_cost(self, dr, dc):
        """Diagonal moves cost sqrt(2), cardinal moves cost 1."""
        return math.sqrt(2) if (dr != 0 and dc != 0) else 1.0

    def search(self, start, goal):
        """
        Returns: (path, nodes_explored)
        Implements f(n) = g(n) + W * h(n)
        """
        # Priority queue: (f_value, tie_breaker, node)
        counter = 0
        open_heap = []
        h0 = self.heuristic_fn(start, goal)
        heapq.heappush(open_heap, (h0, counter, start))

        g_cost = {start: 0.0}
        parent = {start: None}
        closed = set()
        nodes_explored = 0

        while open_heap:
            f, _, cur = heapq.heappop(open_heap)

            if cur in closed:
                continue
            closed.add(cur)
            nodes_explored += 1

            if cur == goal:
                return self._reconstruct(parent, goal), nodes_explored

            r, c = cur
            for dr, dc in self.moves:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)

                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.grid[nr][nc] == 0:
                    continue
                if nxt in closed:
                    continue

                # Diagonal move: check both adjacent cells to prevent corner cutting
                if dr != 0 and dc != 0:
                    if self.grid[r + dr][c] == 0 or self.grid[r][c + dc] == 0:
                        continue

                new_g = g_cost[cur] + self._move_cost(dr, dc)

                if new_g < g_cost.get(nxt, float('inf')):
                    g_cost[nxt] = new_g
                    h = self.heuristic_fn(nxt, goal)
                    f_new = new_g + self.weight * h
                    counter += 1
                    heapq.heappush(open_heap, (f_new, counter, nxt))
                    parent[nxt] = cur

        return None, nodes_explored

    def _reconstruct(self, parent, goal):
        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        return path[::-1]
