"""
Bidirectional A* Search
-----------------------
Searches simultaneously from START → GOAL and GOAL → START.
Meets in the middle — drastically reduces nodes explored on large grids.

Why faster?
  Normal A*:     explores radius R from start    → ~π·R² nodes
  Bidirectional: two searches of radius R/2 each → ~2·π·(R/2)² = πR²/2 nodes
  → roughly 2x fewer nodes on open grids, much more on complex floorplans.

Real-world use: Google Maps (meet-in-the-middle routing),
                highway-node hierarchies, large warehouse navigation.
"""

import heapq
import math


def octile(a, b):
    dr, dc = abs(a[0]-b[0]), abs(a[1]-b[1])
    return max(dr, dc) + (math.sqrt(2)-1)*min(dr, dc)


class BidirectionalAStar:
    MOVES_4 = [(-1,0),(1,0),(0,-1),(0,1)]
    MOVES_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def __init__(self, grid, heuristic_fn=None, allow_diagonal=True):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.h    = heuristic_fn or octile
        self.allow_diagonal = allow_diagonal
        self.moves = self.MOVES_8 if allow_diagonal else self.MOVES_4

    def _cost(self, dr, dc):
        return math.sqrt(2) if dr != 0 and dc != 0 else 1.0

    def _neighbours(self, r, c):
        for dr, dc in self.moves:
            nr, nc = r+dr, c+dc
            if not (0 <= nr < self.rows and 0 <= nc < self.cols): continue
            if self.grid[nr][nc] == 0: continue
            # Prevent corner-cutting
            if self.allow_diagonal and dr != 0 and dc != 0:
                if self.grid[r+dr][c] == 0 or self.grid[r][c+dc] == 0: continue
            yield (nr, nc), self._cost(dr, dc)

    def _reconstruct(self, par_fwd, par_bwd, meeting):
        """Stitch forward and backward paths at meeting point."""
        # Forward path: start → meeting
        path_fwd = []
        node = meeting
        while node is not None:
            path_fwd.append(node)
            node = par_fwd[node]
        path_fwd.reverse()

        # Backward path: meeting → goal
        path_bwd = []
        node = par_bwd[meeting]
        while node is not None:
            path_bwd.append(node)
            node = par_bwd[node]

        return path_fwd + path_bwd

    def search(self, start, goal):
        if start == goal:
            return [start], 0

        # Forward search (start → goal)
        g_fwd  = {start: 0.0}
        par_fwd = {start: None}
        open_fwd = [(self.h(start, goal), 0, start)]
        closed_fwd = set()

        # Backward search (goal → start)
        g_bwd  = {goal: 0.0}
        par_bwd = {goal: None}
        open_bwd = [(self.h(goal, start), 0, goal)]
        closed_bwd = set()

        best_cost = float('inf')
        meeting   = None
        counter   = 0
        nodes_explored = 0

        while open_fwd and open_bwd:
            # Always expand the frontier with lower f-value
            f_fwd = open_fwd[0][0] if open_fwd else float('inf')
            f_bwd = open_bwd[0][0] if open_bwd else float('inf')

            # Termination: both frontiers exceed best found path
            if f_fwd + f_bwd >= best_cost:
                break

            if f_fwd <= f_bwd:
                # Expand forward
                _, _, cur = heapq.heappop(open_fwd)
                if cur in closed_fwd: continue
                closed_fwd.add(cur); nodes_explored += 1

                for nxt, cost in self._neighbours(*cur):
                    ng = g_fwd[cur] + cost
                    if ng < g_fwd.get(nxt, float('inf')):
                        g_fwd[nxt] = ng; par_fwd[nxt] = cur; counter += 1
                        heapq.heappush(open_fwd, (ng + self.h(nxt, goal), counter, nxt))

                    # Check if nxt was reached by backward search
                    if nxt in closed_bwd:
                        total = g_fwd[cur] + cost + g_bwd[nxt]
                        if total < best_cost:
                            best_cost = total; meeting = nxt
                            par_fwd[nxt] = cur
            else:
                # Expand backward
                _, _, cur = heapq.heappop(open_bwd)
                if cur in closed_bwd: continue
                closed_bwd.add(cur); nodes_explored += 1

                for nxt, cost in self._neighbours(*cur):
                    ng = g_bwd[cur] + cost
                    if ng < g_bwd.get(nxt, float('inf')):
                        g_bwd[nxt] = ng; par_bwd[nxt] = cur; counter += 1
                        heapq.heappush(open_bwd, (ng + self.h(nxt, start), counter, nxt))

                    # Check if nxt was reached by forward search
                    if nxt in closed_fwd:
                        total = g_bwd[cur] + cost + g_fwd[nxt]
                        if total < best_cost:
                            best_cost = total; meeting = nxt
                            par_bwd[nxt] = cur

        if meeting is None:
            return None, nodes_explored

        path = self._reconstruct(par_fwd, par_bwd, meeting)
        return path, nodes_explored
