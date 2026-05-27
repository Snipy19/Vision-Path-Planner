"""
BFS - Breadth First Search
--------------------------
Real-world use: Emergency evacuation routes (find nearest exit),
                Network packet routing (shortest hops)

AI Category: Uninformed / Blind Search
Guarantee: Always finds shortest path (in unweighted graphs)
Time: O(V + E),  Space: O(V)

Implementation: Pure Python, NO external search libraries.
Uses a manual FIFO queue (list-based deque logic).
"""

class BFS:
    """
    Explores all neighbors level-by-level.
    Imagine dropping a stone in water — ripples expand uniformly.
    The first ripple to reach the goal = shortest path.
    """

    MOVES_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]           # 4-directional
    MOVES_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]  # 8-directional

    def __init__(self, grid, allow_diagonal=False):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.moves = self.MOVES_8 if allow_diagonal else self.MOVES_4

    def search(self, start, goal):
        """
        Returns: (path as list of (row,col), nodes_explored)
        path = None if unreachable
        """
        # Manual queue — head pointer avoids O(n) list.pop(0)
        queue = [start]
        head = 0
        visited = {start}
        parent = {start: None}
        nodes_explored = 0

        while head < len(queue):
            cur = queue[head]
            head += 1
            nodes_explored += 1

            if cur == goal:
                return self._reconstruct(parent, start, goal), nodes_explored

            r, c = cur
            for dr, dc in self.moves:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if (0 <= nr < self.rows and
                    0 <= nc < self.cols and
                    self.grid[nr][nc] == 1 and
                    nxt not in visited):
                    visited.add(nxt)
                    parent[nxt] = cur
                    queue.append(nxt)

        return None, nodes_explored   # No path found

    def _reconstruct(self, parent, start, goal):
        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        return path[::-1]
