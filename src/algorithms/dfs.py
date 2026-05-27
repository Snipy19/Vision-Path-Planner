"""
DFS - Depth First Search + IDDFS (Iterative Deepening DFS)
-----------------------------------------------------------
Real-world use: Maze generation, topological sorting,
                game tree exploration (Chess/Go move search)

AI Category: Uninformed / Blind Search
DFS:   May find any path (NOT shortest). Good for deep trees.
IDDFS: Combines DFS memory-efficiency + BFS optimality.
       Used in real chess engines (Alpha-Beta pruning base).

Time: O(V + E),  Space: O(depth) — MUCH less memory than BFS
"""

class DFS:
    """
    Goes as deep as possible before backtracking.
    Like solving a maze by always turning left.
    """

    MOVES_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    MOVES_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def __init__(self, grid, allow_diagonal=False):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.moves = self.MOVES_8 if allow_diagonal else self.MOVES_4

    def search(self, start, goal):
        """Iterative (stack-based) DFS — avoids Python recursion limit."""
        stack = [start]
        visited = set()
        parent = {start: None}
        nodes_explored = 0

        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
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
                    stack.append(nxt)
                    if nxt not in parent:
                        parent[nxt] = cur

        return None, nodes_explored

    def iddfs(self, start, goal, max_depth=500):
        """
        Iterative Deepening DFS — tries depth 1, 2, 3, ...
        Guaranteed shortest path like BFS, but O(d) memory like DFS.
        Used in production game engines.
        """
        total_nodes = 0
        for depth_limit in range(1, max_depth + 1):
            result, nodes = self._dls(start, goal, depth_limit)
            total_nodes += nodes
            if result is not None:
                return result, total_nodes
        return None, total_nodes

    def _dls(self, start, goal, limit):
        """Depth-Limited Search (internal helper for IDDFS)."""
        stack = [(start, [start], 0)]
        visited_at_depth = {}
        nodes = 0

        while stack:
            node, path, depth = stack.pop()
            nodes += 1

            if node == goal:
                return path, nodes

            if depth >= limit:
                continue

            if visited_at_depth.get(node, -1) >= depth:
                continue
            visited_at_depth[node] = depth

            r, c = node
            for dr, dc in self.moves:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if (0 <= nr < self.rows and
                    0 <= nc < self.cols and
                    self.grid[nr][nc] == 1):
                    stack.append((nxt, path + [nxt], depth + 1))

        return None, nodes

    def _reconstruct(self, parent, start, goal):
        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        return path[::-1]
