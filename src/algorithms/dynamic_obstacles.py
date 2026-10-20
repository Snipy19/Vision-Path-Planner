"""
Dynamic Obstacles Manager
--------------------------
Allows runtime addition/removal of obstacle cells on the grid.
When obstacles change, any running algorithm re-routes automatically.

Usage:
    dm = DynamicObstacleManager(base_grid)
    dm.add_obstacle(row, col)
    dm.remove_obstacle(row, col)
    current_grid = dm.get_grid()          # always up-to-date
    dm.reset()                            # back to original
"""


class DynamicObstacleManager:
    def __init__(self, base_grid):
        """
        base_grid: list[list[int]]  (1=walkable, 0=obstacle)
        Stored as flat set of added/removed cells — no deep copy needed.
        """
        self.rows     = len(base_grid)
        self.cols     = len(base_grid[0])
        # Copy both levels: callers can safely mutate their input grid after
        # constructing the manager, and reset() remains deterministic.
        self._base    = [row[:] for row in base_grid]
        self._added   = set()              # cells forced to obstacle
        self._removed = set()              # cells forced to walkable
        # Working copy
        self._grid    = [row[:] for row in base_grid]

    # ── Mutation ──────────────────────────────────────────────────────────

    def add_obstacle(self, r, c):
        """Block cell (r,c) — paint it as obstacle."""
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        self._added.add((r, c))
        self._removed.discard((r, c))
        self._grid[r][c] = 0
        return True

    def remove_obstacle(self, r, c):
        """Unblock cell (r,c) — restore to walkable."""
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        self._removed.add((r, c))
        self._added.discard((r, c))
        self._grid[r][c] = 1
        return True

    def toggle(self, r, c):
        """Toggle obstacle state of cell (r,c)."""
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        if self._grid[r][c] == 0:
            return self.remove_obstacle(r, c)
        else:
            return self.add_obstacle(r, c)

    def reset(self):
        """Remove all dynamic changes — restore original grid."""
        self._added.clear()
        self._removed.clear()
        self._grid = [row[:] for row in self._base]

    # ── Query ─────────────────────────────────────────────────────────────

    def get_grid(self):
        """Return current grid (list-of-lists). Fast — no copy."""
        return self._grid

    def is_obstacle(self, r, c):
        return self._grid[r][c] == 0

    def added_count(self):
        return len(self._added)

    def removed_count(self):
        return len(self._removed)

    def added_cells(self):
        return list(self._added)

    def stats(self):
        total = self.rows * self.cols
        walkable = sum(self._grid[r][c] for r in range(self.rows) for c in range(self.cols))
        return {
            "total_cells":      total,
            "walkable":         walkable,
            "obstacles":        total - walkable,
            "dynamic_added":    len(self._added),
            "dynamic_removed":  len(self._removed),
        }
