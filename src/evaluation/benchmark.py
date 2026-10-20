"""Deterministic benchmark for the path-planning core.

The benchmark intentionally measures planning behavior on generated occupancy
grids, where the ground-truth shortest path is known.  It does not call any
external pathfinding library and never fabricates accuracy values: every
reported statistic is derived from the paths returned by this repository.

Run from the repository root:

    python -m src.evaluation.benchmark --scenarios 8 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from src.algorithms.astar import AStar
from src.algorithms.bfs import BFS
from src.algorithms.bidirectional import BidirectionalAStar
from src.algorithms.dfs import DFS

Point = Tuple[int, int]
Grid = List[List[int]]


def _path_steps(path: Optional[List[Point]]) -> Optional[int]:
    return None if not path else len(path) - 1


def _reachable(grid: Grid, start: Point, goal: Point) -> bool:
    """Small connectivity check used only while constructing scenarios."""
    queue = [start]
    seen = {start}
    for current in queue:
        if current == goal:
            return True
        r, c = current
        for dr, dc in BFS.MOVES_4:
            nxt = (r + dr, c + dc)
            if (0 <= nxt[0] < len(grid) and 0 <= nxt[1] < len(grid[0])
                    and grid[nxt[0]][nxt[1]] and nxt not in seen):
                seen.add(nxt)
                queue.append(nxt)
    return False


def make_scenario(seed: int, size: int = 48, density: float = 0.16,
                  carve_route: bool = True) -> Tuple[Grid, Point, Point]:
    """Create a connected, deterministic occupancy grid.

    Randomized obstacles provide the variation.  If a requested density makes
    the map disconnected, an optional narrow route is carved as a fallback so
    the benchmark can still compare successful planners.
    """
    if size < 12:
        raise ValueError("size must be at least 12")
    if not 0.0 <= density < 1.0:
        raise ValueError("density must be in the range [0, 1)")
    start, goal = (1, 1), (size - 2, size - 2)
    for attempt in range(100):
        rng = random.Random(seed + attempt)
        grid = [[1 for _ in range(size)] for _ in range(size)]
        for r in range(size):
            grid[r][0] = grid[r][-1] = 0
        for c in range(size):
            grid[0][c] = grid[-1][c] = 0
        for r in range(1, size - 1):
            for c in range(1, size - 1):
                if rng.random() < density:
                    grid[r][c] = 0
        grid[start[0]][start[1]] = 1
        grid[goal[0]][goal[1]] = 1
        if _reachable(grid, start, goal):
            return grid, start, goal
        if carve_route:
            # Connectivity fallback: a reproducible diagonal backbone with
            # two-cell width, still leaving randomized structure around it.
            for i in range(1, size - 1):
                grid[i][i] = 1
                if i + 1 < size - 1:
                    grid[i][i + 1] = 1
            if _reachable(grid, start, goal):
                return grid, start, goal
    raise RuntimeError(f"could not construct a connected scenario for seed {seed}")


def _algorithm_suite(grid: Grid, selected: Optional[Iterable[str]] = None) -> Dict[str, Callable[[Point, Point], Tuple[Optional[List[Point]], int]]]:
    suite = {
        "BFS": lambda s, g: BFS(grid).search(s, g),
        "DFS": lambda s, g: DFS(grid).search(s, g),
        "A*": lambda s, g: AStar(grid, heuristic="manhattan").search(s, g),
        "Weighted A*": lambda s, g: AStar(
            grid, heuristic="manhattan", weight=1.5).search(s, g),
        "Bidirectional A*": lambda s, g: BidirectionalAStar(
            grid, allow_diagonal=False).search(s, g),
    }
    if selected is None:
        return suite
    unknown = sorted(set(selected) - set(suite))
    if unknown:
        raise ValueError(f"unknown algorithm(s): {', '.join(unknown)}")
    return {name: suite[name] for name in selected}


def run_benchmark(scenarios: int = 8, repeats: int = 3,
                  size: int = 48, seed: int = 17, density: float = 0.16,
                  algorithms: Optional[Iterable[str]] = None,
                  carve_route: bool = True) -> dict:
    """Run the benchmark and return a JSON-serializable report."""
    if scenarios <= 0 or repeats <= 0:
        raise ValueError("scenarios and repeats must be positive")

    rows = []
    for scenario_index in range(scenarios):
        grid, start, goal = make_scenario(seed + scenario_index, size,
                                           density, carve_route)
        optimal_path, optimal_nodes = BFS(grid).search(start, goal)
        if not optimal_path:
            raise RuntimeError("benchmark scenario unexpectedly has no path")
        optimal_steps = len(optimal_path) - 1

        for name, solve in _algorithm_suite(grid, algorithms).items():
            for repeat in range(repeats):
                began = time.perf_counter()
                path, nodes = solve(start, goal)
                elapsed_ms = (time.perf_counter() - began) * 1000.0
                steps = _path_steps(path)
                rows.append({
                    "scenario": scenario_index + 1,
                    "repeat": repeat + 1,
                    "algorithm": name,
                    "success": bool(path and path[-1] == goal),
                    "path_steps": steps,
                    "optimal_steps": optimal_steps,
                    "optimality_pct": round(
                        100.0 * optimal_steps / steps, 3)
                        if steps else None,
                    "nodes_expanded": nodes,
                    "time_ms": round(elapsed_ms, 4),
                })

    summary = {}
    for name in sorted({row["algorithm"] for row in rows}):
        values = [row for row in rows if row["algorithm"] == name]
        times = [row["time_ms"] for row in values]
        nodes = [row["nodes_expanded"] for row in values]
        paths = [row["path_steps"] for row in values if row["path_steps"] is not None]
        optimality = [row["optimality_pct"] for row in values
                      if row["optimality_pct"] is not None]
        summary[name] = {
            "runs": len(values),
            "success_rate_pct": round(100.0 * sum(r["success"] for r in values) / len(values), 2),
            "mean_time_ms": round(statistics.mean(times), 4),
            "p95_time_ms": round(sorted(times)[max(0, math.ceil(0.95 * len(times)) - 1)], 4),
            "mean_nodes_expanded": round(statistics.mean(nodes), 2),
            "mean_path_steps": round(statistics.mean(paths), 2) if paths else None,
            "mean_optimality_pct": round(statistics.mean(optimality), 3) if optimality else None,
        }

    if "BFS" not in summary:
        raise ValueError("BFS must be included as the ground-truth algorithm")
    bfs_nodes = summary["BFS"]["mean_nodes_expanded"]
    for name, stats in summary.items():
        stats["node_reduction_vs_bfs_pct"] = round(
            100.0 * (bfs_nodes - stats["mean_nodes_expanded"]) / bfs_nodes, 2
        ) if bfs_nodes else 0.0

    return {
        "benchmark": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "scenarios": scenarios,
            "repeats": repeats,
            "grid_size": f"{size}x{size}",
            "obstacle_density": density,
            "algorithms": list(summary),
            "connectivity_fallback": carve_route,
            "movement": "4-directional, unit edge cost",
            "ground_truth": "BFS shortest path per scenario",
        },
        "summary": summary,
        "runs": rows,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark path-planning algorithms")
    parser.add_argument("--scenarios", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--density", type=float, default=0.16,
                        help="Obstacle probability in each generated map")
    parser.add_argument("--algorithms", type=str, default=None,
                        help="Comma-separated subset; BFS is always required")
    parser.add_argument("--no-route-carve", action="store_true",
                        help="Fail disconnected maps instead of carving a fallback route")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/benchmarks/latest.json"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    selected = [name.strip() for name in args.algorithms.split(",") if name.strip()] \
        if args.algorithms else None
    if selected and "BFS" not in selected:
        selected.insert(0, "BFS")
    report = run_benchmark(args.scenarios, args.repeats, args.size, args.seed,
                           args.density, selected, not args.no_route_carve)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved benchmark report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
