"""Validate image-to-grid planning on real bundled images.

This is deliberately separate from the synthetic benchmark: it reports what
the current image segmentation produced, rather than pretending that an image
without human labels has a ground-truth perception accuracy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from src.algorithms.astar import AStar
from src.algorithms.bfs import BFS
from src.algorithms.bidirectional import BidirectionalAStar
from src.image_processing.grid import ImageToGrid

Point = Tuple[int, int]


def _largest_component(grid: List[List[int]]) -> List[Point]:
    rows, cols = len(grid), len(grid[0])
    remaining = {(r, c) for r in range(rows) for c in range(cols)
                 if grid[r][c] == 1}
    largest: List[Point] = []
    while remaining:
        seed = next(iter(remaining))
        component = []
        queue = [seed]
        remaining.remove(seed)
        for current in queue:
            component.append(current)
            r, c = current
            for dr, dc in BFS.MOVES_4:
                nxt = (r + dr, c + dc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    queue.append(nxt)
        if len(component) > len(largest):
            largest = component
    return largest


def _farthest(grid: List[List[int]], source: Point) -> Point:
    queue = [source]
    distance = {source: 0}
    for current in queue:
        r, c = current
        for dr, dc in BFS.MOVES_4:
            nxt = (r + dr, c + dc)
            if (0 <= nxt[0] < len(grid) and 0 <= nxt[1] < len(grid[0])
                    and grid[nxt[0]][nxt[1]] and nxt not in distance):
                distance[nxt] = distance[current] + 1
                queue.append(nxt)
    return max(distance, key=distance.get)


def select_component_endpoints(grid: List[List[int]]) -> Tuple[Point, Point]:
    component = _largest_component(grid)
    if len(component) < 2:
        raise ValueError("segmentation produced fewer than two connected cells")
    start = component[0]
    start = _farthest(grid, start)
    goal = _farthest(grid, start)
    return start, goal


def valid_path(grid, path: Optional[List[Point]], start: Point, goal: Point) -> bool:
    if not path or path[0] != start or path[-1] != goal:
        return False
    return all(
        grid[r][c] == 1 and abs(r - pr) + abs(c - pc) == 1
        for (pr, pc), (r, c) in zip(path, path[1:])
    )


def validate_image(image_path: Path, mode: str = "3d",
                   max_dimension: int = 160) -> dict:
    grid, binary, image, gray = ImageToGrid(
        str(image_path), max_dimension=max_dimension, mode=mode).process()
    start, goal = select_component_endpoints(grid)
    algorithms = {
        "BFS": lambda: BFS(grid).search(start, goal),
        "A*": lambda: AStar(grid, heuristic="manhattan").search(start, goal),
        "Weighted A*": lambda: AStar(
            grid, heuristic="manhattan", weight=1.5).search(start, goal),
        "Bidirectional A*": lambda: BidirectionalAStar(
            grid, allow_diagonal=False).search(start, goal),
    }
    results = {}
    for name, solve in algorithms.items():
        began = time.perf_counter()
        path, nodes = solve()
        elapsed_ms = (time.perf_counter() - began) * 1000
        results[name] = {
            "success": valid_path(grid, path, start, goal),
            "path_steps": len(path) - 1 if path else None,
            "nodes_expanded": nodes,
            "time_ms": round(elapsed_ms, 4),
        }
    return {
        "image": str(image_path),
        "mode": mode,
        "grid_shape": [len(grid), len(grid[0])],
        "walkable_pct": round(100 * float(np.mean(np.asarray(grid))), 3),
        "start": list(start),
        "goal": list(goal),
        "algorithms": results,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate bundled images")
    parser.add_argument("--image", nargs="+", type=Path, required=True)
    parser.add_argument("--mode", default="3d",
                        choices=["auto", "simple", "3d", "map", "occupancy"])
    parser.add_argument("--max-dimension", type=int, default=160)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/benchmarks/image_validation.json"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = {
        "validation": {
            "mode": args.mode,
            "max_dimension": args.max_dimension,
            "ground_truth": "none; this report validates paths on the generated grid",
        },
        "images": [validate_image(path, args.mode, args.max_dimension)
                   for path in args.image],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved image validation report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
