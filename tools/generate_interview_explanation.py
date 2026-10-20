"""Generate a plain, interview-focused explanation PDF from measured JSON artifacts."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PAGE_SIZE = (8.27, 11.69)  # A4 portrait
FONT = "DejaVu Sans"


def wrap(text: str, width: int = 94) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=width,
                                       break_long_words=False,
                                       break_on_hyphens=False))
    return lines


def add_page(pdf: PdfPages, title: str, sections: list[tuple[str, str]]) -> None:
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    fig.text(0.08, 0.95, title, fontsize=17, weight="bold",
             color="black", fontfamily=FONT)
    y = 0.90
    for heading, body in sections:
        if heading:
            if y < 0.10:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
                y = 0.95
            fig.text(0.08, y, heading, fontsize=11, weight="bold",
                     color="black", fontfamily=FONT)
            y -= 0.034
        for line in wrap(body):
            if y < 0.075:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
                y = 0.95
            fig.text(0.08, y, line, fontsize=9.2, color="black",
                     fontfamily=FONT, va="top")
            y -= 0.021
        y -= 0.018
    pdf.savefig(fig)
    plt.close(fig)


def pct(value: float) -> str:
    return f"{value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path,
                        default=Path("outputs/benchmarks/benchmark_latest.json"))
    parser.add_argument("--image-report", type=Path,
                        default=Path("outputs/benchmarks/image_validation.json"))
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    summary = benchmark["summary"]
    meta = benchmark["benchmark"]
    image_report = None
    if args.image_report.exists():
        image_report = json.loads(args.image_report.read_text(encoding="utf-8"))

    best_name = max(summary, key=lambda name: summary[name]["node_reduction_vs_bfs_pct"])
    best = summary[best_name]
    bfs = summary.get("BFS", {})
    total_runs = sum(row["runs"] for row in summary.values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        add_page(pdf, "Vision Path Planner - Interview Explanation", [
            ("Authors", "Dhruv Raj - https://github.com/Snipy19 | Yashica Gupta - https://github.com/gyashica1hub"),
            ("Repository", "https://github.com/Snipy19/Vision-Path-Planner"),
            ("One-sentence explanation", "This project converts an input image into a walkable occupancy grid and compares multiple search, planning, reinforcement-learning, and learned-perception approaches for finding a route from a start cell to a goal cell."),
            ("What I should say first", "The main engineering contribution is the complete, measurable pipeline: image preprocessing, grid construction, route planning, visualization, reproducible benchmarking, and validation on real bundled images."),
            ("Important honesty statement", "The reported numbers are generated from the repository's benchmark JSON and real image-validation JSON. They are not universal guarantees. Runtime and search effort change with the map, grid size, obstacle density, seed, movement model, and hardware."),
        ])

        add_page(pdf, "1. Problem, input, and output", [
            ("Problem", "Given a floorplan, maze, map, or rendered scene, determine which pixels/cells are traversable and find a route between two endpoints."),
            ("Input", "A JPG or PNG image. The image pipeline resizes it safely, segments it into walkable and blocked regions, and applies morphological cleanup so small image artifacts do not dominate the grid."),
            ("Internal representation", "A NumPy occupancy grid. A value of 1 means walkable and 0 means blocked. Coordinates are kept consistent between the resized image, the grid, the selected endpoints, and the rendered path overlay."),
            ("Output", "A path as grid coordinates, expanded-node count, elapsed time, path length, success status, and optional visual overlays. Benchmark runs are also saved as JSON so the result can be inspected rather than guessed."),
            ("Why this representation", "A grid makes the planning problem explicit and gives every algorithm the same environment. This is useful for fair comparisons because the planner is not silently changing the map for different algorithms."),
        ])

        add_page(pdf, "2. Architecture and data flow", [
            ("Step 1 - image processing", "ImageToGrid receives the image, limits its maximum dimension, converts it into a segmentation-friendly color representation, identifies floor-like and obstacle-like regions, and performs morphology."),
            ("Step 2 - endpoint handling", "The system selects or receives start and goal coordinates and checks that they are within bounds and on usable cells. The benchmark uses deterministic scenario generation and route carving to avoid accidental impossible cases."),
            ("Step 3 - planning", "The same grid is passed to BFS, DFS, A*, Weighted A*, Bidirectional A*, MDP value iteration, Q-Learning, or the hybrid NN plus A* flow, depending on the selected experiment."),
            ("Step 4 - evaluation", "Every run records success, route length, expanded nodes, elapsed time, and optimality where a BFS shortest path is available as ground truth."),
            ("Step 5 - interface and artifacts", "The PyQt5 dashboard runs longer computations in worker threads, supports visualization and dynamic obstacle edits, and the evaluation tools export JSON evidence and a plain interview report."),
            ("Design principle", "Perception and planning are separated. This makes it possible to improve image segmentation without rewriting the search algorithms and makes planner comparisons reproducible."),
        ])

        add_page(pdf, "3. Algorithms and when to use them", [
            ("BFS", "Uninformed breadth-first search. With uniform four-direction movement it gives a shortest path and is used as the ground-truth reference. It can expand many nodes."),
            ("DFS and IDDFS", "Depth-first methods use less frontier memory but do not generally return the shortest route. IDDFS increases the depth limit gradually and is useful when a depth bound is meaningful."),
            ("A*", "A* combines path cost so far with a heuristic estimate to the goal. With an admissible heuristic it preserves shortest-path optimality while usually expanding fewer nodes than BFS."),
            ("Weighted A*", "Weighted A* increases the heuristic influence. It usually searches faster and expands fewer nodes, but it may return a longer path. The benchmark reports this trade-off instead of calling it universally better."),
            ("Bidirectional A*", "Search grows from both endpoints and attempts to meet in the middle. On suitable maps it reduces explored work while preserving shortest paths under the implemented cost and heuristic assumptions."),
            ("MDP and Q-Learning", "MDP value iteration assumes a model of transitions and rewards. Q-Learning learns action values from experience and is useful when the transition model is not supplied, although training quality depends on the environment and hyperparameters."),
            ("NN plus A*", "The manual classifier creates an alternative grid representation that is then given to A*. It demonstrates a hybrid perception-and-planning design; it is not presented as a production vision model or as a certified accuracy result."),
        ])

        add_page(pdf, "4. Measured benchmark results", [
            ("Benchmark configuration", f"{meta['scenarios']} scenarios x {meta['repeats']} repeats, grid {meta['grid_size']}, random seed {meta['seed']}, obstacle density {meta['obstacle_density']:.3f}, movement model {meta['movement']}. Total recorded planner runs: {total_runs}."),
            ("Reference comparison", f"BFS achieved {bfs.get('success_rate_pct', 0):.0f}% success, mean {bfs.get('mean_nodes_expanded', 0):,.2f} expanded nodes, mean path length {bfs.get('mean_path_steps', 0):,.2f}, and {bfs.get('mean_time_ms', 0):.4f} ms in this run."),
            ("Best search-effort reduction", f"{best_name} reduced expanded nodes by {best['node_reduction_vs_bfs_pct']:.2f}% versus BFS in this measured suite. Its mean success was {best['success_rate_pct']:.0f}%, mean optimality was {best['mean_optimality_pct']:.3f}%, and mean time was {best['mean_time_ms']:.4f} ms."),
            ("How to explain the trade-off", "Weighted A* is the efficiency example: it searches much less than BFS but can give up some shortest-path optimality. Bidirectional A* is the shortest-path example: it reduces work while retaining the shortest-path result on the benchmark suite."),
            ("Do not overclaim", "These values describe this seeded suite, not every possible map. A resume claim should include the benchmark conditions or say 'in a measured reference run'."),
        ])

        image_sections: list[tuple[str, str]] = []
        if image_report:
            checked = 0
            passed = 0
            for item in image_report["images"]:
                image_sections.append((Path(item["image"]).name, f"Grid {item['grid_shape'][0]} x {item['grid_shape'][1]}, walkable {item['walkable_pct']:.2f}%, start {item['start']}, goal {item['goal']}."))
                for name, result in item["algorithms"].items():
                    checked += 1
                    passed += int(result["success"])
                    image_sections.append((f"  {name}", f"{'PASS' if result['success'] else 'FAIL'}; path steps {result.get('path_steps', 'n/a')}; expanded nodes {result.get('nodes_expanded', 'n/a')}; time {result.get('time_ms', 0):.4f} ms."))
            image_sections.append(("Interpretation", f"{passed}/{checked} image-planner checks passed. This validates the generated occupancy grid and route validity on bundled images. It is not a pixel-level perception accuracy score because no human-labelled ground-truth masks are included."))
        else:
            image_sections.append(("No image report found", "Run src.evaluation.image_validation before regenerating this report."))
        add_page(pdf, "5. Real 3D-render validation", image_sections)

        add_page(pdf, "6. Interview walkthrough and limitations", [
            ("A concise walkthrough", "I start with an image, convert it into a binary occupancy grid, validate the endpoint coordinates, run several planners on the same grid, and record route quality and search effort. I then compare the algorithms using reproducible JSON rather than a single visual example."),
            ("Why use BFS as a reference", "BFS is easy to reason about on uniform-cost grids and supplies shortest-path ground truth for the benchmark. That lets me calculate optimality for other methods."),
            ("What I improved", "I made image resizing coordinate-safe, added shared 3D-style segmentation, isolated dynamic obstacle state, fixed bidirectional movement handling, added deterministic benchmark controls, added image validation, and added regression tests."),
            ("Limitations", "The perception pipeline is heuristic and not trained on a labelled dataset. The manual NN is educational rather than production-grade. Results are hardware- and map-dependent. Learned planners need stronger evaluation before being used in safety-critical navigation."),
            ("Good next steps", "Add a labelled segmentation dataset, report precision/recall/IoU, add property-based grid tests, evaluate larger maps, profile memory, add continuous-world planning, and compare against established robotics planners."),
            ("Resume-safe summary", "Built an image-to-occupancy-grid navigation system with classical search, planning under uncertainty, reinforcement learning, a hybrid classifier, a PyQt5 dashboard, reproducible benchmarks, and real-image validation. In a measured seeded suite, Weighted A* reduced expanded nodes by the value reported in this document versus BFS, with the stated optimality trade-off."),
        ])

    print(f"Wrote plain interview report: {args.output}")


if __name__ == "__main__":
    main()
