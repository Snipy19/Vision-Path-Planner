"""Build the external project report PDF.

The PDF is intentionally generated outside the repository.  The HTML
architecture page remains the interactive/animated artifact; the PDF embeds a
static perspective snapshot because PDF is a paginated format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Polygon


NAVY = "#0b1020"
PANEL = "#121a2b"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
BLUE = "#6ea8fe"
PURPLE = "#a78bfa"
GREEN = "#22c55e"
ORANGE = "#f59e0b"


def page_base(title: str, subtitle: str = ""):
    fig = plt.figure(figsize=(11.69, 8.27), facecolor=NAVY)
    fig.text(0.055, 0.93, title, color=TEXT, fontsize=22, weight="bold")
    if subtitle:
        fig.text(0.055, 0.895, subtitle, color=MUTED, fontsize=10)
    fig.text(0.945, 0.04, "Vision Path Planner", color="#64748b", fontsize=8,
             ha="right")
    return fig


def add_card(fig, x, y, w, h, value, label, color):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
        transform=fig.transFigure, facecolor=PANEL, edgecolor=color,
        linewidth=1.2))
    fig.text(x + 0.02, y + h * 0.53, value, color=color, fontsize=20,
             weight="bold")
    fig.text(x + 0.02, y + h * 0.2, label, color=MUTED, fontsize=9)


def architecture_snapshot(fig):
    ax = fig.add_axes([0.06, 0.12, 0.88, 0.72], facecolor=NAVY)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, label, color):
        depth = 0.14
        ax.add_patch(Polygon([(x, y), (x + w, y), (x + w + depth, y + depth),
                              (x + depth, y + depth)], closed=True,
                             facecolor="#0a0f1d", edgecolor=color, linewidth=0.8))
        ax.add_patch(Polygon([(x + w, y), (x + w, y + h),
                              (x + w + depth, y + h + depth),
                              (x + w + depth, y + depth)], closed=True,
                             facecolor="#0a0f1d", edgecolor=color, linewidth=0.8))
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=PANEL, edgecolor=color,
                                    linewidth=1.5))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                color=TEXT, fontsize=9, weight="bold", wrap=True)

    def arrow(x1, y1, x2, y2, color=BLUE):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.3))

    box(0.4, 5.4, 1.5, 0.8, "Image\nJPG / PNG", BLUE)
    box(2.45, 5.4, 1.7, 0.8, "ImageToGrid\nscale + morphology", ORANGE)
    box(4.7, 5.4, 1.6, 0.8, "Occupancy\ngrid", GREEN)
    arrow(1.9, 5.8, 2.45, 5.8)
    arrow(4.15, 5.8, 4.7, 5.8)

    box(0.6, 3.5, 1.45, 0.85, "BFS / DFS", BLUE)
    box(2.45, 3.5, 1.45, 0.85, "A* / WA*", BLUE)
    box(4.3, 3.5, 1.7, 0.85, "Bidirectional\nA*", PURPLE)
    box(6.45, 3.5, 1.55, 0.85, "MDP /\nQ-Learning", PURPLE)
    box(8.35, 3.5, 1.1, 0.85, "NN +\nA*", ORANGE)
    for x in (1.3, 3.15, 5.15, 7.2, 8.9):
        arrow(5.5, 5.4, x, 4.35, ORANGE)

    box(1.5, 1.45, 2.1, 0.85, "PyQt5 dashboard\nQThread workers", BLUE)
    box(4.0, 1.45, 2.0, 0.85, "Metrics\npath / nodes / time", GREEN)
    box(6.4, 1.45, 2.1, 0.85, "Artifacts\noverlays / JSON / PDF", GREEN)
    for x in (2.55, 5.15, 8.0):
        arrow(x, 3.5, x, 2.3, GREEN)
    ax.text(5.1, 6.65, "Runtime architecture (static perspective snapshot)",
            ha="center", color=MUTED, fontsize=10)


def methods_page(pdf):
    fig = page_base("A-Z project explanation", "From pixels to a route, with measurable trade-offs")
    text = [
        ("A — Input", "Accepts a floorplan, maze, map, or rendered scene."),
        ("B — Binary grid", "Thresholding and morphology create 1=walkable / 0=obstacle cells."),
        ("C — Coordinates", "Start and goal are selected manually or deterministically."),
        ("D — Deterministic search", "BFS establishes shortest-path ground truth; A* adds heuristics."),
        ("E — Efficiency", "Expanded nodes, wall-clock time, path steps, and optimality are recorded."),
        ("F — Feedback", "MDP and Q-Learning model stochastic or learned navigation."),
        ("G — Grid changes", "Dynamic obstacles trigger a clean re-plan without mutating the base map."),
        ("H — Hybrid perception", "The manual 3x3-patch NN produces a second grid for A* validation."),
        ("I — Interactive UI", "PyQt5 workers keep the dashboard responsive during computation."),
        ("J — JSON evidence", "Seed, scenario, raw runs, and aggregates are exported for review."),
        ("K — Known limitation", "Perception accuracy is not claimed without human-labelled data."),
        ("L — Reproduction", "One command reruns the benchmark with the same seed."),
    ]
    y = 0.82
    for head, body in text:
        fig.text(0.08, y, head, color=BLUE if head[0] in "ABCD" else GREEN,
                 fontsize=11, weight="bold")
        fig.text(0.25, y, body, color=TEXT, fontsize=10)
        y -= 0.055
    pdf.savefig(fig, facecolor=NAVY)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path,
                        default=Path("outputs/benchmarks/benchmark_latest.json"))
    parser.add_argument("--comparison", type=Path,
                        default=Path("outputs/01_algorithm_comparison.png"))
    parser.add_argument("--image-report", type=Path,
                        default=Path("outputs/benchmarks/image_validation.json"))
    args = parser.parse_args()
    report = json.loads(args.benchmark.read_text(encoding="utf-8"))
    summary = report["summary"]
    meta = report["benchmark"]
    total_runs = sum(row["runs"] for row in summary.values())
    total_success = sum(row["success_rate_pct"] * row["runs"]
                        for row in summary.values()) / max(1, total_runs)
    best_reduction_name = max(summary, key=lambda name: summary[name]["node_reduction_vs_bfs_pct"])
    best_reduction = summary[best_reduction_name]["node_reduction_vs_bfs_pct"]
    image_report = None
    if args.image_report.exists():
        image_report = json.loads(args.image_report.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(args.output) as pdf:
        fig = page_base("VISION PATH PLANNER", "Technical project report | measured, reproducible, and resume-ready")
        fig.text(0.06, 0.75, "Image-to-path navigation across classical AI,\nplanning under uncertainty, and learned perception.",
                 color=TEXT, fontsize=24, weight="bold", linespacing=1.35)
        fig.text(0.06, 0.58, "The repository now includes a scale-safe image pipeline,\nvalidated planner implementations, a seeded benchmark harness,\nand a polished interactive architecture view.", color=MUTED, fontsize=12,
                 linespacing=1.6)
        fig.text(0.06, 0.49, "Authors: Dhruv Raj (github.com/Snipy19) · Yashica Gupta (github.com/gyashica1hub)", color=BLUE, fontsize=10.5, weight="bold")
        add_card(fig, 0.06, 0.25, 0.18, 0.12, str(total_runs), "measured planner runs", BLUE)
        add_card(fig, 0.27, 0.25, 0.18, 0.12, f"{best_reduction:.2f}%", f"{best_reduction_name} node reduction", GREEN)
        add_card(fig, 0.48, 0.25, 0.18, 0.12, f"{total_success:.0f}%", "weighted route success", PURPLE)
        add_card(fig, 0.69, 0.25, 0.18, 0.12, meta["grid_size"], "benchmark grid", ORANGE)
        pdf.savefig(fig, facecolor=NAVY)
        plt.close(fig)

        fig = page_base("System architecture", "Static 3D-style export; open architecture.html for the animated browser version")
        architecture_snapshot(fig)
        pdf.savefig(fig, facecolor=NAVY)
        plt.close(fig)

        methods_page(pdf)

        fig = page_base("Benchmark evidence", f"{meta['scenarios']} scenarios × {meta['repeats']} repeats, seed {meta['seed']}, density {meta['obstacle_density']:.3f}, {meta['movement']}")
        ax = fig.add_axes([0.07, 0.27, 0.86, 0.48])
        ax.axis("off")
        names = list(summary)
        columns = ["Algorithm", "Success", "Mean nodes", "Reduction", "Optimality", "Mean time"]
        table = [columns]
        for name in names:
            row = summary[name]
            table.append([name, f"{row['success_rate_pct']:.0f}%",
                          f"{row['mean_nodes_expanded']:,.2f}",
                          f"{row['node_reduction_vs_bfs_pct']:.2f}%",
                          f"{row['mean_optimality_pct']:.3f}%",
                          f"{row['mean_time_ms']:.4f} ms"])
        tbl = ax.table(cellText=table, loc="center", cellLoc="center",
                       colWidths=[0.23, 0.13, 0.17, 0.17, 0.17, 0.17])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 2.0)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#263653")
            cell.set_facecolor("#172033" if r else "#24365c")
            cell.get_text().set_color(TEXT if r else BLUE)
            if r == 0:
                cell.get_text().set_weight("bold")
        fig.text(0.08, 0.17, "Interpretation: Weighted A* trades a small amount of path optimality for a large reduction in search effort. Bidirectional A* preserves shortest paths while exploring about half as many nodes as BFS on this suite.", color=MUTED, fontsize=10, wrap=True)
        pdf.savefig(fig, facecolor=NAVY)
        plt.close(fig)

        fig = page_base("Updated measured output", "Current benchmark evidence generated from JSON — no stale comparison screenshot")
        real_images = len(image_report["images"]) if image_report else 0
        valid_paths = 0
        checked_paths = 0
        if image_report:
            for item in image_report["images"]:
                for result in item["algorithms"].values():
                    checked_paths += 1
                    valid_paths += int(result["success"])
        add_card(fig, 0.07, 0.66, 0.19, 0.13, str(total_runs), "synthetic planner runs", BLUE)
        add_card(fig, 0.29, 0.66, 0.19, 0.13, f"{best_reduction:.2f}%", "best node reduction", GREEN)
        add_card(fig, 0.51, 0.66, 0.19, 0.13, str(real_images), "real 3D images", PURPLE)
        add_card(fig, 0.73, 0.66, 0.19, 0.13, f"{valid_paths}/{checked_paths}", "valid 3D paths", ORANGE)
        evidence = [
            ("Benchmark", f"{meta['scenarios']} seeded scenarios × {meta['repeats']} repeats on {meta['grid_size']} grids"),
            ("Algorithms", "BFS, DFS, A*, Weighted A*, Bidirectional A*, with measured nodes, latency, and optimality"),
            ("3D validation", "Real bundled renders passed through the shared image-to-occupancy-grid pipeline"),
            ("Reproducibility", f"Seed {meta['seed']} · obstacle density {meta['obstacle_density']:.3f} · {meta['movement']} movement"),
        ]
        y = 0.51
        for label, value in evidence:
            fig.text(0.09, y, label, color=BLUE, fontsize=12, weight="bold")
            fig.text(0.25, y, value, color=TEXT, fontsize=11)
            y -= 0.09
        fig.text(0.09, 0.12, "The original comparison image remains available in outputs/ for local inspection; this report page is intentionally generated from the latest measured artifacts.", color=MUTED, fontsize=10, wrap=True)
        pdf.savefig(fig, facecolor=NAVY)
        plt.close(fig)

        if image_report:
            fig = page_base("3D-render validation", "Real bundled images; success means a valid path on the generated occupancy grid")
            ax = fig.add_axes([0.07, 0.3, 0.86, 0.43])
            ax.axis("off")
            rows = [["Image", "Grid", "Walkable", "BFS", "A*", "Weighted A*", "Bidir A*"]]
            for item in image_report["images"]:
                algos = item["algorithms"]
                short = Path(item["image"]).name
                rows.append([
                    short,
                    "x".join(map(str, item["grid_shape"])),
                    f"{item['walkable_pct']:.2f}%",
                    f"{'PASS' if algos['BFS']['success'] else 'FAIL'} / {algos['BFS']['path_steps']}",
                    f"{'PASS' if algos['A*']['success'] else 'FAIL'} / {algos['A*']['path_steps']}",
                    f"{'PASS' if algos['Weighted A*']['success'] else 'FAIL'} / {algos['Weighted A*']['path_steps']}",
                    f"{'PASS' if algos['Bidirectional A*']['success'] else 'FAIL'} / {algos['Bidirectional A*']['path_steps']}",
                ])
            tbl = ax.table(cellText=rows, loc="center", cellLoc="center",
                           colWidths=[0.18, 0.12, 0.13, 0.14, 0.14, 0.16, 0.16])
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1, 2.0)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor("#263653")
                cell.set_facecolor("#172033" if r else "#24365c")
                cell.get_text().set_color(TEXT if r else BLUE)
                if r == 0:
                    cell.get_text().set_weight("bold")
            fig.text(0.08, 0.18, "This is a segmentation-and-planning validation, not a perception accuracy score: no pixel-level labels are available for these images. The report preserves the selected mode, grid dimensions, walkable ratio, endpoints, path validity, nodes, and latency.", color=MUTED, fontsize=10, wrap=True)
            pdf.savefig(fig, facecolor=NAVY)
            plt.close(fig)

        fig = page_base("Resume-ready summary", "Use the measured claims; keep the methodology attached to the numbers")
        bullets = [
            "Built an image-to-occupancy-grid navigation system in Python/OpenCV with BFS, A*, Weighted A*, Bidirectional A*, MDP value iteration, Q-Learning, and manual backpropagation.",
            f"Reduced expanded nodes by {best_reduction:.2f}% with {best_reduction_name} versus BFS in this measured run; the exact result changes with map size, density, seed, and hardware.",
            "Delivered a PyQt5 dashboard with live obstacle edits, stepwise animation, RL/MDP heatmaps, JSON benchmark export, and regression tests.",
        ]
        y = 0.76
        for bullet in bullets:
            fig.text(0.09, y, "•", color=GREEN, fontsize=20, weight="bold")
            fig.text(0.125, y, bullet, color=TEXT, fontsize=13, wrap=True, va="top")
            y -= 0.16
        fig.text(0.09, 0.22, "Reproduce this exact report: python -m src.evaluation.benchmark --scenarios " + str(meta["scenarios"]) + " --repeats " + str(meta["repeats"]) + " --size " + str(meta["grid_size"].split("x")[0]) + " --seed " + str(meta["seed"]) + " --density " + str(meta["obstacle_density"]), color=BLUE, fontsize=10)
        fig.text(0.09, 0.15, "Important honesty note: no perception accuracy is claimed without a human-labelled dataset.", color=ORANGE, fontsize=10)
        pdf.savefig(fig, facecolor=NAVY)
        plt.close(fig)

    print(f"Wrote report: {args.output}")


if __name__ == "__main__":
    main()
