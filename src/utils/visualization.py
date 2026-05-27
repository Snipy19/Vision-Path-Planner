"""
Visualization — Draw paths, heatmaps, comparisons on images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os


# Color palette (BGR for OpenCV, RGB for matplotlib)
COLORS_BGR = {
    'bfs':      (255,  80,  20),   # Blue-orange
    'dfs':      ( 20, 140, 255),   # Orange
    'astar':    ( 30, 200,  60),   # Green
    'wastar':   (200,  30, 200),   # Purple
    'mdp':      (  0, 200, 200),   # Cyan
    'qlearn':   (255, 180,   0),   # Gold
    'nn':       (180,   0, 255),   # Violet
    'start':    (  0,   0, 255),   # Red
    'goal':     (  0, 255,   0),   # Green
}

COLORS_RGB = {k: (v[2], v[1], v[0]) for k, v in COLORS_BGR.items()}


def draw_path_on_image(image, path, start, goal, color_bgr, thickness=2):
    """Draw a path as polyline on image (OpenCV BGR)."""
    display = image.copy()
    if image.ndim == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    if path and len(path) > 1:
        pts = np.array([(c, r) for (r, c) in path], np.int32)
        cv2.polylines(display, [pts], False, color_bgr, thickness)

    # Start (red circle), Goal (green circle)
    cv2.circle(display, (start[1], start[0]), 6, COLORS_BGR['start'], -1)
    cv2.circle(display, (goal[1],  goal[0]),  6, COLORS_BGR['goal'],  -1)
    cv2.circle(display, (start[1], start[0]), 6, (255,255,255), 1)
    cv2.circle(display, (goal[1],  goal[0]),  6, (255,255,255), 1)
    return display


def draw_value_heatmap(grid, value_dict, title="Value Heatmap", cmap='hot'):
    """Draw MDP/Q-value heatmap over grid."""
    rows, cols = len(grid), len(grid[0])
    heatmap = np.zeros((rows, cols))
    vmin, vmax = float('inf'), float('-inf')

    for (r, c), v in value_dict.items():
        if v is not None:
            heatmap[r][c] = v
            vmin = min(vmin, v)
            vmax = max(vmax, v)

    # Mask obstacles
    mask = np.array([[1 if grid[r][c] == 0 else 0
                      for c in range(cols)]
                     for r in range(rows)], dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.imshow(mask, cmap='binary', alpha=0.4)
    plt.colorbar(im, ax=ax, label='Value')
    ax.set_title(title)
    ax.axis('off')
    return fig


def comparison_plot(image, results, start, goal, output_path=None):
    """
    Multi-panel comparison of all algorithms side-by-side.
    results: dict {name: {'path': [...], 'color': (r,g,b), 'label': str}}
    """
    names = list(results.keys())
    n = len(names)
    cols_per_row = min(4, n)
    rows_needed = math.ceil(n / cols_per_row) if n > 0 else 1

    import math
    fig, axes = plt.subplots(rows_needed, cols_per_row,
                             figsize=(6 * cols_per_row, 5 * rows_needed))
    if n == 1:
        axes = [[axes]]
    elif rows_needed == 1:
        axes = [axes]

    for idx, name in enumerate(names):
        row = idx // cols_per_row
        col = idx % cols_per_row
        ax = axes[row][col]

        r = results[name]
        path = r.get('path')
        color_bgr = r.get('color_bgr', (0, 200, 0))

        drawn = draw_path_on_image(image, path, start, goal, color_bgr, thickness=2)
        ax.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))

        label = r.get('label', name)
        stats = []
        if path:
            stats.append(f"Path: {len(path)} cells")
        else:
            stats.append("No path")
        if 'nodes' in r:
            stats.append(f"Explored: {r['nodes']}")
        if 'time' in r:
            stats.append(f"Time: {r['time']:.4f}s")

        ax.set_title(f"{label}\n" + " | ".join(stats), fontsize=9)
        ax.axis('off')

    # Hide unused axes
    for idx in range(n, rows_needed * cols_per_row):
        row = idx // cols_per_row
        col = idx % cols_per_row
        axes[row][col].axis('off')

    plt.suptitle("Vision Path Planner — AI Algorithm Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {output_path}")

    return fig


def learning_curve_plot(episode_rewards, title="Q-Learning: Reward per Episode"):
    """Plot Q-learning training curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episode_rewards, alpha=0.4, color='steelblue', linewidth=0.8, label='Raw')

    # Smoothed curve (moving average)
    window = min(30, len(episode_rewards) // 5 + 1)
    if len(episode_rewards) >= window:
        smooth = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(episode_rewards)), smooth,
                color='darkorange', linewidth=2, label=f'Avg-{window}')

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def print_comparison_table(results):
    """Print formatted performance table to console."""
    print("\n" + "=" * 75)
    print("                  VISION PATH PLANNER — PERFORMANCE REPORT")
    print("=" * 75)
    print(f"{'Algorithm':<22} {'Path Len':>10} {'Nodes':>10} {'Time(s)':>10} {'Efficiency':>12}")
    print("-" * 75)

    for name, r in results.items():
        path = r.get('path')
        nodes = r.get('nodes', '-')
        time_s = r.get('time', 0)
        plen = len(path) if path else 'NO PATH'
        eff = f"{len(path)/nodes:.4f}" if path and nodes and nodes > 0 else '-'
        print(f"{name:<22} {str(plen):>10} {str(nodes):>10} {time_s:>10.4f} {eff:>12}")

    print("=" * 75)
