"""
╔══════════════════════════════════════════════════════════════════════╗
║          VISION PATH PLANNER — AI Fundamentals Showcase             ║
║                                                                      ║
║  A real-world navigation system demonstrating:                       ║
║    1. BFS       — Uninformed Search (shortest path guarantee)        ║
║    2. DFS+IDDFS — Uninformed Search (memory-efficient)               ║
║    3. A* (5 heuristics) — Informed Heuristic Search                  ║
║    4. Weighted A* — Speed-quality tradeoff                           ║
║    5. MDP + Value Iteration — Stochastic decision making             ║
║    6. Q-Learning — Model-free Reinforcement Learning                 ║
║    7. Neural Network — Learned obstacle classification               ║
╚══════════════════════════════════════════════════════════════════════╝

Problem Formulation:
  An autonomous agent must navigate from Start → Goal through a
  real floorplan image. The image is converted to a 2D grid.
  Multiple AI paradigms solve this and are compared.

Environment:
  - State:  (row, col) cell on the grid
  - Action: {UP, DOWN, LEFT, RIGHT} (+ diagonals for some algorithms)
  - Reward: +100 at goal, -0.04 per step (MDP/RL only)
  - Transition: Deterministic (BFS/DFS/A*), Stochastic (MDP/Q-Learning)

Run:   python main.py
       python main.py --image data/maze/image1.jpg
       python main.py --image data/floorplans/fp2.jpg --rl-episodes 200
"""

import sys
import os
import time
import argparse
import math

# ── Path setup (allows running from any directory) ──────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from image_processing.grid import ImageToGrid
from algorithms.bfs import BFS
from algorithms.dfs import DFS
from algorithms.astar import AStar, HEURISTICS, HEURISTIC_INFO
from mdp.mdp import MDP
from mdp.qlearning import QLearningAgent
from neural.nn_classifier import NeuralNetClassifier, build_training_data, nn_predict_grid
from utils.point_selector import auto_select_points, fix_to_walkable, select_points


# ── Color palette (BGR for OpenCV) ──────────────────────────────────────
ALGO_COLORS = {
    'BFS':              (255,  80,  20),
    'DFS':              ( 20, 140, 255),
    'IDDFS':            ( 80,  80, 255),
    'A*(Manhattan)':    ( 30, 200,  60),
    'A*(Euclidean)':    ( 60, 170,  30),
    'A*(Chebyshev)':    (  0, 220, 110),
    'A*(Octile)':       ( 10, 180, 180),
    'A*(Dijkstra)':     (150, 200,   0),
    'WA*(w=2)':         (200,  30, 200),
    'MDP':              (  0, 200, 200),
    'Q-Learning':       (  0, 180, 255),
    'NN+A*':            (180,  60, 255),
}


# ─────────────────────────────────────────────────────────────────────────
def draw(image, path, start, goal, color_bgr, thickness=2):
    display = image.copy()
    if display.ndim == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    if path and len(path) > 1:
        pts = np.array([(c, r) for (r, c) in path], np.int32)
        cv2.polylines(display, [pts.reshape((-1,1,2))], False, color_bgr, thickness)

    cv2.circle(display, (start[1], start[0]), 7, (0, 0, 255), -1)
    cv2.circle(display, (goal[1],  goal[0]),  7, (0, 255, 0), -1)
    cv2.circle(display, (start[1], start[0]), 7, (255,255,255), 1)
    cv2.circle(display, (goal[1],  goal[0]),  7, (255,255,255), 1)
    return display


def path_length_px(path):
    """Euclidean length of path in pixels."""
    total = 0.0
    for i in range(1, len(path)):
        dr = path[i][0] - path[i-1][0]
        dc = path[i][1] - path[i-1][1]
        total += math.sqrt(dr*dr + dc*dc)
    return round(total, 2)


def show_grid_visual(img, binary, grid):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15,5))

    # Original
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    # Binary
    axs[1].imshow(binary, cmap='gray')
    axs[1].set_title("Binary (Walkable vs Obstacle)")
    axs[1].axis('off')

    # Grid
    axs[2].imshow(grid, cmap='gray')
    axs[2].set_title("Grid (1 = Path, 0 = Block)")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

    
# ─────────────────────────────────────────────────────────────────────────
def run(image_path, rl_episodes=200, interactive_points=False):

    print("\n" + "="*70)
    print("  VISION PATH PLANNER — Full AI Pipeline")
    print("="*70)

    # ── 1. Load & Process Image ──────────────────────────────────────────
    print("\n[1/7] Loading and preprocessing image...")
    processor = ImageToGrid(image_path, white_thresh=200)
    grid, binary, img, gray = processor.process()
    rows, cols = len(grid), len(grid[0])
    walkable_count = sum(grid[r][c] for r in range(rows) for c in range(cols))
    print(f"      Grid size: {rows}×{cols} | Walkable cells: {walkable_count}")

    # ── 2. Select Points ─────────────────────────────────────────────────
    if interactive_points:
        s, g = select_points(img)
        if s is None:
            s, g = auto_select_points(grid)
    else:
        s, g = auto_select_points(grid)

    start = fix_to_walkable(grid, s)
    goal  = fix_to_walkable(grid, g)
    print(f"      Start: {start} | Goal: {goal}")

    results = {}     # {name: {path, nodes, time, color_bgr}}

    # ── 3. BFS ──────────────────────────────────────────────────────────
    print("\n[2/7] Running Uninformed Search (BFS, DFS, IDDFS)...")
    t = time.time()
    bfs_path, bfs_nodes = BFS(grid).search(start, goal)
    bfs_time = time.time() - t
    results['BFS'] = dict(path=bfs_path, nodes=bfs_nodes, time=bfs_time,
                          color_bgr=ALGO_COLORS['BFS'])
    print(f"      BFS   → Path: {len(bfs_path) if bfs_path else 'None':>5} cells | "
          f"Explored: {bfs_nodes:>6} | Time: {bfs_time:.4f}s")

    t = time.time()
    dfs_path, dfs_nodes = DFS(grid).search(start, goal)
    dfs_time = time.time() - t
    results['DFS'] = dict(path=dfs_path, nodes=dfs_nodes, time=dfs_time,
                          color_bgr=ALGO_COLORS['DFS'])
    print(f"      DFS   → Path: {len(dfs_path) if dfs_path else 'None':>5} cells | "
          f"Explored: {dfs_nodes:>6} | Time: {dfs_time:.4f}s")

    t = time.time()
    iddfs_path, iddfs_nodes = DFS(grid).iddfs(start, goal, max_depth=300)
    iddfs_time = time.time() - t
    results['IDDFS'] = dict(path=iddfs_path, nodes=iddfs_nodes, time=iddfs_time,
                             color_bgr=ALGO_COLORS['IDDFS'])
    print(f"      IDDFS → Path: {len(iddfs_path) if iddfs_path else 'None':>5} cells | "
          f"Explored: {iddfs_nodes:>6} | Time: {iddfs_time:.4f}s")

    # ── 4. A* with all heuristics ────────────────────────────────────────
    print("\n[3/7] Running A* with all heuristics...")
    for hname in ['manhattan', 'euclidean', 'chebyshev', 'octile', 'dijkstra']:
        t = time.time()
        path, nodes = AStar(grid, heuristic=hname).search(start, goal)
        elapsed = time.time() - t
        label = f"A*({hname.capitalize()})"
        key = f"A*({hname.capitalize()})"
        results[key] = dict(path=path, nodes=nodes, time=elapsed,
                            color_bgr=ALGO_COLORS.get(f'A*({hname.capitalize()})',
                                                        (50, 200, 80)))
        plen = len(path) if path else 'None'
        print(f"      {label:<22} → Path: {str(plen):>5} | Nodes: {nodes:>6} | {elapsed:.4f}s")
        print(f"         [{HEURISTIC_INFO[hname]}]")

    # Weighted A* (w=2): faster but slightly longer path
    t = time.time()
    wa_path, wa_nodes = AStar(grid, heuristic='octile', weight=2.0).search(start, goal)
    wa_time = time.time() - t
    results['WA*(w=2)'] = dict(path=wa_path, nodes=wa_nodes, time=wa_time,
                                color_bgr=ALGO_COLORS['WA*(w=2)'])
    print(f"      {'WA*(w=2,Octile)':<22} → Path: {str(len(wa_path) if wa_path else 'None'):>5} "
          f"| Nodes: {wa_nodes:>6} | {wa_time:.4f}s")
    print(f"         [Weighted A*: f=g+2h. Faster than A*, sub-optimal but bounded.]")

    # ── 5. MDP + Value Iteration ─────────────────────────────────────────
    print("\n[4/7] Running MDP + Value Iteration (stochastic navigation)...")
    print("      Building MDP (S, A, R, P) over walkable cells...")
    mdp = MDP(grid, goal,
              gamma=0.95,
              slip_prob=0.10,
              convergence_thresh=1e-3,
              max_iterations=300)
    t = time.time()
    iters = mdp.value_iteration()
    mdp_time = time.time() - t
    mdp_path, mdp_steps = mdp.extract_path(start)
    results['MDP'] = dict(path=mdp_path, nodes=len(mdp.states),
                           time=mdp_time, color_bgr=ALGO_COLORS['MDP'])
    print(f"      Converged in {iters} iterations | Time: {mdp_time:.4f}s")
    print(f"      Path length: {len(mdp_path)} | Slip probability: 10%")
    print(f"      States: {len(mdp.states)} | γ={mdp.gamma}")

    # ── 6. Q-Learning ────────────────────────────────────────────────────
    print(f"\n[5/7] Training Q-Learning agent ({rl_episodes} episodes)...")
    qagent = QLearningAgent(grid, goal,
                             alpha=0.1,
                             gamma=0.95,
                             epsilon_start=1.0,
                             epsilon_decay=0.995,
                             epsilon_min=0.05)

    def rl_progress(ep, total, eps, avg_r):
        print(f"      Episode {ep:>4}/{total} | ε={eps:.3f} | Avg Reward={avg_r:>8.2f}")

    t = time.time()
    qagent.train(num_episodes=rl_episodes, max_steps_per_episode=400,
                 start_fixed=start, progress_callback=rl_progress)
    ql_time = time.time() - t

    ql_path, ql_steps = qagent.extract_path(start)
    results['Q-Learning'] = dict(path=ql_path, nodes=rl_episodes,
                                  time=ql_time, color_bgr=ALGO_COLORS['Q-Learning'])
    conv = qagent.convergence_rate()
    print(f"      Training done in {ql_time:.2f}s")
    print(f"      Path length: {len(ql_path)} | Convergence gain: {conv}")

    # ── 7. Neural Network ────────────────────────────────────────────────
    print("\n[6/7] Training Neural Network obstacle classifier...")
    print("      Extracting pixel patch features from image...")
    gray_list = gray.tolist()
    X, y, (pos, neg) = build_training_data(gray_list, threshold=200, sample_rate=0.03)
    print(f"      Training samples: {len(X)} | Walkable: {pos} | Obstacle: {neg}")

    nn = NeuralNetClassifier(input_size=9, hidden_size=16, lr=0.05)
    t = time.time()
    nn.train(X, y, epochs=20)
    nn_train_time = time.time() - t

    final_loss = nn.train_losses[-1] if nn.train_losses else 0
    print(f"      Training done | Epochs: 20 | Final Loss: {final_loss:.4f} | Time: {nn_train_time:.2f}s")

    # Apply NN to predict grid, then run A* on it
    print("      Applying NN to classify all pixels → A* on NN grid...")
    t = time.time()
    nn_grid = nn_predict_grid(nn, gray_list, threshold=0.5)
    nn_grid_numpy = np.array(nn_grid, dtype=np.int8)
    # Border cleanup
    nn_grid_numpy[:3, :] = 0; nn_grid_numpy[-3:, :] = 0
    nn_grid_numpy[:, :3] = 0; nn_grid_numpy[:, -3:] = 0
    nn_grid_list = nn_grid_numpy.tolist()

    nn_start = fix_to_walkable(nn_grid_list, start)
    nn_goal  = fix_to_walkable(nn_grid_list, goal)
    nn_path, nn_nodes = AStar(nn_grid_list, heuristic='octile').search(nn_start, nn_goal)
    nn_total_time = time.time() - t + nn_train_time
    results['NN+A*'] = dict(path=nn_path, nodes=nn_nodes,
                             time=nn_total_time, color_bgr=ALGO_COLORS['NN+A*'])
    print(f"      NN+A* path: {len(nn_path) if nn_path else 'None'} | Time (incl. training): {nn_total_time:.2f}s")

    # ── Print Summary Table ──────────────────────────────────────────────
    print("\n" + "="*80)
    print("  COMPLETE PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Algorithm':<24} {'Path Cells':>12} {'Path(px)':>10} {'Nodes':>8} {'Time(s)':>10} {'Eff':>10}")
    print("-"*80)
    for name, r in results.items():
        p = r['path']
        n = r.get('nodes', 0) or 1
        t_s = r.get('time', 0)
        plen = len(p) if p else 0
        ppx  = path_length_px(p) if p else 0
        eff  = f"{plen/n:.4f}" if p and n else '-'
        print(f"{name:<24} {str(plen if plen else 'NO PATH'):>12} {str(ppx):>10} "
              f"{n:>8} {t_s:>10.4f} {eff:>10}")
    print("="*80)

    # ── Visualize ────────────────────────────────────────────────────────
    print("\n[7/7] Generating visualizations...")
    _visualize_all(img, grid, mdp, qagent, nn, results, start, goal, image_path)

    print("\n✓ Done! Check the 'outputs/' folder for saved images.")
    return results


# ─────────────────────────────────────────────────────────────────────────
def _visualize_all(img, grid, mdp, qagent, nn, results, start, goal, image_path):
    """Generate all visualization panels."""

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Panel 1: Algorithm comparison grid
    names = list(results.keys())
    n = len(names)
    cols_per_row = 4
    nrows = math.ceil(n / cols_per_row)

    fig = plt.figure(figsize=(6 * cols_per_row, 5 * nrows + 1))
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(nrows, cols_per_row, figure=fig,
                           hspace=0.4, wspace=0.15)

    for idx, name in enumerate(names):
        row = idx // cols_per_row
        col = idx % cols_per_row
        ax = fig.add_subplot(gs[row, col])
        r = results[name]
        drawn = draw(img, r['path'], start, goal, r['color_bgr'], thickness=2)
        ax.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))

        p = r['path']
        plen = len(p) if p else 0
        nodes = r.get('nodes', 0)
        t_s   = r.get('time', 0)

        info = f"Path: {plen} | Nodes: {nodes}\nTime: {t_s:.4f}s"
        ax.set_title(name, color='white', fontsize=10, fontweight='bold', pad=3)
        ax.set_xlabel(info, color='#aaaaaa', fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            r_val = results[name]['color_bgr']
            spine.set_edgecolor(f"#{r_val[2]:02x}{r_val[1]:02x}{r_val[0]:02x}")
            spine.set_linewidth(2)

    # Hide unused subplots
    for idx in range(n, nrows * cols_per_row):
        fig.add_subplot(gs[idx // cols_per_row, idx % cols_per_row]).axis('off')

    fig.suptitle("Vision Path Planner — AI Algorithm Comparison",
                 color='white', fontsize=16, fontweight='bold', y=1.01)
    out1 = os.path.join(out_dir, "01_algorithm_comparison.png")
    plt.savefig(out1, dpi=130, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  Saved: {out1}")

    # Panel 2: MDP Value Heatmap
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    fig2.patch.set_facecolor('#0d1117')

    rows, cols = len(grid), len(grid[0])
    vgrid = np.zeros((rows, cols))
    for (r, c), v in mdp.V.items():
        vgrid[r][c] = v
    obstacle_mask = np.array([[grid[r][c] == 0 for c in range(cols)]
                               for r in range(rows)])
    vgrid[obstacle_mask] = np.nan

    im = axes2[0].imshow(vgrid, cmap='plasma', interpolation='nearest')
    axes2[0].set_title("MDP Value Function V(s)\n(Bellman Value Iteration)",
                        color='white', fontsize=11)
    axes2[0].axis('off')
    cbar = plt.colorbar(im, ax=axes2[0], fraction=0.046)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.set_label("V(s) value", color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    axes2[0].set_facecolor('#111111')

    # Q-Learning heatmap
    qgrid = np.array(qagent.get_q_value_grid())
    qgrid[obstacle_mask] = np.nan
    im2 = axes2[1].imshow(qgrid, cmap='viridis', interpolation='nearest')
    axes2[1].set_title("Q-Learning Max Q-Value\n(ε-greedy, temporal difference)",
                        color='white', fontsize=11)
    axes2[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes2[1], fraction=0.046)
    cbar2.ax.yaxis.set_tick_params(color='white')
    cbar2.set_label("max Q(s,a)", color='white')
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color='white')

    fig2.suptitle("MDP & Reinforcement Learning — Value Maps",
                  color='white', fontsize=14, fontweight='bold')
    out2 = os.path.join(out_dir, "02_mdp_rl_heatmaps.png")
    plt.savefig(out2, dpi=130, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  Saved: {out2}")

    # Panel 3: Q-Learning training curve
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    fig3.patch.set_facecolor('#0d1117')
    ax3.set_facecolor('#161b22')
    rewards = qagent.episode_rewards
    ax3.plot(rewards, alpha=0.3, color='#58a6ff', linewidth=0.6, label='Episode Reward')
    if len(rewards) > 20:
        w = 20
        smooth = np.convolve(rewards, np.ones(w)/w, mode='valid')
        ax3.plot(range(w-1, len(rewards)), smooth, color='#f78166',
                 linewidth=2, label=f'Smoothed (window={w})')
    ax3.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax3.set_xlabel("Episode", color='white')
    ax3.set_ylabel("Total Reward", color='white')
    ax3.set_title("Q-Learning Training Curve\n(Reward improves as agent learns optimal policy)",
                  color='white', fontsize=12)
    ax3.tick_params(colors='white')
    ax3.legend(facecolor='#161b22', edgecolor='#444', labelcolor='white')
    ax3.grid(True, alpha=0.2, color='#444')
    for sp in ax3.spines.values():
        sp.set_edgecolor('#444')
    out3 = os.path.join(out_dir, "03_qlearning_training.png")
    plt.savefig(out3, dpi=130, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  Saved: {out3}")

    # Panel 4: Neural Network training loss
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    fig4.patch.set_facecolor('#0d1117')
    ax4.set_facecolor('#161b22')
    ax4.plot(nn.train_losses, color='#3fb950', linewidth=2, marker='o', markersize=4)
    ax4.set_xlabel("Epoch", color='white')
    ax4.set_ylabel("Binary Cross-Entropy Loss", color='white')
    ax4.set_title("Neural Network Training Loss\n(2-layer NN, backpropagation, SGD optimizer)",
                  color='white', fontsize=12)
    ax4.tick_params(colors='white')
    ax4.grid(True, alpha=0.2, color='#444')
    for sp in ax4.spines.values():
        sp.set_edgecolor('#444')
    out4 = os.path.join(out_dir, "04_nn_training_loss.png")
    plt.savefig(out4, dpi=130, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  Saved: {out4}")

        # ─────────────────────────────────────────────
    # NEW: Save individual algorithm path images
    # ─────────────────────────────────────────────
    single_dir = os.path.join(out_dir, "individual_paths")
    os.makedirs(single_dir, exist_ok=True)

    print("\n  Saving individual algorithm paths...")

    for name, r in results.items():
        path = r['path']

        # Handle no path case
        if path is None or len(path) == 0:
            drawn = draw(img, [], start, goal, (0, 0, 255), thickness=2)
        else:
            drawn = draw(img, path, start, goal, r['color_bgr'], thickness=2)

        # Clean filename
        safe_name = (
            name.replace("*", "")
                .replace("(", "")
                .replace(")", "")
                .replace("=", "")
                .replace(",", "")
                .replace(" ", "_")
        )

        save_path = os.path.join(single_dir, f"{safe_name}.png")

        cv2.imwrite(save_path, drawn)

        print(f"    ✔ Saved: {save_path}")

    # Panel 5: Heuristic comparison bar chart
    heuristic_results = {k: v for k, v in results.items() if k.startswith('A*')}
    if heuristic_results:
        fig5, axes5 = plt.subplots(1, 2, figsize=(12, 5))
        fig5.patch.set_facecolor('#0d1117')

        h_names  = list(heuristic_results.keys())
        h_nodes  = [heuristic_results[h]['nodes'] for h in h_names]
        h_times  = [heuristic_results[h]['time'] * 1000 for h in h_names]  # ms

        colors_bar = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff', '#ffa657']

        for ax5, (vals, ylabel, title) in zip(axes5, [
            (h_nodes, 'Nodes Explored', 'Nodes Explored per Heuristic'),
            (h_times, 'Time (ms)',      'Execution Time per Heuristic'),
        ]):
            ax5.set_facecolor('#161b22')
            bars = ax5.bar(range(len(h_names)), vals, color=colors_bar[:len(h_names)])
            ax5.set_xticks(range(len(h_names)))
            ax5.set_xticklabels(h_names, rotation=20, ha='right', color='white', fontsize=9)
            ax5.set_ylabel(ylabel, color='white')
            ax5.set_title(title, color='white', fontsize=11)
            ax5.tick_params(colors='white')
            for sp in ax5.spines.values():
                sp.set_edgecolor('#444')
            ax5.grid(True, alpha=0.2, axis='y', color='#444')
            for bar, val in zip(bars, vals):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f'{val:.1f}', ha='center', va='bottom',
                         color='white', fontsize=8)

        fig5.suptitle("A* Heuristic Comparison: Which heuristic is best?",
                      color='white', fontsize=13, fontweight='bold')
        out5 = os.path.join(out_dir, "05_heuristic_comparison.png")
        plt.savefig(out5, dpi=130, bbox_inches='tight', facecolor='#0d1117')
        plt.close()
        print(f"  Saved: {out5}")

    # Show all plots
    try:
        for fname in ["01_algorithm_comparison.png", "03_qlearning_training.png",
                      "05_heuristic_comparison.png"]:
            fpath = os.path.join(out_dir, fname)
            if os.path.exists(fpath):
                img_show = cv2.imread(fpath)
                if img_show is not None:
                    cv2.namedWindow(fname, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(fname, min(1400, img_show.shape[1]),
                                     min(800, img_show.shape[0]))
                    cv2.imshow(fname, img_show)
        print("\n  Press any key in image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision Path Planner — AI Showcase")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to floorplan/maze image")
    parser.add_argument("--rl-episodes", type=int, default=200,
                        help="Number of Q-Learning training episodes (default: 200)")
    parser.add_argument("--interactive", action="store_true",
                        help="Click to select start/goal points (requires display)")
    args = parser.parse_args()

    # Locate image
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.image:
        image_path = args.image
        if not os.path.isabs(image_path):
            image_path = os.path.join(base_dir, image_path)
    else:
        # Default: try floorplan, then maze
        candidates = [
            os.path.join(base_dir, "data", "floorplans", "fp4.jpg"),
            os.path.join(base_dir, "data", "floorplans", "fp1.jpg"),
            os.path.join(base_dir, "data", "maze", "image1.jpg"),
        ]
        image_path = next((p for p in candidates if os.path.exists(p)), None)

    if image_path is None or not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        print("  Usage: python main.py --image data/floorplans/fp1.jpg")
        sys.exit(1)

    print(f"\nUsing image: {image_path}")
    run(image_path,
        rl_episodes=args.rl_episodes,
        interactive_points=args.interactive)
