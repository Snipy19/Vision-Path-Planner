"""
==============================================================================
          VISION PATH PLANNER - Interactive AI Dashboard (V3 FAST)
==============================================================================

V3 CHANGES:
1. Algorithm Selector  -- choose which algos to run (MDP/QL can be skipped)
2. MDP: pre-built transition table + heapq path extraction (no slow BFS)
3. Q-Learning: early stopping, pre-built move table, bounded BFS fallback
4. AlgoWorker: only runs selected algorithms
5. Dashboard updates only the tabs relevant to selected algos
"""

import sys, os, time, math
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, BASE_DIR)

import cv2, numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QSplitter,
    QScrollArea, QSpinBox, QGroupBox, QGridLayout, QFrame,
    QProgressBar, QTextEdit, QSizePolicy, QTableWidget,
    QTableWidgetItem, QHeaderView, QComboBox, QCheckBox, QMessageBox, QAbstractScrollArea
)
from PyQt5.QtCore  import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui   import QPixmap, QImage, QFont, QColor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from algorithms.bfs       import BFS
from algorithms.dfs       import DFS
from algorithms.astar     import AStar, HEURISTIC_INFO
from mdp.mdp              import MDP
from mdp.qlearning        import QLearningAgent
from neural.nn_classifier import NeuralNetClassifier, build_training_data, nn_predict_grid
from utils.point_selector import auto_select_points, fix_to_walkable

from src.neural.nn_classifier import NeuralNetClassifier, build_training_data, nn_predict_grid


# ---------- colour palette -----------------------------------------------
DARK_BG  = "#0b1020"
PANEL_BG = "#121a2b"
CARD_BG  = "#172033"

BORDER   = "#202b45"

ACCENT   = "#6ea8fe"
ACCENT2  = "#8b5cf6"

TEXT     = "#f8fafc"
MUTED    = "#94a3b8"

GREEN    = "#22c55e"
RED      = "#ef4444"
ORANGE   = "#f59e0b"

ALGO_COLORS_BGR = {
    'BFS':           (255, 80,  20),
    'DFS':           (20,  140, 255),
    'A*(Manhattan)': (30,  200, 60),
    'A*(Euclidean)': (60,  170, 30),
    'A*(Chebyshev)': (34, 197, 94),
    'MDP':           (0,   200, 200),
    'Q-Learning':    (0,   180, 255),
    'NN+A*':         (168, 85, 247),

}
ALGO_COLORS_HEX = {k: f"#{v[2]:02x}{v[1]:02x}{v[0]:02x}"
                   for k, v in ALGO_COLORS_BGR.items()}

ALL_ALGOS = ['BFS', 'DFS', 'A*(Manhattan)', 'A*(Euclidean)',
             'A*(Chebyshev)', 'MDP', 'Q-Learning', 'NN+A*']


# ===========================================================================
#  Background Worker
# ===========================================================================
class AlgoWorker(QThread):
    log  = pyqtSignal(str)
    done = pyqtSignal(dict)
    err  = pyqtSignal(str)

    def __init__(self, grid, img, gray, start, goal, rl_eps, selected):
        super().__init__()
        self.grid, self.img, self.gray = grid, img, gray
        self.start_pt, self.goal_pt   = start, goal
        self.rl_eps  = rl_eps
        self.selected = selected          # set of algo names to run

    def run(self):
        try:
            R  = {}
            g  = self.grid
            s  = self.start_pt
            gl = self.goal_pt
            sel = self.selected

            # ---- BFS ------------------------------------------------
            if 'BFS' in sel:
                self.log.emit("Running BFS ...")
                t = time.time()
                p, n = BFS(g).search(s, gl)
                R['BFS'] = dict(path=p, nodes=n, time=time.time()-t,
                                color_bgr=ALGO_COLORS_BGR['BFS'])
                self.log.emit(f"BFS  -> path={len(p) if p else 'None'}  nodes={n}")

            # ---- DFS ------------------------------------------------
            if 'DFS' in sel:
                self.log.emit("Running DFS ...")
                t = time.time()
                p, n = DFS(g).search(s, gl)
                R['DFS'] = dict(path=p, nodes=n, time=time.time()-t,
                                color_bgr=ALGO_COLORS_BGR['DFS'])
                self.log.emit(f"DFS  -> path={len(p) if p else 'None'}  nodes={n}")

            # ---- A* x 3 ---------------------------------------------
            for h in ['manhattan', 'euclidean', 'chebyshev']:
                lbl = f"A*({h.capitalize()})"
                if lbl not in sel:
                    continue
                self.log.emit(f"Running {lbl} ...")
                t = time.time()
                p, n = AStar(g, heuristic=h).search(s, gl)
                R[lbl] = dict(path=p, nodes=n, time=time.time()-t,
                              color_bgr=ALGO_COLORS_BGR.get(lbl, (50,200,80)))
                self.log.emit(f"{lbl}  path={len(p) if p else 'None'}  nodes={n}")

            # ---- MDP ------------------------------------------------
            if 'MDP' in sel:
                self.log.emit("Running MDP + Value Iteration (V3 Fast) ...")
                mdp = MDP(g, gl, gamma=0.95, slip_prob=0.10,
                          convergence_thresh=1e-3, max_iterations=300)
                t = time.time()
                iters = mdp.value_iteration()
                self.log.emit(f"  Value iteration: {iters} iters")
                mp, _ = mdp.extract_path(s)
                mdp_t = time.time() - t
                self.log.emit(f"MDP  -> {iters} iters  path={len(mp) if mp else 0}  t={mdp_t:.3f}s")
                R['MDP'] = dict(path=mp, nodes=len(mdp.states),
                                time=mdp_t,
                                color_bgr=ALGO_COLORS_BGR['MDP'],
                                mdp_obj=mdp, mdp_iters=iters)

            # ---- Q-Learning -----------------------------------------
            if 'Q-Learning' in sel:
                eps = self.rl_eps
                self.log.emit(f"Training Q-Learning ({eps} episodes, V3 Fast) ...")
                qa = QLearningAgent(g, gl,
                                    alpha=0.3, gamma=0.95,
                                    epsilon_start=1.0,
                                    epsilon_decay=0.9995,
                                    epsilon_min=0.01,
                                    goal_reward=1000.0,
                                    step_reward=-1.0,
                                    wall_penalty=-10.0)
                t = time.time()
                qa.train(num_episodes=eps,
                         max_steps_per_episode=1000,
                         start_fixed=s)
                qp, _ = qa.extract_path(s)
                ql_t  = time.time() - t
                self.log.emit(
                    f"QL   -> path={len(qp) if qp else 0}  "
                    f"trained={qa.episodes_trained}ep  t={ql_t:.3f}s  "
                    f"best_train={qa.best_path_length if qa.best_path_found else 'N/A'}")
                R['Q-Learning'] = dict(path=qp, nodes=qa.episodes_trained,
                                       time=ql_t,
                                       color_bgr=ALGO_COLORS_BGR['Q-Learning'],
                                       qagent=qa)

            # ---- NN + A* --------------------------------------------
            if 'NN+A*' in sel:
                self.log.emit("Training Neural Network classifier ...")
                gl2  = self.gray.tolist()
                X, y, _ = build_training_data(gl2, threshold=200, sample_rate=0.03)
                nn = NeuralNetClassifier(input_size=9, hidden_size=16, lr=0.05)
                t  = time.time()
                nn.train(X, y, epochs=20)
                nt = time.time() - t
                self.log.emit(f"   NN trained  loss={nn.train_losses[-1]:.4f}")

                t2       = time.time()
                nn_grid  = nn_predict_grid(nn, gl2, threshold=0.5)
                nn_arr   = np.array(nn_grid, dtype=np.int8)
                nn_arr[:3,:]  = 1; nn_arr[-3:,:] = 1
                nn_arr[:,:3]  = 1; nn_arr[:,-3:] = 1
                nn_list  = nn_arr.tolist()
                ns = fix_to_walkable(nn_list, s)
                ng = fix_to_walkable(nn_list, gl)
                np2, nn2 = AStar(nn_list, heuristic='manhattan').search(ns, ng)
                if not np2:
                    self.log.emit("NN+A* failed on NN grid, falling back to original grid...")
                    np2, nn2 = AStar(g, heuristic='manhattan').search(s, gl)
                R['NN+A*'] = dict(path=np2, nodes=nn2,
                                  time=time.time()-t2+nt,
                                  color_bgr=ALGO_COLORS_BGR['NN+A*'],
                                  nn_obj=nn)
                self.log.emit(f"NN+A*-> path={len(np2) if np2 else 'None'}  nodes={nn2}")

            self.log.emit("------------------------------")
            self.log.emit(f"Done! ({len(R)} algorithms)")
            self.done.emit(R)

        except Exception:
            import traceback
            self.err.emit(traceback.format_exc())


# ===========================================================================
#  Canvas & utility
# ===========================================================================
class Canvas(FigureCanvas):
    def __init__(self, w=14, h=8, dpi=90):
        self.fig = Figure(figsize=(w, h), dpi=dpi,
                          facecolor=DARK_BG, tight_layout=True)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


class ClickLabel(QLabel):
    clicked_norm = pyqtSignal(float, float)
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and self.pixmap():
            pm = self.pixmap()
            pm_rect = pm.rect()
            pm_rect.moveCenter(self.rect().center())
            if not pm_rect.contains(ev.pos()):
                return
            nx = (ev.x() - pm_rect.x()) / pm_rect.width()
            ny = (ev.y() - pm_rect.y()) / pm_rect.height()
            self.clicked_norm.emit(nx, ny)
        super().mousePressEvent(ev)


def path_px(path):
    if not path: return 0.0
    return round(sum(
        math.sqrt((path[i][0]-path[i-1][0])**2 + (path[i][1]-path[i-1][1])**2)
        for i in range(1, len(path))), 1)


def score_algo(r):
    if not r.get('path'):
        return float('inf')
    return (0.40 * len(r['path']) +
            0.35 * (r.get('nodes') or 0) +
            0.25 * r.get('time', 0) * 10000)


def best_algo(results):
    valid = {k: v for k, v in results.items() if v.get('path')}
    if not valid:
        return None, "No algorithm found a path"
    scores    = {k: score_algo(v) for k, v in valid.items()}
    winner    = min(scores, key=scores.get)
    w         = valid[winner]
    min_path  = min(len(v['path']) for v in valid.values())
    min_nodes = min((v.get('nodes') or 0) for v in valid.values())
    min_time  = min(v.get('time', 0) for v in valid.values())
    reasons   = []
    if len(w['path']) == min_path:   reasons.append(f"shortest path ({min_path} cells)")
    if (w.get('nodes') or 0) == min_nodes: reasons.append(f"fewest nodes ({min_nodes})")
    if abs(w.get('time',0) - min_time) < 0.001: reasons.append(f"fastest ({min_time:.4f}s)")
    reason = " + ".join(reasons) if reasons else "best composite score"
    return winner, reason


# ===========================================================================
#  MAIN DASHBOARD
# ===========================================================================
class Dashboard(QMainWindow):

    STYLE = f"""

QMainWindow, QWidget {{
    background-color: rgba(17, 24, 39, 0.88);
    color: {TEXT};
    font-family: 'Inter';
    font-size: 12px;
}}

QFrame {{
    background: {PANEL_BG};
    border-radius: 16px;
}}

QGroupBox {{
    background-color: rgba(17, 24, 39, 0.88);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    margin-top: 16px;
    padding-top: 18px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    color: {ACCENT};
}}


QFrame {{
    background-color: rgba(17, 24, 39, 0.92);
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.05);
}}


QPushButton {{
    background-color: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 12px;
    color: white;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: rgba(255,255,255,0.08);
}}

QPushButton#run {{
    background-color: #2563eb;
    border: none;
    border-radius: 12px;
    color: white;
    font-size: 14px;
    font-weight: 700;
    padding: 14px;
}}

QPushButton#run:hover {{
    background-color: #3b82f6;
}}

QTabWidget::pane {{
    border: none;
    background: transparent;
}}

QTabBar::tab {{
    background-color: transparent;
    color: #94a3b8;

    min-width: 140px;
    min-height: 42px;

    padding: 12px 20px;

    margin-right: 8px;

    border-radius: 12px;

    font-size: 12px;
    font-weight: 600;
}}

QTabBar::tab:selected {{
    background-color: rgba(255,255,255,0.08);
    color: white;
}}

QTabBar::tab:hover {{
    background-color: rgba(255,255,255,0.05);
}}

QTextEdit {{
    background: #081120;
    border:1px solid #1e293b;
    border-radius: 14px;
    padding: 10px;
    color: #38ff88;
    font-family: 'Inter';
}}

QComboBox,
QSpinBox {{
    background: {CARD_BG};
    border:1px solid #1e293b;
    border-radius: 10px;
    padding: 8px;
    min-height: 20px;
}}

QComboBox:hover,
QSpinBox:hover {{
    border: 1px solid {ACCENT};
}}

QCheckBox {{
    spacing: 10px;
    font-weight: 600;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border:1px solid #1e293b;
    background: {CARD_BG};
}}

QCheckBox::indicator:checked {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
}}

# QTableWidget {{
#     background: {PANEL_BG};
#     border:1px solid #1e293b;
#     border-radius: 14px;
#     gridline-color: {BORDER};

#     alternate-background-color: #172033;
#     color: white;
# }}

QTableWidget {{
    background: #121a2b;
    alternate-background-color: #172033;
    color: white;
    border:1px solid #1e293b;
    border-radius:14px;
    gridline-color:#202b45;
}}

QTableWidget::item {{
    padding: 6px;
    border: none;
}}

QTableWidget::item:selected {{
    background:#2563eb;
    color:white;
}}

QHeaderView::section {{
    background: {CARD_BG};
    color: {ACCENT};
    padding: 12px;
    border: none;
    font-weight: bold;
}}

QScrollArea {{
    border: none;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
}}

QScrollBar::handle:vertical {{
    background: {ACCENT};
    border-radius: 5px;
    min-height: 30px;
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0px;
}}

"""

    def __init__(self):
        super().__init__()
        self.setWindowOpacity(0.98)
        self.setWindowTitle("Vision Path Planner - V3 Fast")
        self.resize(1600, 900)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(self.STYLE)

        self.img = self.gray = self.binary = self.grid = None
        self.start = self.goal = self.results = None
        self._click_mode = None

        self.output_dir    = os.path.join(BASE_DIR, "outputs")
        self.individual_dir = os.path.join(self.output_dir, "individual_paths")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.individual_dir, exist_ok=True)
        self._run_timestamp = None

        self._build_ui()
        self._status("Upload an image to begin")

    # -----------------------------------------------------------------------
    def _build_ui(self):
        c = QWidget(); self.setCentralWidget(c)
        root = QVBoxLayout(c); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        hdr = QFrame(); hdr.setFixedHeight(72)
        hdr.setStyleSheet(
        '''
        background-color: qlineargradient(
            x1:0,y1:0,
            x2:1,y2:0,
            stop:0 #0f172a,
            stop:1 #111827
        );

        border-bottom: 1px solid rgba(255,255,255,0.06);
        '''
        )
        hl  = QHBoxLayout(hdr); hl.setContentsMargins(20,0,20,0)
        t1  = QLabel("Vision-Based Grid Environment Modeling and Optimal Path Planning Using Heuristic Search Algorithms")
        t1.setStyleSheet(
        f"""
        font-size:28px;
        font-weight:800;
        color:{TEXT};
        letter-spacing:1px;
        """
        )
        t2  = QLabel("")
        t2.setStyleSheet(f"font-size:11px;color:{MUTED};")
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet(f"font-size:12px;color:{ACCENT};font-weight:bold;")
        hl.addWidget(t1); hl.addSpacing(12);
        hl.addStretch(); hl.addWidget(self._status_lbl)
        root.addWidget(hdr)

        body = QSplitter(Qt.Horizontal)

        # LEFT PANEL SCROLL AREA
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setFrameShape(QFrame.NoFrame)

        left_panel_widget = self._left_panel()
        left_scroll.setWidget(left_panel_widget)

        body.addWidget(left_scroll)
        body.addWidget(self._right_tabs())
        body.setSizes([280, 1400])

        root.addWidget(body, 1)

        self.statusBar().setStyleSheet(
            f"background:{PANEL_BG};color:{MUTED};border-top:1px solid {BORDER};")
        

    # -----------------------------------------------------------------------
    def _left_panel(self):
        p = QWidget()
        p.setMinimumWidth(260)
        p.setMaximumWidth(300)
        p.setStyleSheet(
    '''
    background-color:#0b1120;
    border-right:1px solid rgba(255,255,255,0.05);
    '''
)
        lay = QVBoxLayout(p); lay.setContentsMargins(14,14,14,14); lay.setSpacing(10)

        # Image
        g = QGroupBox("Image Input"); gl_ = QVBoxLayout(g)
        self._upload_btn = QPushButton("Upload Floorplan / Maze / Map Image")
        self._upload_btn.clicked.connect(self._upload)
        self._img_lbl = QLabel("No image loaded")
        self._img_lbl.setStyleSheet(f"color:{MUTED};font-size:10px;")
        self._img_lbl.setWordWrap(True)
        gl_.addWidget(self._upload_btn); gl_.addWidget(self._img_lbl)
        lay.addWidget(g)

        # Points
        g2 = QGroupBox("Start & Goal"); g2l = QVBoxLayout(g2)
        hint = QLabel("1. Set Start -> click image\n2. Set Goal  -> click image\nOR press Auto-Select")
        hint.setStyleSheet(f"color:{MUTED};font-size:10px;"); g2l.addWidget(hint)
        self._btn_start = QPushButton("Set Start"); self._btn_start.setEnabled(False)
        self._btn_start.clicked.connect(lambda: self._arm_click('start'))
        self._btn_goal  = QPushButton("Set Goal");  self._btn_goal.setEnabled(False)
        self._btn_goal.clicked.connect(lambda: self._arm_click('goal'))
        self._btn_auto  = QPushButton("Auto-Select Points"); self._btn_auto.setEnabled(False)
        self._btn_auto.clicked.connect(self._auto_pts)
        self._pts_lbl   = QLabel("Start: -\nGoal:  -")
        self._pts_lbl.setStyleSheet(f"color:{MUTED};font-size:10px;font-family:monospace;")
        self._click_hint = QLabel("")
        self._click_hint.setStyleSheet(f"color:{ORANGE};font-size:11px;")
        self._click_hint.setWordWrap(True)
        for w in [self._btn_start, self._btn_goal, self._btn_auto,
                  self._pts_lbl, self._click_hint]:
            g2l.addWidget(w)
        lay.addWidget(g2)

        # ---- ALGORITHM SELECTOR ----------------------------------------
        g_sel = QGroupBox("Select Algorithms to Run")
        g_sel_l = QVBoxLayout(g_sel)

        # Quick presets row
        preset_row = QHBoxLayout()
        self._btn_all    = QPushButton("All");    self._btn_all.setFixedHeight(24)
        self._btn_fast   = QPushButton("Fast");   self._btn_fast.setFixedHeight(24)
        self._btn_rl_only = QPushButton("RL Only"); self._btn_rl_only.setFixedHeight(24)
        self._btn_classic = QPushButton("Classic"); self._btn_classic.setFixedHeight(24)
        for b in [self._btn_all, self._btn_fast, self._btn_rl_only, self._btn_classic]:
            b.setStyleSheet(f"font-size:10px;padding:2px 6px;")
        self._btn_all.clicked.connect(lambda: self._preset('all'))
        self._btn_fast.clicked.connect(lambda: self._preset('fast'))
        self._btn_rl_only.clicked.connect(lambda: self._preset('rl'))
        self._btn_classic.clicked.connect(lambda: self._preset('classic'))
        preset_row.addWidget(self._btn_all)
        preset_row.addWidget(self._btn_fast)
        preset_row.addWidget(self._btn_rl_only)
        preset_row.addWidget(self._btn_classic)
        g_sel_l.addLayout(preset_row)

        self._algo_checks = {}
        for name in ALL_ALGOS:
            cb = QCheckBox(name)
            cb.setChecked(True)
            hex_c = ALGO_COLORS_HEX.get(name, ACCENT)
            cb.setStyleSheet(
                f"QCheckBox{{color:{hex_c};font-weight:bold;}}"
                f"QCheckBox::indicator:checked{{background:{hex_c};border-color:{hex_c};}}"
            )
            if name in ('MDP', 'Q-Learning'):
                cb.setChecked(False)   # unchecked by default (slow)
            self._algo_checks[name] = cb
            g_sel_l.addWidget(cb)

        lay.addWidget(g_sel)

        # Settings
        g3 = QGroupBox("Settings"); g3l = QGridLayout(g3)
        g3l.addWidget(QLabel("RL Episodes:"), 0, 0)
        self._rl_spin = QSpinBox()
        self._rl_spin.setRange(50, 5000); self._rl_spin.setValue(500); self._rl_spin.setSingleStep(50)
        g3l.addWidget(self._rl_spin, 0, 1)
        g3l.addWidget(QLabel("White Threshold:"), 1, 0)
        self._thresh_spin = QSpinBox()
        self._thresh_spin.setRange(100, 254); self._thresh_spin.setValue(200)
        g3l.addWidget(self._thresh_spin, 1, 1)
        g3l.addWidget(QLabel("Image Type:"), 2, 0)
        self._img_type = QComboBox()
        self._img_type.addItems(["Auto Detect","Simple (B&W)","Complex (3D Render)",
                                  "Satellite/Map (Roads)","Occupancy Grid (Robotics)"])
        g3l.addWidget(self._img_type, 2, 1)
        g3l.addWidget(QLabel("Invert Grid:"), 3, 0)
        self._invert_check = QComboBox()
        self._invert_check.addItems(["Auto","Normal (White=Walkable)","Inverted (Black=Walkable)"])
        g3l.addWidget(self._invert_check, 3, 1)
        lay.addWidget(g3)

        self._run_btn = QPushButton("Run Selected Algorithms")
        self._run_btn.setObjectName("run"); self._run_btn.setFixedHeight(46)
        self._run_btn.setEnabled(False); self._run_btn.clicked.connect(self._run)
        lay.addWidget(self._run_btn)

        self._prog = QProgressBar(); self._prog.setRange(0, 0); self._prog.setVisible(False)
        lay.addWidget(self._prog)

        g4 = QGroupBox("Live Log"); g4l = QVBoxLayout(g4)
        self._log = QTextEdit(); self._log.setReadOnly(True); self._log.setMinimumHeight(120)
        g4l.addWidget(self._log)
        lay.addWidget(g4)
        lay.addStretch()
        return p

    # -----------------------------------------------------------------------
    def _preset(self, mode):
        checks = self._algo_checks
        if mode == 'all':
            for cb in checks.values(): cb.setChecked(True)
        elif mode == 'fast':
            fast = {'BFS', 'DFS', 'A*(Manhattan)', 'A*(Euclidean)', 'A*(Chebyshev)', 'NN+A*'}
            for k, cb in checks.items(): cb.setChecked(k in fast)
        elif mode == 'rl':
            for k, cb in checks.items(): cb.setChecked(k in {'MDP', 'Q-Learning'})
        elif mode == 'classic':
            classic = {'BFS', 'DFS', 'A*(Manhattan)', 'A*(Euclidean)', 'A*(Chebyshev)'}
            for k, cb in checks.items(): cb.setChecked(k in classic)

    def _selected_algos(self):
        return {k for k, cb in self._algo_checks.items() if cb.isChecked()}

    # -----------------------------------------------------------------------
    def _right_tabs(self):
        self._tabs = QTabWidget()
        self._tabs.addTab(self._tab_grid(),         "Grid View")
        self._canvas_compare = Canvas(14, 10)
        self._tabs.addTab(self._wrap_canvas(self._canvas_compare), "Algorithm Comparison")
        self._canvas_heat = Canvas(14, 7)
        self._tabs.addTab(self._wrap_canvas(self._canvas_heat),    "MDP & RL Heatmaps")
        self._canvas_train = Canvas(14, 6)
        self._tabs.addTab(self._wrap_canvas(self._canvas_train),   "Training Curves")
        self._canvas_heur = Canvas(14, 6)
        self._tabs.addTab(self._wrap_canvas(self._canvas_heur),    "Heuristic Analysis")
        self._tabs.addTab(self._tab_perf(),         "Performance Table")
        self._tabs.addTab(self._tab_best(),         "Best Algorithm")
        self._tabs.setUsesScrollButtons(True)
        self._tabs.tabBar().setElideMode(Qt.ElideNone)
        self._tabs.setTabPosition(QTabWidget.North)
        self._tabs.setMovable(True)
        return self._tabs

    def _wrap_canvas(self, canvas):
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(4,4,4,4)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border:none;")
        inner = QWidget(); il = QVBoxLayout(inner); il.setContentsMargins(0,0,0,0)
        il.addWidget(canvas); scroll.setWidget(inner); l.addWidget(scroll)
        return w

    def _tab_grid(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(10,10,10,10)
        info = QLabel("Upload image -> Set Start + Goal -> Run Selected Algorithms")
        info.setStyleSheet(f"color:{MUTED};font-size:11px;"); lay.addWidget(info)
        panels = QHBoxLayout(); panels.setSpacing(8)
        def make_panel(title):
            frm = QFrame()
            frm.setStyleSheet(
            f"""
            background:{PANEL_BG};
            border:1px solid {BORDER};
            border-radius:18px;
            padding:10px;
            """
            )
            vl = QVBoxLayout(frm); vl.setContentsMargins(6,6,6,6)
            hdr = QLabel(title); hdr.setAlignment(Qt.AlignCenter)
            hdr.setStyleSheet(f"color:{ACCENT};font-weight:bold;font-size:11px;margin-bottom:4px;")
            vl.addWidget(hdr)
            lbl = ClickLabel(); lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(250, 220)
            lbl.setScaledContents(False)
            lbl.setStyleSheet(
            '''
            background-color: #020617;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.06);
            padding: 12px;
            '''
            )
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setCursor(Qt.CrossCursor); vl.addWidget(lbl)
            return frm, lbl
        f1, self._lbl_orig  = make_panel("Original Image")
        f2, self._lbl_bin   = make_panel("Binary Grid (Walkable=White)")
        f3, self._lbl_blend = make_panel("Grid Overlay")
        for lbl in [self._lbl_orig, self._lbl_bin, self._lbl_blend]:
            lbl.clicked_norm.connect(self._img_clicked)
        panels.addWidget(f1); panels.addWidget(f2); panels.addWidget(f3)
        lay.addLayout(panels, 1)
        return w

    def _tab_perf(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(16,16,16,16)
        t = QLabel("Full Algorithm Performance Table")
        t.setStyleSheet(f"font-size:15px;font-weight:bold;color:{TEXT};margin-bottom:8px;")
        lay.addWidget(t)
        self._perf_table = QTableWidget()
        self._perf_table.setColumnCount(7)
        self._perf_table.setHorizontalHeaderLabels(
            ["Algorithm","Path (cells)","Path (px)","Nodes Explored","Time (s)","Efficiency","Optimal?"])
        self._perf_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._perf_table.setAlternatingRowColors(True)
        lay.addWidget(self._perf_table, 1)
        return w
    


    def _tab_best(self):

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24,24,24,24)
        lay.setSpacing(18)

        title = QLabel("Best Algorithm Analysis")
        title.setStyleSheet(
            f"font-size:18px;font-weight:bold;color:{TEXT};margin-bottom:8px;"
        )
        lay.addWidget(title)

        self._best_main = QLabel("Run the algorithms to see results")
        self._best_main.setStyleSheet(
            f"font-size:36px;font-weight:bold;color:{ACCENT};"
            f"background:{PANEL_BG};border:2px solid {ACCENT};"
            f"border-radius:12px;padding:20px;"
        )

        self._best_main.setAlignment(Qt.AlignCenter)
        self._best_main.setWordWrap(True)

        lay.addWidget(self._best_main)

        self._best_reason = QLabel("")
        self._best_reason.setStyleSheet(
            f"font-size:13px;color:{MUTED};margin-top:8px;"
        )

        self._best_reason.setAlignment(Qt.AlignCenter)
        self._best_reason.setWordWrap(True)

        lay.addWidget(self._best_reason)

        rt = QLabel(
            "Full Ranking (composite: path 40% + nodes 35% + time 25%)"
        )

        rt.setStyleSheet(
            f"font-size:12px;color:{MUTED};margin-top:20px;margin-bottom:6px;"
        )

        lay.addWidget(rt)

        self._rank_table = QTableWidget()
        self._rank_table.setColumnCount(5)

        self._rank_table.setHorizontalHeaderLabels([
            "Rank",
            "Algorithm",
            "Path (cells)",
            "Nodes",
            "Time (s)"
        ])

        self._rank_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        self._rank_table.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Minimum
        )
        self._rank_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._rank_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._rank_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        lay.addWidget(self._rank_table)

        vc = QLabel("Best Algorithm Path:")

        vc.setStyleSheet(
            f"font-size:18px;color:{MUTED};margin-top:16px;"
        )

        lay.addWidget(vc)

        self._best_img_lbl = QLabel()

        self._best_img_lbl.setAlignment(Qt.AlignCenter)

        self._best_img_lbl.setMinimumHeight(700)

        self._best_img_lbl.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        self._best_img_lbl.setScaledContents(False)

        self._best_img_lbl.setStyleSheet(
            f"""
            background:#010409;
            border-radius:6px;
            border:1px solid {BORDER};
            padding:10px;
            """
        )

        lay.addWidget(self._best_img_lbl)

        scroll.setWidget(w)

        return scroll


    # def _tab_best(self):
    #     w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(24,24,24,24)
    #     title = QLabel("Best Algorithm Analysis")
    #     title.setStyleSheet(f"font-size:18px;font-weight:bold;color:{TEXT};margin-bottom:8px;")
    #     lay.addWidget(title)
    #     self._best_main = QLabel("Run the algorithms to see results")
    #     self._best_main.setStyleSheet(
    #         f"font-size:36px;font-weight:bold;color:{ACCENT};"
    #         f"background:{PANEL_BG};border:2px solid {ACCENT};"
    #         f"border-radius:12px;padding:20px;")
    #     self._best_main.setAlignment(Qt.AlignCenter); self._best_main.setWordWrap(True)
    #     lay.addWidget(self._best_main)
    #     self._best_reason = QLabel("")
    #     self._best_reason.setStyleSheet(f"font-size:13px;color:{MUTED};margin-top:8px;")
    #     self._best_reason.setAlignment(Qt.AlignCenter); self._best_reason.setWordWrap(True)
    #     lay.addWidget(self._best_reason)
    #     rt = QLabel("Full Ranking (composite: path 40% + nodes 35% + time 25%)")
    #     rt.setStyleSheet(f"font-size:12px;color:{MUTED};margin-top:20px;margin-bottom:6px;")
    #     lay.addWidget(rt)
    #     self._rank_table = QTableWidget()
    #     self._rank_table.setColumnCount(5)
    #     self._rank_table.setHorizontalHeaderLabels(["Rank","Algorithm","Path (cells)","Nodes","Time (s)"])
    #     self._rank_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    #     self._rank_table.setMaximumHeight(380); lay.addWidget(self._rank_table)
    #     vc = QLabel("Best Algorithm Path:")
    #     vc.setStyleSheet(f"font-size:12px;color:{MUTED};margin-top:16px;"); lay.addWidget(vc)
    #     self._best_img_lbl = QLabel()
    #     self._best_img_lbl.setAlignment(Qt.AlignCenter)

    #     # RESPONSIVE IMAGE AREA
    #     self._best_img_lbl.setMinimumHeight(450)
    #     self._best_img_lbl.setSizePolicy(
    #         QSizePolicy.Expanding,
    #         QSizePolicy.Expanding
    #     )

    #     self._best_img_lbl.setScaledContents(False)

    #     self._best_img_lbl.setStyleSheet(
    #         f"""
    #         background:#010409;
    #         border-radius:6px;
    #         border:1px solid {BORDER};
    #         padding:10px;
    #         """
    #     )

    #     lay.addWidget(self._best_img_lbl, 1)
    #     return w

    # -----------------------------------------------------------------------
    def _upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.jpg *.jpeg *.png *.bmp *.tif)")
        if path: self._load(path)

    def _load(self, path):
        try:
            img = cv2.imread(path)
            if img is None: raise ValueError("Could not read image")
            self.img = img
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            mode = self._img_type.currentText()
            invert_mode = self._invert_check.currentText()
            if mode == "Auto Detect":
                unique = len(np.unique(gray))
                if unique < 20 and np.mean(gray) > 200:    mode = "Occupancy Grid (Robotics)"
                elif unique > 150 and len(img.shape) == 3: mode = "Satellite/Map (Roads)"
                elif unique > 80:                          mode = "Complex (3D Render)"
                else:                                      mode = "Simple (B&W)"
                self._log_msg(f"Auto-detect: {mode}")
            if mode == "Occupancy Grid (Robotics)":
                self.gray = gray
                _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  np.ones((2,2), np.uint8))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
                if invert_mode == "Auto":
                    if np.sum(binary > 127) / binary.size < 0.3: binary = 255 - binary
                elif invert_mode == "Inverted (Black=Walkable)": binary = 255 - binary
            elif mode == "Complex (3D Render)":
                self.gray = gray; proc = gray.copy(); scale = 1.0
                if max(h, w) > 800:
                    scale = 800 / max(h, w)
                    proc = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                smooth = cv2.bilateralFilter(proc, 9, 75, 75)
                edges  = cv2.Canny(smooth, 40, 120)
                kernel = np.ones((5, 5), np.uint8)
                walls  = cv2.dilate(edges, kernel, iterations=2)
                walls  = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
                _, binary = cv2.threshold(walls, 1, 255, cv2.THRESH_BINARY_INV)
                h_ff, w_ff = binary.shape
                mask = np.zeros((h_ff+2, w_ff+2), np.uint8)
                for sx, sy in [(0,0),(w_ff-1,0),(0,h_ff-1),(w_ff-1,h_ff-1)]:
                    if 0<=sx<w_ff and 0<=sy<h_ff and binary[sy,sx]==255:
                        cv2.floodFill(binary, mask, (sx, sy), 0)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
                if scale != 1.0: binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                self.gray = gray
                _, binary = cv2.threshold(gray, self._thresh_spin.value(), 255, cv2.THRESH_BINARY)
            self.binary = binary
            grid_arr = (binary > 127).astype(np.int8)
            self.grid = grid_arr.tolist()
            rows, cols = len(self.grid), len(self.grid[0])
            wk = int(grid_arr.sum())
            self._img_lbl.setText(f"OK  {os.path.basename(path)}\n{rows}x{cols}  |  {wk} walkable cells")
            self._log_msg(f"Loaded: {os.path.basename(path)}  {rows}x{cols}")
            self._refresh_grid_panels()
            for b in [self._btn_start, self._btn_goal, self._btn_auto]: b.setEnabled(True)
            self._status("Image loaded - set start & goal")
        except Exception as e:
            import traceback; self._log_msg(f"ERROR: {e}\n{traceback.format_exc()}")

    def _refresh_grid_panels(self):
        if self.img is None: return
        h, w = self.img.shape[:2]; r = max(4, w // 100)
        def mark(im):
            d = im.copy()
            if d.ndim == 2: d = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
            if self.start: cv2.circle(d,(self.start[1],self.start[0]),r,(0,0,255),-1); cv2.circle(d,(self.start[1],self.start[0]),r,(255,255,255),1)
            if self.goal:  cv2.circle(d,(self.goal[1], self.goal[0]), r,(0,255,0),-1); cv2.circle(d,(self.goal[1], self.goal[0]), r,(255,255,255),1)
            return d
        self._cv2_to_qlabel(self._lbl_orig,  mark(self.img))
        self._cv2_to_qlabel(self._lbl_bin,   mark(self.binary))
        blend = cv2.addWeighted(self.img, 0.35,
            cv2.cvtColor(np.array(self.grid,dtype=np.uint8)*255, cv2.COLOR_GRAY2BGR), 0.65, 0)
        self._cv2_to_qlabel(self._lbl_blend, mark(blend))

    def _cv2_to_qlabel(self, lbl, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); h, w, _ = rgb.shape
        qi  = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
        px  = QPixmap.fromImage(qi)
        lw, lh = lbl.width(), lbl.height()
        if lw > 10 and lh > 10: px = px.scaled(lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lbl.setPixmap(px)

    def _arm_click(self, mode):
        self._click_mode = mode
        if mode == 'start':
            self._click_hint.setText("Click on image -> place START")
            self._btn_start.setStyleSheet(f"background:#3d1f1f;border:1px solid {RED};")
            self._btn_goal.setStyleSheet("")
        else:
            self._click_hint.setText("Click on image -> place GOAL")
            self._btn_goal.setStyleSheet(f"background:#1f3d1f;border:1px solid {GREEN};")
            self._btn_start.setStyleSheet("")

    def _img_clicked(self, nx, ny):
        if self.grid is None or self._click_mode is None: return
        rows, cols = len(self.grid), len(self.grid[0])
        pt = fix_to_walkable(self.grid,
                             (max(0,min(rows-1,int(ny*rows))),
                              max(0,min(cols-1,int(nx*cols)))))
        if self._click_mode == 'start':
            self.start = pt; self._log_msg(f"Start -> {pt}")
            self._click_mode = 'goal'
            self._click_hint.setText("Now click image -> place GOAL")
            self._btn_start.setStyleSheet(""); self._btn_goal.setStyleSheet(f"background:#1f3d1f;border:1px solid {GREEN};")
        else:
            self.goal = pt; self._log_msg(f"Goal  -> {pt}")
            self._click_mode = None
            self._click_hint.setText("Points set - press Run Selected Algorithms")
            self._btn_goal.setStyleSheet("")
        self._pts_lbl.setText(f"Start: {self.start or '-'}\nGoal:  {self.goal or '-'}")
        self._refresh_grid_panels()
        if self.start and self.goal: self._run_btn.setEnabled(True)

    def _auto_pts(self):
        if self.grid is None: return
        self.start, self.goal = auto_select_points(self.grid)
        self._log_msg(f"Auto -> Start:{self.start}  Goal:{self.goal}")
        self._click_hint.setText("Auto-selected - press Run Selected Algorithms")
        self._pts_lbl.setText(f"Start: {self.start}\nGoal:  {self.goal}")
        self._refresh_grid_panels(); self._run_btn.setEnabled(True)

    def _run(self):
        if not all([self.grid, self.start, self.goal]):
            self._log_msg("WARNING: Load image and set start/goal first"); return
        sel = self._selected_algos()
        if not sel:
            self._log_msg("WARNING: No algorithms selected!"); return

        self._run_btn.setEnabled(False); self._prog.setVisible(True)
        self._log.clear()
        self._log_msg(f"V3 Fast — Running {len(sel)} algorithm(s): {', '.join(sorted(sel))}")
        if 'MDP' in sel:
            self._log_msg("  MDP: pre-built trans table + heapq path extraction")
        if 'Q-Learning' in sel:
            self._log_msg(f"  Q-Learning: {self._rl_spin.value()} eps, early-stop enabled")
        self._status("Running ...")
        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._worker = AlgoWorker(
            self.grid, self.img, self.gray,
            self.start, self.goal,
            self._rl_spin.value(), sel)
        self._worker.log.connect(self._log_msg)
        self._worker.done.connect(self._on_done)
        self._worker.err.connect(self._on_err)
        self._worker.start()

    def _on_done(self, R):
        self.results = R
        self._prog.setVisible(False); self._run_btn.setEnabled(True); self._status("Complete!")
        self._render_compare(R); self._render_heatmaps(R)
        self._render_training(R); self._render_heuristics(R)
        self._render_perf_table(R); self._render_best(R)
        self._tabs.setCurrentIndex(1)

    def _on_err(self, msg):
        self._prog.setVisible(False); self._run_btn.setEnabled(True)
        self._log_msg(f"ERROR:\n{msg}"); self._status("Error - check log")

    def _save_fig(self, fig, name, subfolder=None):
        if self._run_timestamp is None: return
        folder = self.individual_dir if subfolder == "individual" else self.output_dir
        fname  = os.path.join(folder, f"{self._run_timestamp}_{name}.png")
        fig.savefig(fname, dpi=150, facecolor=DARK_BG, edgecolor='none', bbox_inches='tight')
        self._log_msg(f"Saved: {fname}")

    def _save_algo_image(self, name, bgr_image):
        if self._run_timestamp is None: return
        fname = os.path.join(self.individual_dir, f"{self._run_timestamp}_{name}_path.png")
        cv2.imwrite(fname, bgr_image); self._log_msg(f"Saved: {fname}")

    def _draw(self, path, color_bgr, thickness=2):
        vis = self.img.copy()
        if path and len(path) > 1:
            pts = np.array([(c,r) for r,c in path], np.int32)
            cv2.polylines(vis, [pts.reshape(-1,1,2)], False, color_bgr, thickness)
        r = max(4, self.img.shape[1]//80)
        for pt, col in [(self.start,(0,0,255)),(self.goal,(0,255,0))]:
            cv2.circle(vis,(pt[1],pt[0]),r,col,-1); cv2.circle(vis,(pt[1],pt[0]),r,(255,255,255),1)
        return vis

    def _render_compare(self, R):
        c = self._canvas_compare; c.fig.clear()
        c.fig.patch.set_facecolor(DARK_BG)
        names = list(R.keys()); n = len(names); nc = 4; nr = math.ceil(n/nc)
        gs = gridspec.GridSpec(nr, nc, figure=c.fig, hspace=0.55, wspace=0.12)
        c.fig.suptitle("Vision Path Planner — Algorithm Comparison (V3 Fast)",
                        color='white', fontsize=14, fontweight='bold')
        for i, name in enumerate(names):
            ax = c.fig.add_subplot(gs[i//nc, i%nc]); rv = R[name]
            vis = self._draw(rv['path'], rv['color_bgr'])
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            pl = len(rv['path']) if rv['path'] else 0
            ax.set_title(name, color='white', fontsize=8, fontweight='bold', pad=2)
            ax.set_xlabel(f"Path:{pl}  Nodes:{rv.get('nodes',0)}\n{rv.get('time',0):.4f}s",
                          color='#aaa', fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            hx = ALGO_COLORS_HEX.get(name, '#58a6ff')
            for sp in ax.spines.values(): sp.set_edgecolor(hx); sp.set_linewidth(2)
            self._save_algo_image(name.replace('(','').replace(')','').replace('*','star'), vis)
        for i in range(n, nr*nc): c.fig.add_subplot(gs[i//nc, i%nc]).axis('off')
        self._save_fig(c.fig, "01_algorithm_comparison"); c.draw()

    def _render_heatmaps(self, R):
        c = self._canvas_heat; c.fig.clear(); c.fig.patch.set_facecolor(DARK_BG)
        axes = c.fig.subplots(1, 2)
        rows, cols = len(self.grid), len(self.grid[0])
        omask = np.array([[self.grid[r][c]==0 for c in range(cols)] for r in range(rows)])
        if 'MDP' in R and 'mdp_obj' in R['MDP']:
            vg = np.zeros((rows, cols))
            for (r2,c2), v in R['MDP']['mdp_obj'].V.items(): vg[r2][c2] = v
            vg[omask] = np.nan
            im = axes[0].imshow(vg, cmap='plasma', interpolation='nearest')
            axes[0].set_title("MDP Value V(s)\nheapq path extraction (V3)", color='white', fontsize=10)
            c.fig.colorbar(im, ax=axes[0]).set_label("V(s)", color='white')
        else:
            axes[0].set_title("MDP not run", color=MUTED, fontsize=10)
        axes[0].axis('off'); axes[0].set_facecolor('#111')
        if 'Q-Learning' in R and 'qagent' in R['Q-Learning']:
            qg = np.array(R['Q-Learning']['qagent'].get_q_value_grid())
            qg[omask] = np.nan
            im2 = axes[1].imshow(qg, cmap='viridis', interpolation='nearest')
            axes[1].set_title("Q-Learning max Q(s,a)\nearly-stop + bounded BFS (V3)", color='white', fontsize=10)
            c.fig.colorbar(im2, ax=axes[1]).set_label("max Q", color='white')
        else:
            axes[1].set_title("Q-Learning not run", color=MUTED, fontsize=10)
        axes[1].axis('off')
        c.fig.suptitle("MDP & RL — Value Maps", color='white', fontsize=13, fontweight='bold')
        self._save_fig(c.fig, "02_mdp_rl_heatmaps"); c.draw()

    def _render_training(self, R):
        c = self._canvas_train; c.fig.clear(); c.fig.patch.set_facecolor(DARK_BG)
        ax1, ax2 = c.fig.subplots(1, 2)
        for ax in [ax1, ax2]:
            ax.set_facecolor(PANEL_BG); ax.tick_params(colors='white')
            ax.grid(True, alpha=0.2, color='#444')
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        if 'Q-Learning' in R and 'qagent' in R['Q-Learning']:
            rw = R['Q-Learning']['qagent'].episode_rewards
            ax1.plot(rw, alpha=0.3, color='#58a6ff', lw=0.7, label='Reward')
            if len(rw) > 20:
                w = 20
                sm = np.convolve(rw, np.ones(w)/w, mode='valid')
                ax1.plot(range(w-1, len(rw)), sm, color='#f78166', lw=2, label=f'Avg-{w}')
            ax1.axhline(0, color='gray', lw=0.5, ls='--')
            ax1.legend(facecolor=PANEL_BG, labelcolor='white')
            ax1.set_title(
                f"Q-Learning Training Curve\n({R['Q-Learning']['qagent'].episodes_trained} eps trained)",
                color='white', fontsize=10)
        else:
            ax1.set_title("Q-Learning not run", color=MUTED, fontsize=10)
        ax1.set_xlabel("Episode", color='white'); ax1.set_ylabel("Reward", color='white')
        if 'NN+A*' in R and 'nn_obj' in R['NN+A*']:
            ax2.plot(R['NN+A*']['nn_obj'].train_losses, color=GREEN, lw=2, marker='o', markersize=3)
            ax2.set_title("Neural Network Loss", color='white', fontsize=10)
        else:
            ax2.set_title("NN+A* not run", color=MUTED, fontsize=10)
        ax2.set_xlabel("Epoch", color='white'); ax2.set_ylabel("BCE Loss", color='white')
        c.fig.suptitle("Training Curves", color='white', fontsize=13, fontweight='bold')
        self._save_fig(c.fig, "03_training_curves"); c.draw()

    def _render_heuristics(self, R):
        c = self._canvas_heur; c.fig.clear(); c.fig.patch.set_facecolor(DARK_BG)
        hr = {k: v for k, v in R.items() if k.startswith('A*')}
        if not hr:
            ax = c.fig.add_subplot(111); ax.set_facecolor(PANEL_BG)
            ax.text(0.5, 0.5, "No A* variants selected", ha='center', va='center',
                    color=MUTED, transform=ax.transAxes, fontsize=14)
            ax.axis('off'); c.draw(); return
        axes = c.fig.subplots(1, 3); names = list(hr.keys())
        short = [n.replace('A*(','').replace(')','') for n in names]; bc = ['#58a6ff','#3fb950','#f78166']
        def bar(ax, vals, ylabel, title):
            ax.set_facecolor(PANEL_BG)
            bars = ax.bar(range(len(names)), vals, color=bc[:len(names)])
            ax.set_xticks(range(len(names))); ax.set_xticklabels(short, rotation=20, ha='right', color='white', fontsize=9)
            ax.set_ylabel(ylabel, color='white'); ax.set_title(title, color='white', fontsize=10)
            ax.tick_params(colors='white'); ax.grid(True, alpha=0.2, axis='y', color='#444')
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            for b, v in zip(bars, vals):
                ax.text(b.get_x()+b.get_width()/2, b.get_height(), f'{v:.1f}',
                        ha='center', va='bottom', color='white', fontsize=8)
        bar(axes[0], [hr[h]['nodes'] for h in names], 'Nodes Explored', 'Nodes Explored')
        bar(axes[1], [hr[h]['time']*1000 for h in names], 'Time (ms)', 'Exec Time (ms)')
        bar(axes[2], [len(hr[h]['path']) if hr[h]['path'] else 0 for h in names], 'Path (cells)', 'Path Length')
        c.fig.suptitle("A* Heuristic Comparison", color='white', fontsize=13, fontweight='bold')
        self._save_fig(c.fig, "04_heuristic_comparison"); c.draw()

    def _render_perf_table(self, R):
        tbl = self._perf_table; tbl.setRowCount(0)
        OPTIMAL = {'BFS','A*(Manhattan)','A*(Euclidean)','A*(Chebyshev)'}
        for name, r in R.items():
            row = tbl.rowCount(); tbl.insertRow(row)
            pl = len(r['path']) if r.get('path') else 0
            pp = path_px(r.get('path')); nd = r.get('nodes', 0) or 0
            ts = r.get('time', 0); eff = f"{pl/nd:.5f}" if pl and nd else '-'
            opt = "Yes" if name in OPTIMAL else "Approx"
            vals = [name, str(pl) if pl else 'NO PATH', str(pp), str(nd), f"{ts:.4f}", eff, opt]
            hex_c = ALGO_COLORS_HEX.get(name, '#444')
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val); item.setTextAlignment(Qt.AlignCenter)
                if row % 2 == 1:
                    item.setForeground(QColor("#000000"))
                else:
                    item.setForeground(QColor("#ffffff"))
                if col == 0:
                    item.setForeground(QColor(hex_c)); item.setFont(QFont("Segoe UI", 10, QFont.Bold))
                if not r.get('path'): item.setForeground(QColor(RED))
                tbl.setItem(row, col, item)

    def _render_best(self, R):
        winner, reason = best_algo(R)
        if winner is None: self._best_main.setText("No path found by any algorithm"); return
        hex_c = ALGO_COLORS_HEX.get(winner, ACCENT); wdata = R[winner]
        pl = len(wdata['path']) if wdata.get('path') else 0
        nd = wdata.get('nodes', 0); ts = wdata.get('time', 0)
        self._best_main.setStyleSheet(
            f"font-size:32px;font-weight:bold;color:{hex_c};"
            f"background:{PANEL_BG};border:3px solid {hex_c};background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #111827,stop:1 #0f172a);"
            
            f"border-radius:12px;padding:20px;")
        self._best_main.setText(
            f"BEST: {winner}\n\nPath: {pl} cells  |  Nodes: {nd}  |  Time: {ts:.4f}s")
        self._best_reason.setText(
            f"Why best?  ->  {reason}\n\n"
            f"Scoring: 40% path length + 35% nodes explored + 25% time")
        valid   = [(k,v) for k,v in R.items() if v.get('path')]
        ranked  = sorted(valid, key=lambda kv: score_algo(kv[1]))
        all_algos = [(k,v,True) for k,v in ranked] + [(k,v,False) for k,v in R.items() if not v.get('path')]
        self._rank_table.setRowCount(0); rank_i = 0
        for k, v, found in all_algos:
            row = self._rank_table.rowCount(); self._rank_table.insertRow(row)
            rank_str = str(rank_i+1) if found else "-"
            pl2 = len(v['path']) if found else "NO PATH"
            nd2 = v.get('nodes', 0); ts2 = v.get('time', 0)
            for col, val in enumerate([rank_str, k, str(pl2), str(nd2), f"{ts2:.4f}"]):
                item = QTableWidgetItem(val); item.setTextAlignment(Qt.AlignCenter)
                if k == winner:
                    item.setBackground(QColor(hex_c+"33")); item.setFont(QFont("Segoe UI", 10, QFont.Bold))
                self._rank_table.setItem(row, col, item)
            if found: rank_i += 1
        if wdata.get('path') and self.img is not None:
            vis = self._draw(wdata['path'], wdata['color_bgr'], thickness=3)
            h, w = vis.shape[:2]
            cv2.putText(vis, f"BEST: {winner}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            self._save_algo_image("BEST_"+winner.replace('(','').replace(')','').replace('*','star'), vis)
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            qi  = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
            px  = QPixmap.fromImage(qi)
            lw, lh = self._best_img_lbl.width(), self._best_img_lbl.height()
            if lw > 10 and lh > 10:
                px = px.scaled(
                lw - 20,
                lh - 20,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
                self._rank_table.resizeRowsToContents()

        table_height = (
            self._rank_table.horizontalHeader().height()
            + self._rank_table.rowCount() * self._rank_table.rowHeight(0)
            + 10
        )

        self._rank_table.setMinimumHeight(table_height)
        self._rank_table.setMaximumHeight(table_height)
        self._best_img_lbl.setPixmap(px)

    def _log_msg(self, msg):
        self._log.append(msg)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _status(self, msg):
        self._status_lbl.setText(msg)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.img is not None: QTimer.singleShot(80, self._refresh_grid_panels)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = Dashboard()
    win.showMaximized()
    sys.exit(app.exec_())