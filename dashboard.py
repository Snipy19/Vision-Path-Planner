"""
==============================================================================
       VISION PATH PLANNER - Interactive AI Dashboard (V4)
==============================================================================
NEW in V4:
  1. Bidirectional A*  — meet-in-the-middle search (new algorithm)
  2. Dynamic Obstacles — click to paint/erase walls live, auto re-route
  3. Step-by-step Animation — watch any algorithm expand node-by-node
==============================================================================
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
    QTableWidgetItem, QHeaderView, QComboBox, QCheckBox, QSlider
)
from PyQt5.QtCore  import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui   import QPixmap, QImage, QFont, QColor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from algorithms.bfs            import BFS
from algorithms.dfs            import DFS
from algorithms.astar          import AStar, HEURISTIC_INFO
from algorithms.bidirectional  import BidirectionalAStar
from algorithms.dynamic_obstacles import DynamicObstacleManager
from mdp.mdp                   import MDP
from mdp.qlearning             import QLearningAgent
from neural.nn_classifier      import NeuralNetClassifier, build_training_data, nn_predict_grid
from utils.point_selector      import auto_select_points, fix_to_walkable
from image_processing.grid     import ImageToGrid

# ── Colours ────────────────────────────────────────────────────────────────
DARK_BG  = "#0b1020"; PANEL_BG = "#121a2b"; CARD_BG  = "#172033"
BORDER   = "#202b45"; ACCENT   = "#6ea8fe"; ACCENT2  = "#8b5cf6"
TEXT     = "#f8fafc"; MUTED    = "#94a3b8"
GREEN    = "#22c55e"; RED      = "#ef4444"; ORANGE   = "#f59e0b"
MAX_GRID_DIMENSION = 160

ALGO_COLORS_BGR = {
    'BFS':              (255, 80,  20),
    'DFS':              (20,  140, 255),
    'A*(Manhattan)':    (30,  200, 60),
    'A*(Euclidean)':    (60,  170, 30),
    'A*(Chebyshev)':    (34,  197, 94),
    'Bidir A*':         (255, 165, 0),
    'MDP':              (0,   200, 200),
    'Q-Learning':       (0,   180, 255),
    'NN+A*':            (168, 85,  247),
}
ALGO_COLORS_HEX = {k: f"#{v[2]:02x}{v[1]:02x}{v[0]:02x}" for k, v in ALGO_COLORS_BGR.items()}

ALL_ALGOS = ['BFS','DFS','A*(Manhattan)','A*(Euclidean)','A*(Chebyshev)',
             'Bidir A*','MDP','Q-Learning','NN+A*']

STYLE = f"""
QMainWindow,QWidget{{background:{DARK_BG};color:{TEXT};font-family:'Inter',sans-serif;font-size:12px;}}
QFrame{{background:{PANEL_BG};border-radius:16px;border:1px solid rgba(255,255,255,0.05);}}
QGroupBox{{background:rgba(17,24,39,0.88);border:1px solid rgba(255,255,255,0.06);border-radius:16px;margin-top:16px;padding-top:18px;}}
QGroupBox::title{{subcontrol-origin:margin;left:14px;padding:0 8px;color:{ACCENT};}}
QPushButton{{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:10px;color:white;font-weight:600;}}
QPushButton:hover{{background:rgba(255,255,255,0.09);}}
QPushButton#run{{background:#2563eb;border:none;border-radius:12px;color:white;font-size:14px;font-weight:700;padding:14px;}}
QPushButton#run:hover{{background:#3b82f6;}}
QPushButton#run:disabled{{background:#1c2128;color:{MUTED};}}
QTabWidget::pane{{border:none;background:transparent;}}
QTabBar::tab{{background:transparent;color:{MUTED};min-width:120px;min-height:38px;padding:10px 16px;margin-right:6px;border-radius:10px;font-size:11px;font-weight:600;}}
QTabBar::tab:selected{{background:rgba(255,255,255,0.08);color:white;}}
QTabBar::tab:hover{{background:rgba(255,255,255,0.05);}}
QTextEdit{{background:#081120;border:1px solid #1e293b;border-radius:12px;padding:8px;color:#38ff88;font-family:'Consolas',monospace;font-size:11px;}}
QComboBox,QSpinBox{{background:{CARD_BG};border:1px solid #1e293b;border-radius:10px;padding:6px;min-height:20px;color:{TEXT};}}
QComboBox:hover,QSpinBox:hover{{border:1px solid {ACCENT};}}
QCheckBox{{spacing:8px;font-weight:600;}}
QCheckBox::indicator{{width:16px;height:16px;border-radius:4px;border:1px solid #1e293b;background:{CARD_BG};}}
QCheckBox::indicator:checked{{background:{ACCENT};border:1px solid {ACCENT};}}
QTableWidget{{background:{PANEL_BG};border:1px solid #1e293b;border-radius:12px;gridline-color:{BORDER};}}
QHeaderView::section{{background:{CARD_BG};color:{ACCENT};padding:10px;border:none;font-weight:bold;}}
QScrollArea{{border:none;}}
QScrollBar:vertical{{background:transparent;width:8px;}}
QScrollBar::handle:vertical{{background:{ACCENT};border-radius:4px;min-height:24px;}}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0px;}}
QSlider::groove:horizontal{{height:4px;background:{BORDER};border-radius:2px;}}
QSlider::handle:horizontal{{background:{ACCENT};width:14px;height:14px;border-radius:7px;margin:-5px 0;}}
QSlider::sub-page:horizontal{{background:{ACCENT};border-radius:2px;}}
"""

# ══════════════════════════════════════════════════════════════════════════════
#  Animation Worker — yields explored nodes one-by-one
# ══════════════════════════════════════════════════════════════════════════════
class AnimWorker(QThread):
    step  = pyqtSignal(object, object, int)   # (explored_so_far, path_so_far, node_count)
    done  = pyqtSignal(object, int, float)    # (final_path, nodes, time)
    err   = pyqtSignal(str)

    def __init__(self, algo_name, grid, start, goal, delay_ms=30):
        super().__init__()
        self.algo_name  = algo_name
        self.grid       = grid
        self.start      = start
        self.goal       = goal
        self.delay_ms   = delay_ms
        self._paused    = False
        self._stop      = False

    def pause(self):  self._paused = True
    def resume(self): self._paused = False
    def stop(self):   self._stop   = True

    def _bfs_gen(self):
        from collections import deque
        grid, start, goal = self.grid, self.start, self.goal
        rows, cols = len(grid), len(grid[0])
        moves = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        queue = deque([start]); visited = {start}; parent = {start: None}
        t0 = time.time()
        while queue:
            cur = queue.popleft()
            yield cur, None
            if cur == goal:
                path = []; n = goal
                while n: path.append(n); n = parent[n]
                path.reverse()
                return path, time.time()-t0
            r,c = cur
            for dr,dc in moves:
                nr,nc = r+dr,c+dc
                nxt=(nr,nc)
                if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]==1 and nxt not in visited:
                    visited.add(nxt); parent[nxt]=cur; queue.append(nxt)
        return None, time.time()-t0

    def _astar_gen(self, heuristic='manhattan'):
        import heapq, math
        def h(a,b):
            dr,dc=abs(a[0]-b[0]),abs(a[1]-b[1])
            return max(dr,dc)+(math.sqrt(2)-1)*min(dr,dc)
        grid,start,goal=self.grid,self.start,self.goal
        rows,cols=len(grid),len(grid[0])
        moves=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        heap=[(h(start,goal),0,start)]; g={start:0.0}; par={start:None}
        closed=set(); cnt=0; t0=time.time()
        while heap:
            _,__,cur=heapq.heappop(heap)
            if cur in closed: continue
            closed.add(cur)
            yield cur, None
            if cur==goal:
                path=[]; n=goal
                while n is not None: path.append(n); n=par[n]
                path.reverse()
                return path,time.time()-t0
            r,c=cur
            for dr,dc in moves:
                nr,nc=r+dr,c+dc
                nxt=(nr,nc)
                if not(0<=nr<rows and 0<=nc<cols): continue
                if grid[nr][nc]==0 or nxt in closed: continue
                if dr!=0 and dc!=0:
                    if grid[r+dr][c]==0 or grid[r][c+dc]==0: continue
                cost=math.sqrt(2) if dr!=0 and dc!=0 else 1.0
                ng=g[cur]+cost
                if ng<g.get(nxt,float('inf')):
                    g[nxt]=ng; par[nxt]=cur; cnt+=1
                    heapq.heappush(heap,(ng+h(nxt,goal),cnt,nxt))
        return None,time.time()-t0

    def _dfs_gen(self):
        grid,start,goal=self.grid,self.start,self.goal
        rows,cols=len(grid),len(grid[0])
        moves=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        stack=[start]; visited=set(); parent={start:None}; t0=time.time()
        while stack:
            cur=stack.pop()
            if cur in visited: continue
            visited.add(cur)
            yield cur,None
            if cur==goal:
                path=[]; n=goal
                while n is not None: path.append(n); n=parent[n]
                path.reverse()
                return path,time.time()-t0
            r,c=cur
            for dr,dc in moves:
                nr,nc=r+dr,c+dc
                nxt=(nr,nc)
                if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]==1 and nxt not in visited:
                    stack.append(nxt)
                    if nxt not in parent: parent[nxt]=cur
        return None,time.time()-t0

    def run(self):
        try:
            explored = []
            name = self.algo_name
            if name == 'BFS':      gen = self._bfs_gen()
            elif name == 'DFS':    gen = self._dfs_gen()
            else:                  gen = self._astar_gen()

            final_path = None
            elapsed    = 0.0
            t0         = time.time()

            for item in gen:
                if self._stop: return
                while self._paused:
                    if self._stop: return
                    time.sleep(0.05)

                if isinstance(item, tuple) and len(item)==2 and isinstance(item[1], list):
                    # generator returned (path, time)  — Python generators can't really return mid-iteration
                    # handle via StopIteration below
                    pass
                else:
                    node, _ = item
                    explored.append(node)
                    self.step.emit(list(explored), None, len(explored))
                    time.sleep(self.delay_ms / 1000.0)

            # Extract final path — re-run fast to get result
            if name == 'BFS':
                final_path, n = BFS(self.grid).search(self.start, self.goal)
            elif name == 'DFS':
                final_path, n = DFS(self.grid).search(self.start, self.goal)
            else:
                final_path, n = AStar(self.grid, heuristic='octile').search(self.start, self.goal)

            elapsed = time.time() - t0
            self.step.emit(list(explored), final_path, len(explored))
            self.done.emit(final_path, len(explored), elapsed)

        except Exception:
            import traceback
            self.err.emit(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
#  Algorithm Worker (background, all algos)
# ══════════════════════════════════════════════════════════════════════════════
class AlgoWorker(QThread):
    log  = pyqtSignal(str)
    done = pyqtSignal(dict)
    err  = pyqtSignal(str)

    def __init__(self, grid, img, gray, start, goal, rl_eps, selected):
        super().__init__()
        self.grid=grid; self.img=img; self.gray=gray
        self.start_pt=start; self.goal_pt=goal
        self.rl_eps=rl_eps; self.selected=selected

    def run(self):
        try:
            R={}; g=self.grid; s=self.start_pt; gl=self.goal_pt; sel=self.selected

            if 'BFS' in sel:
                self.log.emit("Running BFS ...")
                t=time.time(); p,n=BFS(g).search(s,gl)
                R['BFS']=dict(path=p,nodes=n,time=time.time()-t,color_bgr=ALGO_COLORS_BGR['BFS'])
                self.log.emit(f"BFS  -> path={len(p) if p else 'None'}  nodes={n}")

            if 'DFS' in sel:
                self.log.emit("Running DFS ...")
                t=time.time(); p,n=DFS(g).search(s,gl)
                R['DFS']=dict(path=p,nodes=n,time=time.time()-t,color_bgr=ALGO_COLORS_BGR['DFS'])
                self.log.emit(f"DFS  -> path={len(p) if p else 'None'}  nodes={n}")

            for h in ['manhattan','euclidean','chebyshev']:
                lbl=f"A*({h.capitalize()})"
                if lbl not in sel: continue
                self.log.emit(f"Running {lbl} ...")
                t=time.time(); p,n=AStar(g,heuristic=h).search(s,gl)
                R[lbl]=dict(path=p,nodes=n,time=time.time()-t,color_bgr=ALGO_COLORS_BGR.get(lbl,(50,200,80)))
                self.log.emit(f"{lbl}  path={len(p) if p else 'None'}  nodes={n}")

            if 'Bidir A*' in sel:
                self.log.emit("Running Bidirectional A* ...")
                t=time.time(); p,n=BidirectionalAStar(g).search(s,gl)
                R['Bidir A*']=dict(path=p,nodes=n,time=time.time()-t,color_bgr=ALGO_COLORS_BGR['Bidir A*'])
                self.log.emit(f"Bidir A* -> path={len(p) if p else 'None'}  nodes={n}")

            if 'MDP' in sel:
                self.log.emit("Running MDP + Value Iteration ...")
                mdp=MDP(g,gl,gamma=0.95,slip_prob=0.10,convergence_thresh=1e-3,max_iterations=300)
                t=time.time(); iters=mdp.value_iteration()
                mp,_=mdp.extract_path(s); mdp_t=time.time()-t
                self.log.emit(f"MDP  -> {iters} iters  path={len(mp) if mp else 0}  t={mdp_t:.3f}s")
                R['MDP']=dict(path=mp,nodes=len(mdp.states),time=mdp_t,
                              color_bgr=ALGO_COLORS_BGR['MDP'],mdp_obj=mdp,mdp_iters=iters)

            if 'Q-Learning' in sel:
                eps=self.rl_eps
                self.log.emit(f"Training Q-Learning ({eps} episodes) ...")
                qa=QLearningAgent(g,gl,alpha=0.3,gamma=0.95,epsilon_start=1.0,
                                  epsilon_decay=0.9995,epsilon_min=0.01,
                                  goal_reward=1000.0,step_reward=-1.0,wall_penalty=-10.0)
                t=time.time(); qa.train(num_episodes=eps,max_steps_per_episode=1000,start_fixed=s)
                qp,_=qa.extract_path(s); ql_t=time.time()-t
                self.log.emit(f"QL   -> path={len(qp) if qp else 0}  t={ql_t:.3f}s")
                R['Q-Learning']=dict(path=qp,nodes=qa.episodes_trained,time=ql_t,
                                     color_bgr=ALGO_COLORS_BGR['Q-Learning'],qagent=qa)

            if 'NN+A*' in sel:
                self.log.emit("Training Neural Network classifier ...")
                gl2=self.gray.tolist()
                X,y,_=build_training_data(gl2,threshold=200,sample_rate=0.03)
                nn=NeuralNetClassifier(input_size=9,hidden_size=16,lr=0.05)
                t=time.time(); nn.train(X,y,epochs=20); nt=time.time()-t
                self.log.emit(f"   NN trained  loss={nn.train_losses[-1]:.4f}")
                t2=time.time()
                nn_grid=nn_predict_grid(nn,gl2,threshold=0.5)
                nn_arr=np.array(nn_grid,dtype=np.int8)
                nn_arr[:3,:]=1; nn_arr[-3:,:]=1; nn_arr[:,:3]=1; nn_arr[:,-3:]=1
                nn_list=nn_arr.tolist()
                ns=fix_to_walkable(nn_list,s); ng=fix_to_walkable(nn_list,gl)
                np2,nn2=AStar(nn_list,heuristic='manhattan').search(ns,ng)
                if not np2:
                    self.log.emit("NN+A* fallback to original grid ...")
                    np2,nn2=AStar(g,heuristic='manhattan').search(s,gl)
                R['NN+A*']=dict(path=np2,nodes=nn2,time=time.time()-t2+nt,
                                color_bgr=ALGO_COLORS_BGR['NN+A*'],nn_obj=nn)
                self.log.emit(f"NN+A*-> path={len(np2) if np2 else 'None'}")

            self.log.emit(f"Done! ({len(R)} algorithms)")
            self.done.emit(R)
        except Exception:
            import traceback; self.err.emit(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
#  Canvas & helpers
# ══════════════════════════════════════════════════════════════════════════════
class Canvas(FigureCanvas):
    def __init__(self, w=14, h=8, dpi=90):
        self.fig=Figure(figsize=(w,h),dpi=dpi,facecolor=DARK_BG,tight_layout=True)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)


class ClickLabel(QLabel):
    clicked_norm = pyqtSignal(float,float)
    def mousePressEvent(self,ev):
        if ev.button()==Qt.LeftButton and self.pixmap():
            pm=self.pixmap(); pm_rect=pm.rect(); pm_rect.moveCenter(self.rect().center())
            if not pm_rect.contains(ev.pos()): return
            nx=(ev.x()-pm_rect.x())/pm_rect.width()
            ny=(ev.y()-pm_rect.y())/pm_rect.height()
            self.clicked_norm.emit(nx,ny)
        super().mousePressEvent(ev)


def path_px(path):
    if not path: return 0.0
    return round(sum(math.sqrt((path[i][0]-path[i-1][0])**2+(path[i][1]-path[i-1][1])**2)
                     for i in range(1,len(path))),1)

def score_algo(r):
    if not r.get('path'): return float('inf')
    return 0.40*len(r['path'])+0.35*(r.get('nodes') or 0)+0.25*r.get('time',0)*10000

def best_algo(results):
    valid={k:v for k,v in results.items() if v.get('path')}
    if not valid: return None,"No algorithm found a path"
    scores={k:score_algo(v) for k,v in valid.items()}
    winner=min(scores,key=scores.get); w=valid[winner]
    min_path=min(len(v['path']) for v in valid.values())
    min_nodes=min((v.get('nodes') or 0) for v in valid.values())
    min_time=min(v.get('time',0) for v in valid.values())
    reasons=[]
    if len(w['path'])==min_path:     reasons.append(f"shortest path ({min_path} cells)")
    if (w.get('nodes') or 0)==min_nodes: reasons.append(f"fewest nodes ({min_nodes})")
    if abs(w.get('time',0)-min_time)<0.001: reasons.append(f"fastest ({min_time:.4f}s)")
    return winner," + ".join(reasons) if reasons else "best composite score"


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Path Planner — V4  |  Bidir A* · Dynamic Obstacles · Animation")
        self.resize(1650,950); self.setMinimumSize(1100,700)
        self.setStyleSheet(STYLE)

        self.img=self.gray=self.binary=self.grid=None
        self.start=self.goal=self.results=None
        self._click_mode=None
        self._obstacle_mode=False
        self._dm=None                  # DynamicObstacleManager

        # Animation state
        self._anim_worker=None
        self._anim_explored=[]
        self._anim_path=None

        self.output_dir=os.path.join(BASE_DIR,"outputs")
        self.individual_dir=os.path.join(self.output_dir,"individual_paths")
        os.makedirs(self.output_dir,exist_ok=True)
        os.makedirs(self.individual_dir,exist_ok=True)
        self._run_timestamp=None

        self._build_ui()
        self._status("Upload an image to begin")

    # ── UI build ─────────────────────────────────────────────────────────

    def _build_ui(self):
        c=QWidget(); self.setCentralWidget(c)
        root=QVBoxLayout(c); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        hdr=QFrame(); hdr.setFixedHeight(68)
        hdr.setStyleSheet("background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0f172a,stop:1 #111827);border-bottom:1px solid rgba(255,255,255,0.06);border-radius:0;")
        hl=QHBoxLayout(hdr); hl.setContentsMargins(20,0,20,0)
        t1=QLabel("Vision-Based Grid Environment Modeling and Optimal Path Planning Using Heuristic Search Algorithms")
        t1.setStyleSheet(f"font-size:18px;font-weight:800;color:{TEXT};letter-spacing:0.5px;")
        badge=QLabel("V4  |  Bidir A* · Dynamic Obstacles · Animation")
        badge.setStyleSheet(f"font-size:10px;color:{ACCENT};background:rgba(110,168,254,0.12);border-radius:6px;padding:3px 10px;")
        self._status_lbl=QLabel("Ready")
        self._status_lbl.setStyleSheet(f"font-size:12px;color:{ACCENT};font-weight:bold;")
        hl.addWidget(t1); hl.addSpacing(12); hl.addWidget(badge)
        hl.addStretch(); hl.addWidget(self._status_lbl)
        root.addWidget(hdr)

        body=QSplitter(Qt.Horizontal)
        ls=QScrollArea(); ls.setWidgetResizable(True)
        ls.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ls.setFrameShape(QFrame.NoFrame)
        ls.setWidget(self._left_panel())
        body.addWidget(ls); body.addWidget(self._right_tabs())
        body.setSizes([290,1360])
        root.addWidget(body,1)

        self.statusBar().setStyleSheet(f"background:{PANEL_BG};color:{MUTED};border-top:1px solid {BORDER};")

    def _left_panel(self):
        p=QWidget(); p.setMinimumWidth(265); p.setMaximumWidth(305)
        p.setStyleSheet(f"background:#0b1120;border-right:1px solid rgba(255,255,255,0.05);")
        lay=QVBoxLayout(p); lay.setContentsMargins(14,14,14,14); lay.setSpacing(10)

        # Image upload
        g=QGroupBox("Image Input"); gl_=QVBoxLayout(g)
        self._upload_btn=QPushButton("Upload Floorplan / Maze / Map")
        self._upload_btn.clicked.connect(self._upload)
        self._img_lbl=QLabel("No image loaded")
        self._img_lbl.setStyleSheet(f"color:{MUTED};font-size:10px;"); self._img_lbl.setWordWrap(True)
        gl_.addWidget(self._upload_btn); gl_.addWidget(self._img_lbl)
        lay.addWidget(g)

        # Points
        g2=QGroupBox("Start & Goal"); g2l=QVBoxLayout(g2)
        hint=QLabel("1. Set Start → click image\n2. Set Goal → click image\nOR press Auto-Select")
        hint.setStyleSheet(f"color:{MUTED};font-size:10px;"); g2l.addWidget(hint)
        self._btn_start=QPushButton("Set Start"); self._btn_start.setEnabled(False)
        self._btn_start.clicked.connect(lambda:self._arm_click('start'))
        self._btn_goal=QPushButton("Set Goal"); self._btn_goal.setEnabled(False)
        self._btn_goal.clicked.connect(lambda:self._arm_click('goal'))
        self._btn_auto=QPushButton("Auto-Select Points"); self._btn_auto.setEnabled(False)
        self._btn_auto.clicked.connect(self._auto_pts)
        self._pts_lbl=QLabel("Start: -\nGoal:  -")
        self._pts_lbl.setStyleSheet(f"color:{MUTED};font-size:10px;font-family:monospace;")
        self._click_hint=QLabel("")
        self._click_hint.setStyleSheet(f"color:{ORANGE};font-size:11px;"); self._click_hint.setWordWrap(True)
        for w in [self._btn_start,self._btn_goal,self._btn_auto,self._pts_lbl,self._click_hint]:
            g2l.addWidget(w)
        lay.addWidget(g2)

        # ── Dynamic Obstacles ──
        g_dyn=QGroupBox("Dynamic Obstacles  [NEW]"); g_dyn_l=QVBoxLayout(g_dyn)
        dyn_hint=QLabel("Enable → click image to paint/erase walls\nAlgorithms auto re-route on Run")
        dyn_hint.setStyleSheet(f"color:{MUTED};font-size:10px;"); g_dyn_l.addWidget(dyn_hint)
        self._obs_toggle=QPushButton("Enable Obstacle Mode")
        self._obs_toggle.setEnabled(False)
        self._obs_toggle.clicked.connect(self._toggle_obstacle_mode)
        self._obs_reset=QPushButton("Reset All Obstacles")
        self._obs_reset.setEnabled(False)
        self._obs_reset.clicked.connect(self._reset_obstacles)
        self._obs_lbl=QLabel("No obstacles added")
        self._obs_lbl.setStyleSheet(f"color:{MUTED};font-size:10px;")
        for w in [self._obs_toggle,self._obs_reset,self._obs_lbl]: g_dyn_l.addWidget(w)
        lay.addWidget(g_dyn)

        # Algo selector
        g_sel=QGroupBox("Select Algorithms"); g_sel_l=QVBoxLayout(g_sel)
        prow=QHBoxLayout()
        for lbl,mode in [("All","all"),("Fast","fast"),("RL","rl"),("Classic","classic")]:
            b=QPushButton(lbl); b.setFixedHeight(22)
            b.setStyleSheet("font-size:10px;padding:2px 6px;")
            b.clicked.connect(lambda _,m=mode:self._preset(m))
            prow.addWidget(b)
        g_sel_l.addLayout(prow)
        self._algo_checks={}
        for name in ALL_ALGOS:
            cb=QCheckBox(name); cb.setChecked(name not in ('MDP','Q-Learning'))
            hex_c=ALGO_COLORS_HEX.get(name,ACCENT)
            cb.setStyleSheet(f"QCheckBox{{color:{hex_c};font-weight:bold;}}QCheckBox::indicator:checked{{background:{hex_c};border-color:{hex_c};}}")
            self._algo_checks[name]=cb; g_sel_l.addWidget(cb)
        lay.addWidget(g_sel)

        # Settings
        g3=QGroupBox("Settings"); g3l=QGridLayout(g3)
        g3l.addWidget(QLabel("RL Episodes:"),0,0)
        self._rl_spin=QSpinBox(); self._rl_spin.setRange(50,5000); self._rl_spin.setValue(500); self._rl_spin.setSingleStep(50)
        g3l.addWidget(self._rl_spin,0,1)
        g3l.addWidget(QLabel("White Threshold:"),1,0)
        self._thresh_spin=QSpinBox(); self._thresh_spin.setRange(100,254); self._thresh_spin.setValue(200)
        g3l.addWidget(self._thresh_spin,1,1)
        g3l.addWidget(QLabel("Image Type:"),2,0)
        self._img_type=QComboBox()
        self._img_type.addItems(["Auto Detect","Simple (B&W)","Complex (3D Render)","Satellite/Map","Occupancy Grid"])
        g3l.addWidget(self._img_type,2,1)
        lay.addWidget(g3)

        self._run_btn=QPushButton("Run Selected Algorithms")
        self._run_btn.setObjectName("run"); self._run_btn.setFixedHeight(46)
        self._run_btn.setEnabled(False); self._run_btn.clicked.connect(self._run)
        lay.addWidget(self._run_btn)

        self._prog=QProgressBar(); self._prog.setRange(0,0); self._prog.setVisible(False)
        lay.addWidget(self._prog)

        g4=QGroupBox("Live Log"); g4l=QVBoxLayout(g4)
        self._log=QTextEdit(); self._log.setReadOnly(True); self._log.setMinimumHeight(100)
        g4l.addWidget(self._log)
        lay.addWidget(g4)
        lay.addStretch()
        return p

    def _right_tabs(self):
        self._tabs=QTabWidget()
        self._tabs.addTab(self._tab_grid(),              "Grid View")
        self._canvas_compare=Canvas(14,10)
        self._tabs.addTab(self._wrap_canvas(self._canvas_compare),"Algorithm Comparison")
        self._canvas_heat=Canvas(14,7)
        self._tabs.addTab(self._wrap_canvas(self._canvas_heat),   "MDP & RL Heatmaps")
        self._canvas_train=Canvas(14,6)
        self._tabs.addTab(self._wrap_canvas(self._canvas_train),  "Training Curves")
        self._canvas_heur=Canvas(14,6)
        self._tabs.addTab(self._wrap_canvas(self._canvas_heur),   "Heuristic Analysis")
        self._tabs.addTab(self._tab_perf(),              "Performance Table")
        self._tabs.addTab(self._tab_best(),              "Best Algorithm")
        self._tabs.addTab(self._tab_animation(),         "Step Animation  [NEW]")
        self._tabs.setUsesScrollButtons(True)
        self._tabs.tabBar().setElideMode(Qt.ElideNone)
        self._tabs.setMovable(True)
        return self._tabs

    def _wrap_canvas(self,canvas):
        w=QWidget(); l=QVBoxLayout(w); l.setContentsMargins(4,4,4,4)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setStyleSheet("border:none;")
        inner=QWidget(); il=QVBoxLayout(inner); il.setContentsMargins(0,0,0,0)
        il.addWidget(canvas); scroll.setWidget(inner); l.addWidget(scroll)
        return w

    def _tab_grid(self):
        w=QWidget(); lay=QVBoxLayout(w); lay.setContentsMargins(10,10,10,10)
        info=QLabel("Upload image → Set Start + Goal → Run Algorithms  |  Enable Obstacle Mode to paint walls")
        info.setStyleSheet(f"color:{MUTED};font-size:11px;"); lay.addWidget(info)
        panels=QHBoxLayout(); panels.setSpacing(8)
        def make_panel(title):
            frm=QFrame()
            frm.setStyleSheet(f"background:{PANEL_BG};border:1px solid {BORDER};border-radius:18px;padding:10px;")
            vl=QVBoxLayout(frm); vl.setContentsMargins(6,6,6,6)
            hdr=QLabel(title); hdr.setAlignment(Qt.AlignCenter)
            hdr.setStyleSheet(f"color:{ACCENT};font-weight:bold;font-size:11px;margin-bottom:4px;")
            vl.addWidget(hdr)
            lbl=ClickLabel(); lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(250,220); lbl.setScaledContents(False)
            lbl.setStyleSheet("background:#020617;border-radius:12px;border:1px solid rgba(255,255,255,0.06);padding:8px;")
            lbl.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
            lbl.setCursor(Qt.CrossCursor); vl.addWidget(lbl)
            return frm,lbl
        f1,self._lbl_orig =make_panel("Original Image")
        f2,self._lbl_bin  =make_panel("Binary Grid (White=Walkable)")
        f3,self._lbl_blend=make_panel("Grid Overlay")
        for lbl in [self._lbl_orig,self._lbl_bin,self._lbl_blend]:
            lbl.clicked_norm.connect(self._img_clicked)
        panels.addWidget(f1); panels.addWidget(f2); panels.addWidget(f3)
        lay.addLayout(panels,1)
        return w

    def _tab_perf(self):
        w=QWidget(); lay=QVBoxLayout(w); lay.setContentsMargins(16,16,16,16)
        t=QLabel("Full Algorithm Performance Table")
        t.setStyleSheet(f"font-size:15px;font-weight:bold;color:{TEXT};margin-bottom:8px;")
        lay.addWidget(t)
        self._perf_table=QTableWidget()
        self._perf_table.setColumnCount(7)
        self._perf_table.setHorizontalHeaderLabels(["Algorithm","Path (cells)","Path (px)","Nodes Explored","Time (s)","Efficiency","Optimal?"])
        self._perf_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._perf_table.setAlternatingRowColors(True)
        lay.addWidget(self._perf_table,1)
        return w

    def _tab_best(self):
        w=QWidget(); lay=QVBoxLayout(w); lay.setContentsMargins(24,24,24,24)
        lay.addWidget(QLabel("Best Algorithm Analysis"))
        self._best_main=QLabel("Run the algorithms to see results")
        self._best_main.setStyleSheet(f"font-size:32px;font-weight:bold;color:{ACCENT};background:{PANEL_BG};border:2px solid {ACCENT};border-radius:12px;padding:20px;")
        self._best_main.setAlignment(Qt.AlignCenter); self._best_main.setWordWrap(True)
        lay.addWidget(self._best_main)
        self._best_reason=QLabel("")
        self._best_reason.setStyleSheet(f"font-size:13px;color:{MUTED};margin-top:8px;")
        self._best_reason.setAlignment(Qt.AlignCenter); self._best_reason.setWordWrap(True)
        lay.addWidget(self._best_reason)
        lay.addWidget(QLabel("Full Ranking (path 40% + nodes 35% + time 25%)"))
        self._rank_table=QTableWidget(); self._rank_table.setColumnCount(5)
        self._rank_table.setHorizontalHeaderLabels(["Rank","Algorithm","Path (cells)","Nodes","Time (s)"])
        self._rank_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._rank_table.setMaximumHeight(340); lay.addWidget(self._rank_table)
        self._best_img_lbl=QLabel(); self._best_img_lbl.setAlignment(Qt.AlignCenter)
        self._best_img_lbl.setMinimumHeight(400)
        self._best_img_lbl.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        self._best_img_lbl.setStyleSheet(f"background:#010409;border-radius:6px;border:1px solid {BORDER};padding:8px;")
        lay.addWidget(self._best_img_lbl,1)
        return w

    # ── Animation Tab ─────────────────────────────────────────────────────
    def _tab_animation(self):
        w=QWidget(); lay=QVBoxLayout(w); lay.setContentsMargins(16,16,16,16); lay.setSpacing(10)

        # Controls row
        ctrl=QHBoxLayout()
        ctrl.addWidget(QLabel("Algorithm:"))
        self._anim_combo=QComboBox()
        self._anim_combo.addItems(['BFS','DFS','A* (octile)'])
        ctrl.addWidget(self._anim_combo)

        ctrl.addWidget(QLabel("Speed:"))
        self._anim_speed=QSlider(Qt.Horizontal)
        self._anim_speed.setRange(1,200); self._anim_speed.setValue(30)
        self._anim_speed.setFixedWidth(120)
        ctrl.addWidget(self._anim_speed)
        ctrl.addWidget(QLabel("ms/step"))

        self._anim_play=QPushButton("▶  Play")
        self._anim_play.setEnabled(False)
        self._anim_play.clicked.connect(self._anim_start)

        self._anim_pause=QPushButton("⏸  Pause")
        self._anim_pause.setEnabled(False)
        self._anim_pause.clicked.connect(self._anim_pause_resume)

        self._anim_reset_btn=QPushButton("⏹  Reset")
        self._anim_reset_btn.setEnabled(False)
        self._anim_reset_btn.clicked.connect(self._anim_reset)

        for b in [self._anim_play,self._anim_pause,self._anim_reset_btn]: ctrl.addWidget(b)
        ctrl.addStretch()
        lay.addLayout(ctrl)

        # Stats bar
        self._anim_stats=QLabel("Load image + set points → press Play")
        self._anim_stats.setStyleSheet(f"color:{MUTED};font-size:11px;")
        lay.addWidget(self._anim_stats)

        # Main image display
        self._anim_lbl=QLabel(); self._anim_lbl.setAlignment(Qt.AlignCenter)
        self._anim_lbl.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        self._anim_lbl.setMinimumHeight(500)
        self._anim_lbl.setStyleSheet("background:#010409;border-radius:12px;border:1px solid rgba(255,255,255,0.05);")
        lay.addWidget(self._anim_lbl,1)
        return w

    # ── Dynamic Obstacle Methods ──────────────────────────────────────────

    def _toggle_obstacle_mode(self):
        self._obstacle_mode=not self._obstacle_mode
        if self._obstacle_mode:
            self._obs_toggle.setText("✏  Obstacle Mode: ON  (click image)")
            self._obs_toggle.setStyleSheet(f"background:#7c2d12;border:1px solid {RED};color:white;font-weight:bold;border-radius:12px;padding:10px;")
            self._click_hint.setText("🔴 Obstacle Mode: Left-click to add wall, click existing wall to remove")
        else:
            self._obs_toggle.setText("Enable Obstacle Mode")
            self._obs_toggle.setStyleSheet("")
            self._click_hint.setText("")

    def _reset_obstacles(self):
        if self._dm:
            self._dm.reset()
            self.grid=self._dm.get_grid()
            self._obs_lbl.setText("All obstacles removed")
            self._refresh_grid_panels()
            self._log_msg("Dynamic obstacles reset")

    def _update_obs_label(self):
        if self._dm:
            stats=self._dm.stats()
            self._obs_lbl.setText(f"Added: {stats['dynamic_added']} cells  |  Total walls: {stats['obstacles']}")

    # ── Image handling ────────────────────────────────────────────────────

    def _upload(self):
        path,_=QFileDialog.getOpenFileName(self,"Select Image","","Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        if path: self._load(path)

    def _load(self,path):
        try:
            img=cv2.imread(path)
            if img is None: raise ValueError("Could not read image")
            # Keep interactive search responsive on camera/floorplan images
            # without changing the coordinate system used by the overlays.
            h0, w0 = img.shape[:2]
            if max(h0, w0) > MAX_GRID_DIMENSION:
                scale = MAX_GRID_DIMENSION / max(h0, w0)
                img = cv2.resize(img, (max(1, int(w0 * scale)),
                                       max(1, int(h0 * scale))),
                                 interpolation=cv2.INTER_AREA)
            self.img=img; gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); h,w=gray.shape
            mode=self._img_type.currentText()
            if mode=="Auto Detect":
                unique=len(np.unique(gray))
                if unique<20 and np.mean(gray)>200:    mode="Occupancy Grid"
                elif unique>150 and len(img.shape)==3: mode="Satellite/Map"
                elif unique>80:                         mode="Complex (3D Render)"
                else:                                   mode="Simple (B&W)"
                self._log_msg(f"Auto-detect: {mode}")
            if mode=="Occupancy Grid":
                self.gray=gray
                _,binary=cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
                binary=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
            elif mode=="Complex (3D Render)":
                # Use the same tested HSV/morphology segmenter as the CLI and
                # image-validation report, so GUI and headless results agree.
                segmenter = ImageToGrid("", white_thresh=self._thresh_spin.value,
                                        mode="3d")
                self.gray, binary = segmenter.preprocess(img, mode="3d")
            else:
                self.gray=gray
                _,binary=cv2.threshold(gray,self._thresh_spin.value(),255,cv2.THRESH_BINARY)
            self.binary=binary
            grid_arr=(binary>127).astype(np.int8)
            self.grid=grid_arr.tolist()
            # Init dynamic obstacle manager
            self._dm=DynamicObstacleManager(self.grid)
            rows,cols=len(self.grid),len(self.grid[0])
            wk=int(grid_arr.sum())
            self._img_lbl.setText(f"OK  {os.path.basename(path)}\n{rows}×{cols}  |  {wk} walkable cells")
            self._log_msg(f"Loaded: {os.path.basename(path)}  {rows}×{cols}")
            self._refresh_grid_panels()
            for b in [self._btn_start,self._btn_goal,self._btn_auto,self._obs_toggle,self._obs_reset]: b.setEnabled(True)
            self._status("Image loaded — set start & goal")
        except Exception as e:
            import traceback; self._log_msg(f"ERROR: {e}\n{traceback.format_exc()}")

    def _refresh_grid_panels(self):
        if self.img is None: return
        h,w=self.img.shape[:2]; r=max(4,w//100)
        # Build obstacle overlay
        current_grid=self._dm.get_grid() if self._dm else self.grid
        def mark(im,draw_obs=False):
            d=im.copy()
            if d.ndim==2: d=cv2.cvtColor(d,cv2.COLOR_GRAY2BGR)
            # Draw dynamic obstacles in red
            if draw_obs and self._dm:
                for (or_,oc) in self._dm.added_cells():
                    cv2.rectangle(d,(oc-1,or_-1),(oc+1,or_+1),(0,0,200),-1)
            if self.start: cv2.circle(d,(self.start[1],self.start[0]),r,(0,0,255),-1); cv2.circle(d,(self.start[1],self.start[0]),r,(255,255,255),1)
            if self.goal:  cv2.circle(d,(self.goal[1], self.goal[0]), r,(0,255,0),-1); cv2.circle(d,(self.goal[1], self.goal[0]), r,(255,255,255),1)
            return d
        self._cv2_to_qlabel(self._lbl_orig,mark(self.img,draw_obs=True))
        bin_img=cv2.cvtColor(np.array(current_grid,dtype=np.uint8)*255,cv2.COLOR_GRAY2BGR)
        self._cv2_to_qlabel(self._lbl_bin,mark(bin_img))
        blend=cv2.addWeighted(self.img,0.35,bin_img,0.65,0)
        self._cv2_to_qlabel(self._lbl_blend,mark(blend))

    def _cv2_to_qlabel(self,lbl,bgr):
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); h,w,_=rgb.shape
        qi=QImage(rgb.data,w,h,3*w,QImage.Format_RGB888)
        px=QPixmap.fromImage(qi)
        lw,lh=lbl.width(),lbl.height()
        if lw>10 and lh>10: px=px.scaled(lw,lh,Qt.KeepAspectRatio,Qt.SmoothTransformation)
        lbl.setPixmap(px)

    def _arm_click(self,mode):
        self._obstacle_mode=False
        self._obs_toggle.setStyleSheet("") if self._obs_toggle.isEnabled() else None
        self._click_mode=mode
        if mode=='start':
            self._click_hint.setText("Click on image → place START")
            self._btn_start.setStyleSheet(f"background:#3d1f1f;border:1px solid {RED};")
            self._btn_goal.setStyleSheet("")
        else:
            self._click_hint.setText("Click on image → place GOAL")
            self._btn_goal.setStyleSheet(f"background:#1f3d1f;border:1px solid {GREEN};")
            self._btn_start.setStyleSheet("")

    def _img_clicked(self,nx,ny):
        if self.grid is None: return
        rows,cols=len(self.grid),len(self.grid[0])
        r_click=max(0,min(rows-1,int(ny*rows)))
        c_click=max(0,min(cols-1,int(nx*cols)))

        # Obstacle mode: paint/erase wall
        if self._obstacle_mode and self._dm:
            self._dm.toggle(r_click,c_click)
            self.grid=self._dm.get_grid()
            self._update_obs_label()
            self._refresh_grid_panels()
            return

        if self._click_mode is None: return
        pt=fix_to_walkable(self.grid,(r_click,c_click))
        if self._click_mode=='start':
            self.start=pt; self._log_msg(f"Start → {pt}")
            self._click_mode='goal'
            self._click_hint.setText("Now click image → place GOAL")
            self._btn_start.setStyleSheet(""); self._btn_goal.setStyleSheet(f"background:#1f3d1f;border:1px solid {GREEN};")
        else:
            self.goal=pt; self._log_msg(f"Goal  → {pt}")
            self._click_mode=None
            self._click_hint.setText("Points set — press Run")
            self._btn_goal.setStyleSheet("")
        self._pts_lbl.setText(f"Start: {self.start or '-'}\nGoal:  {self.goal or '-'}")
        self._refresh_grid_panels()
        if self.start and self.goal:
            self._run_btn.setEnabled(True)
            self._anim_play.setEnabled(True)

    def _auto_pts(self):
        if self.grid is None: return
        self.start,self.goal=auto_select_points(self.grid)
        self._log_msg(f"Auto → Start:{self.start}  Goal:{self.goal}")
        self._click_hint.setText("Auto-selected — press Run")
        self._pts_lbl.setText(f"Start: {self.start}\nGoal:  {self.goal}")
        self._refresh_grid_panels()
        self._run_btn.setEnabled(True)
        self._anim_play.setEnabled(True)

    # ── Preset ────────────────────────────────────────────────────────────

    def _preset(self,mode):
        checks=self._algo_checks
        if mode=='all':
            for cb in checks.values(): cb.setChecked(True)
        elif mode=='fast':
            fast={'BFS','DFS','A*(Manhattan)','A*(Euclidean)','A*(Chebyshev)','Bidir A*','NN+A*'}
            for k,cb in checks.items(): cb.setChecked(k in fast)
        elif mode=='rl':
            for k,cb in checks.items(): cb.setChecked(k in {'MDP','Q-Learning'})
        elif mode=='classic':
            classic={'BFS','DFS','A*(Manhattan)','A*(Euclidean)','A*(Chebyshev)'}
            for k,cb in checks.items(): cb.setChecked(k in classic)

    def _selected_algos(self):
        return {k for k,cb in self._algo_checks.items() if cb.isChecked()}

    # ── Run ───────────────────────────────────────────────────────────────

    def _run(self):
        if not all([self.grid,self.start,self.goal]):
            self._log_msg("WARNING: Load image and set start/goal first"); return
        sel=self._selected_algos()
        if not sel:
            self._log_msg("WARNING: No algorithms selected!"); return
        # Use current dynamic grid
        current_grid=self._dm.get_grid() if self._dm else self.grid
        self._run_btn.setEnabled(False); self._prog.setVisible(True)
        self._log.clear()
        self._log_msg(f"Running {len(sel)} algorithm(s): {', '.join(sorted(sel))}")
        if self._dm and self._dm.added_count()>0:
            self._log_msg(f"Dynamic obstacles active: {self._dm.added_count()} cells blocked")
        self._status("Running ...")
        self._run_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        self._worker=AlgoWorker(current_grid,self.img,self.gray,self.start,self.goal,self._rl_spin.value(),sel)
        self._worker.log.connect(self._log_msg)
        self._worker.done.connect(self._on_done)
        self._worker.err.connect(self._on_err)
        self._worker.start()

    def _on_done(self,R):
        self.results=R; self._prog.setVisible(False); self._run_btn.setEnabled(True); self._status("Complete!")
        self._render_compare(R); self._render_heatmaps(R)
        self._render_training(R); self._render_heuristics(R)
        self._render_perf_table(R); self._render_best(R)
        self._tabs.setCurrentIndex(1)

    def _on_err(self,msg):
        self._prog.setVisible(False); self._run_btn.setEnabled(True)
        self._log_msg(f"ERROR:\n{msg}"); self._status("Error — check log")

    # ── Animation ─────────────────────────────────────────────────────────

    def _anim_start(self):
        if not all([self.grid,self.start,self.goal]):
            self._log_msg("Set start and goal first"); return
        self._anim_reset()
        name=self._anim_combo.currentText().replace(" (octile)","")
        delay=self._anim_speed.value()
        current_grid=self._dm.get_grid() if self._dm else self.grid
        self._anim_worker=AnimWorker(name,current_grid,self.start,self.goal,delay_ms=delay)
        self._anim_worker.step.connect(self._anim_on_step)
        self._anim_worker.done.connect(self._anim_on_done)
        self._anim_worker.err.connect(lambda e:self._log_msg(f"Anim error: {e}"))
        self._anim_worker.start()
        self._anim_play.setEnabled(False); self._anim_pause.setEnabled(True); self._anim_reset_btn.setEnabled(True)
        self._anim_stats.setText(f"Animating {name} ...")
        self._log_msg(f"Animation started: {name}")

    def _anim_pause_resume(self):
        if not self._anim_worker: return
        if self._anim_worker._paused:
            self._anim_worker.resume(); self._anim_pause.setText("⏸  Pause")
        else:
            self._anim_worker.pause(); self._anim_pause.setText("▶  Resume")

    def _anim_reset(self):
        if self._anim_worker and self._anim_worker.isRunning():
            self._anim_worker.stop(); self._anim_worker.wait(500)
        self._anim_explored=[]; self._anim_path=None
        self._anim_play.setEnabled(True); self._anim_pause.setEnabled(False)
        self._anim_pause.setText("⏸  Pause")
        self._anim_stats.setText("Reset — press Play to start")
        self._anim_lbl.clear()

    def _anim_on_step(self,explored,path,count):
        self._anim_explored=explored; self._anim_path=path
        self._draw_animation_frame(explored,path)
        self._anim_stats.setText(f"Nodes explored: {count}")

    def _anim_on_done(self,path,nodes,elapsed):
        self._anim_path=path
        self._draw_animation_frame(self._anim_explored,path)
        pl=len(path) if path else 0
        self._anim_stats.setText(f"Done! Nodes: {nodes}  Path: {pl} cells  Time: {elapsed:.3f}s")
        self._anim_play.setEnabled(True); self._anim_pause.setEnabled(False)
        self._log_msg(f"Animation done — nodes:{nodes} path:{pl} t:{elapsed:.3f}s")

    def _draw_animation_frame(self,explored,path):
        if self.img is None: return
        vis=self.img.copy()
        # Draw explored nodes as semi-transparent blue dots
        for (r,c) in explored:
            cv2.circle(vis,(c,r),1,(255,180,50),-1)
        # Draw path in bright green
        if path and len(path)>1:
            for i in range(1,len(path)):
                cv2.line(vis,(path[i-1][1],path[i-1][0]),(path[i][1],path[i][0]),(0,255,80),2)
        # Start/goal
        sz=max(4,self.img.shape[1]//80)
        if self.start: cv2.circle(vis,(self.start[1],self.start[0]),sz,(0,0,255),-1)
        if self.goal:  cv2.circle(vis,(self.goal[1], self.goal[0]), sz,(0,255,0),-1)
        # Show on label
        rgb=cv2.cvtColor(vis,cv2.COLOR_BGR2RGB); h,w,_=rgb.shape
        qi=QImage(rgb.data,w,h,3*w,QImage.Format_RGB888)
        px=QPixmap.fromImage(qi)
        lw,lh=self._anim_lbl.width(),self._anim_lbl.height()
        if lw>10 and lh>10: px=px.scaled(lw,lh,Qt.KeepAspectRatio,Qt.SmoothTransformation)
        self._anim_lbl.setPixmap(px)

    # ── Render helpers ────────────────────────────────────────────────────

    def _save_fig(self,fig,name):
        if not self._run_timestamp: return
        fname=os.path.join(self.output_dir,f"{self._run_timestamp}_{name}.png")
        fig.savefig(fname,dpi=150,facecolor=DARK_BG,edgecolor='none',bbox_inches='tight')
        self._log_msg(f"Saved: {os.path.basename(fname)}")

    def _save_algo_image(self,name,bgr):
        if not self._run_timestamp: return
        fname=os.path.join(self.individual_dir,f"{self._run_timestamp}_{name}_path.png")
        cv2.imwrite(fname,bgr)

    def _draw(self,path,color_bgr,thickness=2):
        vis=self.img.copy()
        if path and len(path)>1:
            pts=np.array([(c,r) for r,c in path],np.int32)
            cv2.polylines(vis,[pts.reshape(-1,1,2)],False,color_bgr,thickness)
        r=max(4,self.img.shape[1]//80)
        for pt,col in [(self.start,(0,0,255)),(self.goal,(0,255,0))]:
            cv2.circle(vis,(pt[1],pt[0]),r,col,-1); cv2.circle(vis,(pt[1],pt[0]),r,(255,255,255),1)
        return vis

    def _render_compare(self,R):
        c=self._canvas_compare; c.fig.clear(); c.fig.patch.set_facecolor(DARK_BG)
        names=list(R.keys()); n=len(names); nc=4; nr=math.ceil(n/nc)
        gs=gridspec.GridSpec(nr,nc,figure=c.fig,hspace=0.55,wspace=0.12)
        c.fig.suptitle("Vision Path Planner — Algorithm Comparison (V4)",color='white',fontsize=13,fontweight='bold')
        for i,name in enumerate(names):
            ax=c.fig.add_subplot(gs[i//nc,i%nc]); rv=R[name]
            vis=self._draw(rv['path'],rv['color_bgr'])
            ax.imshow(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB))
            pl=len(rv['path']) if rv['path'] else 0
            ax.set_title(name,color='white',fontsize=8,fontweight='bold',pad=2)
            ax.set_xlabel(f"Path:{pl}  Nodes:{rv.get('nodes',0)}\n{rv.get('time',0):.4f}s",color='#aaa',fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            hx=ALGO_COLORS_HEX.get(name,'#58a6ff')
            for sp in ax.spines.values(): sp.set_edgecolor(hx); sp.set_linewidth(2)
            self._save_algo_image(name.replace('(','').replace(')','').replace('*','star'),vis)
        for i in range(n,nr*nc): c.fig.add_subplot(gs[i//nc,i%nc]).axis('off')
        self._save_fig(c.fig,"01_algorithm_comparison"); c.draw()

    def _render_heatmaps(self,R):
        c=self._canvas_heat; c.fig.clear(); c.fig.patch.set_facecolor(DARK_BG)
        axes=c.fig.subplots(1,2)
        rows,cols=len(self.grid),len(self.grid[0])
        omask=np.array([[self.grid[r][c]==0 for c in range(cols)] for r in range(rows)])
        if 'MDP' in R and 'mdp_obj' in R['MDP']:
            vg=np.zeros((rows,cols))
            for (r2,c2),v in R['MDP']['mdp_obj'].V.items(): vg[r2][c2]=v
            vg[omask]=np.nan
            im=axes[0].imshow(vg,cmap='plasma',interpolation='nearest')
            axes[0].set_title("MDP Value V(s)\nBellman Value Iteration",color='white',fontsize=10)
            c.fig.colorbar(im,ax=axes[0]).set_label("V(s)",color='white')
        else:
            axes[0].set_title("MDP not run",color=MUTED,fontsize=10)
        axes[0].axis('off'); axes[0].set_facecolor('#111')
        if 'Q-Learning' in R and 'qagent' in R['Q-Learning']:
            qg=np.array(R['Q-Learning']['qagent'].get_q_value_grid()); qg[omask]=np.nan
            im2=axes[1].imshow(qg,cmap='viridis',interpolation='nearest')
            axes[1].set_title("Q-Learning max Q(s,a)",color='white',fontsize=10)
            c.fig.colorbar(im2,ax=axes[1]).set_label("max Q",color='white')
        else:
            axes[1].set_title("Q-Learning not run",color=MUTED,fontsize=10)
        axes[1].axis('off')
        c.fig.suptitle("MDP & RL — Value Maps",color='white',fontsize=13,fontweight='bold')
        self._save_fig(c.fig,"02_mdp_rl_heatmaps"); c.draw()

    def _render_training(self,R):
        c=self._canvas_train; c.fig.clear(); c.fig.patch.set_facecolor(DARK_BG)
        ax1,ax2=c.fig.subplots(1,2)
        for ax in [ax1,ax2]:
            ax.set_facecolor(PANEL_BG); ax.tick_params(colors='white')
            ax.grid(True,alpha=0.2,color='#444')
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        if 'Q-Learning' in R and 'qagent' in R['Q-Learning']:
            rw=R['Q-Learning']['qagent'].episode_rewards
            ax1.plot(rw,alpha=0.3,color='#58a6ff',lw=0.7,label='Reward')
            if len(rw)>20:
                w=20; sm=np.convolve(rw,np.ones(w)/w,mode='valid')
                ax1.plot(range(w-1,len(rw)),sm,color='#f78166',lw=2,label=f'Avg-{w}')
            ax1.axhline(0,color='gray',lw=0.5,ls='--')
            ax1.legend(facecolor=PANEL_BG,labelcolor='white')
            ax1.set_title(f"Q-Learning Training Curve\n({R['Q-Learning']['qagent'].episodes_trained} eps)",color='white',fontsize=10)
        else:
            ax1.set_title("Q-Learning not run",color=MUTED,fontsize=10)
        ax1.set_xlabel("Episode",color='white'); ax1.set_ylabel("Reward",color='white')
        if 'NN+A*' in R and 'nn_obj' in R['NN+A*']:
            ax2.plot(R['NN+A*']['nn_obj'].train_losses,color=GREEN,lw=2,marker='o',markersize=3)
            ax2.set_title("Neural Network Loss\n2-layer NN · manual backprop",color='white',fontsize=10)
        else:
            ax2.set_title("NN+A* not run",color=MUTED,fontsize=10)
        ax2.set_xlabel("Epoch",color='white'); ax2.set_ylabel("BCE Loss",color='white')
        c.fig.suptitle("Training Curves",color='white',fontsize=13,fontweight='bold')
        self._save_fig(c.fig,"03_training_curves"); c.draw()

    def _render_heuristics(self,R):
        c=self._canvas_heur; c.fig.clear(); c.fig.patch.set_facecolor(DARK_BG)
        hr={k:v for k,v in R.items() if k.startswith('A*') and 'Bidir' not in k}
        if not hr:
            ax=c.fig.add_subplot(111); ax.set_facecolor(PANEL_BG)
            ax.text(0.5,0.5,"No A* variants run",ha='center',va='center',color=MUTED,transform=ax.transAxes,fontsize=14)
            ax.axis('off'); c.draw(); return
        axes=c.fig.subplots(1,3); names=list(hr.keys())
        short=[n.replace('A*(','').replace(')','') for n in names]; bc=['#58a6ff','#3fb950','#f78166']
        def bar(ax,vals,ylabel,title):
            ax.set_facecolor(PANEL_BG)
            bars=ax.bar(range(len(names)),vals,color=bc[:len(names)])
            ax.set_xticks(range(len(names))); ax.set_xticklabels(short,rotation=20,ha='right',color='white',fontsize=9)
            ax.set_ylabel(ylabel,color='white'); ax.set_title(title,color='white',fontsize=10)
            ax.tick_params(colors='white'); ax.grid(True,alpha=0.2,axis='y',color='#444')
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,b.get_height(),f'{v:.1f}',ha='center',va='bottom',color='white',fontsize=8)
        bar(axes[0],[hr[h]['nodes'] for h in names],'Nodes','Nodes Explored')
        bar(axes[1],[hr[h]['time']*1000 for h in names],'Time (ms)','Exec Time (ms)')
        bar(axes[2],[len(hr[h]['path']) if hr[h]['path'] else 0 for h in names],'Path (cells)','Path Length')
        c.fig.suptitle("A* Heuristic Comparison",color='white',fontsize=13,fontweight='bold')
        self._save_fig(c.fig,"04_heuristic_comparison"); c.draw()

    def _render_perf_table(self,R):
        tbl=self._perf_table; tbl.setRowCount(0)
        OPTIMAL={'BFS','A*(Manhattan)','A*(Euclidean)','A*(Chebyshev)','Bidir A*'}
        for name,r in R.items():
            row=tbl.rowCount(); tbl.insertRow(row)
            pl=len(r['path']) if r.get('path') else 0
            pp=path_px(r.get('path')); nd=r.get('nodes',0) or 0
            ts=r.get('time',0); eff=f"{pl/nd:.5f}" if pl and nd else '-'
            opt="Yes" if name in OPTIMAL else "Approx"
            vals=[name,str(pl) if pl else 'NO PATH',str(pp),str(nd),f"{ts:.4f}",eff,opt]
            hex_c=ALGO_COLORS_HEX.get(name,'#444')
            for col,val in enumerate(vals):
                item=QTableWidgetItem(val); item.setTextAlignment(Qt.AlignCenter)
                if col==0: item.setForeground(QColor(hex_c)); item.setFont(QFont("Segoe UI",10,QFont.Bold))
                if not r.get('path'): item.setForeground(QColor(RED))
                tbl.setItem(row,col,item)

    def _render_best(self,R):
        winner,reason=best_algo(R)
        if winner is None: self._best_main.setText("No path found"); return
        hex_c=ALGO_COLORS_HEX.get(winner,ACCENT); wdata=R[winner]
        pl=len(wdata['path']) if wdata.get('path') else 0
        nd=wdata.get('nodes',0); ts=wdata.get('time',0)
        self._best_main.setStyleSheet(f"font-size:30px;font-weight:bold;color:{hex_c};background:{PANEL_BG};border:3px solid {hex_c};border-radius:12px;padding:20px;")
        self._best_main.setText(f"BEST: {winner}\n\nPath: {pl} cells  |  Nodes: {nd}  |  Time: {ts:.4f}s")
        self._best_reason.setText(f"Why best? → {reason}\n\nScoring: 40% path + 35% nodes + 25% time")
        valid=[(k,v) for k,v in R.items() if v.get('path')]
        ranked=sorted(valid,key=lambda kv:score_algo(kv[1]))
        all_algos=[(k,v,True) for k,v in ranked]+[(k,v,False) for k,v in R.items() if not v.get('path')]
        self._rank_table.setRowCount(0); rank_i=0
        for k,v,found in all_algos:
            row=self._rank_table.rowCount(); self._rank_table.insertRow(row)
            rank_str=str(rank_i+1) if found else "-"
            pl2=len(v['path']) if found else "NO PATH"; nd2=v.get('nodes',0); ts2=v.get('time',0)
            for col,val in enumerate([rank_str,k,str(pl2),str(nd2),f"{ts2:.4f}"]):
                item=QTableWidgetItem(val); item.setTextAlignment(Qt.AlignCenter)
                if k==winner: item.setBackground(QColor(hex_c+"33")); item.setFont(QFont("Segoe UI",10,QFont.Bold))
                self._rank_table.setItem(row,col,item)
            if found: rank_i+=1
        if wdata.get('path') and self.img is not None:
            vis=self._draw(wdata['path'],wdata['color_bgr'],thickness=3)
            h,w=vis.shape[:2]
            cv2.putText(vis,f"BEST: {winner}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            self._save_algo_image("BEST_"+winner.replace('(','').replace(')','').replace('*','star'),vis)
            rgb=cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)
            qi=QImage(rgb.data,w,h,3*w,QImage.Format_RGB888)
            px=QPixmap.fromImage(qi)
            lw,lh=self._best_img_lbl.width(),self._best_img_lbl.height()
            if lw>10 and lh>10: px=px.scaled(lw-20,lh-20,Qt.KeepAspectRatio,Qt.SmoothTransformation)
            self._best_img_lbl.setPixmap(px)

    def _log_msg(self,msg):
        self._log.append(msg)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _status(self,msg): self._status_lbl.setText(msg)

    def resizeEvent(self,ev):
        super().resizeEvent(ev)
        if self.img is not None: QTimer.singleShot(80,self._refresh_grid_panels)


if __name__=="__main__":
    app=QApplication(sys.argv)
    app.setStyle("Fusion")
    win=Dashboard()
    win.showMaximized()
    sys.exit(app.exec_())
