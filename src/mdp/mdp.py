"""
MDP - Markov Decision Process + Value Iteration (V3 - FAST FIX)
-----------------------------------------------------------------------
SPEED FIXES:
1. value_iteration() -- pure dict ops, pre-built transition table
2. extract_path() -- A*-style heapq guided by V-values (NOT unbounded BFS)
3. _v_value_astar: bounded heap, stops as soon as goal dequeued
"""

import heapq


class MDP:

    ACTIONS = {
        'UP':    (-1,  0),
        'DOWN':  ( 1,  0),
        'LEFT':  ( 0, -1),
        'RIGHT': ( 0,  1),
    }

    def __init__(self, grid, goal,
                 gamma=0.95,
                 step_reward=-0.04,
                 goal_reward=100.0,
                 obstacle_penalty=-1.0,
                 slip_prob=0.1,
                 convergence_thresh=1e-3,
                 max_iterations=300):
        self.grid  = grid
        self.rows  = len(grid)
        self.cols  = len(grid[0])
        self.goal  = goal
        self.gamma = gamma
        self.step_reward      = step_reward
        self.goal_reward      = goal_reward
        self.obstacle_penalty = obstacle_penalty
        self.slip_prob        = slip_prob
        self.convergence_thresh = convergence_thresh
        self.max_iterations     = max_iterations

        self.states    = [(r, c)
                          for r in range(self.rows)
                          for c in range(self.cols)
                          if grid[r][c] == 1]
        self.state_set = set(self.states)

        self.V      = {s: 0.0 for s in self.states}
        self.policy = {}
        self.iterations_run = 0

        # Pre-build neighbour table once
        self._neighbors = {}
        for s in self.states:
            r, c = s
            nb = {}
            for a, (dr, dc) in self.ACTIONS.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and grid[nr][nc] == 1:
                    nb[a] = (nr, nc)
                else:
                    nb[a] = s
            self._neighbors[s] = nb

        # Pre-build transition table once
        action_list = list(self.ACTIONS.keys())
        self._trans = {}
        for s in self.states:
            self._trans[s] = {}
            for idx, a in enumerate(action_list):
                if s == self.goal:
                    self._trans[s][a] = [(1.0, s)]
                    continue
                la = action_list[(idx - 1) % 4]
                ra = action_list[(idx + 1) % 4]
                p  = self.slip_prob
                self._trans[s][a] = [
                    (1.0 - 2 * p, self._neighbors[s][a]),
                    (p,           self._neighbors[s][la]),
                    (p,           self._neighbors[s][ra]),
                ]

    def R(self, state):
        return self.goal_reward if state == self.goal else self.step_reward

    def value_iteration(self):
        V      = self.V
        trans  = self._trans
        gamma  = self.gamma
        thresh = self.convergence_thresh
        goal   = self.goal
        gr     = self.goal_reward
        sr     = self.step_reward

        for iteration in range(self.max_iterations):
            max_delta = 0.0
            new_V = {}
            for s in self.states:
                if s == goal:
                    new_V[s] = gr
                    continue
                best = float('-inf')
                for a, outcomes in trans[s].items():
                    val = sr + gamma * (
                        outcomes[0][0] * V[outcomes[0][1]] +
                        outcomes[1][0] * V[outcomes[1][1]] +
                        outcomes[2][0] * V[outcomes[2][1]]
                    )
                    if val > best:
                        best = val
                new_V[s] = best
                d = abs(best - V[s])
                if d > max_delta:
                    max_delta = d

            V = new_V
            self.iterations_run = iteration + 1
            if max_delta < thresh:
                break

        self.V = V
        self._extract_policy()
        return self.iterations_run

    def _extract_policy(self):
        V     = self.V
        trans = self._trans
        for s in self.states:
            if s == self.goal:
                self.policy[s] = 'GOAL'
                continue
            best_a, best_val = None, float('-inf')
            for a, outcomes in trans[s].items():
                val = sum(p * V.get(sp, 0) for p, sp in outcomes)
                if val > best_val:
                    best_val = val
                    best_a   = a
            self.policy[s] = best_a

    def extract_path(self, start, max_steps=None):
        path = self._policy_walk(start)
        if path and path[-1] == self.goal and len(path) >= 5:
            return path, len(path)
        path = self._v_value_astar(start)
        return path, len(path)

    def _policy_walk(self, start):
        path, cur = [start], start
        visited   = {start}
        limit     = self.rows * self.cols
        for _ in range(limit):
            if cur == self.goal:
                break
            a = self.policy.get(cur)
            if not a or a == 'GOAL':
                break
            nxt = self._neighbors[cur][a]
            if nxt in visited:
                break
            visited.add(nxt)
            path.append(nxt)
            cur = nxt
        return path

    def _v_value_astar(self, start):
        """Priority-queue search: priority = -V(s). Stops at first goal hit."""
        if start == self.goal:
            return [start]
        V, nb, goal = self.V, self._neighbors, self.goal
        ctr  = 0
        heap = [(-V.get(start, 0), ctr, start, [start])]
        seen = {start}
        while heap:
            _, _, cur, path = heapq.heappop(heap)
            if cur == goal:
                return path
            if len(path) > self.rows * self.cols:
                continue
            for nxt in nb[cur].values():
                if nxt not in seen:
                    seen.add(nxt)
                    ctr += 1
                    heapq.heappush(heap,
                        (-V.get(nxt, 0), ctr, nxt, path + [nxt]))
        return [start]

    def get_value_grid(self):
        vg = [[None] * self.cols for _ in range(self.rows)]
        for (r, c), v in self.V.items():
            vg[r][c] = round(v, 3)
        return vg

    # kept for compatibility
    def transitions(self, state, action_name):
        return self._trans.get(state, {}).get(action_name, [(1.0, state)])