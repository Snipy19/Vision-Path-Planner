"""
Q-Learning V3 - FAST FIX
-------------------------
Speed fixes:
1. Early stopping: if goal reached N times consecutively -> done
2. Smarter start: always from fixed start (avoids wasted random resets)
3. Efficient Q lookup with defaultdict
4. BFS fallback is bounded (not full-grid BFS)
5. best_path tracked inline, extract_path is O(path_length)
"""

import random
from collections import defaultdict, deque


class QLearningAgent:

    ACTIONS = {
        'UP':    (-1,  0),
        'DOWN':  ( 1,  0),
        'LEFT':  ( 0, -1),
        'RIGHT': ( 0,  1),
    }

    def __init__(self, grid, goal,
                 alpha=0.3,
                 gamma=0.95,
                 epsilon_start=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.9995,
                 goal_reward=1000.0,
                 step_reward=-1.0,
                 wall_penalty=-10.0):
        self.grid        = grid
        self.rows        = len(grid)
        self.cols        = len(grid[0])
        self.goal        = goal
        self.alpha       = alpha
        self.gamma       = gamma
        self.epsilon     = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.wall_penalty = wall_penalty

        self.action_names = list(self.ACTIONS.keys())

        # defaultdict avoids repeated key-existence checks
        # Optimistic init = 1.0
        self.Q = defaultdict(lambda: {a: 1.0 for a in self.action_names})

        self.walkable = [(r, c) for r in range(self.rows)
                         for c in range(self.cols) if grid[r][c] == 1]

        # Pre-build move table for speed
        self._move = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r][c] != 1:
                    continue
                mv = {}
                for a, (dr, dc) in self.ACTIONS.items():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and grid[nr][nc] == 1:
                        mv[a] = ((nr, nc), self.step_reward, False)
                    else:
                        mv[a] = ((r, c),   self.wall_penalty, False)
                # goal override
                for a in self.action_names:
                    ns, rw, _ = mv[a]
                    if ns == goal:
                        mv[a] = (ns, self.goal_reward, True)
                self._move[(r, c)] = mv

        self.episode_rewards = []
        self.episode_steps   = []
        self.episodes_trained = 0
        self.best_path_found  = None
        self.best_path_length = float('inf')

    # ------------------------------------------------------------------
    def choose_action(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.choice(self.action_names)
        q = self.Q[state]
        return max(q, key=q.get)

    def step(self, state, action):
        return self._move[state][action]

    def update_Q(self, state, action, reward, next_state):
        q      = self.Q[state]
        max_nq = max(self.Q[next_state].values())
        q[action] += self.alpha * (reward + self.gamma * max_nq - q[action])

    # ------------------------------------------------------------------
    def train(self, num_episodes=500, max_steps_per_episode=1000,
              start_fixed=None, progress_callback=None):
        """
        Faster training:
        - Early stopping after 30 consecutive goal-hits without improvement
        - Default episodes lowered to 500 (sufficient with strong params)
        """
        goal              = self.goal
        epsilon_decay     = self.epsilon_decay
        epsilon_min       = self.epsilon_min
        consecutive_hits  = 0
        last_best_episode = 0

        for episode in range(num_episodes):
            state = start_fixed if start_fixed else random.choice(self.walkable)
            total_reward = 0.0
            steps        = 0
            ep_path      = [state]

            for _ in range(max_steps_per_episode):
                action = self.choose_action(state)
                next_state, reward, done = self.step(state, action)
                self.update_Q(state, action, reward, next_state)
                state        = next_state
                total_reward += reward
                steps        += 1
                ep_path.append(next_state)

                if done:
                    if len(ep_path) < self.best_path_length:
                        self.best_path_length = len(ep_path)
                        self.best_path_found  = ep_path[:]
                        last_best_episode     = episode
                    consecutive_hits += 1
                    break
            else:
                consecutive_hits = 0

            self.epsilon = max(epsilon_min, self.epsilon * epsilon_decay)
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            self.episodes_trained += 1

            if progress_callback and (episode % 100 == 0 or episode == num_episodes - 1):
                avg_r = sum(self.episode_rewards[-100:]) / min(100, len(self.episode_rewards))
                progress_callback(episode + 1, num_episodes, self.epsilon, avg_r)

            # Early stopping: 50 consecutive goal-hits, no improvement in 200 ep
            if (consecutive_hits >= 50 and
                    episode - last_best_episode > 200 and
                    self.best_path_found is not None):
                break

    # ------------------------------------------------------------------
    def extract_path(self, start, max_steps=None):
        if max_steps is None:
            max_steps = self.rows * self.cols * 2

        # Method 1: greedy walk
        path = self._greedy_walk(start, max_steps)
        if path and path[-1] == self.goal and len(path) >= 5:
            return path, len(path)

        # Method 2: best path from training
        if self.best_path_found and self.best_path_found[0] == start:
            return self.best_path_found, len(self.best_path_found)

        # Method 3: Q-guided BFS (bounded depth)
        path = self._q_guided_bfs(start)
        if path and path[-1] == self.goal:
            return path, len(path)

        return (path or [start]), len(path or [start])

    def _greedy_walk(self, start, max_steps):
        path  = [start]
        state = start
        seen  = {start: 1}
        for _ in range(max_steps):
            if state == self.goal:
                break
            action     = self.choose_action(state, greedy=True)
            next_state, _, done = self.step(state, action)
            seen[next_state] = seen.get(next_state, 0) + 1
            if seen[next_state] >= 3:
                break
            path.append(next_state)
            state = next_state
            if done:
                break
        return path

    def _q_guided_bfs(self, start):
        """Bounded BFS: only explores up to rows*cols nodes."""
        if start == self.goal:
            return [start]
        goal    = self.goal
        Q       = self.Q
        limit   = self.rows * self.cols
        queue   = deque([(start, [start])])
        visited = {start}
        while queue:
            cur, path = queue.popleft()
            if len(path) > limit:
                continue
            if cur == goal:
                return path
            # sort neighbours by max Q descending
            nbrs = []
            for a, (dr, dc) in self.ACTIONS.items():
                nr, nc = cur[0] + dr, cur[1] + dc
                if (0 <= nr < self.rows and 0 <= nc < self.cols and
                        self.grid[nr][nc] == 1 and (nr, nc) not in visited):
                    q_val = Q[cur].get(a, 0)
                    nbrs.append((q_val, (nr, nc)))
            nbrs.sort(reverse=True)
            for _, nxt in nbrs:
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))
        return [start]

    def get_q_value_grid(self):
        qg = [[0.0] * self.cols for _ in range(self.rows)]
        for (r, c), acts in self.Q.items():
            qg[r][c] = max(acts.values())
        return qg

    def convergence_rate(self):
        if len(self.episode_rewards) < 100:
            return None
        first = sum(self.episode_rewards[:50]) / 50
        last  = sum(self.episode_rewards[-50:]) / 50
        return round(last - first, 2)