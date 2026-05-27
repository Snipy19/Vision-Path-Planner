import random
from collections import defaultdict, deque
import numpy as np


class QLearningAgent:

    ACTIONS = {
        'UP':    (-1, 0),
        'DOWN':  (1, 0),
        'LEFT':  (0, -1),
        'RIGHT': (0, 1),
    }

    def __init__(
        self,
        grid,
        goal,

        alpha=0.25,
        gamma=0.95,

        epsilon_start=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.02,

        goal_reward=1000.0,
        step_reward=-0.05,
        wall_penalty=-3.0,
    ):

        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

        self.goal = goal

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.wall_penalty = wall_penalty

        self.action_names = list(self.ACTIONS.keys())

        # ---------------------------
        # Q TABLE
        # ---------------------------
        self.Q = defaultdict(
            lambda: {a: 0.0 for a in self.action_names}
        )

        # ---------------------------
        # WALKABLE STATES
        # ---------------------------
        self.walkable = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if grid[r][c] == 1
        ]

        # ---------------------------
        # FAST MOVE TABLE
        # ---------------------------
        self.move_table = {}

        for r in range(self.rows):
            for c in range(self.cols):

                if grid[r][c] != 1:
                    continue

                self.move_table[(r, c)] = {}

                for action, (dr, dc) in self.ACTIONS.items():

                    nr = r + dr
                    nc = c + dc

                    if (
                        0 <= nr < self.rows
                        and 0 <= nc < self.cols
                        and grid[nr][nc] == 1
                    ):

                        reward = self.step_reward
                        done = False

                        if (nr, nc) == goal:
                            reward = self.goal_reward
                            done = True

                        self.move_table[(r, c)][action] = (
                            (nr, nc),
                            reward,
                            done
                        )

                    else:

                        self.move_table[(r, c)][action] = (
                            (r, c),
                            self.wall_penalty,
                            False
                        )

        # ---------------------------
        # STATS
        # ---------------------------
        self.episode_rewards = []
        self.episode_steps = []

        self.episodes_trained = 0

        self.best_path_found = None
        self.best_path_length = float('inf')

    # ==========================================================
    # ACTION SELECTION
    # ==========================================================

    def choose_action(self, state, greedy=False):

        # Exploration
        if not greedy and random.random() < self.epsilon:
            return random.choice(self.action_names)

        qvals = self.Q[state]

        max_q = max(qvals.values())

        best_actions = [
            a for a, q in qvals.items()
            if q == max_q
        ]

        return random.choice(best_actions)

    # ==========================================================
    # ENVIRONMENT STEP
    # ==========================================================

    def step(self, state, action):

        return self.move_table[state][action]

    # ==========================================================
    # Q UPDATE
    # ==========================================================

    def update_q(self, state, action, reward, next_state):

        old_q = self.Q[state][action]

        next_max = max(
            self.Q[next_state].values()
        )

        new_q = old_q + self.alpha * (
            reward
            + self.gamma * next_max
            - old_q
        )

        self.Q[state][action] = new_q

    # ==========================================================
    # TRAIN
    # ==========================================================

    def train(
        self,
        num_episodes=500,
        max_steps_per_episode=200,
        start_fixed=None,
        progress_callback=None
    ):

        consecutive_success = 0

        for episode in range(num_episodes):

            if start_fixed:
                state = start_fixed
            else:
                state = random.choice(self.walkable)

            total_reward = 0
            path = [state]

            for step_i in range(max_steps_per_episode):

                action = self.choose_action(state)

                next_state, reward, done = self.step(
                    state,
                    action
                )

                self.update_q(
                    state,
                    action,
                    reward,
                    next_state
                )

                state = next_state

                total_reward += reward

                path.append(state)

                if done:

                    consecutive_success += 1

                    if len(path) < self.best_path_length:

                        self.best_path_length = len(path)

                        self.best_path_found = path[:]

                    break

            else:
                consecutive_success = 0

            # Epsilon decay
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )

            self.episode_rewards.append(total_reward)
            self.episode_steps.append(len(path))

            self.episodes_trained += 1

            # Progress callback
            if progress_callback and episode % 25 == 0:

                avg_reward = np.mean(
                    self.episode_rewards[-25:]
                )

                progress_callback(
                    episode + 1,
                    num_episodes,
                    self.epsilon,
                    avg_reward
                )

            # Early stopping
            if consecutive_success >= 40:

                print(
                    f"[Q-Learning] Early convergence at episode {episode}"
                )

                break

    # ==========================================================
    # EXTRACT PATH
    # ==========================================================

    def extract_path(
        self,
        start,
        max_steps=None
    ):

        if max_steps is None:
            max_steps = self.rows + self.cols + 200

        state = start

        path = [state]

        visited = set()

        for _ in range(max_steps):

            if state == self.goal:
                break

            visited.add(state)

            action = self.choose_action(
                state,
                greedy=True
            )

            next_state, _, _ = self.step(
                state,
                action
            )

            # Loop prevention
            if next_state in visited:
                break

            path.append(next_state)

            state = next_state

        # If failed -> fallback BFS
        if path[-1] != self.goal:

            fallback = self.q_guided_bfs(start)

            if fallback:
                return fallback, len(fallback)

        # Smooth path
        path = self.smooth_path(path)

        return path, len(path)

    # ==========================================================
    # Q GUIDED BFS
    # ==========================================================

    def q_guided_bfs(self, start):

        q = deque()

        q.append((start, [start]))

        visited = {start}

        while q:

            current, path = q.popleft()

            if current == self.goal:
                return path

            neighbors = []

            for action, (dr, dc) in self.ACTIONS.items():

                nr = current[0] + dr
                nc = current[1] + dc

                if (
                    0 <= nr < self.rows
                    and 0 <= nc < self.cols
                    and self.grid[nr][nc] == 1
                    and (nr, nc) not in visited
                ):

                    score = self.Q[current][action]

                    neighbors.append(
                        (score, (nr, nc))
                    )

            neighbors.sort(reverse=True)

            for _, nxt in neighbors:

                visited.add(nxt)

                q.append(
                    (nxt, path + [nxt])
                )

        return None

    # ==========================================================
    # PATH SMOOTHING
    # ==========================================================

    def smooth_path(self, path):

        if len(path) <= 2:
            return path

        smooth = [path[0]]

        prev_dir = None

        for i in range(1, len(path)):

            dr = path[i][0] - path[i - 1][0]
            dc = path[i][1] - path[i - 1][1]

            curr_dir = (dr, dc)

            if curr_dir != prev_dir:
                smooth.append(path[i])

            prev_dir = curr_dir

        if smooth[-1] != path[-1]:
            smooth.append(path[-1])

        return smooth

    # ==========================================================
    # HEATMAP SUPPORT
    # ==========================================================

    def get_q_value_grid(self):

        qgrid = np.zeros(
            (self.rows, self.cols),
            dtype=np.float32
        )

        for (r, c), actions in self.Q.items():

            qgrid[r][c] = max(actions.values())

        return qgrid