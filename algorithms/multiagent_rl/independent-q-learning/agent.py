"""
Tabular Q-learning agent for independent multi-agent RL.
"""

import random
from collections import defaultdict
from typing import Dict

import numpy as np

from config import CFG, NUM_ACTIONS
from env import CoinCollectionEnv


class QLearningAgent:
    """
    Independent Q-Learning agent (tabular, off-policy TD).

    Each agent treats the other as part of its environment.
    This is the standard "decentralised execution" baseline and
    deliberately ignores the non-stationarity — a key teaching point.

    Q-update (Bellman)
    ------------------
    Q(s,a) <- Q(s,a) + alpha [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
    """

    def __init__(
        self,
        agent_id:      int,
        alpha:         float = CFG.alpha,
        gamma:         float = CFG.gamma,
        epsilon_start: float = CFG.epsilon_start,
        epsilon_end:   float = CFG.epsilon_end,
        epsilon_decay: float = CFG.epsilon_decay,
    ):
        self.agent_id      = agent_id
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        # Q-table: state -> array of Q-values (one per action)
        self.q_table: Dict = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        # running stats
        self.total_updates = 0

    # ── action selection ────────────────────────────────────────

    def select_action(self, state, greedy: bool = False) -> int:
        """epsilon-greedy (or fully greedy if greedy=True)."""
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        return int(np.argmax(self.q_table[state]))

    # ── learning ────────────────────────────────────────────────

    def update(self, state, action: int, reward: float, next_state):
        """Single Q-learning update step."""
        best_next                    = float(np.max(self.q_table[next_state]))
        td_target                    = reward + self.gamma * best_next
        td_error                     = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        self.total_updates          += 1

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ── introspection ────────────────────────────────────────────

    @property
    def q_table_size(self) -> int:
        return len(self.q_table)

    def action_distribution(self, env: CoinCollectionEnv, n_rollouts: int = 100) -> np.ndarray:
        """Empirical action frequencies over greedy rollouts."""
        counts = np.zeros(NUM_ACTIONS)
        for _ in range(n_rollouts):
            s = env.reset()
            for _ in range(CFG.max_steps):
                a = self.select_action(s, greedy=True)
                counts[a] += 1
                s, _, _, _ = env.step([a, a])
        return counts / counts.sum()

    def value_map(self, grid_size: int) -> np.ndarray:
        """
        Return a (grid_size x grid_size) array of mean max-Q values.
        Averages over all Q-table entries where this agent is at (r, c).
        """
        sums   = np.zeros((grid_size, grid_size))
        counts = np.zeros((grid_size, grid_size))
        idx    = self.agent_id      # position index in state tuple
        for state, q in self.q_table.items():
            r, c = state[idx]
            if 0 <= r < grid_size and 0 <= c < grid_size:
                sums[r, c]   += float(np.max(q))
                counts[r, c] += 1
        with np.errstate(invalid="ignore"):
            vmap = np.where(counts > 0, sums / counts, np.nan)
        return vmap
