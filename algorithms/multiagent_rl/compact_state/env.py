"""
Grid-world environment with compact state representation.

Key difference from the original:
  The state replaces frozenset(coin_positions) — which has C(36,4) = 58,905
  configurations on a 6x6 grid — with the Manhattan-nearest coin delta (dr, dc)
  for each agent. This reduces the effective visited state space from ~76M to
  ~10k-50k entries, allowing Q-values to generalise across situations.
"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple

from config import ACTIONS, CFG, Config


@dataclass
class CoinCollectionEnv:
    """
    Grid world — two agents collect respawning coins.

    State (compact)
    ---------------
    (agent0_pos, agent1_pos, nearest_coin_delta_0, nearest_coin_delta_1)

    where:
      - agent_pos is (row, col)
      - nearest_coin_delta is (dr, dc) — the row/col offset from the agent
        to its closest coin by Manhattan distance.

    Rewards
    -------
    Competitive : +1.0 to the collector, 0 to the other.
    Cooperative : +0.5 to BOTH agents when any coin is collected.

    Episode structure
    -----------------
    Continuous task — coins respawn to keep the board full.
    Episodes end after `max_steps` steps (externally controlled).
    """

    grid_size:   int  = field(default_factory=lambda: CFG.grid_size)
    num_coins:   int  = field(default_factory=lambda: CFG.num_coins)
    cooperative: bool = field(default_factory=lambda: CFG.cooperative)

    # mutable state — reset each episode
    agent_positions: List[Tuple[int, int]] = field(default_factory=list)
    coin_positions:  set                   = field(default_factory=set)

    # ── episode bookkeeping ──────────────────────────────────────

    def reset(self) -> tuple:
        """Randomly place agents and coins; return initial compact state."""
        all_cells = [(r, c) for r in range(self.grid_size)
                             for c in range(self.grid_size)]
        chosen = random.sample(all_cells, 2 + self.num_coins)
        self.agent_positions = [chosen[0], chosen[1]]
        self.coin_positions  = set(chosen[2:])
        return self._state()

    def _nearest_coin_delta(self, agent_idx: int) -> Tuple[int, int]:
        """
        Return (dr, dc) from agent to its nearest coin by Manhattan distance.
        Returns (0, 0) if no coins exist (should not happen in normal play).
        """
        pos = self.agent_positions[agent_idx]
        if not self.coin_positions:
            return (0, 0)
        return min(
            ((cp[0] - pos[0], cp[1] - pos[1]) for cp in self.coin_positions),
            key=lambda d: abs(d[0]) + abs(d[1]),
        )

    def _state(self) -> tuple:
        """
        Compact state: agent positions + nearest-coin offset for each agent.
        Positions stay at indices 0 and 1 so value_map() works unchanged.
        """
        return (
            self.agent_positions[0],
            self.agent_positions[1],
            self._nearest_coin_delta(0),
            self._nearest_coin_delta(1),
        )

    # ── step ────────────────────────────────────────────────────

    def step(self, actions: List[int]) -> Tuple[tuple, List[float], bool, dict]:
        """
        Parameters
        ----------
        actions : [action_idx_agent0, action_idx_agent1]

        Returns
        -------
        next_state, rewards, done, info
        """
        rewards         = [0.0, 0.0]
        coins_collected = 0

        # 1. Move both agents (bounded by grid walls)
        new_positions = []
        for pos, action in zip(self.agent_positions, actions):
            dr, dc = ACTIONS[action]
            nr = max(0, min(self.grid_size - 1, pos[0] + dr))
            nc = max(0, min(self.grid_size - 1, pos[1] + dc))
            new_positions.append((nr, nc))
        self.agent_positions = new_positions

        # 2. Collect coins
        for i, pos in enumerate(self.agent_positions):
            if pos in self.coin_positions:
                self.coin_positions.discard(pos)
                coins_collected += 1
                if self.cooperative:
                    rewards[0] += 0.5
                    rewards[1] += 0.5
                else:
                    rewards[i] += 1.0

        # 3. Respawn coins so the board stays full
        self._respawn_coins()

        done = False  # episode termination is controlled externally
        return self._state(), rewards, done, {"coins_collected": coins_collected}

    def _respawn_coins(self):
        occupied = set(self.agent_positions) | self.coin_positions
        empty = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in occupied
        ]
        while len(self.coin_positions) < self.num_coins and empty:
            choice = random.choice(empty)
            self.coin_positions.add(choice)
            empty.remove(choice)

    # ── helpers ─────────────────────────────────────────────────

    def render_ascii(self) -> str:
        """Return a multi-line ASCII string of the current grid."""
        AGENT = ["A", "B"]
        COIN  = "●"
        EMPTY = "·"
        g = [[EMPTY] * self.grid_size for _ in range(self.grid_size)]
        for pos in self.coin_positions:
            g[pos[0]][pos[1]] = COIN
        for i, pos in enumerate(self.agent_positions):
            g[pos[0]][pos[1]] = AGENT[i]
        border = "  ┌" + "──" * self.grid_size + "┐"
        rows   = ["  │ " + " ".join(row) + " │" for row in g]
        bottom = "  └" + "──" * self.grid_size + "┘"
        return "\n".join([border] + rows + [bottom])
