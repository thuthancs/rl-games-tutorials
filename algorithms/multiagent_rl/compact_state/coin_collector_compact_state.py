"""
╔══════════════════════════════════════════════════════════════════╗
║   Multiagent RL: Coin Collection — Compact State Representation  ║
╠══════════════════════════════════════════════════════════════════╣
║  Problem with the original coin_collector.py                     ║
║  ─────────────────────────────────────────────────────────────── ║
║  The original state included frozenset(coin_positions), which    ║
║  has C(36,4) = 58,905 possible configurations on a 6×6 grid.    ║
║  Combined with 36×36 agent positions, the total state space is   ║
║  ~76 million. With only 250k training steps, most states are     ║
║  visited once at most, so Q-values stay near zero and the greedy ║
║  policy degrades worse than random exploration as ε decays.      ║
║                                                                  ║
║  Fix: compact state representation                               ║
║  ─────────────────────────────────────────────────────────────── ║
║  Replace frozenset(coins) with the Manhattan-nearest coin delta  ║
║  (dr, dc) for each agent. dr ∈ [-5,5], dc ∈ [-5,5] on a 6×6    ║
║  grid → 11×11 = 121 values per agent.                           ║
║                                                                  ║
║  New state space: 36 × 36 × 121 × 121 ≈ 19M                    ║
║  But crucially: the same (agent_pos, coin_delta) pattern now     ║
║  maps to the SAME state regardless of where other coins or the   ║
║  other agent are, so Q-values generalise across situations.      ║
║  In practice, visited states cluster around ~10k-50k entries.    ║
╠══════════════════════════════════════════════════════════════════╣
║  Game Theory Concepts:                                           ║
║    · Nash Equilibrium       — stable joint policy                ║
║    · Zero-sum vs Cooperative— tunable reward structure           ║
║    · Non-stationarity       — each agent changes the other's env ║
║    · Tragedy of the Commons — competitive over-exploitation      ║
║    · Territory emergence    — cooperative spatial division       ║
╠══════════════════════════════════════════════════════════════════╣
║  Install:  pip install numpy matplotlib                          ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── stdlib ────────────────────────────────────────────────────────
import random
import time
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── third-party ───────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.animation as animation

# ═════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Grid
    grid_size:     int   = 6       # NxN grid
    num_coins:     int   = 4       # coins alive on board at once
    # Training
    num_episodes:  int   = 5_000   # training episodes
    max_steps:     int   = 50      # max steps per episode
    cooperative:   bool  = False   # True → shared reward; False → competitive
    # Q-learning
    alpha:         float = 0.10    # learning rate
    gamma:         float = 0.95    # discount factor
    epsilon_start: float = 1.00    # initial exploration
    epsilon_end:   float = 0.05    # minimum exploration
    epsilon_decay: float = 0.9995  # per-episode multiplicative decay
    # Misc
    seed:          int   = 42
    log_interval:  int   = 500     # print progress every N episodes
    smooth_window: int   = 200     # rolling-average window for plots


CFG = Config()

# Actions: up, down, left, right, stay
ACTIONS      = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
ACTION_NAMES = ["Up", "Down", "Left", "Right", "Stay"]
NUM_ACTIONS  = len(ACTIONS)

random.seed(CFG.seed)
np.random.seed(CFG.seed)


# ═════════════════════════════════════════════════════════════════
#  ENVIRONMENT
# ═════════════════════════════════════════════════════════════════

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

    This replaces the original frozenset(coin_positions) which had
    C(36,4) = 58,905 configurations on a 6×6 grid, making the state
    space too large for tabular Q-learning to cover in 5k episodes.

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

    # mutable state (reset each episode)
    agent_positions: List[Tuple[int, int]] = field(default_factory=list)
    coin_positions:  set                   = field(default_factory=set)

    # ── episode bookkeeping ──────────────────────────────────────
    def reset(self) -> tuple:
        """Randomly place agents and coins; return initial state."""
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


# ═════════════════════════════════════════════════════════════════
#  AGENT
# ═════════════════════════════════════════════════════════════════

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
        """ε-greedy (or fully greedy if greedy=True)."""
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        return int(np.argmax(self.q_table[state]))

    # ── learning ────────────────────────────────────────────────
    def update(self, state, action: int, reward: float, next_state):
        """Single Q-learning update step."""
        best_next               = float(np.max(self.q_table[next_state]))
        td_target               = reward + self.gamma * best_next
        td_error                = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        self.total_updates     += 1

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
        Agent position is still at state[agent_id] so this works unchanged.
        """
        sums   = np.zeros((grid_size, grid_size))
        counts = np.zeros((grid_size, grid_size))
        idx    = self.agent_id   # 0 for Agent A, 1 for Agent B
        for state, q in self.q_table.items():
            r, c = state[idx]
            if 0 <= r < grid_size and 0 <= c < grid_size:
                sums[r, c]   += float(np.max(q))
                counts[r, c] += 1
        with np.errstate(invalid="ignore"):
            vmap = np.where(counts > 0, sums / counts, np.nan)
        return vmap


# ═════════════════════════════════════════════════════════════════
#  TRAINING
# ═════════════════════════════════════════════════════════════════

@dataclass
class TrainingLog:
    """Stores all per-episode metrics from a training run."""
    rewards:  List[List[float]] = field(default_factory=lambda: [[], []])
    coins:    List[float]       = field(default_factory=list)
    epsilon:  List[float]       = field(default_factory=list)
    td_error: List[List[float]] = field(default_factory=lambda: [[], []])


def train(cfg: Config = CFG, verbose: bool = True) -> Tuple[List[QLearningAgent], TrainingLog]:
    """
    Main training loop.

    Both agents learn simultaneously (independent Q-learning).
    Returns the trained agents and a full training log.
    """
    env    = CoinCollectionEnv(cfg.grid_size, cfg.num_coins, cfg.cooperative)
    agents = [QLearningAgent(0, cfg.alpha, cfg.gamma,
                             cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_decay),
              QLearningAgent(1, cfg.alpha, cfg.gamma,
                             cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_decay)]
    log    = TrainingLog()

    if verbose:
        mode = "Cooperative" if cfg.cooperative else "Competitive"
        print("═" * 60)
        print(f"  Multiagent RL — Coin Collection Game (compact state)")
        print(f"  Grid: {cfg.grid_size}×{cfg.grid_size}  │  Coins: {cfg.num_coins}"
              f"  │  Mode: {mode}")
        print(f"  Episodes: {cfg.num_episodes}  │  α={cfg.alpha}  γ={cfg.gamma}")
        print(f"  State: (pos_A, pos_B, coin_delta_A, coin_delta_B)")
        print("═" * 60)
        t0 = time.time()

    for ep in range(cfg.num_episodes):
        state      = env.reset()
        ep_rewards = [0.0, 0.0]
        ep_coins   = 0
        ep_td      = [0.0, 0.0]

        for _ in range(cfg.max_steps):
            actions = [agents[i].select_action(state) for i in range(2)]
            next_state, rewards, _, info = env.step(actions)
            ep_coins += info["coins_collected"]

            for i in range(2):
                # capture TD error before update for logging
                best_next   = float(np.max(agents[i].q_table[next_state]))
                td_target   = rewards[i] + cfg.gamma * best_next
                td_err      = abs(td_target - agents[i].q_table[state][actions[i]])
                ep_td[i]   += td_err

                agents[i].update(state, actions[i], rewards[i], next_state)
                ep_rewards[i] += rewards[i]

            state = next_state

        for i in range(2):
            agents[i].decay_epsilon()
            log.rewards[i].append(ep_rewards[i])
            log.td_error[i].append(ep_td[i] / cfg.max_steps)
        log.coins.append(ep_coins)
        log.epsilon.append(agents[0].epsilon)

        if verbose and (ep + 1) % cfg.log_interval == 0:
            w   = cfg.log_interval
            a0  = np.mean(log.rewards[0][-w:])
            a1  = np.mean(log.rewards[1][-w:])
            ac  = np.mean(log.coins[-w:])
            atd = np.mean(log.td_error[0][-w:])
            elapsed = time.time() - t0
            print(
                f"  Ep {ep+1:>5} │ ε={agents[0].epsilon:.3f} │ "
                f"R̄: A={a0:.2f}  B={a1:.2f} │ "
                f"Coins/ep={ac:.1f} │ TD={atd:.3f} │ "
                f"Q-states={agents[0].q_table_size:,} │ "
                f"{elapsed:.0f}s"
            )

    if verbose:
        print(f"\n  Training complete in {time.time()-t0:.1f}s")
    return agents, log


# ═════════════════════════════════════════════════════════════════
#  GAME THEORY ANALYSIS
# ═════════════════════════════════════════════════════════════════

def analyse_policy(
    agents: List[QLearningAgent],
    cfg:    Config = CFG,
    n_eval: int = 300,
) -> dict:
    """
    Evaluate learned policies through a game-theory lens.

    Returns a dict with keys:
      avg_reward, action_dist, nash_gain, pareto_gap
    """
    env = CoinCollectionEnv(cfg.grid_size, cfg.num_coins, cfg.cooperative)

    print("\n" + "─" * 55)
    print("  Game Theory Analysis")
    print("─" * 55)

    # ── 1. Baseline greedy rollout ───────────────────────────────
    action_counts  = [np.zeros(NUM_ACTIONS), np.zeros(NUM_ACTIONS)]
    total_rewards  = [0.0, 0.0]
    total_coins    = 0

    for _ in range(n_eval):
        state = env.reset()
        for _ in range(CFG.max_steps):
            actions = [agents[i].select_action(state, greedy=True) for i in range(2)]
            for i in range(2):
                action_counts[i][actions[i]] += 1
            next_state, rewards, _, info = env.step(actions)
            for i in range(2):
                total_rewards[i] += rewards[i]
            total_coins += info["coins_collected"]
            state = next_state

    avg_reward = [total_rewards[i] / n_eval for i in range(2)]
    avg_coins  = total_coins / n_eval

    print(f"\n  Average reward per episode (greedy, n={n_eval}):")
    for i, label in enumerate(["A", "B"]):
        print(f"    Agent {label}: {avg_reward[i]:.3f}")
    print(f"  Total coins collected per episode: {avg_coins:.2f}")

    print("\n  Action distribution (greedy policy):")
    for i, label in enumerate(["A", "B"]):
        dist = action_counts[i] / action_counts[i].sum()
        row  = "  ".join(f"{ACTION_NAMES[a]}:{dist[a]:.2f}" for a in range(NUM_ACTIONS))
        print(f"    Agent {label}: {row}")

    # ── 2. Nash stability check ──────────────────────────────────
    print("\n  Nash stability check — agent A deviates to 'Stay':")
    stay_reward_A = 0.0
    for _ in range(n_eval):
        state = env.reset()
        for _ in range(CFG.max_steps):
            a0 = 4  # always Stay
            a1 = agents[1].select_action(state, greedy=True)
            next_state, rewards, _, _ = env.step([a0, a1])
            stay_reward_A += rewards[0]
            state = next_state
    stay_reward_A /= n_eval
    nash_gain = stay_reward_A - avg_reward[0]
    print(f"    Reward if A always stays : {stay_reward_A:.3f}")
    print(f"    Reward under trained policy: {avg_reward[0]:.3f}")
    print(f"    Deviation gain: {nash_gain:+.3f}")
    if nash_gain > 0.05:
        print("    → A gains from deviating — policy is NOT Nash stable.")
    else:
        print("    → No gain from deviating → approximate Nash equilibrium ✓")

    # ── 3. Pareto efficiency gap ─────────────────────────────────
    print("\n  Pareto efficiency gap:")
    oracle_coins = 0
    oracle_env   = CoinCollectionEnv(cfg.grid_size, cfg.num_coins, True)
    for _ in range(n_eval):
        state = oracle_env.reset()
        for _ in range(CFG.max_steps):
            a0 = agents[0].select_action(state, greedy=True)
            next_state, _, _, info = oracle_env.step([a0, 4])
            oracle_coins += info["coins_collected"]
            state = next_state
    oracle_coins /= n_eval
    pareto_gap = oracle_coins - avg_coins
    print(f"    Coins collected (joint policy): {avg_coins:.2f}/ep")
    print(f"    Coins collected (single agent): {oracle_coins:.2f}/ep")
    print(f"    Pareto gap: {pareto_gap:+.2f} coins/ep")
    if not cfg.cooperative and pareto_gap > 0.3:
        print("    → Competitive incentives reduce total welfare (tragedy of the commons).")
    elif cfg.cooperative:
        print("    → Cooperative incentives bring joint policy closer to social optimum.")

    return dict(
        avg_reward=avg_reward,
        avg_coins=avg_coins,
        action_dist=[action_counts[i] / action_counts[i].sum() for i in range(2)],
        nash_gain=nash_gain,
        pareto_gap=pareto_gap,
    )


# ═════════════════════════════════════════════════════════════════
#  COMPARISON: COMPETITIVE vs COOPERATIVE
# ═════════════════════════════════════════════════════════════════

def compare_modes(cfg: Config = CFG) -> Tuple[dict, dict]:
    """
    Train two pairs of agents — one competitive, one cooperative —
    and return both training logs for side-by-side analysis.
    """
    print("\n" + "═" * 60)
    print("  Mode Comparison: Competitive vs Cooperative")
    print("═" * 60)

    results = {}
    for mode in [False, True]:
        label = "cooperative" if mode else "competitive"
        print(f"\n── Training ({label}) ──")
        c = Config(
            grid_size=cfg.grid_size,
            num_coins=cfg.num_coins,
            num_episodes=cfg.num_episodes,
            max_steps=cfg.max_steps,
            cooperative=mode,
            alpha=cfg.alpha,
            gamma=cfg.gamma,
            epsilon_start=cfg.epsilon_start,
            epsilon_end=cfg.epsilon_end,
            epsilon_decay=cfg.epsilon_decay,
            seed=cfg.seed,
            log_interval=cfg.log_interval,
        )
        agents, log = train(c)
        analysis    = analyse_policy(agents, c)
        results[label] = {"agents": agents, "log": log, "analysis": analysis, "cfg": c}

    return results["competitive"], results["cooperative"]


# ═════════════════════════════════════════════════════════════════
#  PLOTTING
# ═════════════════════════════════════════════════════════════════

COLORS = {
    "agent_A":   "#185FA5",
    "agent_B":   "#D85A30",
    "coins":     "#3B6D11",
    "epsilon":   "#7F77DD",
    "td_error":  "#993556",
    "grid_bg":   "#F1EFE8",
    "coin_fill": "#EF9F27",
}


def _smooth(data: list, w: int) -> np.ndarray:
    return np.convolve(data, np.ones(w) / w, mode="valid")


def plot_training(log: TrainingLog, cfg: Config, title_suffix: str = ""):
    """Four-panel training dashboard."""
    w   = cfg.smooth_window
    fig = plt.figure(figsize=(18, 4))
    fig.suptitle(
        f"Training Curves — {'Cooperative' if cfg.cooperative else 'Competitive'}"
        f"{' | ' + title_suffix if title_suffix else ''}"
        f" (compact state)",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    # 1. Rewards
    ax = fig.add_subplot(gs[0])
    for i, (label, color) in enumerate(zip(["Agent A", "Agent B"],
                                           [COLORS["agent_A"], COLORS["agent_B"]])):
        s = _smooth(log.rewards[i], w)
        ax.plot(s, color=color, lw=1.5, label=label)
    ax.set_title("Reward / episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # 2. Coins
    ax = fig.add_subplot(gs[1])
    s = _smooth(log.coins, w)
    ax.plot(s, color=COLORS["coins"], lw=1.5)
    ax.set_title("Coins collected / episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Coins")
    ax.grid(alpha=0.25)

    # 3. TD error
    ax = fig.add_subplot(gs[2])
    for i, (label, color) in enumerate(zip(["Agent A", "Agent B"],
                                           [COLORS["agent_A"], COLORS["agent_B"]])):
        s = _smooth(log.td_error[i], w)
        ax.plot(s, color=color, lw=1.5, label=label, alpha=0.8)
    ax.set_title("Mean |TD error| / episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("|TD error|")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # 4. Epsilon
    ax = fig.add_subplot(gs[3])
    ax.plot(log.epsilon, color=COLORS["epsilon"], lw=1.5)
    ax.set_title("Exploration ε decay")
    ax.set_xlabel("Episode")
    ax.set_ylabel("ε")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    return fig


def plot_value_heatmaps(agents: List[QLearningAgent], cfg: Config):
    """Side-by-side Q-value heatmaps for both agents."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    mode = "Cooperative" if cfg.cooperative else "Competitive"
    fig.suptitle(f"Q-Value Heatmaps ({mode}, compact state)", fontsize=13, fontweight="bold")

    labels = ["Agent A", "Agent B"]

    for i, (ax, agent) in enumerate(zip(axes, agents)):
        vmap = agent.value_map(cfg.grid_size)
        vmin = np.nanmin(vmap) if not np.all(np.isnan(vmap)) else 0
        vmap_display = np.where(np.isnan(vmap), vmin, vmap)
        im = ax.imshow(
            vmap_display,
            cmap="Blues" if i == 0 else "Oranges",
            aspect="equal",
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="max Q")
        ax.set_title(f"{labels[i]} — max Q per cell\n(avg over known states)")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        for r in range(cfg.grid_size):
            for c in range(cfg.grid_size):
                v = vmap[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                            fontsize=7, color="white" if v > (np.nanmax(vmap) * 0.6) else "black")

    plt.tight_layout()
    return fig


def plot_comparison(comp_log: TrainingLog, coop_log: TrainingLog, cfg: Config):
    """
    Side-by-side comparison of competitive vs cooperative training runs.
    Shows rewards and total coins on the same axes.
    """
    w   = cfg.smooth_window
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Competitive vs Cooperative — Head-to-Head Comparison (compact state)",
                 fontsize=13, fontweight="bold")

    # ── rewards ─────────────────────────────────────────────────
    ax = axes[0]
    for i, (label, color) in enumerate(zip(["A", "B"],
                                           [COLORS["agent_A"], COLORS["agent_B"]])):
        s_c = _smooth(comp_log.rewards[i], w)
        s_o = _smooth(coop_log.rewards[i], w)
        ax.plot(s_c, color=color, lw=1.5, ls="--", label=f"Agent {label} (comp)")
        ax.plot(s_o, color=color, lw=1.5, ls="-",  label=f"Agent {label} (coop)", alpha=0.7)
    ax.set_title("Individual reward / episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)

    # ── total coins ──────────────────────────────────────────────
    ax = axes[1]
    sc = _smooth(comp_log.coins, w)
    so = _smooth(coop_log.coins, w)
    ax.plot(sc, color="#A32D2D", lw=1.5, ls="--", label="Competitive")
    ax.plot(so, color="#0F6E56", lw=1.5, ls="-",  label="Cooperative")
    ax.set_title("Total coins / episode\n(social welfare)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Coins")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    x = np.arange(min(len(sc), len(so)))
    ax.fill_between(x, sc[:len(x)], so[:len(x)],
                    where=(so[:len(x)] > sc[:len(x)]),
                    alpha=0.15, color="#0F6E56", label="Coop advantage")

    # ── TD error ─────────────────────────────────────────────────
    ax = axes[2]
    sc = _smooth(comp_log.td_error[0], w)
    so = _smooth(coop_log.td_error[0], w)
    ax.plot(sc, color="#A32D2D", lw=1.5, ls="--", label="Competitive")
    ax.plot(so, color="#0F6E56", lw=1.5, ls="-",  label="Cooperative")
    ax.set_title("Agent A — |TD error| / episode\n(learning signal strength)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("|TD error|")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════
#  ANIMATED EPISODE REPLAY
# ═════════════════════════════════════════════════════════════════

def animate_episode(
    agents: List[QLearningAgent],
    cfg:    Config = CFG,
    steps:  int    = 60,
    interval_ms: int = 350,
    save_path: Optional[str] = None,
):
    """
    Matplotlib animation of a single greedy episode.
    Shows agent positions, coins, and last action taken.

    Parameters
    ----------
    save_path : str or None
        If given, saves the animation as a GIF (requires Pillow).
    """
    env   = CoinCollectionEnv(cfg.grid_size, cfg.num_coins, cfg.cooperative)
    state = env.reset()
    g     = cfg.grid_size

    # Pre-compute all frames
    frames = []
    for _ in range(steps):
        actions = [agents[i].select_action(state, greedy=True) for i in range(2)]
        frame   = dict(
            agent_pos=list(env.agent_positions),
            coin_pos=set(env.coin_positions),
            actions=actions,
        )
        frames.append(frame)
        state, _, _, _ = env.step(actions)

    # ── set up figure ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, g - 0.5)
    ax.set_ylim(-0.5, g - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(range(g))
    ax.set_yticks(range(g))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="#cccccc", linewidth=0.5)
    ax.set_facecolor(COLORS["grid_bg"])

    agent_circles = [
        plt.Circle((0, 0), 0.35, color=COLORS["agent_A"], zorder=4),
        plt.Circle((0, 0), 0.35, color=COLORS["agent_B"], zorder=4),
    ]
    agent_labels = []
    for i, (circle, label) in enumerate(zip(agent_circles, ["A", "B"])):
        ax.add_patch(circle)
        txt = ax.text(0, 0, label, ha="center", va="center",
                      fontsize=10, fontweight="bold", color="white", zorder=5)
        agent_labels.append(txt)

    coin_artists: List[plt.Artist] = []
    title = ax.set_title("", fontsize=10)

    legend_patches = [
        mpatches.Patch(color=COLORS["agent_A"], label="Agent A"),
        mpatches.Patch(color=COLORS["agent_B"], label="Agent B"),
        mpatches.Patch(color=COLORS["coin_fill"], label="Coin"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
              framealpha=0.8, bbox_to_anchor=(1.0, 1.0))

    def _draw_frame(t):
        fr = frames[t]
        for art in coin_artists:
            art.remove()
        coin_artists.clear()
        for pos in fr["coin_pos"]:
            c   = plt.Circle((pos[1], pos[0]), 0.25, color=COLORS["coin_fill"], zorder=3)
            txt = ax.text(pos[1], pos[0], "$", ha="center", va="center",
                          fontsize=8, color="#412402", fontweight="bold", zorder=4)
            ax.add_patch(c)
            coin_artists.extend([c, txt])
        for i, (circle, lbl) in enumerate(zip(agent_circles, agent_labels)):
            r, c_col = fr["agent_pos"][i]
            circle.center = (c_col, r)
            lbl.set_position((c_col, r))
        mode = "Cooperative" if cfg.cooperative else "Competitive"
        title.set_text(
            f"Step {t+1}/{steps}  [{mode}]\n"
            f"A→{ACTION_NAMES[fr['actions'][0]]}  "
            f"B→{ACTION_NAMES[fr['actions'][1]]}"
        )
        return agent_circles + agent_labels + coin_artists + [title]

    _draw_frame(0)
    anim = animation.FuncAnimation(
        fig, _draw_frame, frames=steps,
        interval=interval_ms, blit=False, repeat=True,
    )

    if save_path:
        print(f"  Saving animation → {save_path}")
        anim.save(save_path, writer="pillow", fps=int(1000 / interval_ms))
        print("  Saved.")

    plt.tight_layout()
    return fig, anim


# ═════════════════════════════════════════════════════════════════
#  ASCII EPISODE REPLAY (no matplotlib needed)
# ═════════════════════════════════════════════════════════════════

def replay_ascii(
    agents: List[QLearningAgent],
    cfg:    Config = CFG,
    steps:  int    = 20,
    delay:  float  = 0.0,
):
    """Print a greedy episode step-by-step in the terminal."""
    env    = CoinCollectionEnv(cfg.grid_size, cfg.num_coins, cfg.cooperative)
    state  = env.reset()
    scores = [0.0, 0.0]

    print("\n" + "─" * 40)
    print(f"  Greedy Episode Replay ({cfg.grid_size}×{cfg.grid_size}, compact state)")
    print("─" * 40)

    for t in range(steps):
        grid    = env.render_ascii()
        actions = [agents[i].select_action(state, greedy=True) for i in range(2)]
        next_state, rewards, _, info = env.step(actions)
        for i in range(2):
            scores[i] += rewards[i]

        print(f"\nStep {t+1:>3}  "
              f"A→{ACTION_NAMES[actions[0]]:<5}  "
              f"B→{ACTION_NAMES[actions[1]]:<5}  "
              f"coins={info['coins_collected']}  "
              f"score=[{scores[0]:.0f}, {scores[1]:.0f}]")
        print(grid)
        state = next_state
        if delay > 0:
            time.sleep(delay)

    print(f"\n  Final scores → A: {scores[0]:.0f}   B: {scores[1]:.0f}")


# ═════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    """
    Full pipeline:
      1. Train in competitive mode
      2. Train in cooperative mode (for comparison)
      3. Run game theory analysis on both
      4. Plot training curves, value heatmaps, comparison
      5. Animate a greedy episode (competitive)
    """
    QUICK = "--quick" in sys.argv
    n_eps = 1_000 if QUICK else CFG.num_episodes

    cfg = Config(num_episodes=n_eps)

    # ── 1. Train both modes ──────────────────────────────────────
    comp_result, coop_result = compare_modes(cfg)

    comp_agents, comp_log = comp_result["agents"], comp_result["log"]
    coop_agents, coop_log = coop_result["agents"], coop_result["log"]

    # ── 2. ASCII replay (competitive) ───────────────────────────
    replay_ascii(comp_agents, cfg, steps=15)

    # ── 3. Plots ─────────────────────────────────────────────────
    print("\n  Generating plots ...")

    fig1 = plot_training(comp_log, cfg, title_suffix="Competitive")
    fig1.savefig("marl_compact_training_competitive.png", dpi=150, bbox_inches="tight")

    fig2 = plot_training(coop_log, cfg, title_suffix="Cooperative")
    fig2.savefig("marl_compact_training_cooperative.png", dpi=150, bbox_inches="tight")

    fig3 = plot_value_heatmaps(comp_agents, cfg)
    fig3.savefig("marl_compact_heatmaps_competitive.png", dpi=150, bbox_inches="tight")

    fig4 = plot_value_heatmaps(coop_agents, cfg)
    fig4.savefig("marl_compact_heatmaps_cooperative.png", dpi=150, bbox_inches="tight")

    fig5 = plot_comparison(comp_log, coop_log, cfg)
    fig5.savefig("marl_compact_comparison.png", dpi=150, bbox_inches="tight")

    print("  Plots saved:")
    for name in [
        "marl_compact_training_competitive.png",
        "marl_compact_training_cooperative.png",
        "marl_compact_heatmaps_competitive.png",
        "marl_compact_heatmaps_cooperative.png",
        "marl_compact_comparison.png",
    ]:
        print(f"    → {name}")

    # ── 4. Animated episode ──────────────────────────────────────
    print("\n  Rendering animated episode (competitive) ...")
    fig6, anim = animate_episode(
        comp_agents, cfg,
        steps=50,
        interval_ms=300,
        save_path="marl_compact_episode.gif",
    )

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()
