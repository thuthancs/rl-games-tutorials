"""
Training loop and mode-comparison utilities.
"""

import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from agent import QLearningAgent
from analysis import analyse_policy
from config import CFG, Config
from env import CoinCollectionEnv


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
    agents = [
        QLearningAgent(0, cfg.alpha, cfg.gamma,
                       cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_decay),
        QLearningAgent(1, cfg.alpha, cfg.gamma,
                       cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_decay),
    ]
    log = TrainingLog()

    if verbose:
        mode = "Cooperative" if cfg.cooperative else "Competitive"
        print("=" * 60)
        print(f"  Multiagent RL - Coin Collection Game (compact state)")
        print(f"  Grid: {cfg.grid_size}x{cfg.grid_size}  |  Coins: {cfg.num_coins}"
              f"  |  Mode: {mode}")
        print(f"  Episodes: {cfg.num_episodes}  |  alpha={cfg.alpha}  gamma={cfg.gamma}")
        print(f"  State: (pos_A, pos_B, coin_delta_A, coin_delta_B)")
        print("=" * 60)
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
                best_next = float(np.max(agents[i].q_table[next_state]))
                td_target = rewards[i] + cfg.gamma * best_next
                td_err    = abs(td_target - agents[i].q_table[state][actions[i]])
                ep_td[i] += td_err

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
                f"  Ep {ep+1:>5} | eps={agents[0].epsilon:.3f} | "
                f"R: A={a0:.2f}  B={a1:.2f} | "
                f"Coins/ep={ac:.1f} | TD={atd:.3f} | "
                f"Q-states={agents[0].q_table_size:,} | "
                f"{elapsed:.0f}s"
            )

    if verbose:
        print(f"\n  Training complete in {time.time()-t0:.1f}s")
    return agents, log


def compare_modes(cfg: Config = CFG) -> Tuple[dict, dict]:
    """
    Train two pairs of agents - one competitive, one cooperative -
    and return both result dicts for side-by-side analysis.
    """
    print("\n" + "=" * 60)
    print("  Mode Comparison: Competitive vs Cooperative")
    print("=" * 60)

    results = {}
    for mode in [False, True]:
        label = "cooperative" if mode else "competitive"
        print(f"\n-- Training ({label}) --")
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
