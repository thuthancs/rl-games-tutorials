"""
Game-theory analysis of learned policies.

Provides metrics for Nash stability, Pareto efficiency, and action distributions.
"""

from typing import List

import numpy as np

from agent import QLearningAgent
from config import ACTION_NAMES, CFG, Config, NUM_ACTIONS
from env import CoinCollectionEnv


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

    print("\n" + "-" * 55)
    print("  Game Theory Analysis")
    print("-" * 55)

    # ── 1. Baseline greedy rollout ───────────────────────────────
    action_counts = [np.zeros(NUM_ACTIONS), np.zeros(NUM_ACTIONS)]
    total_rewards = [0.0, 0.0]
    total_coins   = 0

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
    # Test: does agent A gain by switching to a fixed "Stay" policy?
    print("\n  Nash stability check - agent A deviates to 'Stay':")
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
        print("    -> A gains from deviating - policy is NOT Nash stable.")
    else:
        print("    -> No gain from deviating -> approximate Nash equilibrium")

    # ── 3. Pareto efficiency gap ─────────────────────────────────
    # Upper bound: what if we ran a single oracle agent collecting all coins?
    print("\n  Pareto efficiency gap:")
    oracle_coins = 0
    oracle_env   = CoinCollectionEnv(cfg.grid_size, cfg.num_coins, True)
    for _ in range(n_eval):
        state = oracle_env.reset()
        for _ in range(CFG.max_steps):
            # Oracle: agent A uses its policy; B stays put
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
        print("    -> Competitive incentives reduce total welfare (tragedy of the commons).")
    elif cfg.cooperative:
        print("    -> Cooperative incentives bring joint policy closer to social optimum.")

    return dict(
        avg_reward=avg_reward,
        avg_coins=avg_coins,
        action_dist=[action_counts[i] / action_counts[i].sum() for i in range(2)],
        nash_gain=nash_gain,
        pareto_gap=pareto_gap,
    )
