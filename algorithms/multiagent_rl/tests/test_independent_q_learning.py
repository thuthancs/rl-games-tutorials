"""Tests for independent-q-learning (full joint state). Run from repo: see README."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_IQL = _ROOT / "independent-q-learning"
if str(_IQL) not in sys.path:
    sys.path.insert(0, str(_IQL))

from agent import QLearningAgent  # noqa: E402
from config import Config, NUM_ACTIONS  # noqa: E402
from env import CoinCollectionEnv  # noqa: E402


@pytest.fixture
def cfg() -> Config:
    return Config(grid_size=4, num_coins=2, cooperative=False, seed=123)


@pytest.fixture(autouse=True)
def set_seeds(cfg: Config) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)


def test_reset_returns_three_tuple_with_frozenset_coins(cfg: Config) -> None:
    env = CoinCollectionEnv(cfg.grid_size, cfg.num_coins, cfg.cooperative)
    s = env.reset()
    assert len(s) == 3
    assert len(s[2]) == cfg.num_coins


def test_step_keeps_agents_on_grid(cfg: Config) -> None:
    env = CoinCollectionEnv(cfg.grid_size, cfg.num_coins, cfg.cooperative)
    env.reset()
    for _ in range(20):
        actions = [random.randrange(NUM_ACTIONS) for _ in range(2)]
        _, _, _, _ = env.step(actions)
        for pos in env.agent_positions:
            r, c = pos
            assert 0 <= r < cfg.grid_size
            assert 0 <= c < cfg.grid_size


def test_q_update_changes_value(cfg: Config) -> None:
    agent = QLearningAgent(
        0,
        alpha=0.5,
        gamma=0.9,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1.0,
    )
    agent.epsilon = 0.0
    state = ((0, 0), (1, 1), frozenset({(2, 2), (3, 3)}))
    next_state = ((0, 1), (1, 1), frozenset({(2, 2), (3, 3)}))
    action = 0
    before = float(agent.q_table[state][action])
    agent.update(state, action, 1.0, next_state)
    after = float(agent.q_table[state][action])
    assert after != before


def test_greedy_select_argmax(cfg: Config) -> None:
    agent = QLearningAgent(
        0,
        alpha=0.1,
        gamma=0.95,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1.0,
    )
    agent.epsilon = 0.0
    state = ((0, 0), (1, 1), frozenset({(2, 2)}))
    agent.q_table[state][:] = 0.0
    agent.q_table[state][3] = 99.0
    assert agent.select_action(state, greedy=True) == 3
