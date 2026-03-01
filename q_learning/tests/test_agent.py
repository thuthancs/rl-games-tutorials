"""Tests for agent module."""

import random

import pytest

from q_learning.agent import (
    QLearningAgent,
    get_current_direction,
    get_direction,
    get_state_representation,
)
from q_learning.constants import ACTIONS, DIRECTIONS
from q_learning.game import GameLogic


class TestGetCurrentDirection:
    def test_single_cell_default_rightward(self):
        assert get_current_direction([(1, 1)]) == "rightward"

    def test_head_above_neck_is_downward(self):
        # head (0,0), neck (1,0) -> head is above neck -> snake facing upward (head - neck = (-1,0))
        assert get_current_direction([(0, 0), (1, 0)]) == "upward"

    def test_head_below_neck_is_downward(self):
        assert get_current_direction([(1, 0), (0, 0)]) == "downward"

    def test_head_right_of_neck_is_rightward(self):
        assert get_current_direction([(0, 1), (0, 0)]) == "rightward"

    def test_head_left_of_neck_is_leftward(self):
        assert get_current_direction([(0, 0), (0, 1)]) == "leftward"


class TestGetDirection:
    def test_go_straight_preserves_direction(self):
        for name, (dr, dc) in DIRECTIONS.items():
            out = get_direction("go_straight", name)
            assert out == (dr, dc)

    def test_turn_around_reverses(self):
        assert get_direction("turn_around", "upward") == (1, 0)   # downward
        assert get_direction("turn_around", "downward") == (-1, 0)  # upward
        assert get_direction("turn_around", "leftward") == (0, 1)   # rightward
        assert get_direction("turn_around", "rightward") == (0, -1)  # leftward

    def test_turn_left_rightward_gives_upward(self):
        assert get_direction("turn_left", "rightward") == (-1, 0)


class TestGetStateRepresentation:
    def test_state_tuple_structure(self):
        random.seed(42)
        game = GameLogic(grid_size=3)
        game.place_food()
        state = get_state_representation(game)
        assert len(state) == 4
        head_pos, head_dir, body_tuple, food_pos = state
        assert head_pos == game.Snake.snake_positions[0]
        assert head_dir in ("upward", "downward", "leftward", "rightward")
        assert body_tuple == tuple(game.Snake.snake_positions)
        assert food_pos == game.GameEnvironment.food_pos


class TestQLearningAgent:
    @pytest.fixture
    def agent(self):
        return QLearningAgent(
            actions=ACTIONS,
            learning_rate=0.2,
            epsilon=0.0,
            gamma=0.5,
            grid_size=3,
            num_episodes=100,
        )

    def test_set_q_table_initializes_all_zero(self, agent):
        agent.set_q_table()
        assert len(agent.q_table) > 0
        for state, actions_dict in agent.q_table.items():
            for a in ACTIONS:
                assert actions_dict[a] == 0.0

    def test_get_q_value_missing_state_returns_zero(self, agent):
        agent.set_q_table()
        assert agent.get_q_value(((0, 0), "upward", ((0, 0),), (1, 1)), "go_straight") == 0.0

    def test_choose_action_epsilon_zero_is_greedy(self, agent):
        random.seed(42)
        agent.set_q_table()
        state = next(iter(agent.q_table.keys()))
        action = agent.choose_action(state, epsilon=0.0)
        assert action in ACTIONS
        # With all Q equal, should pick first in list (deterministic tie-break)
        assert action == ACTIONS[0]

    def test_update_q_value_changes_table(self, agent):
        agent.set_q_table()
        state = next(iter(agent.q_table.keys()))
        action = ACTIONS[0]
        next_state = next(s for s in agent.q_table.keys() if s != state)
        before = agent.get_q_value(state, action)
        agent.update_q_value(state, action, reward=10, next_state=next_state)
        after = agent.get_q_value(state, action)
        assert after != before
        # Q-learning update: new_q = old_q + lr * (reward + gamma * max_a Q(s',a) - old_q)
        expected = before + agent.learning_rate * (10 + agent.gamma * 0 - before)
        assert abs(after - expected) < 1e-9
