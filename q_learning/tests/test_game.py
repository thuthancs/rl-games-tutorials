"""Tests for game module (Snake, GameEnvironment, GameLogic)."""

import random

import pytest

from q_learning.game import GameEnvironment, GameLogic, Snake


class TestSnake:
    def test_initial_state(self):
        s = Snake()
        assert s.snake_positions == [(1, 1)]
        assert s.just_ate_food is False


class TestGameEnvironment:
    def test_grid_size(self):
        env = GameEnvironment(4)
        assert env.grid_size == 4
        assert env.size == 4
        assert env.score == 0
        assert env.food_pos == (2, 2)


class TestGameLogic:
    def test_move_right_in_bounds(self):
        random.seed(42)
        game = GameLogic(grid_size=3)
        game.place_food()
        # Snake starts at (1,1). Move right -> (1, 2)
        game.move(game.RIGHT)
        assert game.Snake.snake_positions[0] == (1, 2)
        assert len(game.Snake.snake_positions) == 1

    def test_wall_collision_raises(self):
        random.seed(42)
        game = GameLogic(grid_size=2)
        game.place_food()
        # Snake at (1,1). Move right -> (1, 2) is out of bounds for 2x2 (cols 0-1)
        with pytest.raises(Exception, match="Wall"):
            game.move(game.RIGHT)

    def test_self_collision_raises(self):
        random.seed(42)
        game = GameLogic(grid_size=3)
        game.place_food()
        # Set snake explicitly: head (1,2), body (1,1). Moving LEFT goes to (1,1) = body.
        game.Snake.snake_positions = [(1, 2), (1, 1)]
        with pytest.raises(Exception, match="Self"):
            game.move(game.LEFT)

    def test_eat_food_increases_score_and_length(self):
        random.seed(42)
        game = GameLogic(grid_size=3)
        # Force food to a known position: place snake at (1,1), put food at (1,2)
        game.Snake.snake_positions = [(1, 1)]
        game.GameEnvironment.food_pos = (1, 2)
        assert game.GameEnvironment.score == 0
        game.move(game.RIGHT)  # (1,1) -> (1,2) eats food
        assert game.GameEnvironment.score == 1
        assert len(game.Snake.snake_positions) == 2
        assert game.Snake.snake_positions[0] == (1, 2)

    def test_get_state_keys(self):
        random.seed(42)
        game = GameLogic(grid_size=3)
        game.place_food()
        state = game.get_state()
        assert "snake" in state
        assert "food" in state
        assert "score" in state
        assert "grid_size" in state
        assert state["grid_size"] == 3
