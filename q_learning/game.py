"""Snake game environment and logic for Q-Learning."""

import os
import random
from typing import Optional


class Snake:
    """The snake entity: list of cell positions (head first) and eating flag."""

    def __init__(self, initial_length: int = 1) -> None:
        """Initialize snake with default length and single cell at (1, 1)."""
        self.length = initial_length
        self.just_ate_food = False
        self.snake_positions = [(1, 1)]


class GameEnvironment:
    """Grid world: size, score, and current food position."""

    def __init__(self, grid_size: int) -> None:
        """Create a grid_size x grid_size environment with food at center."""
        self.grid_size = grid_size
        # Backwards-compatible alias used throughout the codebase
        self.size = grid_size
        self.score = 0
        self.grid = [[0] * grid_size for _ in range(grid_size)]
        self.food_pos = (grid_size // 2, grid_size // 2)


class GameLogic:
    """Game logic for the Snake environment."""

    # make sure that the down is 1 and up is -1 for the row index
    DOWN = (1, 0)
    UP = (-1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    def __init__(self, grid_size: int, score_history: Optional[list] = None) -> None:
        """Create game with given grid size; optionally pass score history (unused)."""
        if score_history is None:
            score_history = []
        self.GameEnvironment = GameEnvironment(grid_size)
        self.Snake = Snake()
        self.directions = [self.DOWN, self.UP, self.LEFT, self.RIGHT]
        self.highest_score = self.load_high_score()

    def load_high_score(self) -> int:
        """Load highest score from highscore.txt; return 0 if missing or invalid."""
        if os.path.exists("highscore.txt"):
            with open("highscore.txt", "r") as f:
                try:
                    return int(f.read().strip())
                except ValueError:
                    return 0
        return 0

    def save_high_score(self) -> None:
        """Write current score to highscore.txt if it exceeds the stored high score."""
        if self.GameEnvironment.score > self.highest_score:
            self.highest_score = self.GameEnvironment.score
            with open("highscore.txt", "w") as f:
                f.write(str(self.highest_score))

    def place_food(self) -> None:
        """Set food to a random empty cell (not occupied by the snake)."""
        empty = [
            (r, c)
            for r in range(self.GameEnvironment.grid_size)
            for c in range(self.GameEnvironment.grid_size)
            if (r, c) not in self.Snake.snake_positions
        ]
        self.GameEnvironment.food_pos = random.choice(empty)

    def move(self, direction: tuple) -> None:
        """Move the snake one step in the given (dr, dc) direction.

        Raises:
            Exception: "Game Over: Self-collision" or "Game Over: Wall-collision" on invalid move.
        """
        head = self.Snake.snake_positions[0]

        # Calculate the new head position based on the direction
        new_head = (head[0] + direction[0], head[1] + direction[1])

        # Check self-collision
        if new_head in self.Snake.snake_positions[1:]:
            raise Exception("Game Over: Self-collision")

        # Check wall-collision
        if not (
            0 <= new_head[0] < self.GameEnvironment.grid_size
            and 0 <= new_head[1] < self.GameEnvironment.grid_size
        ):
            raise Exception("Game Over: Wall-collision")

        # Add the new head position to the snake's positions
        self.Snake.snake_positions.insert(0, new_head)

        # Check if the snake has eaten food
        if new_head == self.GameEnvironment.food_pos:
            self.Snake.just_ate_food = True
            self.GameEnvironment.score += 1
            self.place_food()

        # If the snake has not just eaten food, remove the last position
        if not self.Snake.just_ate_food:
            self.Snake.snake_positions.pop()
        else:
            self.Snake.just_ate_food = False

    def stop_game(self) -> int:
        """Save high score if applicable and return the current score."""
        self.save_high_score()
        return self.GameEnvironment.score

    def get_state(self) -> dict:
        """Return a dict with keys: snake, food, score, grid_size."""
        return {
            "snake": self.Snake.snake_positions,
            "food": self.GameEnvironment.food_pos,
            "score": self.GameEnvironment.score,
            "grid_size": self.GameEnvironment.size,
        }

    def render(self) -> None:
        """Print an ASCII grid with 'S' for snake and 'F' for food."""
        size = self.GameEnvironment.grid_size
        grid = [["."] * size for _ in range(size)]

        for r, c in self.Snake.snake_positions:
            grid[r][c] = "S"

        fr, fc = self.GameEnvironment.food_pos
        grid[fr][fc] = "F"

        for row in grid:
            print(" ".join(row))
        print()
