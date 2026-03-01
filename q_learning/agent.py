"""Q-Learning agent for the Snake game."""

import random
from typing import Dict, Tuple

from q_learning.constants import DIRECTIONS
from q_learning.game import GameLogic
from q_learning.state_generation import generate_all_valid_states


def get_current_direction(snake_positions: list) -> str:
    """Determine the snake's current facing from head and neck positions.

    Args:
        snake_positions: List of (row, col) cells, head first.

    Returns:
        One of 'upward', 'downward', 'leftward', 'rightward'. Defaults to 'rightward' if length < 2.
    """
    if len(snake_positions) < 2:
        return "rightward"  # Default

    head, neck = snake_positions[0], snake_positions[1]
    dr, dc = head[0] - neck[0], head[1] - neck[1]

    if dr == -1:
        return "upward"
    if dr == 1:
        return "downward"
    if dc == 1:
        return "rightward"
    return "leftward"


def get_state_representation(game: GameLogic) -> Tuple:
    """Convert the current game state to the Q-learning state tuple.

    Args:
        game: Current GameLogic instance.

    Returns:
        (head_pos, head_dir, body_tuple, food_pos) for use as Q-table key.
    """
    head_pos = game.Snake.snake_positions[0]
    head_dir = get_current_direction(game.Snake.snake_positions)

    body_tuple = tuple(game.Snake.snake_positions)
    food_pos = game.GameEnvironment.food_pos

    return (head_pos, head_dir, body_tuple, food_pos)


def get_direction(action: str, current_direction: str) -> Tuple[int, int]:
    """Convert a relative action into an absolute (dr, dc) direction.

    Args:
        action: One of 'turn_left', 'go_straight', 'turn_right', 'turn_around'.
        current_direction: Current facing: 'upward', 'downward', 'leftward', or 'rightward'.

    Returns:
        (dr, dc) tuple for applying to head position.
    """
    turns = {
        "upward": {
            "turn_left": "leftward",
            "go_straight": "upward",
            "turn_right": "rightward",
            "turn_around": "downward",
        },
        "downward": {
            "turn_left": "rightward",
            "go_straight": "downward",
            "turn_right": "leftward",
            "turn_around": "upward",
        },
        "leftward": {
            "turn_left": "downward",
            "go_straight": "leftward",
            "turn_right": "upward",
            "turn_around": "rightward",
        },
        "rightward": {
            "turn_left": "upward",
            "go_straight": "rightward",
            "turn_right": "downward",
            "turn_around": "leftward",
        },
    }
    new_dir = turns[current_direction][action]
    return DIRECTIONS[new_dir]


class QLearningAgent:
    """Epsilon-greedy Q-learning agent with tabular Q(s, a)."""

    REWARD_FOOD = 10
    REWARD_DEATH = -10
    REWARD_STEP = 0

    def __init__(
        self,
        actions: list,
        learning_rate: float,
        epsilon: float,
        gamma: float,
        grid_size: int,
        num_episodes: int,
    ):
        """Define the attributes of the learning agent.

        Args:
            actions: a list of possible actions
            learning_rate: how much to update the q-values at each step
            epsilon: the probability of choosing a random action (exploration)
            gamma: the discounting factor for future rewards
            grid_size: the size of the game grid
            num_episodes: the number of training episodes
        """
        self.q_table: Dict = {}
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.actions = actions
        self.grid_size = grid_size
        self.num_episodes = num_episodes

    def set_q_table(self) -> Dict:
        """Initialize the Q-table with all valid game states and zero Q-values. Returns self.q_table."""
        self.q_table = generate_all_valid_states(self.grid_size, self.actions)

        for action in self.actions:
            for state in self.q_table:
                self.q_table[state][action] = 0.0
        return self.q_table

    def get_q_value(self, state: Tuple, action: str) -> float:
        """Return Q(state, action); 0.0 if state or action is missing."""
        return self.q_table.get(state, {}).get(action, 0.0)

    def choose_action(self, state: Tuple, epsilon: float) -> str:
        """Epsilon-greedy action: with probability epsilon random, else greedy (break ties by action order)."""
        if random.uniform(0, 1) < epsilon:
            return random.choice(self.actions)
        else:
            # Initialize state in Q-table if not present (defensive programming)
            if state not in self.q_table:
                self.q_table[state] = {a: 0.0 for a in self.actions}

            # Get all the q-values (state-action) for the current state
            state_actions = self.q_table[state]

            # Select the action with the highest q-value
            max_q = max(state_actions.values())

            # In case of multiple actions with the same max q-value, choose randomly among them
            best_actions = [action for action, q in state_actions.items() if q == max_q]

            # Use deterministic tie-breaking: always pick the first action in the original actions list
            for action in self.actions:
                if action in best_actions:
                    return action

            return self.actions[0]

    def update_q_value(
        self, state: Tuple, action: str, reward: int, next_state: Tuple
    ) -> float:
        """Update the Q-value for a given state-action pair based on the received reward and the maximum future Q-value.

        Args:
            state: the current state
            action: the action taken
            reward: the reward received after taking the action
            next_state: the state resulting from taking the action
        """
        # Initialize state in Q-table if not present
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}

        current_q_value = self.get_q_value(state, action)
        max_future_q = max(
            self.q_table.get(next_state, {}).values(), default=0.0
        )

        # Q-learning formula
        new_q_value = current_q_value + self.learning_rate * (
            reward + self.gamma * max_future_q - current_q_value
        )

        # Update the Q-table
        self.q_table[state][action] = new_q_value

        return self.q_table[state][action]
