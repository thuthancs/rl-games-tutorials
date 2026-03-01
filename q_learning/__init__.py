"""Q-Learning Snake: state generation, game environment, agent, and training."""

from q_learning.agent import (
    QLearningAgent,
    get_current_direction,
    get_direction,
    get_state_representation,
)
from q_learning.constants import (
    ACTIONS,
    DIRECTIONS,
    HEAD_DIRECTIONS,
)
from q_learning.game import GameEnvironment, GameLogic, Snake
from q_learning.state_generation import (
    count_states_by_length,
    dir_from,
    generate_all_valid_states,
    generate_connected_placements,
    head_dir_pairs_for_placement,
    neighbors,
)

# Training helpers (require matplotlib and tqdm) are available as:
# from q_learning.training import run_training, moving_average, plot_learning_trajectories

__all__ = [
    # state_generation
    "neighbors",
    "dir_from",
    "generate_connected_placements",
    "head_dir_pairs_for_placement",
    "generate_all_valid_states",
    "count_states_by_length",
    # game
    "Snake",
    "GameEnvironment",
    "GameLogic",
    # agent
    "get_current_direction",
    "get_state_representation",
    "get_direction",
    "QLearningAgent",
    # constants
    "DIRECTIONS",
    "ACTIONS",
    "HEAD_DIRECTIONS",
]
