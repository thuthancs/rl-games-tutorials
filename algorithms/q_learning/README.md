# Q-Learning Snake

Python package extracted from `notebooks/q_learning_snake.ipynb`. Implements state generation, game environment, Q-learning agent, and training/plotting for the Snake game.

## Structure

- **`state_generation.py`** – Generate all valid game states: `neighbors`, `dir_from`, `generate_connected_placements`, `head_dir_pairs_for_placement`, `generate_all_valid_states`, `count_states_by_length`.
- **`constants.py`** – `DIRECTIONS`, `ACTIONS`, `HEAD_DIRECTIONS`, and default hyperparameters.
- **`game.py`** – `Snake`, `GameEnvironment`, `GameLogic` (move, place_food, get_state, render).
- **`agent.py`** – `get_current_direction`, `get_state_representation`, `get_direction`, `QLearningAgent` (Q-table, epsilon-greedy, Q-learning update).
- **`training.py`** – `run_training`, `moving_average`, `plot_learning_trajectories` (requires `matplotlib`, `tqdm`).
- **`main.py`** – Entry point: train with different epsilon configs and save learning curve plot.

## Install

From repo root:

```bash
pip install -r q_learning/requirements.txt
```

## Run training

From repo root:

```bash
python -m q_learning.main
```

This runs training for three epsilon regimes and saves `learning_trajectories.png` in the current directory.

## Run tests

From repo root (install dependencies first: `pip install -r q_learning/requirements.txt`):

```bash
python -m pytest q_learning/tests -v
```

Tests cover state generation (`neighbors`, `dir_from`, placements, `generate_all_valid_states`, `count_states_by_length`), game logic (move, wall/self collision, eating food), and the agent (`get_current_direction`, `get_direction`, `get_state_representation`, `QLearningAgent` set_q_table, get_q_value, choose_action, update_q_value).

## Use as a library

```python
from q_learning import (
    generate_all_valid_states,
    count_states_by_length,
    QLearningAgent,
    GameLogic,
    get_state_representation,
    ACTIONS,
    HEAD_DIRECTIONS,
)
from q_learning.training import run_training, plot_learning_trajectories
```
