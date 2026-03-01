"""Training loop and plotting for Q-Learning Snake."""

from typing import Dict, List

import matplotlib.pyplot as plt
from tqdm import tqdm

from q_learning.agent import (
    QLearningAgent,
    get_current_direction,
    get_direction,
    get_state_representation,
)
from q_learning.constants import (
    ACTIONS,
    GRID_SIZE,
    MAX_STEPS_PER_EPISODE,
    MOVING_AVG_WINDOW,
    NUM_EPISODES,
)
from q_learning.game import GameLogic


def run_training(
    train_epsilon: float,
    gamma: float,
    num_episodes: int = NUM_EPISODES,
    grid_size: int = GRID_SIZE,
    learning_rate: float = 0.2,
) -> List[float]:
    """Train a Q-learning agent and return evaluation scores per episode.

    For each episode, one training step (with train_epsilon) is followed by one
    evaluation episode (epsilon=0, pure exploitation). The evaluation score
    is appended to the returned list.

    Args:
        train_epsilon: Exploration probability during training.
        gamma: Discount factor for future rewards.
        num_episodes: Number of training (and evaluation) episodes.
        grid_size: Side length of the grid.
        learning_rate: Q-learning step size.

    Returns:
        List of length num_episodes: evaluation score after each training episode.
    """
    agent = QLearningAgent(
        actions=ACTIONS,
        learning_rate=learning_rate,
        epsilon=train_epsilon,
        gamma=gamma,
        grid_size=grid_size,
        num_episodes=num_episodes,
    )

    # Start from a fresh Q-table for each training run
    agent.set_q_table()

    eval_scores: List[float] = []

    for episode in tqdm(
        range(num_episodes), desc=f"Training (epsilon={train_epsilon})"
    ):
        # -------- Training episode (epsilon = train_epsilon) --------
        game = GameLogic(grid_size=agent.grid_size)
        game.place_food()

        steps = 0
        while True:
            steps += 1
            current_state = get_state_representation(game)
            action = agent.choose_action(current_state, train_epsilon)
            current_dir = get_current_direction(game.Snake.snake_positions)
            direction = get_direction(action, current_dir)

            try:
                old_score = game.GameEnvironment.score
                game.move(direction)
                new_score = game.GameEnvironment.score

                if new_score > old_score:
                    reward = 10  # Ate food
                else:
                    reward = 0  # Normal step

                next_state = get_state_representation(game)

            except Exception as e:
                if "Wall" in str(e):
                    reward = -10
                elif "Self" in str(e):
                    reward = -10
                else:
                    reward = -10

                next_state = current_state  # Terminal state
                agent.update_q_value(current_state, action, reward, next_state)
                break

            agent.update_q_value(current_state, action, reward, next_state)

            if steps >= MAX_STEPS_PER_EPISODE:
                break

        # -------- Evaluation episode (epsilon = 0.0, pure exploitation) --------
        eval_game = GameLogic(grid_size=agent.grid_size)
        eval_game.place_food()

        eval_steps = 0
        while True:
            eval_steps += 1
            eval_state = get_state_representation(eval_game)
            eval_action = agent.choose_action(eval_state, 0.0)
            eval_current_dir = get_current_direction(eval_game.Snake.snake_positions)
            eval_direction = get_direction(eval_action, eval_current_dir)

            try:
                eval_game.move(eval_direction)
            except Exception:
                break

            if eval_steps >= MAX_STEPS_PER_EPISODE:
                break

        eval_scores.append(eval_game.GameEnvironment.score)

    return eval_scores


def moving_average(values: List[float], window_size: int) -> List[float]:
    """Compute a simple moving average over the given values.

    Args:
        values: Sequence of numbers to smooth.
        window_size: Number of recent values to average (or fewer at the start).

    Returns:
        List of same length as values; each element is the average of the
        current and previous (window_size - 1) values, or all values so far
        near the start.
    """
    if not values or window_size <= 1:
        return values

    averaged: List[float] = []
    cumulative_sum = 0.0

    for i, v in enumerate(values):
        cumulative_sum += v
        if i >= window_size:
            cumulative_sum -= values[i - window_size]
            averaged.append(cumulative_sum / window_size)
        else:
            averaged.append(cumulative_sum / (i + 1))

    return averaged


def plot_learning_trajectories(
    results: Dict[str, List[float]],
    num_episodes: int = NUM_EPISODES,
    window_size: int = MOVING_AVG_WINDOW,
    save_path: str = "learning_trajectories.png",
) -> None:
    """Plot smoothed (moving-average) score per episode for each series and save to file.

    Args:
        results: Dict mapping legend label to list of per-episode scores.
        num_episodes: Number of episodes (x-axis length).
        window_size: Moving average window for smoothing.
        save_path: Output path for the PNG figure.
    """
    episodes = list(range(1, num_episodes + 1))

    plt.figure(figsize=(10, 6))

    for label, scores in results.items():
        smoothed = moving_average(scores, window_size)
        plt.plot(episodes, smoothed, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Average score per episode (moving average)")
    plt.title(
        f"Snake Q-learning: Learning trajectories (episode 1-{num_episodes})\n"
        f"(window size = {window_size})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
