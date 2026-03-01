"""Entry point for training the Q-Learning Snake agent."""

import random

from q_learning.constants import MOVING_AVG_WINDOW, NUM_EPISODES
from q_learning.training import plot_learning_trajectories, run_training


def main() -> None:
    # Make runs reproducible
    random.seed(42)

    epsilon_configs = {
        "Pure exploitation (epsilon=0.0)": 0.0,
        "Pure exploration (epsilon=1.0)": 1.0,
        "Mixed (epsilon=0.1)": 0.1,
    }

    results: dict = {}

    for label, eps in epsilon_configs.items():
        scores = run_training(
            train_epsilon=eps, gamma=0.5, num_episodes=NUM_EPISODES
        )
        results[label] = scores

    plot_learning_trajectories(
        results,
        num_episodes=NUM_EPISODES,
        window_size=MOVING_AVG_WINDOW,
    )


if __name__ == "__main__":
    main()
