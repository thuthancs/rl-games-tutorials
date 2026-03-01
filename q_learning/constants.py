"""Constants and default configuration for Q-Learning Snake."""

# Direction name -> (dr, dc) for row/col
DIRECTIONS = {
    "upward": (-1, 0),
    "downward": (1, 0),
    "leftward": (0, -1),
    "rightward": (0, 1),
}

# Relative actions (from the agent)
ACTIONS = ["turn_left", "go_straight", "turn_right", "turn_around"]

# Head direction names (for state representation)
HEAD_DIRECTIONS = ["upward", "downward", "rightward", "leftward"]

# Default hyperparameters
GRID_SIZE = 4
LEARNING_RATE = 0.2
NUM_EPISODES = 20000
MAX_STEPS_PER_EPISODE = 200
MOVING_AVG_WINDOW = 50
