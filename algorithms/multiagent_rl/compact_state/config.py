"""
Global configuration, action definitions, and random seed setup.
All other modules import from here — nothing here imports from the project.
"""

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    # Grid
    grid_size:     int   = 6       # NxN grid
    num_coins:     int   = 4       # coins alive on board at once
    # Training
    num_episodes:  int   = 10_000   # training episodes
    max_steps:     int   = 50      # max steps per episode
    cooperative:   bool  = False   # True -> shared reward; False -> competitive
    # Q-learning
    alpha:         float = 0.10    # learning rate
    gamma:         float = 0.95    # discount factor
    epsilon_start: float = 1.00    # initial exploration
    epsilon_end:   float = 0.05    # minimum exploration
    epsilon_decay: float = 0.9995  # per-episode multiplicative decay
    # Misc
    seed:          int   = 42
    log_interval:  int   = 500     # print progress every N episodes
    smooth_window: int   = 200     # rolling-average window for plots


CFG = Config()

# Actions: up, down, left, right, stay
ACTIONS      = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
ACTION_NAMES = ["Up", "Down", "Left", "Right", "Stay"]
NUM_ACTIONS  = len(ACTIONS)

random.seed(CFG.seed)
np.random.seed(CFG.seed)
