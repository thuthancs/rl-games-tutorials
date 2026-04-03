"""
Matplotlib visualizations: training dashboards, Q-value heatmaps,
and competitive-vs-cooperative comparison charts.
"""

from typing import List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from agent import QLearningAgent
from config import Config
from training import TrainingLog

COLORS = {
    "agent_A":   "#185FA5",
    "agent_B":   "#D85A30",
    "coins":     "#3B6D11",
    "epsilon":   "#7F77DD",
    "td_error":  "#993556",
    "grid_bg":   "#F1EFE8",
    "coin_fill": "#EF9F27",
}


def _smooth(data: list, w: int) -> np.ndarray:
    return np.convolve(data, np.ones(w) / w, mode="valid")


def plot_training(log: TrainingLog, cfg: Config, title_suffix: str = ""):
    """Four-panel training dashboard."""
    w   = cfg.smooth_window
    fig = plt.figure(figsize=(18, 4))
    fig.suptitle(
        f"Training Curves - {'Cooperative' if cfg.cooperative else 'Competitive'}"
        f"{' | ' + title_suffix if title_suffix else ''}",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    # 1. Rewards
    ax = fig.add_subplot(gs[0])
    for i, (label, color) in enumerate(zip(["Agent A", "Agent B"],
                                           [COLORS["agent_A"], COLORS["agent_B"]])):
        s = _smooth(log.rewards[i], w)
        ax.plot(s, color=color, lw=1.5, label=label)
    ax.set_title("Reward / episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # 2. Coins
    ax = fig.add_subplot(gs[1])
    s = _smooth(log.coins, w)
    ax.plot(s, color=COLORS["coins"], lw=1.5)
    ax.set_title("Coins collected / episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Coins")
    ax.grid(alpha=0.25)

    # 3. TD error
    ax = fig.add_subplot(gs[2])
    for i, (label, color) in enumerate(zip(["Agent A", "Agent B"],
                                           [COLORS["agent_A"], COLORS["agent_B"]])):
        s = _smooth(log.td_error[i], w)
        ax.plot(s, color=color, lw=1.5, label=label, alpha=0.8)
    ax.set_title("Mean |TD error| / episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("|TD error|")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # 4. Epsilon
    ax = fig.add_subplot(gs[3])
    ax.plot(log.epsilon, color=COLORS["epsilon"], lw=1.5)
    ax.set_title("Exploration epsilon decay")
    ax.set_xlabel("Episode")
    ax.set_ylabel("epsilon")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    return fig


def plot_value_heatmaps(agents: List[QLearningAgent], cfg: Config):
    """Side-by-side Q-value heatmaps for both agents."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    mode = "Cooperative" if cfg.cooperative else "Competitive"
    fig.suptitle(f"Q-Value Heatmaps ({mode})", fontsize=13, fontweight="bold")

    labels = ["Agent A", "Agent B"]

    for i, (ax, agent) in enumerate(zip(axes, agents)):
        vmap = agent.value_map(cfg.grid_size)
        # replace NaN with min for display
        vmin = np.nanmin(vmap) if not np.all(np.isnan(vmap)) else 0
        vmap_display = np.where(np.isnan(vmap), vmin, vmap)
        im = ax.imshow(
            vmap_display,
            cmap="Blues" if i == 0 else "Oranges",
            aspect="equal",
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="max Q")
        ax.set_title(f"{labels[i]} - max Q per cell\n(avg over known states)")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        for r in range(cfg.grid_size):
            for c in range(cfg.grid_size):
                v = vmap[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if v > (np.nanmax(vmap) * 0.6) else "black")

    plt.tight_layout()
    return fig


def plot_comparison(comp_log: TrainingLog, coop_log: TrainingLog, cfg: Config):
    """
    Side-by-side comparison of competitive vs cooperative training runs.
    Shows rewards and total coins on the same axes.
    """
    w   = cfg.smooth_window
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Competitive vs Cooperative - Head-to-Head Comparison",
                 fontsize=13, fontweight="bold")

    # rewards
    ax = axes[0]
    for i, (label, color) in enumerate(zip(["A", "B"],
                                           [COLORS["agent_A"], COLORS["agent_B"]])):
        s_c = _smooth(comp_log.rewards[i], w)
        s_o = _smooth(coop_log.rewards[i], w)
        ax.plot(s_c, color=color, lw=1.5, ls="--", label=f"Agent {label} (comp)")
        ax.plot(s_o, color=color, lw=1.5, ls="-",  label=f"Agent {label} (coop)", alpha=0.7)
    ax.set_title("Individual reward / episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)

    # total coins
    ax = axes[1]
    sc = _smooth(comp_log.coins, w)
    so = _smooth(coop_log.coins, w)
    ax.plot(sc, color="#A32D2D", lw=1.5, ls="--", label="Competitive")
    ax.plot(so, color="#0F6E56", lw=1.5, ls="-",  label="Cooperative")
    ax.set_title("Total coins / episode\n(social welfare)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Coins")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    x = np.arange(min(len(sc), len(so)))
    ax.fill_between(x, sc[:len(x)], so[:len(x)],
                    where=(so[:len(x)] > sc[:len(x)]),
                    alpha=0.15, color="#0F6E56", label="Coop advantage")

    # TD error
    ax = axes[2]
    sc = _smooth(comp_log.td_error[0], w)
    so = _smooth(coop_log.td_error[0], w)
    ax.plot(sc, color="#A32D2D", lw=1.5, ls="--", label="Competitive")
    ax.plot(so, color="#0F6E56", lw=1.5, ls="-",  label="Cooperative")
    ax.set_title("Agent A - |TD error| / episode\n(learning signal strength)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("|TD error|")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    return fig
