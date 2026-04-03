"""
Episode visualization: animated Matplotlib replay and ASCII terminal replay.
"""

import time
from typing import List, Optional

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from agent import QLearningAgent
from config import ACTION_NAMES, CFG, Config
from env import CoinCollectionEnv
from plotting import COLORS


def animate_episode(
    agents:      List[QLearningAgent],
    cfg:         Config = CFG,
    steps:       int    = 60,
    interval_ms: int    = 350,
    save_path:   Optional[str] = None,
):
    """
    Matplotlib animation of a single greedy episode.
    Shows agent positions, coins, and last action taken.

    Parameters
    ----------
    save_path : str or None
        If given, saves the animation as a GIF (requires Pillow).
    """
    env   = CoinCollectionEnv(cfg.grid_size, cfg.num_coins, cfg.cooperative)
    state = env.reset()
    g     = cfg.grid_size

    # Pre-compute all frames
    frames = []
    for _ in range(steps):
        actions = [agents[i].select_action(state, greedy=True) for i in range(2)]
        frame   = dict(
            agent_pos=list(env.agent_positions),
            coin_pos=set(env.coin_positions),
            actions=actions,
        )
        frames.append(frame)
        state, _, _, _ = env.step(actions)

    # set up figure
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, g - 0.5)
    ax.set_ylim(-0.5, g - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(range(g))
    ax.set_yticks(range(g))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="#cccccc", linewidth=0.5)
    ax.set_facecolor(COLORS["grid_bg"])

    agent_circles = [
        plt.Circle((0, 0), 0.35, color=COLORS["agent_A"], zorder=4),
        plt.Circle((0, 0), 0.35, color=COLORS["agent_B"], zorder=4),
    ]
    agent_labels = []
    for circle, label in zip(agent_circles, ["A", "B"]):
        ax.add_patch(circle)
        txt = ax.text(0, 0, label, ha="center", va="center",
                      fontsize=10, fontweight="bold", color="white", zorder=5)
        agent_labels.append(txt)

    coin_artists: List[plt.Artist] = []
    title = ax.set_title("", fontsize=10)

    legend_patches = [
        mpatches.Patch(color=COLORS["agent_A"], label="Agent A"),
        mpatches.Patch(color=COLORS["agent_B"], label="Agent B"),
        mpatches.Patch(color=COLORS["coin_fill"], label="Coin"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
              framealpha=0.8, bbox_to_anchor=(1.0, 1.0))

    def _draw_frame(t):
        fr = frames[t]
        for art in coin_artists:
            art.remove()
        coin_artists.clear()
        for pos in fr["coin_pos"]:
            c   = plt.Circle((pos[1], pos[0]), 0.25, color=COLORS["coin_fill"], zorder=3)
            txt = ax.text(pos[1], pos[0], "$", ha="center", va="center",
                          fontsize=8, color="#412402", fontweight="bold", zorder=4)
            ax.add_patch(c)
            coin_artists.extend([c, txt])
        for i, (circle, lbl) in enumerate(zip(agent_circles, agent_labels)):
            r, c_col = fr["agent_pos"][i]
            circle.center = (c_col, r)
            lbl.set_position((c_col, r))
        mode = "Cooperative" if cfg.cooperative else "Competitive"
        title.set_text(
            f"Step {t+1}/{steps}  [{mode}, compact state]\n"
            f"A->{ACTION_NAMES[fr['actions'][0]]}  "
            f"B->{ACTION_NAMES[fr['actions'][1]]}"
        )
        return agent_circles + agent_labels + coin_artists + [title]

    _draw_frame(0)
    anim = animation.FuncAnimation(
        fig, _draw_frame, frames=steps,
        interval=interval_ms, blit=False, repeat=True,
    )

    if save_path:
        print(f"  Saving animation -> {save_path}")
        anim.save(save_path, writer="pillow", fps=int(1000 / interval_ms))
        print("  Saved.")

    plt.tight_layout()
    return fig, anim


def replay_ascii(
    agents: List[QLearningAgent],
    cfg:    Config = CFG,
    steps:  int    = 20,
    delay:  float  = 0.0,   # seconds between frames (0 = no delay)
):
    """Print a greedy episode step-by-step in the terminal."""
    env    = CoinCollectionEnv(cfg.grid_size, cfg.num_coins, cfg.cooperative)
    state  = env.reset()
    scores = [0.0, 0.0]

    print("\n" + "-" * 40)
    print(f"  Greedy Episode Replay ({cfg.grid_size}x{cfg.grid_size}, compact state)")
    print("-" * 40)

    for t in range(steps):
        grid    = env.render_ascii()
        actions = [agents[i].select_action(state, greedy=True) for i in range(2)]
        next_state, rewards, _, info = env.step(actions)
        for i in range(2):
            scores[i] += rewards[i]

        print(f"\nStep {t+1:>3}  "
              f"A->{ACTION_NAMES[actions[0]]:<5}  "
              f"B->{ACTION_NAMES[actions[1]]:<5}  "
              f"coins={info['coins_collected']}  "
              f"score=[{scores[0]:.0f}, {scores[1]:.0f}]")
        print(grid)
        state = next_state
        if delay > 0:
            time.sleep(delay)

    print(f"\n  Final scores -> A: {scores[0]:.0f}   B: {scores[1]:.0f}")
