"""
╔══════════════════════════════════════════════════════════════════╗
║   Multiagent RL: Coin Collection — Compact State Representation  ║
╠══════════════════════════════════════════════════════════════════╣
║  Problem with the original coin_collector.py                     ║
║  The original state included frozenset(coin_positions), which    ║
║  has C(36,4) = 58,905 possible configurations on a 6x6 grid.    ║
║  Combined with 36x36 agent positions, the total state space is   ║
║  ~76 million. With only 250k training steps, most states are     ║
║  visited once at most, so Q-values stay near zero and the greedy ║
║  policy degrades worse than random exploration as epsilon decays.║
║                                                                  ║
║  Fix: compact state representation                               ║
║  Replace frozenset(coins) with the Manhattan-nearest coin delta  ║
║  (dr, dc) for each agent. dr in [-5,5], dc in [-5,5] on a 6x6   ║
║  grid -> 11x11 = 121 values per agent.                           ║
║                                                                  ║
║  New state space: 36 x 36 x 121 x 121 ~ 19M                     ║
║  But crucially: the same (agent_pos, coin_delta) pattern now     ║
║  maps to the SAME state regardless of where other coins are,     ║
║  so Q-values generalise. Visited states cluster at ~10k-50k.     ║
╠══════════════════════════════════════════════════════════════════╣
║  Install:  pip install numpy matplotlib                          ║
╚══════════════════════════════════════════════════════════════════╝

Full pipeline:
  1. Train in competitive mode
  2. Train in cooperative mode (for comparison)
  3. Run game theory analysis on both
  4. Plot training curves, value heatmaps, comparison
  5. Animate a greedy episode (competitive)

Usage:
  python main.py           # full run
  python main.py --quick   # 1 000 episodes for rapid testing
"""

import os
import sys

import matplotlib.pyplot as plt

from config import Config
from plotting import plot_comparison, plot_training, plot_value_heatmaps
from training import compare_modes
from visualization import animate_episode, replay_ascii

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")


def _img(name: str) -> str:
    """Return full path inside the images/ directory."""
    return os.path.join(IMAGES_DIR, name)


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    QUICK = "--quick" in sys.argv
    n_eps = 1_000 if QUICK else 10_000

    cfg = Config(num_episodes=n_eps)

    # 1. Train both modes
    comp_result, coop_result = compare_modes(cfg)

    comp_agents, comp_log = comp_result["agents"], comp_result["log"]
    coop_agents, coop_log = coop_result["agents"], coop_result["log"]

    # 2. ASCII replay (competitive)
    replay_ascii(comp_agents, cfg, steps=15)

    # 3. Plots
    print("\n  Generating plots ...")

    fig1 = plot_training(comp_log, cfg, title_suffix="Competitive")
    fig1.savefig(_img("marl_compact_training_competitive.png"), dpi=150, bbox_inches="tight")

    fig2 = plot_training(coop_log, cfg, title_suffix="Cooperative")
    fig2.savefig(_img("marl_compact_training_cooperative.png"), dpi=150, bbox_inches="tight")

    fig3 = plot_value_heatmaps(comp_agents, cfg)
    fig3.savefig(_img("marl_compact_heatmaps_competitive.png"), dpi=150, bbox_inches="tight")

    fig4 = plot_value_heatmaps(coop_agents, cfg)
    fig4.savefig(_img("marl_compact_heatmaps_cooperative.png"), dpi=150, bbox_inches="tight")

    fig5 = plot_comparison(comp_log, coop_log, cfg)
    fig5.savefig(_img("marl_compact_comparison.png"), dpi=150, bbox_inches="tight")

    print("  Plots saved to images/:")
    for name in [
        "marl_compact_training_competitive.png",
        "marl_compact_training_cooperative.png",
        "marl_compact_heatmaps_competitive.png",
        "marl_compact_heatmaps_cooperative.png",
        "marl_compact_comparison.png",
    ]:
        print(f"    -> images/{name}")

    # 4. Animated episode
    print("\n  Rendering animated episode (competitive) ...")
    _, _ = animate_episode(
        comp_agents, cfg,
        steps=50,
        interval_ms=300,
        save_path=_img("marl_compact_episode.gif"),
    )

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()
