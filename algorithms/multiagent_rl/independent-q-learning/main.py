"""
╔══════════════════════════════════════════════════════════════════╗
║   Multiagent Reinforcement Learning: Coin Collection Game        ║
║   Independent Q-Learning  |  Game Theory Analysis  |  Viz       ║
╠══════════════════════════════════════════════════════════════════╣
║  Game Theory Concepts:                                           ║
║    · Nash Equilibrium       - stable joint policy                ║
║    · Zero-sum vs Cooperative- tunable reward structure           ║
║    · Non-stationarity       - each agent changes the other's env ║
║    · Tragedy of the Commons - competitive over-exploitation      ║
║    · Territory emergence    - cooperative spatial division       ║
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

import sys

import matplotlib.pyplot as plt

from config import Config
from plotting import plot_comparison, plot_training, plot_value_heatmaps
from training import compare_modes
from visualization import animate_episode, replay_ascii


def main():
    QUICK = "--quick" in sys.argv
    n_eps = 1_000 if QUICK else 5_000

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
    fig1.savefig("marl_training_competitive.png", dpi=150, bbox_inches="tight")

    fig2 = plot_training(coop_log, cfg, title_suffix="Cooperative")
    fig2.savefig("marl_training_cooperative.png", dpi=150, bbox_inches="tight")

    fig3 = plot_value_heatmaps(comp_agents, cfg)
    fig3.savefig("marl_heatmaps_competitive.png", dpi=150, bbox_inches="tight")

    fig4 = plot_value_heatmaps(coop_agents, cfg)
    fig4.savefig("marl_heatmaps_cooperative.png", dpi=150, bbox_inches="tight")

    fig5 = plot_comparison(comp_log, coop_log, cfg)
    fig5.savefig("marl_comparison.png", dpi=150, bbox_inches="tight")

    print("  Plots saved:")
    for name in [
        "marl_training_competitive.png",
        "marl_training_cooperative.png",
        "marl_heatmaps_competitive.png",
        "marl_heatmaps_cooperative.png",
        "marl_comparison.png",
    ]:
        print(f"    -> {name}")

    # 4. Animated episode
    print("\n  Rendering animated episode (competitive) ...")
    _, _ = animate_episode(
        comp_agents, cfg,
        steps=50,
        interval_ms=300,
        save_path="marl_episode.gif",
    )

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()
