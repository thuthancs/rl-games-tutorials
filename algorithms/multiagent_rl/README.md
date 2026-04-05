# Multi-Agent RL: Coin Collection

Two **Independent Q-Learning (IQL)** implementations on a shared coin-collection grid world: **full joint state** vs. **compact nearest-coin state**. See **[ALGORITHM.md](ALGORITHM.md)** for the ALGO-HC-style specification (inputs/outputs, steps, complexity, flowcharts, tests).

## Subprojects

| Folder | State representation |
| ------ | -------------------- |
| [`independent-q-learning/`](independent-q-learning/) | `(pos0, pos1, frozenset(coins))` |
| [`compact_state/`](compact_state/) | `(pos0, pos1, nearest_coin_delta0, nearest_coin_delta1)` |

Each is a self-contained script package: run from **inside** that folder so imports (`config`, `env`, …) resolve.

## Dependencies

```bash
pip install -r requirements.txt
```

For development (pytest):

```bash
pip install -r requirements-dev.txt
```

Requires **Python 3.9+**, **NumPy**, **Matplotlib**.

## Run training and plots

From `algorithms/multiagent_rl/independent-q-learning/` (or `compact_state/`):

```bash
python main.py           # full run (see each main.py for episode count)
python main.py --quick   # shorter run for smoke testing
```

## Run tests

From `algorithms/multiagent_rl/`:

```bash
python -m pytest tests -v
```

[`pytest.ini`](pytest.ini) disables the `langsmith` pytest plugin if present (avoids a known collection-time conflict on some setups). See [ALGORITHM.md](ALGORITHM.md) (section on tests) for what the tests cover.
