"""Smoke tests for compact_state without mixing imports with independent-q-learning."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_COMPACT = _ROOT / "compact_state"


def test_compact_reset_state_has_four_components() -> None:
    code = """
import random
random.seed(0)
from env import CoinCollectionEnv
e = CoinCollectionEnv(4, 2, False)
s = e.reset()
assert len(s) == 4, s
assert len(s[2]) == 2 and len(s[3]) == 2
"""
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_COMPACT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_compact_step_positions_bounded() -> None:
    code = """
import random
random.seed(42)
from env import CoinCollectionEnv
from config import NUM_ACTIONS
e = CoinCollectionEnv(4, 2, False)
e.reset()
for _ in range(30):
    a = [random.randrange(NUM_ACTIONS) for _ in range(2)]
    e.step(a)
    for pos in e.agent_positions:
        r, c = pos
        assert 0 <= r < 4 and 0 <= c < 4
"""
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_COMPACT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
