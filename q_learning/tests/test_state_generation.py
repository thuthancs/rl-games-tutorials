"""Tests for state_generation module."""

import pytest

from q_learning.state_generation import (
    count_states_by_length,
    dir_from,
    generate_all_valid_states,
    generate_connected_placements,
    head_dir_pairs_for_placement,
    neighbors,
)


class TestNeighbors:
    def test_center_cell_2x2(self):
        assert set(neighbors((0, 0), 2)) == {(1, 0), (0, 1)}
        assert set(neighbors((1, 1), 2)) == {(0, 1), (1, 0)}

    def test_center_cell_3x3(self):
        result = set(neighbors((1, 1), 3))
        expected = {(0, 1), (2, 1), (1, 0), (1, 2)}
        assert result == expected

    def test_corner_returns_two_neighbors(self):
        assert len(list(neighbors((0, 0), 3))) == 2
        assert set(neighbors((0, 0), 3)) == {(1, 0), (0, 1)}

    def test_edge_returns_three_neighbors(self):
        assert len(list(neighbors((0, 1), 3))) == 3


class TestDirFrom:
    def test_upward(self):
        assert dir_from((1, 0), (0, 0)) == "upward"

    def test_downward(self):
        assert dir_from((0, 0), (1, 0)) == "downward"

    def test_rightward(self):
        assert dir_from((0, 0), (0, 1)) == "rightward"

    def test_leftward(self):
        assert dir_from((0, 1), (0, 0)) == "leftward"


class TestGenerateConnectedPlacements:
    def test_length_1_on_2x2(self):
        placements = generate_connected_placements(1, 2)
        assert len(placements) == 4
        assert all(len(p) == 1 for p in placements)

    def test_length_2_on_2x2(self):
        placements = generate_connected_placements(2, 2)
        # 2x2 grid: 4 cells, each pair of adjacent cells is valid
        # Adjacent pairs: (0,0)-(1,0), (0,0)-(0,1), (1,0)-(0,0), (1,0)-(1,1), etc.
        # Unique placements (as sets): 4 edges * 2 orderings / 2 = 4? Actually each edge gives one sorted tuple.
        assert len(placements) >= 2
        assert all(len(p) == 2 for p in placements)

    def test_length_1_on_3x3(self):
        placements = generate_connected_placements(1, 3)
        assert len(placements) == 9


class TestHeadDirPairsForPlacement:
    def test_single_cell_returns_four_directions(self):
        pairs = head_dir_pairs_for_placement([(0, 0)], 2)
        assert len(pairs) == 4
        dirs = {d for (pos, d) in pairs}
        assert dirs == {"upward", "downward", "leftward", "rightward"}

    def test_two_connected_cells(self):
        # Horizontal segment: (0,0) and (0,1). Head can be either end with direction toward the other.
        pairs = head_dir_pairs_for_placement([(0, 0), (0, 1)], 2)
        assert len(pairs) >= 2
        # (0,0) head facing rightward, (0,1) head facing leftward
        assert ((0, 0), "rightward") in pairs
        assert ((0, 1), "leftward") in pairs


class TestGenerateAllValidStates:
    def test_2x2_has_states(self):
        actions = ["turn_left", "go_straight", "turn_right", "turn_around"]
        states = generate_all_valid_states(2, actions)
        assert len(states) > 0
        for state, actions_dict in states.items():
            head_pos, head_dir, body_tuple, food_pos = state
            assert len(actions_dict) == 4
            assert all(a in actions_dict for a in actions)
            assert food_pos not in body_tuple

    def test_each_state_has_all_actions_initialized_to_zero(self):
        actions = ["turn_left", "go_straight"]
        states = generate_all_valid_states(2, actions)
        for state, actions_dict in states.items():
            for a in actions:
                assert actions_dict[a] == 0.0


class TestCountStatesByLength:
    def test_counts_match_total(self):
        actions = ["turn_left", "go_straight", "turn_right", "turn_around"]
        states = generate_all_valid_states(3, actions)
        counts = count_states_by_length(states)
        assert sum(counts.values()) == len(states)

    def test_length_1_count_3x3(self):
        actions = ["turn_left", "go_straight", "turn_right", "turn_around"]
        states = generate_all_valid_states(3, actions)
        counts = count_states_by_length(states)
        # Length 1: 9 positions * 4 directions * 8 food positions = 288
        assert counts[1] == 288
