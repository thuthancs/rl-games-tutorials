"""State generation utilities for Q-Learning Snake Game.

This module contains functions to generate all valid game states for a snake game
on a grid of any size. States are represented as tuples of:
(head_position, head_direction, body_tuple, food_position)
"""

from typing import List, Tuple, Set, Dict


def neighbors(cell: Tuple[int, int], grid_size: int):
    """Generate valid neighboring cells within the grid.

    Args:
        cell: (row, col) tuple representing the current cell
        grid_size: size of the grid

    Yields:
        Tuple[int, int]: valid neighboring cells
    """
    r, c = cell

    # Loop through all possible movements: up, down, left, right
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc

        # Check if the new position is within the grid boundaries
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            yield (nr, nc)


def dir_from(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
    """Determine direction from one position to another.

    Args:
        from_pos: the coordinates of the starting position
        to_pos: the coordinates of the target position

    Returns:
        str: the direction from from_pos to to_pos ('upward', 'downward', 'rightward', 'leftward')
    """
    dr, dc = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]

    if dr == -1:
        return "upward"
    if dr == 1:
        return "downward"
    if dc == 1:
        return "rightward"
    return "leftward"


def generate_connected_placements(length: int, grid_size: int) -> List[Tuple]:
    """Generate all connected placements of a snake of given length on the grid.

    Args:
        length: the length of the snake
        grid_size: size of the grid

    Returns:
        List[Tuple]: list of unique snake placements (as sorted tuples of cells)
    """
    cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    placements = set()

    def dfs(path):
        # If the path has reached the desired length, record it
        if len(path) == length:
            placements.add(tuple(sorted(path)))
            return

        # If the path is not yet complete, extend it
        tail = path[-1]
        for nb in neighbors(tail, grid_size):
            if nb not in path:
                dfs(path + [nb])

    # Start DFS from each cell in the grid
    for cell in cells:
        dfs([cell])

    return [tuple(p) for p in placements]


def head_dir_pairs_for_placement(
    shape_cells: List[Tuple[int, int]], grid_size: int
) -> Set:
    """Generate all valid (head_pos, head_dir) pairs for a given snake shape.

    Args:
        shape_cells: a list of (r, c) tuples representing the snake's body cells
        grid_size: size of the grid

    Returns:
        set: a set of (head_pos, head_dir) pairs
    """
    shape_cells = set(shape_cells)
    pairs = set()

    # Special-case single cell: any facing is valid
    if len(shape_cells) == 1:
        only = next(iter(shape_cells))
        for d in ["upward", "downward", "rightward", "leftward"]:
            pairs.add((only, d))
        return pairs

    def dfs(path):
        """Recursive DFS to build paths through the shape cells."""
        # If the path covers all shape cells, record the head and direction
        if len(path) == len(shape_cells):
            head, second = path[0], path[1]
            pairs.add((head, dir_from(head, second)))
            return

        # If not complete, extend the path
        tail = path[-1]
        for nb in neighbors(tail, grid_size):
            # Only consider neighbors that are part of the shape and not already in the path
            if nb in shape_cells and nb not in path:
                dfs(path + [nb])

    # Start DFS from each cell in the shape
    for start in shape_cells:
        dfs([start])
    return pairs


def generate_all_valid_states(grid_size: int, actions: List[str]) -> Dict:
    """Generate all valid game states for a given grid size.

    Args:
        grid_size: size of the grid
        actions: list of available actions (used to initialize Q-values)

    Returns:
        Dict: dictionary mapping states to action dictionaries initialized to 0.0
              Format: {(head_pos, head_dir, body_tuple, food_pos): {action: 0.0, ...}, ...}
    """
    game_states = {}
    food_positions = [(x, y) for x in range(grid_size) for y in range(grid_size)]

    for length in range(1, grid_size * grid_size + 1):
        placements = generate_connected_placements(length, grid_size)

        for placement in placements:
            pairs = head_dir_pairs_for_placement(placement, grid_size)

            for head_pos, head_dir in pairs:
                body_tuple = tuple(placement)

                for food_pos in food_positions:
                    if food_pos not in placement:
                        state = (head_pos, head_dir, body_tuple, food_pos)
                        game_states[state] = {action: 0.0 for action in actions}

    return game_states


def count_states_by_length(game_states: Dict) -> Dict[int, int]:
    """Count the number of valid states for each snake length.

    Args:
        game_states: dictionary of game states

    Returns:
        Dict[int, int]: {length: count, ...}
    """
    length_counts = {}

    for (head_pos, head_dir, body_tuple, food_pos) in game_states.keys():
        length = len(body_tuple)
        length_counts[length] = length_counts.get(length, 0) + 1

    return length_counts
