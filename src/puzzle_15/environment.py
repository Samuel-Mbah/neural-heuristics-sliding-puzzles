"""
15-Puzzle Environment
=====================
Defines the Puzzle15 class, which represents the 4x4 sliding tile puzzle
(also known as the 15-puzzle).  The blank tile is encoded as 0 and the
goal state is:

    1  2  3  4
    5  6  7  8
    9 10 11 12
   13 14 15  0

This module also exports two convenience constants used throughout the
other modules:

    NANO_TO_SEC  – conversion factor from nanoseconds to seconds
    INF          – a large integer used as a sentinel "infinity" value
"""

import numpy as np

NANO_TO_SEC = 1_000_000_000
INF = 100_000


class Puzzle15:
    """4x4 sliding tile puzzle (the 15-puzzle).

    The internal state representation is a flat *tuple* of 16 integers
    where 0 denotes the blank tile and tiles 1-15 represent the numbered
    tiles.  The goal state is ``(1, 2, ..., 15, 0)``.
    """

    def __init__(self):
        self.goal_state = tuple(range(1, 16)) + (0,)
        # 2-D numpy array representation of the goal state (used for task
        # generation and pretty-printing). Blank tile (0) at bottom-right [3,3].
        self.goal = np.array(
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12],
             [13, 14, 15, 0]]
        )
        # (row_delta, col_delta) pairs corresponding to up, down, left, right
        self.DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # ------------------------------------------------------------------
    # State inspection helpers
    # ------------------------------------------------------------------

    def is_solved(self, state: tuple) -> bool:
        """Return True when *state* equals the goal state."""
        return state == self.goal_state

    def checkWin(self, state: tuple) -> bool:
        """Alias for :meth:`is_solved` (kept for backward compatibility)."""
        return self.is_solved(state)

    # ------------------------------------------------------------------
    # Move generation
    # ------------------------------------------------------------------

    def get_possible_moves(self, state) -> list:
        """Return a list of valid index-delta moves for the blank tile.

        The blank tile can move:
          -4  (up),  +4 (down),  -1 (left),  +1 (right)
        depending on its current position.

        Parameters
        ----------
        state:
            Either a flat tuple/list of 16 ints *or* a 2-D numpy array.
        """
        if isinstance(state, np.ndarray):
            zero_pos = np.where(state == 0)
            row, col = zero_pos[0][0], zero_pos[1][0]
        else:
            flat = list(state)
            idx = flat.index(0)
            row, col = divmod(idx, 4)

        moves = []
        if row > 0:
            moves.append(-4)   # up
        if row < 3:
            moves.append(4)    # down
        if col > 0:
            moves.append(-1)   # left
        if col < 3:
            moves.append(1)    # right
        return moves

    def apply_move(self, state: tuple, move: int) -> tuple:
        """Return the new state after sliding the blank tile by *move* steps."""
        zero_idx = state.index(0)
        new_idx = zero_idx + move
        new_state = list(state)
        new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
        return tuple(new_state)

    def simulateMove(self, state: tuple, direction: tuple):
        """Simulate moving the blank tile in *direction* = (row_delta, col_delta).

        Returns
        -------
        (valid, new_state) : (bool, tuple | None)
            *valid* is False when the move would leave the board; in that
            case *new_state* is None.
        """
        zero_idx = state.index(0)
        new_idx = zero_idx + direction[0] * 4 + direction[1]

        # Boundary checks
        if new_idx < 0 or new_idx >= 16:
            return False, None
        if direction == (-1, 0) and zero_idx // 4 == 0:
            return False, None
        if direction == (1, 0) and zero_idx // 4 == 3:
            return False, None
        if direction == (0, -1) and zero_idx % 4 == 0:
            return False, None
        if direction == (0, 1) and zero_idx % 4 == 3:
            return False, None

        new_state = list(state)
        new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
        return True, tuple(new_state)

    # ------------------------------------------------------------------
    # Heuristic / cost helpers
    # ------------------------------------------------------------------

    def get_cost_to_goal(self, state: tuple) -> int:
        """Compute the Manhattan-distance heuristic for *state*."""
        distance = 0
        for i, tile in enumerate(state):
            if tile == 0:
                continue
            goal_pos = tile - 1
            current_row, current_col = divmod(i, 4)
            goal_row, goal_col = divmod(goal_pos, 4)
            distance += abs(current_row - goal_row) + abs(current_col - goal_col)
        return distance

    # ------------------------------------------------------------------
    # Feature encoding
    # ------------------------------------------------------------------

    def encode_state(self, state: tuple) -> np.ndarray:
        """Encode *state* as a 1-D numpy array suitable for neural network input."""
        return np.array(state)

    # ------------------------------------------------------------------
    # Misc utilities
    # ------------------------------------------------------------------

    def hash(self, state: tuple, group: list) -> tuple:
        """Return a canonical hash of the tiles in *group* for pattern DBs."""
        return tuple(sorted([state[i] for i in group]))
