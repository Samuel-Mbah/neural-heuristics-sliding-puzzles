"""
24-Puzzle Environment
======================
State representation, utility functions, and feature extraction for the
5x5 sliding tile puzzle (24-puzzle).

The blank tile is encoded as 0 and the tiles are numbered 1–24.  The
goal state is the identity permutation ``[1, 2, …, 24, 0]`` represented
as a flat numpy array of length 25.

Pattern Databases
-----------------
Five pattern groups are defined (``PDB_24_PUZZLE``) as suggested by the
IDA* literature.  :func:`extract_features` maps a board state to a
compact feature vector by summing the tile values within each group.

Solvability
-----------
For a 5x5 board (odd grid width) a state is solvable if and only if the
number of inversions is even (:func:`is_solvable`).

Predecessor Generation
-----------------------
:func:`erev_24_puzzle` returns a placeholder list of predecessor states
(random permutations).  Replace this with an exact implementation when
proper predecessor logic is required.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Pattern Databases
# ---------------------------------------------------------------------------

#: Five disjoint tile groups used as pattern database keys for feature
#: extraction.  Indices are 1-based tile labels (0 = blank).
PDB_24_PUZZLE = [
    [1, 2, 5, 6, 7],
    [3, 4, 8, 9, 14],
    [10, 15, 16, 20, 21],
    [11, 12, 17, 22],
    [13, 18, 23, 24],
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(state: np.ndarray, pdb: list = None) -> np.ndarray:
    """Extract pattern-database features from *state*.

    For each tile group in *pdb*, the function sums the values of the
    non-blank tiles belonging to that group.

    Parameters
    ----------
    state:
        1-D numpy array of length 25 (0-indexed tile values, 0 = blank).
    pdb:
        List of tile groups.  Defaults to :data:`PDB_24_PUZZLE`.

    Returns
    -------
    features : np.ndarray
        1-D array of length ``len(pdb)``.
    """
    if pdb is None:
        pdb = PDB_24_PUZZLE

    features = []
    for pattern in pdb:
        # state is 0-based; tile labels in pdb are 1-based
        features.append(sum(state[i - 1] for i in pattern if state[i - 1] != 0))
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# State generation helpers
# ---------------------------------------------------------------------------

def generate_initial_state_24_puzzle() -> np.ndarray:
    """Return a random *solvable* 24-puzzle state.

    The function samples uniformly at random until a solvable permutation
    is found (expected ~2 attempts on average).

    Returns
    -------
    state : np.ndarray
        1-D array of length 25 (permutation of 0–24).
    """
    while True:
        state = np.random.permutation(25)
        if is_solvable(state):
            return state


def generate_goal_state_24_puzzle() -> np.ndarray:
    """Return the goal state ``[1, 2, …, 24, 0]`` as a 1-D numpy array.

    The blank tile (0) is at the last position (index 24).
    """
    return np.concatenate([np.arange(1, 25), [0]]).astype(int)


# ---------------------------------------------------------------------------
# Solvability check
# ---------------------------------------------------------------------------

def is_solvable(state: np.ndarray) -> bool:
    """Return True when *state* is a solvable 24-puzzle configuration.

    For a 5x5 (odd-width) board, a permutation is solvable iff the number
    of inversions among the non-blank tiles is even.

    Parameters
    ----------
    state:
        1-D array of length 25 (permutation of 0–24).
    """
    inversion_count = 0
    n = len(state)
    for i in range(n):
        for j in range(i + 1, n):
            if state[i] != 0 and state[j] != 0 and state[i] > state[j]:
                inversion_count += 1
    return inversion_count % 2 == 0


# ---------------------------------------------------------------------------
# Predecessor / successor generation (placeholder)
# ---------------------------------------------------------------------------

def erev_24_puzzle(s_prime: np.ndarray, num_predecessors: int = 5) -> list:
    """Generate predecessor states for *s_prime* (placeholder implementation).

    .. warning::
        This is a **placeholder** that returns random permutations.
        Replace with an exact predecessor function for meaningful results.

    Parameters
    ----------
    s_prime:
        Current state as a 1-D numpy array of length 25.
    num_predecessors:
        Number of predecessor states to return.

    Returns
    -------
    predecessors : list of np.ndarray
    """
    return [np.random.permutation(len(s_prime)) for _ in range(num_predecessors)]


# ---------------------------------------------------------------------------
# Cost-to-goal heuristic (placeholder)
# ---------------------------------------------------------------------------

def cost_to_goal(state: np.ndarray) -> float:
    """Compute a rough cost-to-goal estimate for *state* (placeholder).

    .. warning::
        This is a **placeholder** that uses the sum of tile values.
        Replace with the Manhattan-distance or a pattern-database lookup.

    Parameters
    ----------
    state:
        1-D numpy array of length 25.
    """
    return float(np.sum(state))
