"""
Search and Learning Algorithms for the 15-Puzzle
==================================================
This module implements:

1. **GenerateTaskPrac** – generates training tasks at increasing difficulty
   levels by scrambling the goal state with a sequence of random moves.

2. **IDA*** – Iterative-Deepening A* search, with two variants:
   - :func:`ida_star_manhattan` – uses the Manhattan-distance heuristic
     (no neural network; useful for baseline testing).
   - :func:`ida_star_nn` – uses a trained neural network as the heuristic.

3. **LearnHeuristicPrac** – the main learning loop that alternates between
   solving tasks with IDA* and training the neural network heuristics.

All algorithms are adapted from the procedure described in:

    Marom & Rosman (2020). "Utilising Uncertainty for Efficient
    Learning of Likely-Admissible Heuristics."
    https://www.raillab.org/publication/marom-2020-utilising/
"""

import random
import numpy as np
from time import perf_counter_ns

from .environment import Puzzle15, NANO_TO_SEC, INF
from .neural_network import create_ffnn, create_wunn, train_nn


# ---------------------------------------------------------------------------
# Task Generation
# ---------------------------------------------------------------------------

def generate_task_prac(puzzle: Puzzle15, num_tasks_per_iter: int, length_inc: list) -> list:
    """Generate a set of scrambled puzzle tasks (GenerateTaskPrac).

    Each task is obtained by taking the goal state and applying a random
    walk of *inc* moves, where *inc* cycles over the values in
    *length_inc*.  Tasks at increasing difficulty levels support a
    curriculum-learning approach.

    Parameters
    ----------
    puzzle:
        A :class:`~environment.Puzzle15` instance.
    num_tasks_per_iter:
        Number of tasks to generate for each difficulty level.
    length_inc:
        Sequence of scramble lengths, e.g. ``[1, 2, 4, 6, 8, 10]``.

    Returns
    -------
    tasks : list of tuple
        Each element is a flat tuple representing a scrambled board state.
    """
    tasks = []
    for inc in length_inc:
        for _ in range(num_tasks_per_iter):
            state = puzzle.goal_state
            for _ in range(inc):
                moves = puzzle.get_possible_moves(state)
                move = random.choice(moves)
                state = puzzle.apply_move(state, move)
            tasks.append(state)
    return tasks


# ---------------------------------------------------------------------------
# IDA* with Manhattan-distance heuristic (baseline, no model)
# ---------------------------------------------------------------------------

def _h_manhattan(puzzle: Puzzle15, state: tuple) -> int:
    """Compute the Manhattan-distance heuristic for *state*."""
    h = 0
    for idx, tile in enumerate(state):
        if tile == 0:
            continue
        goal_row = (tile - 1) // 4
        goal_col = (tile - 1) % 4
        cur_row, cur_col = divmod(idx, 4)
        h += abs(goal_row - cur_row) + abs(goal_col - cur_col)
    return h


def _search_manhattan(puzzle, path, g, bound, dirs):
    """Recursive search step for :func:`ida_star_manhattan`."""
    cur = path[-1]
    f = g + _h_manhattan(puzzle, cur)

    if f > bound:
        return f
    if puzzle.checkWin(cur):
        return True

    minimum = INF
    for direction in puzzle.DIRECTIONS:
        if dirs and (-direction[0], -direction[1]) == dirs[-1]:
            continue
        valid, next_state = puzzle.simulateMove(cur, direction)
        if not valid or next_state in path:
            continue

        path.append(next_state)
        dirs.append(direction)
        result = _search_manhattan(puzzle, path, g + 1, bound, dirs)
        if result is True:
            return True
        if result < minimum:
            minimum = result
        path.pop()
        dirs.pop()

    return minimum


def ida_star_manhattan(puzzle: Puzzle15, initial_state: tuple):
    """Solve *initial_state* with IDA* using the Manhattan-distance heuristic.

    Parameters
    ----------
    puzzle:
        A :class:`~environment.Puzzle15` instance.
    initial_state:
        Flat tuple representing the starting board configuration.

    Returns
    -------
    dirs : list of (int, int) | None
        Sequence of direction tuples that lead from *initial_state* to the
        goal, or ``None`` when no solution is found within the bound.
    """
    if puzzle.checkWin(initial_state):
        return []

    t_start = perf_counter_ns()
    bound = _h_manhattan(puzzle, initial_state)
    path = [initial_state]
    dirs = []

    while True:
        result = _search_manhattan(puzzle, path, 0, bound, dirs)
        if result is True:
            elapsed = (perf_counter_ns() - t_start) / NANO_TO_SEC
            print(f"Solved in {elapsed:.3f}s with {len(dirs)} moves.")
            return dirs
        if result == INF:
            return None
        bound = result


# ---------------------------------------------------------------------------
# IDA* with learned neural network heuristic
# ---------------------------------------------------------------------------

def _search_nn(puzzle, path, g, bound, dirs, heuristic_fn):
    """Recursive search step for :func:`ida_star_nn`."""
    cur = path[-1]
    f = g + heuristic_fn(cur)

    if f > bound:
        return f
    if puzzle.checkWin(cur):
        return True

    minimum = INF
    for direction in puzzle.DIRECTIONS:
        if dirs and (-direction[0], -direction[1]) == dirs[-1]:
            continue
        valid, next_state = puzzle.simulateMove(cur, direction)
        if not valid or next_state in path:
            continue

        path.append(next_state)
        dirs.append(direction)
        result = _search_nn(puzzle, path, g + 1, bound, dirs, heuristic_fn)
        if result is True:
            return True
        if result < minimum:
            minimum = result
        path.pop()
        dirs.pop()

    return minimum


def ida_star_nn(puzzle: Puzzle15, initial_state: tuple, model):
    """Solve *initial_state* with IDA* using a trained neural network heuristic.

    Parameters
    ----------
    puzzle:
        A :class:`~environment.Puzzle15` instance.
    initial_state:
        Flat tuple representing the starting board configuration.
    model:
        A trained Keras model whose ``predict`` method accepts a 2-D numpy
        array with shape ``(1, 16)`` and returns a scalar cost estimate.

    Returns
    -------
    dirs : list of (int, int) | None
        Solution path or ``None`` if the search exhausted all bounds.
    """
    heuristic_fn = lambda state: model.predict(
        np.array([puzzle.encode_state(state)]), verbose=0
    )[0][0]

    if puzzle.checkWin(initial_state):
        return []

    t_start = perf_counter_ns()
    bound = heuristic_fn(initial_state)
    path = [initial_state]
    dirs = []

    while True:
        result = _search_nn(puzzle, path, 0, bound, dirs, heuristic_fn)
        if result is True:
            elapsed = (perf_counter_ns() - t_start) / NANO_TO_SEC
            print(f"Solved in {elapsed:.3f}s with {len(dirs)} moves.")
            return dirs
        if result == INF:
            return None
        bound = result


def solve_task_with_ida_star(task: tuple, model, alpha, tmax, max_steps, K):
    """Wrapper that solves a single *task* using :func:`ida_star_nn`.

    Parameters
    ----------
    task:
        Flat tuple representing the starting board state.
    model:
        Trained Keras FFNN model.
    alpha, tmax, max_steps, K:
        Parameters reserved for future WUNN-guided exploration (currently
        passed through but not used in the IDA* call itself).

    Returns
    -------
    dirs : list | None
        Solution path from :func:`ida_star_nn`.
    """
    puzzle = Puzzle15()
    return ida_star_nn(puzzle, task, model)


# ---------------------------------------------------------------------------
# LearnHeuristicPrac – main training loop
# ---------------------------------------------------------------------------

def learn_heuristic_prac(
    num_iter: int,
    num_tasks_per_iter: int,
    num_tasks_per_iter_thresh: int,
    alpha0: float,
    delta: float,
    epsilon: float,
    beta0: float,
    gamma: float,
    kappa: float,
    max_steps: int,
    memory_buffer_max_records: int,
    train_iter: int,
    max_train_iter: int,
    minibatch_size: int,
    tmax: int,
    mu0: float,
    sigma0_2: float,
    q: float,
    K: int,
):
    """Train FFNN and WUNN heuristics using the LearnHeuristicPrac algorithm.

    The algorithm iterates between:
      1. Generating tasks and attempting to solve them with IDA* guided by
         the current FFNN heuristic.
      2. Collecting solved state–cost pairs into a memory buffer.
      3. Training both the FFNN and WUNN on the memory buffer.
      4. Adjusting the exploration parameter *alpha* based on the success
         rate (similar to simulated annealing).

    Parameters
    ----------
    num_iter:
        Total number of training iterations.
    num_tasks_per_iter:
        Number of tasks generated per iteration.
    num_tasks_per_iter_thresh:
        Minimum number of tasks that must be solved to keep *alpha* stable.
    alpha0:
        Initial confidence level for the Gaussian quantile heuristic.
    delta:
        Step by which *alpha* is decreased when success rate is too low.
    epsilon:
        Epistemic uncertainty threshold for task generation (WUNN).
    beta0:
        Initial WUNN regularisation weight.
    gamma:
        Multiplicative decay factor for *beta*.
    kappa:
        Scaling factor for the WUNN epistemic uncertainty stopping criterion.
    max_steps:
        Maximum random-walk steps during task generation.
    memory_buffer_max_records:
        Maximum number of records retained in the experience replay buffer.
    train_iter:
        Number of training epochs for the FFNN per iteration.
    max_train_iter:
        Number of training epochs for the WUNN per iteration.
    minibatch_size:
        Mini-batch size used when training both networks.
    tmax:
        Time budget (seconds) for a single IDA* search (currently unused
        in the search itself but retained for interface compatibility).
    mu0:
        Prior mean for WUNN weight initialisation (currently unused).
    sigma0_2:
        Prior variance for WUNN weight initialisation (currently unused).
    q:
        Quantile used to track the *y_q* progress indicator.
    K:
        Number of stochastic forward passes for epistemic uncertainty estimation.

    Returns
    -------
    ffnn, wunn : (keras.Sequential, keras.Sequential)
        The trained FFNN and WUNN models.
    """
    puzzle = Puzzle15()
    ffnn = create_ffnn()
    wunn = create_wunn()

    memory_buffer: list = []
    yq = -float("inf")
    alpha = alpha0
    beta = beta0
    update_beta = True

    for n in range(num_iter):
        num_solved = 0

        tasks = generate_task_prac(
            puzzle,
            num_tasks_per_iter=num_tasks_per_iter,
            length_inc=[1, 2, 4, 6, 8, 10],
        )

        for task in tasks:
            plan = solve_task_with_ida_star(task, ffnn, alpha, tmax, max_steps, K)
            if plan is not None:
                num_solved += 1
                # Reconstruct visited states from the starting task state and
                # the sequence of direction tuples returned by IDA*.
                visited_state = task
                for direction in plan:
                    visited_state = puzzle.simulateMove(visited_state, direction)[1]
                    cost_to_goal = puzzle.get_cost_to_goal(visited_state)
                    memory_buffer.append((puzzle.encode_state(visited_state), cost_to_goal))

        # Trim memory buffer to the most recent records
        if len(memory_buffer) > memory_buffer_max_records:
            memory_buffer = memory_buffer[-memory_buffer_max_records:]

        if num_solved < num_tasks_per_iter_thresh:
            alpha = max(alpha - delta, 0.5)
            update_beta = False
        else:
            update_beta = True

        if memory_buffer:
            train_nn(ffnn, memory_buffer, train_iter)
            train_nn(
                wunn,
                memory_buffer,
                max_train_iter,
                batch_size=minibatch_size,
                kappa=kappa,
                update_beta=update_beta,
                gamma=gamma,
                beta=beta,
            )
            yq = np.quantile([y for _, y in memory_buffer], q)

        print(
            f"Iteration {n + 1}/{num_iter}: "
            f"solved={num_solved}/{len(tasks)}, "
            f"alpha={alpha:.4f}, yq={yq:.4f}, "
            f"buffer_size={len(memory_buffer)}"
        )

    return ffnn, wunn
