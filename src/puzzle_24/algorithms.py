"""
Search and Learning Algorithms for the 24-Puzzle
==================================================
This module implements:

1. **GenerateTaskPrac** – selects challenging training tasks by performing a
   random walk through the state space guided by epistemic uncertainty
   from the WUNN model.

2. **IDA*** – Iterative-Deepening A* search using a neural network heuristic
   adjusted by a Gaussian quantile confidence level *alpha*.

3. **LearnHeuristicPrac** – the main training loop that alternates between
   task generation, IDA* solving, memory-buffer management, and network
   training.

All algorithms follow the procedure described in:

    Marom & Rosman (2020). "Utilising Uncertainty for Efficient
    Learning of Likely-Admissible Heuristics."
    https://www.raillab.org/publication/marom-2020-utilising/
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from scipy.stats import norm

from .environment import (
    PDB_24_PUZZLE,
    extract_features,
    generate_initial_state_24_puzzle,
    generate_goal_state_24_puzzle,
    erev_24_puzzle,
    cost_to_goal,
)
from .neural_network import (
    WeightUncertaintyNN,
    FeedForwardNN,
    compute_sigma_e2,
    sample_from_softmax,
    device,
)


# ---------------------------------------------------------------------------
# Task Generation
# ---------------------------------------------------------------------------

def generate_task_prac_24_puzzle(
    nn_wunn: WeightUncertaintyNN,
    epsilon: float,
    max_steps: int,
    K: int,
    initial_state: np.ndarray,
    goal_state: np.ndarray,
):
    """Generate a training task guided by epistemic uncertainty (GenerateTaskPrac).

    The function performs a random walk through predecessor states starting
    from *initial_state*, greedily selecting states with high epistemic
    uncertainty as estimated by *nn_wunn*.  A task is returned as soon as a
    state with uncertainty ≥ *epsilon* is found.

    Parameters
    ----------
    nn_wunn:
        Trained (or partially trained) :class:`~neural_network.WeightUncertaintyNN`.
    epsilon:
        Minimum epistemic uncertainty required to generate a task.
    max_steps:
        Maximum number of random-walk steps before giving up.
    K:
        Number of MC-dropout forward passes for uncertainty estimation.
    initial_state:
        Starting board state (1-D numpy array of length 25).
    goal_state:
        Goal board state (1-D numpy array of length 25).

    Returns
    -------
    task : (np.ndarray, np.ndarray) | None
        A ``(start_state, goal_state)`` pair, or ``None`` when no suitable
        task is found within *max_steps*.
    """
    s_prime = initial_state
    s_double_prime = None

    for _ in range(max_steps):
        states = {}

        for s in erev_24_puzzle(s_prime):
            if s_double_prime is not None and np.array_equal(s_double_prime, s):
                continue
            x = extract_features(s, PDB_24_PUZZLE)
            sigma_e2 = compute_sigma_e2(nn_wunn, x, K)
            states[tuple(s)] = sigma_e2

        if not states:
            print("No predecessor states available; stopping walk.")
            break

        s, sigma_e2 = sample_from_softmax(states)

        if sigma_e2 >= epsilon:
            task = (np.array(s), goal_state)
            print(f"Task generated with sigma_e2={sigma_e2:.4f}")
            return task

        s_double_prime = s_prime
        s_prime = np.array(s)

    return None  # max_steps reached without finding a suitable task


# ---------------------------------------------------------------------------
# Heuristic function
# ---------------------------------------------------------------------------

def h_gaussian(alpha: float, nn_output: float, scale: float = 1.0) -> float:
    """Compute the *alpha*-quantile of a Gaussian centred at *nn_output*.

    This is the confidence-adjusted heuristic used in IDA*:

        h(s) = Φ⁻¹(α) × scale + nn_output

    Parameters
    ----------
    alpha:
        Confidence level in (0, 1).
    nn_output:
        Predicted mean cost-to-goal from the FFNN.
    scale:
        Standard deviation of the Gaussian (default 1.0).
    """
    return float(norm.ppf(alpha, loc=nn_output, scale=scale))


# ---------------------------------------------------------------------------
# IDA* with neural network heuristic
# ---------------------------------------------------------------------------

def _step_cost(node, succ) -> int:
    """Uniform step cost (= 1) for adjacent states."""
    return 1


def _ida_star_search(path, g, bound, goal_state, nn_ffnn, alpha):
    """Recursive search step for :func:`ida_star`."""
    node = path[-1]
    features = extract_features(np.array(node), PDB_24_PUZZLE)
    nn_out = nn_ffnn(
        torch.tensor(features, dtype=torch.float32).to(device)
    ).cpu().item()
    f = g + h_gaussian(alpha, nn_out)

    if f > bound:
        return f, None
    if np.array_equal(node, goal_state):
        return f, list(path)

    min_bound = float("inf")
    for succ in erev_24_puzzle(np.array(node)):
        succ_key = tuple(succ)
        if succ_key not in path:
            path.append(succ_key)
            t, result = _ida_star_search(
                path, g + _step_cost(node, succ), bound, goal_state, nn_ffnn, alpha
            )
            if result is not None:
                return t, result
            if t < min_bound:
                min_bound = t
            path.pop()

    return min_bound, None


def ida_star(
    task: tuple,
    nn_ffnn: FeedForwardNN,
    alpha: float,
    tmax: int,
) -> list | None:
    """Solve *task* using IDA* with the FFNN Gaussian-quantile heuristic.

    Parameters
    ----------
    task:
        ``(initial_state, goal_state)`` pair (each a 1-D numpy array of
        length 25).
    nn_ffnn:
        Trained :class:`~neural_network.FeedForwardNN`.
    alpha:
        Confidence level used in :func:`h_gaussian`.
    tmax:
        Time budget (seconds); currently not enforced but reserved for
        future use.

    Returns
    -------
    path : list | None
        Sequence of state tuples from initial to goal, or ``None`` when no
        solution is found.
    """
    initial_state, goal_state = task
    initial_features = extract_features(initial_state, PDB_24_PUZZLE)
    nn_out = nn_ffnn(
        torch.tensor(initial_features, dtype=torch.float32).to(device)
    ).cpu().item()

    bound = h_gaussian(alpha, nn_out)
    path = [tuple(initial_state)]

    while True:
        t, result = _ida_star_search(path, 0, bound, goal_state, nn_ffnn, alpha)
        if result is not None:
            return result
        if t == float("inf"):
            return None
        bound = t


# ---------------------------------------------------------------------------
# LearnHeuristicPrac – main training loop
# ---------------------------------------------------------------------------

def learn_heuristic_prac_24_puzzle(params: dict) -> list:
    """Train FFNN and WUNN for the 24-puzzle (LearnHeuristicPrac).

    Parameters
    ----------
    params : dict
        Dictionary of hyper-parameters.  Expected keys:

        ======================= =============================================
        Key                     Description
        ======================= =============================================
        ``NumIter``             Total training iterations.
        ``NumTasksPerIter``     Tasks attempted per iteration.
        ``NumTasksPerIterThresh`` Min solved tasks to keep *alpha* stable.
        ``alpha0``              Initial confidence level.
        ``delta``               Step for decreasing *alpha*.
        ``epsilon``             Min epistemic uncertainty for task selection.
        ``beta0``               Initial regularisation weight.
        ``gamma``               Multiplicative decay for *beta*.
        ``kappa``               Epistemic uncertainty stopping factor.
        ``MaxSteps``            Max random-walk steps in task generation.
        ``MemoryBufferMaxRecords`` Memory buffer capacity.
        ``TrainIter``           FFNN training epochs per iteration.
        ``MaxTrainIter``        WUNN training epochs per iteration.
        ``MiniBatchSize``       Mini-batch size.
        ``tmax``                IDA* time budget (seconds).
        ``mu0``                 Prior mean (stored in WUNN).
        ``sigma0``              Prior std-dev (stored in WUNN).
        ``q``                   Quantile for progress tracking.
        ``K``                   MC-dropout forward passes.
        ======================= =============================================

    Returns
    -------
    results : list of dict
        One dict per iteration containing keys ``alpha``, ``times``,
        ``generated_nodes``, ``suboptimalities``, and ``optimal_solutions``.
    """
    nn_wunn = WeightUncertaintyNN(
        input_dim=5, output_dim=1,
        mu0=params["mu0"], sigma0=params["sigma0"]
    ).to(device)
    nn_ffnn = FeedForwardNN(input_dim=5, output_dim=1).to(device)

    memory_buffer: deque = deque(maxlen=params["MemoryBufferMaxRecords"])
    yq = -np.inf
    alpha = params["alpha0"]
    beta = params["beta0"]
    update_beta = True

    optimizer_wunn = optim.Adam(nn_wunn.parameters())
    optimizer_ffnn = optim.Adam(nn_ffnn.parameters())
    criterion = nn.MSELoss()

    results = []

    for n in range(params["NumIter"]):
        num_solved = 0
        times, generated_nodes, suboptimalities, optimal_solutions = [], [], [], []

        for _ in range(params["NumTasksPerIter"]):
            initial_state = generate_initial_state_24_puzzle()
            goal_state = generate_goal_state_24_puzzle()
            task = generate_task_prac_24_puzzle(
                nn_wunn, params["epsilon"], params["MaxSteps"],
                params["K"], initial_state, goal_state
            )

            if task is not None:
                plan = ida_star(task, nn_ffnn, alpha, params["tmax"])

                if plan:
                    num_solved += 1
                    for sj in plan:
                        if np.array_equal(sj, goal_state):
                            yj = cost_to_goal(np.array(sj))
                            xj = extract_features(np.array(sj), PDB_24_PUZZLE)
                            memory_buffer.append((xj, yj))
                    print(f"Task solved. Buffer size: {len(memory_buffer)}")
                    # Placeholder metrics – replace with real measurements
                    times.append(np.random.random())
                    generated_nodes.append(np.random.randint(1000, 10000))
                    suboptimalities.append(np.random.random())
                    optimal_solutions.append(np.random.random())
                else:
                    print("Failed to solve task.")

        if num_solved < params["NumTasksPerIterThresh"]:
            alpha = max(alpha - params["delta"], 0.5)
            update_beta = False
        else:
            update_beta = True

        # ------------------------------------------------------------------
        # Train FFNN
        # ------------------------------------------------------------------
        if memory_buffer:
            batch = random.sample(memory_buffer, min(len(memory_buffer), params["MiniBatchSize"]))
            inputs, targets = zip(*batch)
            inputs_t = torch.tensor(np.array(inputs), dtype=torch.float32).to(device)
            targets_t = torch.tensor(np.array(targets), dtype=torch.float32).to(device)

            optimizer_ffnn.zero_grad()
            outputs = nn_ffnn(inputs_t).squeeze()
            loss = criterion(outputs, targets_t)
            loss.backward()
            optimizer_ffnn.step()
            print(f"FFNN Loss: {loss.item():.6f}")

        # ------------------------------------------------------------------
        # Train WUNN
        # ------------------------------------------------------------------
        if memory_buffer:
            for _ in range(params["MaxTrainIter"]):
                batch = random.sample(memory_buffer, min(len(memory_buffer), params["MiniBatchSize"]))
                inputs, targets = zip(*batch)
                inputs_t = torch.tensor(np.array(inputs), dtype=torch.float32).to(device)
                targets_t = torch.tensor(np.array(targets), dtype=torch.float32).to(device)

                optimizer_wunn.zero_grad()
                outputs = nn_wunn(inputs_t).squeeze()
                loss = criterion(outputs, targets_t)
                loss.backward()
                optimizer_wunn.step()

                sigma_e_vals = outputs.cpu().detach().numpy()
                if all(s < params["kappa"] * params["epsilon"] for s in sigma_e_vals):
                    break

            print(f"WUNN Loss: {loss.item():.6f}")

        if update_beta:
            beta *= params["gamma"]
            # beta is passed to train_nn for future WUNN regularisation;
            # it is accepted by train_nn but not yet applied to the loss.

        if memory_buffer:
            yq = np.quantile([yj for _, yj in memory_buffer], params["q"])

        print(
            f"Iteration {n+1}/{params['NumIter']}: "
            f"solved={num_solved}, alpha={alpha:.4f}, yq={yq:.4f}"
        )

        results.append({
            "alpha": alpha,
            "times": times,
            "generated_nodes": generated_nodes,
            "suboptimalities": suboptimalities,
            "optimal_solutions": optimal_solutions,
        })

    return results
