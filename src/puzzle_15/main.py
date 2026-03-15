"""
15-Puzzle: Training Entry Point
================================
Run this script to train the FFNN and WUNN heuristics for the 15-puzzle
using the LearnHeuristicPrac algorithm, and then evaluate the trained
FFNN on a held-out test set.

Usage
-----
    python -m src.puzzle_15.main
or simply:
    python src/puzzle_15/main.py
"""

import numpy as np
import tensorflow as tf

from .algorithms import learn_heuristic_prac, solve_task_with_ida_star, generate_task_prac
from .environment import Puzzle15


def check_gpu():
    """Print whether a GPU is available for TensorFlow."""
    if tf.config.list_physical_devices("GPU"):
        print("GPU is available.")
    else:
        print("GPU is not available, using CPU.")


def analyze_results(results: list) -> None:
    """Print a summary table of IDA* evaluation results.

    Parameters
    ----------
    results:
        List of ``(plan | None)`` items returned by
        :func:`~algorithms.solve_task_with_ida_star`.
    """
    generated_nodes = []
    planning_times = []
    suboptimalities = []
    optimal_solved = 0
    avg_optimal_cost = 53.05  # Korf (1985) reports ~53 moves average optimal solution length

    for plan in results:
        if plan is not None:
            num_steps = len(plan)
            generated_nodes.append(num_steps)
            # Use plan length as a proxy for planning effort
            planning_times.append(num_steps)
            suboptimalities.append((num_steps - avg_optimal_cost) / avg_optimal_cost)
            if num_steps <= avg_optimal_cost:
                optimal_solved += 1

    n = len(results)
    if not generated_nodes:
        print("No tasks were solved during evaluation.")
        return

    print("\n=== Evaluation Results ===")
    print(f"Tasks evaluated : {n}")
    print(f"Tasks solved    : {len(generated_nodes)} ({len(generated_nodes)/n*100:.1f}%)")
    print(f"Avg plan length : {np.mean(generated_nodes):.2f}")
    print(f"Avg plan time   : {np.mean(planning_times):.2f} (proxy)")
    print(f"Avg suboptimality: {np.mean(suboptimalities)*100:.2f}%")
    print(f"Optimal solved  : {optimal_solved} ({optimal_solved/n*100:.1f}%)")


def main():
    check_gpu()

    # -----------------------------------------------------------------------
    # Training parameters
    # -----------------------------------------------------------------------
    params = {
        "num_iter": 10,
        "num_tasks_per_iter": 10,
        "num_tasks_per_iter_thresh": 6,
        "alpha0": 0.99,
        "delta": 0.05,
        "epsilon": 0.005,
        "beta0": 0.005,
        "gamma": 0.64,
        "kappa": 0,
        "max_steps": 1000,
        "memory_buffer_max_records": 1000,
        "train_iter": 100,
        "max_train_iter": 100,
        "minibatch_size": 10,
        "tmax": 60,
        "mu0": 0,
        "sigma0_2": 10,
        "q": 0.95,
        "K": 100,
    }

    print("Starting LearnHeuristicPrac training …")
    ffnn, wunn = learn_heuristic_prac(**params)
    print("Training complete.")

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    puzzle = Puzzle15()
    test_tasks = generate_task_prac(
        puzzle,
        num_tasks_per_iter=2,
        length_inc=[5, 10, 15, 20, 25],
    )

    print(f"\nEvaluating on {len(test_tasks)} test tasks …")
    results = [
        solve_task_with_ida_star(task, ffnn, params["alpha0"], params["tmax"], params["max_steps"], params["K"])
        for task in test_tasks
    ]

    analyze_results(results)


if __name__ == "__main__":
    main()
