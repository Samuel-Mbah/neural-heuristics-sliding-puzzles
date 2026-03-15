"""
24-Puzzle: Training Entry Point
================================
Run this script to train the FFNN and WUNN heuristics for the 24-puzzle
and then print a summary of per-iteration results.

Usage
-----
    python -m src.puzzle_24.main
or simply:
    python src/puzzle_24/main.py
"""

import numpy as np
import torch

from .algorithms import learn_heuristic_prac_24_puzzle
from .neural_network import device


def print_results(results: list) -> None:
    """Print a per-iteration summary table.

    Parameters
    ----------
    results:
        List of result dicts returned by
        :func:`~algorithms.learn_heuristic_prac_24_puzzle`.
    """
    print("\n=== Training Results (24-Puzzle) ===")
    print(f"{'Iter':>4}  {'Alpha':>6}  {'Times (mean)':>12}  {'Nodes (mean)':>12}  "
          f"{'Subopt (mean)':>13}  {'Optimal (mean)':>14}")
    print("-" * 70)

    for i, result in enumerate(results):
        alpha = result["alpha"]

        def _safe_mean(arr):
            return f"{np.mean(arr):.4f}" if arr else "N/A"

        print(
            f"{i+1:>4}  {alpha:>6.4f}  "
            f"{_safe_mean(result['times']):>12}  "
            f"{_safe_mean(result['generated_nodes']):>12}  "
            f"{_safe_mean(result['suboptimalities']):>13}  "
            f"{_safe_mean(result['optimal_solutions']):>14}"
        )


def main():
    print(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # Training parameters
    # -----------------------------------------------------------------------
    params = {
        "NumIter": 100,
        "NumTasksPerIter": 10,
        "NumTasksPerIterThresh": 5,
        "alpha0": 0.95,
        "delta": 0.05,
        "epsilon": 0.1,
        "beta0": 1.0,
        "gamma": 0.9,
        "kappa": 0.1,
        "MaxSteps": 100,
        "MemoryBufferMaxRecords": 1000,
        "TrainIter": 10,
        "MaxTrainIter": 10,
        "MiniBatchSize": 32,
        "tmax": 10,
        "mu0": 0,
        "sigma0": 1,
        "q": 0.9,
        "K": 10,
    }

    print("Starting LearnHeuristicPrac training for 24-puzzle …")
    results = learn_heuristic_prac_24_puzzle(params)
    print("Training complete.")

    print_results(results)


if __name__ == "__main__":
    main()
