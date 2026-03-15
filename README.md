# Neural Heuristics for Sliding Puzzles

>
A reproducibility study of the paper
["Utilising Uncertainty for Efficient Learning of Likely-Admissible Heuristics"](https://www.raillab.org/publication/marom-2020-utilising/marom-2020-utilising.pdf)
(Marom & Rosman, 2020).  The project trains neural networks to serve as
admissible heuristics for the IDA* search algorithm on the 15-puzzle and
24-puzzle domains.

---

## Repository Structure

```
.
├── src/
│   ├── puzzle_15/           # 15-puzzle (4×4) implementation
│   │   ├── environment.py   # Puzzle15 class, move generation, heuristics
│   │   ├── neural_network.py# FFNN and WUNN model factories (TensorFlow/Keras)
│   │   ├── algorithms.py    # GenerateTaskPrac, IDA*, LearnHeuristicPrac
│   │   └── main.py          # Training entry point
│   └── puzzle_24/           # 24-puzzle (5×5) implementation
│       ├── environment.py   # State utils, pattern databases, feature extraction
│       ├── neural_network.py# WeightUncertaintyNN, FeedForwardNN (PyTorch)
│       ├── algorithms.py    # GenerateTaskPrac, IDA*, LearnHeuristicPrac
│       └── main.py          # Training entry point
├── notebooks/
│   ├── 15_puzzle.ipynb      # Original exploratory notebook (15-puzzle)
│   └── 24_puzzle_pdb.ipynb  # Original exploratory notebook (24-puzzle)
├── requirements.txt
└── README.md
```

---

## Project Overview

### Problem Domains

| Domain    | Grid  | Tiles | State space |
|-----------|-------|-------|-------------|
| 15-puzzle | 4 × 4 | 15    | ~10¹³       |
| 24-puzzle | 5 × 5 | 24    | ~10²⁵       |

### Algorithm

The **LearnHeuristicPrac** procedure (Algorithm 1 in the paper) iterates
between three phases:

1. **Task generation** (`GenerateTaskPrac`) – scrambles the goal state with
   random moves (15-puzzle) or uses epistemic uncertainty from the WUNN to
   select challenging states (24-puzzle).
2. **Planning** – attempts to solve each task with IDA* using the current
   FFNN heuristic.
3. **Learning** – appends solved state–cost pairs to an experience replay
   buffer and retrains both the FFNN and WUNN on a random mini-batch.

### Neural Network Architectures

| Name  | Framework      | Outputs              | Purpose                       |
|-------|----------------|----------------------|-------------------------------|
| FFNN  | TF/Keras (15)  | scalar cost          | heuristic for IDA*            |
| WUNN  | TF/Keras (15)  | mean + log-variance  | uncertainty-guided exploration|
| FFNN  | PyTorch (24)   | scalar cost          | heuristic for IDA*            |
| WUNN  | PyTorch (24)   | scalar cost (MC drop)| epistemic uncertainty estimate|

---

## Getting Started

### Requirements

- Python 3.9+
- TensorFlow 2.13 (15-puzzle)
- PyTorch 2.0 (24-puzzle)
- NumPy, SciPy

### Installation

```bash
git clone https://github.com/Samuel-Mbah/Reproducibility-Project.git
cd Reproducibility-Project

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Running the Experiments

**15-puzzle**
```bash
python -m src.puzzle_15.main
```

**24-puzzle**
```bash
python -m src.puzzle_24.main
```

### Configuration

Training hyper-parameters are defined as dictionaries inside each `main.py`.
Key parameters:

| Parameter                  | Description                                        |
|----------------------------|----------------------------------------------------|
| `num_iter`                 | Number of training iterations                      |
| `num_tasks_per_iter`       | Tasks attempted per iteration                      |
| `num_tasks_per_iter_thresh`| Min solved tasks to keep α stable                  |
| `alpha0`                   | Initial Gaussian quantile confidence level α       |
| `delta`                    | Step for decreasing α on low success rate          |
| `epsilon`                  | Epistemic uncertainty threshold (task generation)  |
| `memory_buffer_max_records`| Experience replay buffer capacity                  |
| `train_iter`               | FFNN training epochs per iteration                 |
| `max_train_iter`           | WUNN training epochs per iteration                 |
| `K`                        | Monte-Carlo forward passes for uncertainty         |

### Jupyter Notebooks

The original exploratory notebooks are preserved in `notebooks/` for
reference and interactive experimentation:

```bash
jupyter notebook notebooks/15_puzzle.ipynb
jupyter notebook notebooks/24_puzzle_pdb.ipynb
```

---

## Results

Our experiments aimed to reproduce the results reported in Marom & Rosman
(2020). Despite rigorous efforts and multiple trials, we were unable to
fully replicate the paper's findings. Below is a summary:

### What Worked

- ✅ Successfully implemented **GenerateTaskPrac** for both domains.
- ✅ IDA* with Manhattan-distance heuristic solves the 15-puzzle correctly.
- ✅ Training loop runs end-to-end without errors.

### Challenges

- ❌ **LearnHeuristicPrac did not converge** – the algorithm ran for many
  hours without producing a meaningful improvement in the success rate.
- ❌ **WUNN complexity** – integrating aleatoric and epistemic uncertainties
  in the C# → Python port introduced subtle differences in training dynamics.
- ⚠️ **Computational cost** – initial training on an Intel i5-12450H 2.00 GHz
  CPU with 32 GB RAM took ~12 hours per run without positive results.

### Key Metrics

| Metric                  | Paper (claimed) | Our reproduction |
|-------------------------|-----------------|------------------|
| Generated Nodes (mean)  | reported        | N/A (no convergence) |
| Planning Time (mean, s) | reported        | N/A              |
| Suboptimality (%)       | reported        | N/A              |
| Optimal Solved (%)      | reported        | N/A              |

### Conclusion

Re-implementing the learning algorithm from the original C# codebase in
Python introduced challenges that prevented convergence within the available
computational budget. The IDA* search and task-generation components work
correctly; the bottleneck is the WUNN-guided exploration loop.

---

## References

- Marom, O., & Rosman, B. (2020). *Utilising Uncertainty for Efficient
  Learning of Likely-Admissible Heuristics.*
  [Paper](https://www.raillab.org/publication/marom-2020-utilising/marom-2020-utilising.pdf) |
  [Supplementary](https://www.raillab.org/publication/marom-2020-utilising/marom-2020-utilising_supp.pdf)

---

## Contact

For questions or collaboration requests, please contact
[smwmbah@gmail.com](mailto:smwmbah@gmail.com).
