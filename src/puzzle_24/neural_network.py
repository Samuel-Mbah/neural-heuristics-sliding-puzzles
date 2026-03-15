"""
Neural Network Models for the 24-Puzzle
=========================================
Two PyTorch architectures are provided:

1. **FeedForwardNN** (FFNN) – A simple feed-forward network that predicts a
   scalar cost-to-goal estimate.

2. **WeightUncertaintyNN** (WUNN) – A dropout-based network that enables
   Monte-Carlo estimation of epistemic uncertainty via stochastic forward
   passes.

Both networks share the same single linear + dropout architecture; the key
difference is that the WUNN stores prior hyper-parameters (``mu0``,
``sigma0``) and is explicitly kept in *train* mode when sampling, enabling
dropout-based uncertainty estimation.

Utility functions
-----------------
:func:`compute_sigma_e2`
    Estimates the epistemic variance of a WUNN prediction via *K* stochastic
    forward passes.

:func:`sample_from_softmax`
    Selects a state from a dictionary of states weighted by their epistemic
    uncertainty scores using a softmax distribution.
"""

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# Network definitions
# ---------------------------------------------------------------------------

class WeightUncertaintyNN(nn.Module):
    """Dropout-based neural network for epistemic-uncertainty estimation.

    Parameters
    ----------
    input_dim:
        Dimensionality of the input feature vector.
    output_dim:
        Number of output neurons (typically 1 for cost prediction).
    mu0:
        Prior mean for weight initialisation (stored for reference).
    sigma0:
        Prior standard deviation for weight initialisation (stored for
        reference).
    dropout_p:
        Dropout probability applied during both training *and* inference
        when sampling for uncertainty estimation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mu0: float,
        sigma0: float,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.fc(x)


class FeedForwardNN(nn.Module):
    """Simple feed-forward neural network for cost-to-goal prediction.

    Parameters
    ----------
    input_dim:
        Dimensionality of the input feature vector.
    output_dim:
        Number of output neurons (typically 1).
    dropout_p:
        Dropout probability applied during training.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout_p: float = 0.2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Uncertainty estimation
# ---------------------------------------------------------------------------

def compute_sigma_e2(nn_wunn: WeightUncertaintyNN, x: np.ndarray, K: int) -> float:
    """Estimate epistemic variance via *K* Monte-Carlo dropout forward passes.

    Parameters
    ----------
    nn_wunn:
        A :class:`WeightUncertaintyNN` instance.
    x:
        1-D feature vector as a numpy array.
    K:
        Number of stochastic forward passes.

    Returns
    -------
    sigma_e2 : float
        Variance of the *K* predictions (epistemic uncertainty estimate).
    """
    nn_wunn.train()  # keep dropout active
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    samples = np.array(
        [nn_wunn(x_tensor).cpu().detach().numpy() for _ in range(K)]
    )
    nn_wunn.eval()
    return float(np.var(samples))


# ---------------------------------------------------------------------------
# Softmax-weighted state selection
# ---------------------------------------------------------------------------

def sample_from_softmax(states: dict):
    """Sample a state proportionally to its uncertainty score (softmax).

    Parameters
    ----------
    states:
        Mapping from state tuple to uncertainty score (float).

    Returns
    -------
    (selected_state, score) : (tuple, float)
    """
    keys = list(states.keys())
    values = np.array(list(states.values()), dtype=np.float64)

    max_value = np.max(values)
    exp_values = np.exp(values - max_value)  # numerically stable softmax
    softmax_probs = exp_values / np.sum(exp_values)

    selected_index = np.random.choice(len(keys), p=softmax_probs)
    return keys[selected_index], values[selected_index]
