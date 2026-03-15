"""
Neural Network Models for the 15-Puzzle
========================================
Two architectures are provided:

1. **FFNN** – Simple feed-forward neural network with one hidden layer.
   Predicts a single scalar cost-to-goal value for a given board state.

2. **WUNN** – Weighted-Uncertainty Neural Network.
   Predicts a distribution over cost-to-goal values via its mean and
   log-variance outputs, which are used to capture aleatoric and
   epistemic uncertainty as described in:

       Marom & Rosman (2020). "Utilising Uncertainty for Efficient
       Learning of Likely-Admissible Heuristics."
       https://www.raillab.org/publication/marom-2020-utilising/

Both models are built with TensorFlow / Keras.
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


def create_ffnn(input_dim: int = 16, hidden_dim: int = 20, dropout_rate: float = 0.25):
    """Create and compile a feed-forward neural network (FFNN).

    Parameters
    ----------
    input_dim:
        Number of input features (16 for the 15-puzzle flat state).
    hidden_dim:
        Number of neurons in the single hidden layer.
    dropout_rate:
        Dropout probability applied after the hidden layer.

    Returns
    -------
    model : keras.Sequential
        Compiled Keras model ready for training.
    """
    model = Sequential([
        Dense(hidden_dim, activation="relu", input_shape=(input_dim,)),
        Dropout(dropout_rate),
        Dense(1),  # single output: predicted cost-to-goal
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def create_wunn(input_dim: int = 16, hidden_dim: int = 20):
    """Create and compile a Weighted-Uncertainty Neural Network (WUNN).

    The network outputs two values per sample:
      - output[0]: predicted mean cost-to-goal (μ)
      - output[1]: predicted log-variance (log σ²)

    Parameters
    ----------
    input_dim:
        Number of input features (16 for the 15-puzzle flat state).
    hidden_dim:
        Number of neurons in the single hidden layer.

    Returns
    -------
    model : keras.Sequential
        Compiled Keras model ready for training.
    """
    model = Sequential([
        Dense(hidden_dim, activation="relu", input_shape=(input_dim,)),
        Dense(2),  # two outputs: mean and log-variance
    ])
    model.compile(optimizer=Adam(learning_rate=0.000001), loss="mse")
    return model


def train_nn(
    model,
    memory_buffer: list,
    iterations: int,
    batch_size=None,
    kappa=None,
    update_beta: bool = False,
    gamma=None,
    beta=None,
):
    """Train *model* on data stored in *memory_buffer*.

    Parameters
    ----------
    model:
        A compiled Keras model (FFNN or WUNN).
    memory_buffer:
        List of ``(encoded_state, cost_to_goal)`` pairs collected from
        previously solved tasks.
    iterations:
        Number of training epochs.
    batch_size:
        Mini-batch size passed to ``model.fit``.  ``None`` means full batch.
    kappa, update_beta, gamma, beta:
        Reserved for future WUNN-specific training logic (currently unused
        but accepted to maintain a uniform calling signature).
    """
    if not memory_buffer:
        return

    x_train, y_train = zip(*memory_buffer)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    model.fit(x_train, y_train, epochs=iterations, batch_size=batch_size, verbose=0)
