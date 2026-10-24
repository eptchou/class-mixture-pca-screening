"""Keras model builders.

Three networks are used throughout:

* ``create_mlp_classifier`` -- the baseline classifier whose first hidden
  layer (128 units, named ``dense1``) is the screening target; its
  post-activation output is named ``dense1_activation``.
* ``create_finetune_subnet`` -- the Siamese feature-extraction subnetwork.
  Its first layer keeps only the ``k`` retained neurons, is initialised
  with the sliced pre-trained weights and frozen; deeper layers inherit
  the sliced ``dense2`` weights and are fine-tuned.
* ``create_siamese_network`` -- wraps a subnetwork into a twin network
  with an L1-distance head and a sigmoid similarity output.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, layers, models


def create_mlp_classifier(input_shape: Sequence[int], num_classes: int,
                          first_layer_neurons: int = 128):
    """Baseline classifier: Dense(128) + BN + ReLU -> Dense(64) -> softmax."""
    inp = Input(shape=input_shape)
    x = layers.Dense(first_layer_neurons, name="dense1")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu", name="dense1_activation")(x)
    x = layers.Dense(64, activation="relu", name="dense2")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out)


def create_finetune_subnet(input_shape: Sequence[int], k_neurons: int,
                           initial_weights: List[np.ndarray]):
    """Siamese subnetwork with a frozen, pre-trained screened first layer.

    Parameters
    ----------
    initial_weights : list ``[w1, b1, w2]``
        Already sliced to the retained neurons: ``w1`` has shape
        ``(n_features, k_neurons)``, ``b1`` shape ``(k_neurons,)`` and
        ``w2`` shape ``(k_neurons, 64)``.
    """
    w1, b1, w2 = initial_weights
    inp = Input(shape=input_shape)

    dense1 = layers.Dense(k_neurons, name=f"dense1_k{k_neurons}", trainable=False)
    x = dense1(inp)
    dense1.set_weights([w1, b1])

    bn = layers.BatchNormalization(trainable=False)
    x = bn(x)
    x = layers.Activation("relu")(x)

    dense2 = layers.Dense(64, activation="relu", name="dense2")
    x = dense2(x)
    # Migrate the sliced baseline dense2 weights, keeping the fresh bias.
    _, current_b2 = dense2.get_weights()
    if w2.shape[0] >= k_neurons:
        dense2.set_weights([w2[:k_neurons, :], current_b2])

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    return models.Model(inp, x)


def create_siamese_network(subnet, input_shape: Sequence[int]):
    """Twin network with an L1-distance head and sigmoid similarity output."""
    inp_a, inp_b = Input(shape=input_shape), Input(shape=input_shape)
    embed_a, embed_b = subnet(inp_a), subnet(inp_b)
    l1 = layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([embed_a, embed_b])
    out = layers.Dense(1, activation="sigmoid")(l1)
    return models.Model([inp_a, inp_b], out)


def slice_weights(baseline_weights: List[np.ndarray],
                  indices: np.ndarray) -> List[np.ndarray]:
    """Slice ``[W1, b1, W2]`` to the retained neuron indices.

    Returns ``[W1[:, idx], b1[idx], W2[idx, :]]``.
    """
    w1, b1, w2 = baseline_weights
    return [w1[:, indices], b1[indices], w2[indices, :]]
