"""Baseline 1D CNN model architecture and feature extraction interface.

Designed for 2048-sample single-channel CWRU vibration windows.
Provides:
    1. 4-class fault classification output (Softmax probabilities).
    2. Explicit feature representation extraction (learned embedding)
       for downstream novelty estimation.
"""

from typing import Tuple
import keras
from keras import layers
import numpy as np

from src.cnn.config import (
    EMBEDDING_DIM,
    EMBEDDING_LAYER_NAME,
    INPUT_SHAPE,
    LEARNING_RATE,
    NUM_CLASSES,
)


def build_baseline_cnn(
    input_shape: Tuple[int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    embedding_dim: int = EMBEDDING_DIM,
    embedding_layer_name: str = EMBEDDING_LAYER_NAME,
) -> keras.Model:
    """Construct and compile a modular 1D CNN baseline model.

    Architecture:
        Input (2048, 1)
        -> Conv1D (16 filters, kernel 15, stride 2, relu)
        -> MaxPool1D (pool 2)
        -> Conv1D (32 filters, kernel 7, stride 1, relu)
        -> MaxPool1D (pool 2)
        -> Conv1D (64 filters, kernel 3, stride 1, relu)
        -> GlobalAveragePooling1D
        -> Dense (64 units, relu, name="learned_embedding")
        -> Dropout (0.2)
        -> Dense (4 units, softmax, name="output_probabilities")

    Args:
        input_shape: Shape tuple of single observation window, default (2048, 1).
        num_classes: Number of classification targets, default 4.
        learning_rate: Adam optimizer learning rate, default 0.001.
        embedding_dim: Dimensionality of penultimate feature representation, default 64.
        embedding_layer_name: Identifier for the penultimate dense layer.

    Returns:
        keras.Model: Compiled Keras model instance.
    """
    inputs = keras.Input(shape=input_shape, name="vibration_input")

    # Block 1: Large receptive field to capture long-period vibration harmonics
    x = layers.Conv1D(
        filters=16,
        kernel_size=15,
        strides=2,
        padding="same",
        activation="relu",
        name="conv1d_1",
    )(inputs)
    x = layers.MaxPool1D(pool_size=2, name="maxpool_1")(x)

    # Block 2: Intermediate frequency feature extraction
    x = layers.Conv1D(
        filters=32,
        kernel_size=7,
        strides=1,
        padding="same",
        activation="relu",
        name="conv1d_2",
    )(x)
    x = layers.MaxPool1D(pool_size=2, name="maxpool_2")(x)

    # Block 3: Local fine-grained defect signature extraction
    x = layers.Conv1D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding="same",
        activation="relu",
        name="conv1d_3",
    )(x)
    x = layers.GlobalAveragePooling1D(name="global_pool")(x)

    # Penultimate embedding representation (used later for Novelty Detection)
    embedding = layers.Dense(
        embedding_dim,
        activation="relu",
        name=embedding_layer_name,
    )(x)
    x = layers.Dropout(0.2, name="dropout")(embedding)

    # Output probabilities (used later for Classification & Uncertainty Estimation)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="output_probabilities",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cwru_baseline_1d_cnn")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def predict_probabilities(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """Generate 4-class softmax probability array for input vibration windows.

    Args:
        model: Trained Keras CNN model.
        X: Input vibration tensor of shape (N, 2048, 1).

    Returns:
        np.ndarray: Softmax probabilities of shape (N, 4).
    """
    probs = model.predict(X, verbose=0)
    return np.asarray(probs, dtype=np.float32)


def predict_classes(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """Generate discrete predicted class labels for input vibration windows.

    Args:
        model: Trained Keras CNN model.
        X: Input vibration tensor of shape (N, 2048, 1).

    Returns:
        np.ndarray: 1D array of predicted class integers of shape (N,).
    """
    probs = predict_probabilities(model, X)
    return np.argmax(probs, axis=-1).astype(np.int64)


def extract_embeddings(
    model: keras.Model,
    X: np.ndarray,
    layer_name: str = EMBEDDING_LAYER_NAME,
) -> np.ndarray:
    """Extract penultimate learned feature representation from the trained model.

    This function provides the interface required for future novelty detection modules.

    Args:
        model: Trained Keras CNN model.
        X: Input vibration tensor of shape (N, 2048, 1).
        layer_name: Name of the penultimate embedding layer.

    Returns:
        np.ndarray: Extracted embedding vectors of shape (N, embedding_dim).
    """
    embedding_layer = model.get_layer(layer_name)
    feature_extractor = keras.Model(
        inputs=model.inputs,
        outputs=embedding_layer.output,
        name="feature_extractor",
    )
    embeddings = feature_extractor.predict(X, verbose=0)
    return np.asarray(embeddings, dtype=np.float32)
