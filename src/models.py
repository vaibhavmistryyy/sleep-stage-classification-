"""
Deep Learning Model Architectures for Sleep Stage Classification
Author: Vaibhav Mistry

Defines two neural network architectures:
1. Hybrid 1D-CNN + Bidirectional LSTM (Deep Temporal Recurrent Model)
2. 1D-CNN Baseline with Global Average Pooling
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam


def build_cnn_bilstm(
    input_shape: Tuple[int, int],
    num_classes: int = 5,
    learning_rate: float = 0.001
) -> Sequential:
    """
    Construct a hybrid 1D-CNN + Bidirectional LSTM model.

    Rationale:
    - 1D-CNN extracts shift-invariant local waveform morphology (e.g. sleep spindles,
      K-complexes, delta waves) from high-frequency time series (3000 time steps).
    - Max Pooling reduces dimensionality while retaining salient feature peaks.
    - Bidirectional LSTM captures bidirectional temporal dynamics and stage transitions.
    - Dropout regularizes against overfitting on multi-subject recordings.

    Args:
        input_shape: Tuple of (time_steps, n_channels), e.g. (3000, 3).
        num_classes: Number of target sleep stages (default: 5).
        learning_rate: Initial Adam learning rate.

    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential([
        # 1D-CNN Feature Extractor
        Conv1D(filters=64, kernel_size=7, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # First Bidirectional LSTM Layer (returns sequences for stacked recurrent processing)
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),

        # Second Bidirectional LSTM Layer
        Bidirectional(LSTM(32)),
        Dropout(0.3),

        # Dense Classifier Head
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ], name="CNN_BiLSTM_Sleep_Classifier")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_cnn_baseline(
    input_shape: Tuple[int, int],
    num_classes: int = 5,
    learning_rate: float = 0.001
) -> Sequential:
    """
    Construct a baseline pure 1D-CNN model with Global Average Pooling.

    Serves as an ablation comparison to demonstrate the necessity of recurrent
    temporal modeling (BiLSTM) for polysomnographic sleep staging.

    Args:
        input_shape: Tuple of (time_steps, n_channels).
        num_classes: Number of target sleep stages (default: 5).
        learning_rate: Initial Adam learning rate.

    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential([
        Conv1D(filters=64, kernel_size=7, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ], name="CNN_Baseline_Classifier")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
