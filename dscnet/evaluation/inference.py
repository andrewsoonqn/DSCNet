"""Shared raw-volume inference contract for trained DSCNet models."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from dscnet.evaluation.sliding_window import sliding_window_logits


def validate_normalization(mean: float, std: float) -> tuple[float, float]:
    mean_value = float(mean)
    std_value = float(std)
    if not np.isfinite(mean_value):
        raise ValueError("normalization mean must be finite")
    if not np.isfinite(std_value) or std_value <= 0:
        raise ValueError("normalization standard deviation must be finite and positive")
    return mean_value, std_value


def load_normalization(path) -> tuple[float, float]:
    values = np.asarray(np.load(path))
    if values.shape != (2,):
        raise ValueError("normalization file must contain exactly mean and standard deviation")
    return validate_normalization(values[0], values[1])


def predict_probabilities(
    model,
    volume: np.ndarray,
    *,
    mean: float,
    std: float,
    roi_shape: Sequence[int],
    n_classes: int,
    batch_size: int,
    device: torch.device,
    use_amp: bool = False,
) -> np.ndarray:
    """Normalize one raw 3-D volume and return class probabilities."""
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError("model input must be one three-dimensional volume")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("model input must contain numeric values")
    if not np.isfinite(array).all():
        raise ValueError("model input contains non-finite values")
    mean_value, std_value = validate_normalization(mean, std)
    normalized = np.ascontiguousarray(
        (array.astype(np.float32, copy=False) - mean_value) / std_value,
        dtype=np.float32,
    )
    probabilities = sliding_window_logits(
        model,
        normalized,
        roi_shape,
        n_classes,
        batch_size,
        device,
        use_amp=use_amp,
    )
    probabilities = np.ascontiguousarray(probabilities, dtype=np.float32)
    expected = (n_classes, *array.shape)
    if probabilities.shape != expected:
        raise RuntimeError(
            f"model returned probabilities with shape {probabilities.shape}; expected {expected}"
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("model returned non-finite probabilities")
    if np.any(probabilities < -1e-6) or np.any(probabilities > 1 + 1e-6):
        raise RuntimeError("model returned values outside the probability range")
    if not np.allclose(probabilities.sum(axis=0), 1.0, rtol=1e-4, atol=1e-5):
        raise RuntimeError("model class probabilities do not sum to one")
    return probabilities
