"""Shared edge-safe sliding-window inference for DSCNet pipelines."""

from __future__ import annotations

from contextlib import nullcontext
from itertools import product
from typing import Sequence

import numpy as np
import torch


def axis_starts(length: int, window: int) -> tuple[int, ...]:
    """Return deterministic half-window starts, including the far edge exactly."""
    if window <= 0 or length < window:
        raise ValueError("window must be positive and no larger than the image axis")
    stride = max(1, window // 2)
    starts = list(range(0, length - window + 1, stride))
    edge = length - window
    if not starts or starts[-1] != edge:
        starts.append(edge)
    return tuple(starts)


def quartic_center_weights(roi_shape: Sequence[int]) -> np.ndarray:
    if len(roi_shape) != 3 or any(value <= 0 for value in roi_shape):
        raise ValueError("roi_shape must contain three positive dimensions")
    coordinates = np.indices(tuple(roi_shape), dtype=np.float32)
    denominator = np.ones(tuple(roi_shape), dtype=np.float32)
    for axis, length in enumerate(roi_shape):
        denominator += (coordinates[axis] - length // 2) ** 4
    return (1.0 / denominator)[np.newaxis, ...]


def sliding_window_logits(
    model,
    image: np.ndarray,
    roi_shape: Sequence[int],
    n_classes: int,
    batch_size: int,
    device: torch.device,
    *,
    use_amp: bool = False,
) -> np.ndarray:
    """Blend logits over every voxel of one normalized three-dimensional image."""
    if image.ndim != 3:
        raise ValueError("image must be three-dimensional")
    if n_classes <= 1 or batch_size <= 0:
        raise ValueError("n_classes and batch_size must be positive")
    roi = tuple(int(value) for value in roi_shape)
    original_shape = image.shape
    padded_shape = tuple(max(length, window) for length, window in zip(image.shape, roi))
    padded = np.zeros(padded_shape, dtype=np.float32)
    padded[tuple(slice(0, length) for length in image.shape)] = image.astype(
        np.float32, copy=False
    )

    starts = [axis_starts(length, window) for length, window in zip(padded_shape, roi)]
    locations = list(product(*starts))
    weights = quartic_center_weights(roi)
    logits = np.zeros((n_classes, *padded_shape), dtype=np.float32)
    coverage = np.zeros((1, *padded_shape), dtype=np.float32)

    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(locations), batch_size):
            batch_locations = locations[offset : offset + batch_size]
            patches = np.stack(
                [
                    padded[
                        z : z + roi[0],
                        y : y + roi[1],
                        x : x + roi[2],
                    ]
                    for z, y, x in batch_locations
                ]
            )[:, np.newaxis, ...]
            tensor = torch.from_numpy(patches).to(device, non_blocking=True)
            amp_context = (
                torch.autocast(device_type="cuda")
                if use_amp and device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                output = model(tensor)
            values = output.detach().float().cpu().numpy()
            expected = (len(batch_locations), n_classes, *roi)
            if values.shape != expected:
                raise RuntimeError(
                    f"model returned logits with shape {values.shape}; expected {expected}"
                )
            if not np.isfinite(values).all():
                raise RuntimeError("model returned non-finite sliding-window logits")
            for patch_logits, (z, y, x) in zip(values, batch_locations):
                region = (
                    slice(None),
                    slice(z, z + roi[0]),
                    slice(y, y + roi[1]),
                    slice(x, x + roi[2]),
                )
                logits[region] += patch_logits * weights
                coverage[region] += weights

    if not np.isfinite(coverage).all() or np.any(coverage <= 0):
        raise RuntimeError("sliding-window inference did not cover every voxel")
    logits /= coverage
    crop = (slice(None),) + tuple(slice(0, length) for length in original_shape)
    result = logits[crop]
    if not np.isfinite(result).all():
        raise RuntimeError("sliding-window blending produced non-finite logits")
    return result
