"""Pure binary-segmentation metrics for MiniVess."""

import numpy as np
from skimage.morphology import skeletonize


def to_minivess_binary_mask(mask: np.ndarray, name: str = "mask") -> np.ndarray:
    """Validate MiniVess's {0, 1} label contract and return a boolean mask."""
    mask = np.asarray(mask)
    values = np.unique(mask)
    if not np.isin(values, (0, 1)).all():
        raise ValueError(
            f"{name} must contain only MiniVess labels {{0, 1}}; "
            f"found {values.tolist()}"
        )
    return mask.astype(bool)


def _validate_pair(prediction: np.ndarray, target: np.ndarray):
    prediction = to_minivess_binary_mask(prediction, "prediction")
    target = to_minivess_binary_mask(target, "target")
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: "
            f"{prediction.shape} != {target.shape}"
        )
    return prediction, target


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return foreground Dice, treating two empty masks as a perfect match."""
    prediction, target = _validate_pair(prediction, target)
    denominator = prediction.sum() + target.sum()
    if denominator == 0:
        return 1.0
    intersection = np.logical_and(prediction, target).sum()
    return float(2 * intersection / denominator)


def cldice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return the topology-preserving clDice score for two binary masks."""
    prediction, target = _validate_pair(prediction, target)

    prediction_skeleton = skeletonize(prediction)
    target_skeleton = skeletonize(target)
    prediction_length = prediction_skeleton.sum()
    target_length = target_skeleton.sum()

    if prediction_length == 0 and target_length == 0:
        return 1.0
    if prediction_length == 0 or target_length == 0:
        return 0.0

    topology_precision = np.logical_and(prediction_skeleton, target).sum()
    topology_precision /= prediction_length
    topology_sensitivity = np.logical_and(target_skeleton, prediction).sum()
    topology_sensitivity /= target_length

    denominator = topology_precision + topology_sensitivity
    if denominator == 0:
        return 0.0
    return float(2 * topology_precision * topology_sensitivity / denominator)
