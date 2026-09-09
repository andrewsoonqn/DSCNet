"""Shared segmentation metric aggregation for validation and test runs."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from S3_Metrics import cldice_score, dice_score, to_minivess_binary_mask


def summarize_predictions(label_manifest, prediction_dir, mismatch_loader):
    labels = [line for line in Path(label_manifest).read_text().splitlines() if line]
    if not labels:
        raise ValueError(f"empty label manifest: {label_manifest}")

    dice_values = []
    cldice_values = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for label_path in labels:
        name = os.path.basename(label_path)
        prediction_path = Path(prediction_dir) / name
        prediction_image = sitk.ReadImage(str(prediction_path))
        target_image = sitk.ReadImage(label_path)
        if prediction_image.GetSize() == target_image.GetSize():
            prediction = sitk.GetArrayFromImage(prediction_image)
            target = sitk.GetArrayFromImage(target_image)
        else:
            prediction, target = mismatch_loader(str(prediction_path), label_path)

        prediction = to_minivess_binary_mask(prediction, name=f"prediction {name}")
        target = to_minivess_binary_mask(target, name=f"target {name}")
        dice_values.append(dice_score(prediction, target))
        cldice_values.append(cldice_score(prediction, target))
        true_positives += int(np.logical_and(prediction == 1, target == 1).sum())
        false_positives += int(np.logical_and(prediction == 1, target == 0).sum())
        false_negatives += int(np.logical_and(prediction == 0, target == 1).sum())

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = (
        true_positives / precision_denominator if precision_denominator else 1.0
    )
    recall = true_positives / recall_denominator if recall_denominator else 1.0
    return {
        "dice": float(np.mean(dice_values)),
        "cldice": float(np.mean(cldice_values)),
        "precision": float(precision),
        "recall": float(recall),
        "false_positives": float(false_positives),
        "false_negatives": float(false_negatives),
    }


def log_summary(args, prefix, metrics, step=None):
    tracker = getattr(args, "tracker", None)
    if tracker is None:
        return
    for name, value in metrics.items():
        tracker.log_metric(f"{prefix}.{name}", value, step=step)
