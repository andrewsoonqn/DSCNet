from __future__ import annotations

import random
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

CHECKPOINT_FORMAT_VERSION = 2
SAMPLER_IMPLEMENTATION = "grid_sample_v1"


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["algorithm"],
            numpy_state["state"].cpu().numpy(),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _base_checkpoint(model, pipeline):
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "sampler_implementation": SAMPLER_IMPLEMENTATION,
        "pipeline": pipeline,
        "model_state_dict": model.state_dict(),
    }


def _atomic_torch_save(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def save_model_checkpoint(model, path, pipeline):
    """Save model weights with metadata that rejects incompatible samplers."""
    _atomic_torch_save(_base_checkpoint(model, pipeline), path)


def save_training_checkpoint(
    model,
    path,
    pipeline,
    *,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_score,
    config_digest,
    loop_state=None,
):
    """Atomically save every state required to resume the next training step."""
    checkpoint = _base_checkpoint(model, pipeline)
    checkpoint["training_state"] = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_score": float(best_score),
        "rng": capture_rng_state(),
        "config_digest": str(config_digest),
        "loop_state": dict(loop_state or {}),
    }
    _atomic_torch_save(checkpoint, path)


def _load_checkpoint(path, pipeline):
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "sampler_implementation": SAMPLER_IMPLEMENTATION,
        "pipeline": pipeline,
    }
    if not isinstance(checkpoint, dict) or any(
        checkpoint.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError(
            f"incompatible checkpoint {path}; retrain with the current {pipeline} "
            f"sampler (format {CHECKPOINT_FORMAT_VERSION}, "
            f"{SAMPLER_IMPLEMENTATION})"
        )
    if "model_state_dict" not in checkpoint:
        raise RuntimeError(f"checkpoint {path} has no model_state_dict")
    return checkpoint


def load_model_checkpoint(model, path, pipeline):
    """Load weights from a compatible model or training checkpoint."""
    checkpoint = _load_checkpoint(path, pipeline)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def load_training_checkpoint(
    model,
    path,
    pipeline,
    *,
    optimizer,
    scheduler,
    scaler,
    expected_config_digest,
):
    """Restore full training state and return the saved epoch and best score."""
    checkpoint = _load_checkpoint(path, pipeline)
    training_state = checkpoint.get("training_state")
    if not isinstance(training_state, dict):
        raise RuntimeError(f"checkpoint {path} has no resumable training state")
    required = {"optimizer", "scheduler", "scaler", "epoch", "best_score", "rng", "config_digest", "loop_state"}
    missing = required - training_state.keys()
    if missing:
        raise RuntimeError(f"checkpoint {path} is missing training state: {sorted(missing)}")
    if training_state.get("config_digest") != expected_config_digest:
        raise RuntimeError(
            f"checkpoint {path} was created for a different experiment config"
        )
    if (scheduler is None) != (training_state["scheduler"] is None):
        raise RuntimeError(f"checkpoint {path} has incompatible scheduler state")
    if (scaler is None) != (training_state["scaler"] is None):
        raise RuntimeError(f"checkpoint {path} has incompatible scaler state")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(training_state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(training_state["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(training_state["scaler"])
    restore_rng_state(training_state["rng"])
    return {
        "epoch": training_state["epoch"],
        "best_score": training_state["best_score"],
        "config_digest": training_state["config_digest"],
        "loop_state": training_state["loop_state"],
    }
