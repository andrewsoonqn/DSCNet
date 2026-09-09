from pathlib import Path

import torch

CHECKPOINT_FORMAT_VERSION = 1
SAMPLER_IMPLEMENTATION = "grid_sample_v1"


def save_model_checkpoint(model, path, pipeline):
    """Save weights with enough metadata to reject incompatible samplers."""
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "sampler_implementation": SAMPLER_IMPLEMENTATION,
            "pipeline": pipeline,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def load_model_checkpoint(model, path, pipeline):
    """Load only checkpoints produced by the current sampler implementation."""
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
    model.load_state_dict(checkpoint["model_state_dict"])
