"""Package trained DSCNet checkpoints as inference-only MLflow Models."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import mlflow
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
import numpy as np
from omegaconf import OmegaConf
from packaging.requirements import Requirement
import torch

from dscnet.evaluation.inference import load_normalization, predict_probabilities
from dscnet.experiment.config import (
    PROJECT_ROOT,
    ExperimentConfig,
    to_runtime_namespace,
    validate_config,
)
from dscnet.experiment.tracking import sha256_file
from dscnet.models.factory import build_model
from dscnet.training.checkpoints import load_model_checkpoint
from dscnet.workflow import apply_reproducibility


MODEL_PACKAGE_FORMAT_VERSION = 1


def _load_typed_config(path: str | Path):
    loaded = OmegaConf.load(path)
    resolved = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), loaded)
    OmegaConf.resolve(resolved)
    config = OmegaConf.to_object(resolved)
    if not isinstance(config, ExperimentConfig):
        raise TypeError("model package config did not produce ExperimentConfig")
    validate_config(config)
    return config, resolved


def model_signature(n_classes: int) -> ModelSignature:
    return ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float32"), (-1, -1, -1))]),
        outputs=Schema(
            [TensorSpec(np.dtype("float32"), (n_classes, -1, -1, -1))]
        ),
    )


def architecture_text(model, args) -> str:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    header = {
        "pipeline": args.training_pipeline,
        "model_class": type(model).__name__,
        "input_channels": args.n_channels,
        "output_classes": args.n_classes,
        "roi_shape": list(args.ROI_shape),
        "total_parameters": total,
        "trainable_parameters": trainable,
    }
    return json.dumps(header, indent=2, sort_keys=True) + "\n\n" + str(model) + "\n"


def locked_requirements(project_root: str | Path = PROJECT_ROOT) -> list[str]:
    """Export the complete exact environment from the checked-in uv lock."""
    uv = shutil.which("uv")
    if uv is None:
        fallback = Path.home() / ".local" / "bin" / "uv"
        if fallback.is_file():
            uv = str(fallback)
    if uv is None:
        raise RuntimeError("uv executable is required to export model requirements")
    completed = subprocess.run(
        [
            uv,
            "export",
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--no-hashes",
            "--format",
            "requirements-txt",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    requirements = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    for value in requirements:
        requirement = Requirement(value)
        specifiers = list(requirement.specifier)
        if requirement.url is not None or len(specifiers) != 1 or specifiers[0].operator != "==":
            raise RuntimeError(f"uv exported a non-exact model requirement: {value}")
    if not requirements:
        raise RuntimeError("uv exported no model requirements")
    return requirements


class DscnetPythonModel(mlflow.pyfunc.PythonModel):
    """Predict probabilities from one raw three-dimensional float32 volume."""

    def load_context(self, context) -> None:
        config, _ = _load_typed_config(context.artifacts["resolved_config"])
        args = to_runtime_namespace(config)
        apply_reproducibility(
            args.seed, args.deterministic, args.deterministic_warn_only
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(args, args.training_pipeline, self.device)
        load_model_checkpoint(
            self.model,
            context.artifacts["checkpoint"],
            args.training_pipeline,
        )
        self.model.to(self.device)
        self.model.eval()
        self.mean, self.std = load_normalization(context.artifacts["normalization"])
        self.roi_shape = tuple(args.ROI_shape)
        self.n_classes = int(args.n_classes)
        self.predict_batch_size = int(args.predict_batch_size)
        self.use_amp = not bool(args.if_fullprecision)

    def predict(
        self,
        context,
        model_input: np.ndarray,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        del context, params
        return predict_probabilities(
            self.model,
            np.asarray(model_input),
            mean=self.mean,
            std=self.std,
            roi_shape=self.roi_shape,
            n_classes=self.n_classes,
            batch_size=self.predict_batch_size,
            device=self.device,
            use_amp=self.use_amp,
        )


def save_model_package(
    destination: str | Path,
    *,
    checkpoint_path: str | Path,
    resolved_config_path: str | Path,
    normalization_path: str | Path,
    provenance: Mapping[str, Any],
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Create a self-contained pyfunc package without changing MLflow run state."""
    destination = Path(destination)
    checkpoint = Path(checkpoint_path)
    config_path = Path(resolved_config_path)
    normalization = Path(normalization_path)
    for name, path in {
        "checkpoint": checkpoint,
        "resolved config": config_path,
        "normalization": normalization,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing model-package {name}: {path}")

    config, resolved = _load_typed_config(config_path)
    args = to_runtime_namespace(config)
    device = torch.device("cpu")
    model = build_model(args, args.training_pipeline, device)
    loaded = load_model_checkpoint(model, checkpoint, args.training_pipeline)
    model.eval()
    mean, std = load_normalization(normalization)

    evidence = {
        "format_version": MODEL_PACKAGE_FORMAT_VERSION,
        "pipeline": args.training_pipeline,
        "checkpoint_sha256": sha256_file(checkpoint),
        "normalization_sha256": sha256_file(normalization),
        "normalization_mean": mean,
        "normalization_std": std,
        "checkpoint_epoch": loaded.get("training_state", {}).get("epoch"),
        "checkpoint_best_score": loaded.get("training_state", {}).get("best_score"),
        **dict(provenance),
    }
    records = destination.parent / f".{destination.name}-records"
    records.mkdir(parents=True, exist_ok=False)
    try:
        resolved_record = records / "resolved-config.yaml"
        resolved_record.write_text(OmegaConf.to_yaml(resolved, resolve=True, sort_keys=True))
        architecture = records / "architecture.txt"
        architecture.write_text(architecture_text(model, args))
        provenance_path = records / "provenance.json"
        provenance_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        mlflow.pyfunc.save_model(
            path=str(destination),
            python_model=str(Path(project_root) / "dscnet" / "experiment" / "model_definition.py"),
            artifacts={
                "checkpoint": str(checkpoint),
                "resolved_config": str(resolved_record),
                "normalization": str(normalization),
                "architecture": str(architecture),
                "provenance": str(provenance_path),
            },
            code_paths=[str(Path(project_root) / "dscnet")],
            signature=model_signature(args.n_classes),
            input_example=np.zeros((1, 1, 1), dtype=np.float32),
            pip_requirements=locked_requirements(project_root),
            metadata=evidence,
        )
    finally:
        shutil.rmtree(records, ignore_errors=True)
    return evidence


def validate_model_package_fresh(
    model_path: str | Path,
    *,
    checkpoint_path: str | Path,
    resolved_config_path: str | Path,
    normalization_path: str | Path,
    sample_volume: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Load only the packaged predictor in a clean process and compare it."""
    config, _ = _load_typed_config(resolved_config_path)
    args = to_runtime_namespace(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    direct = build_model(args, args.training_pipeline, device)
    load_model_checkpoint(direct, checkpoint_path, args.training_pipeline)
    direct.to(device).eval()
    mean, std = load_normalization(normalization_path)
    expected = predict_probabilities(
        direct,
        sample_volume,
        mean=mean,
        std=std,
        roi_shape=args.ROI_shape,
        n_classes=args.n_classes,
        batch_size=args.predict_batch_size,
        device=device,
        use_amp=not args.if_fullprecision,
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sample_path = root / "sample.npy"
        result_path = root / "prediction.npy"
        origin_path = root / "module-origin.txt"
        script_path = root / "load_saved_model.py"
        np.save(sample_path, np.asarray(sample_volume, dtype=np.float32))
        script_path.write_text(
            "import sys\n"
            "import mlflow.pyfunc\n"
            "import numpy as np\n"
            "model = mlflow.pyfunc.load_model(sys.argv[1])\n"
            "prediction = model.predict(np.load(sys.argv[2]))\n"
            "np.save(sys.argv[3], np.asarray(prediction, dtype=np.float32))\n"
            "module = sys.modules['dscnet.experiment.modeling']\n"
            "open(sys.argv[4], 'w').write(module.__file__)\n"
        )
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                str(Path(model_path).resolve()),
                str(sample_path),
                str(result_path),
                str(origin_path),
            ],
            cwd=root,
            env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            check=True,
        )
        observed = np.load(result_path)
        module_origin = Path(origin_path.read_text()).resolve()
    bundled_code = (Path(model_path).resolve() / "code").resolve()
    if bundled_code not in module_origin.parents:
        raise RuntimeError(
            f"fresh process loaded host code instead of bundled code: {module_origin}"
        )
    if expected.shape != observed.shape:
        raise RuntimeError(
            f"packaged prediction shape {observed.shape} != direct {expected.shape}"
        )
    if not np.isfinite(observed).all():
        raise RuntimeError("packaged prediction contains non-finite values")
    if np.any(observed < -1e-6) or np.any(observed > 1 + 1e-6):
        raise RuntimeError("packaged prediction contains invalid probabilities")
    if not np.allclose(observed.sum(axis=0), 1.0, rtol=1e-4, atol=1e-5):
        raise RuntimeError("packaged class probabilities do not sum to one")
    np.testing.assert_allclose(observed, expected, rtol=rtol, atol=atol)
    if not np.array_equal(observed.argmax(axis=0), expected.argmax(axis=0)):
        raise RuntimeError("packaged and direct class masks differ")
    return {
        "shape": list(observed.shape),
        "rtol": rtol,
        "atol": atol,
        "max_absolute_difference": float(np.max(np.abs(observed - expected))),
        "masks_equal": True,
        "fresh_process": True,
        "bundled_module_origin": str(module_origin),
    }


def log_training_model(
    recorder,
    *,
    args,
    resolved_config,
    experiment_digest: str,
    uv_lock_sha256: str,
):
    """Package, smoke-test, and log the best model from a formal training run."""
    import SimpleITK as sitk

    if recorder.run_id is None or recorder.git_provenance is None:
        raise RuntimeError("MLflow training run is not initialized")
    checkpoint = Path(args.Dir_Weights) / args.model_name_max
    normalization = Path(args.Meanstd_path)
    validation_images = [
        Path(line)
        for line in Path(args.Image_Va_txt).read_text().splitlines()
        if line
    ]
    if not validation_images or not validation_images[0].is_file():
        raise RuntimeError("model packaging requires a validation image")
    sample = sitk.GetArrayFromImage(sitk.ReadImage(str(validation_images[0])))
    provenance = {
        "training_run_id": recorder.run_id,
        "experiment_digest": experiment_digest,
        "git_commit": recorder.git_provenance["commit"],
        "git_dirty": recorder.git_provenance["dirty"],
        "uv_lock_sha256": uv_lock_sha256,
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config_path = root / "resolved-config.yaml"
        config_path.write_text(
            OmegaConf.to_yaml(resolved_config, resolve=True, sort_keys=True)
        )
        package_path = root / "model"
        evidence = save_model_package(
            package_path,
            checkpoint_path=checkpoint,
            resolved_config_path=config_path,
            normalization_path=normalization,
            provenance=provenance,
        )
        parity = validate_model_package_fresh(
            package_path,
            checkpoint_path=checkpoint,
            resolved_config_path=config_path,
            normalization_path=normalization,
            sample_volume=np.asarray(sample, dtype=np.float32),
        )
        logged = recorder.log_model_directory(
            package_path,
            name=f"dscnet-{args.training_pipeline}",
            tags={
                "dscnet.pipeline": args.training_pipeline,
                "dscnet.experiment_digest": experiment_digest,
                "dscnet.checkpoint_sha256": evidence["checkpoint_sha256"],
                "dscnet.validation_smoke": "passed",
            },
            params={
                "checkpoint_epoch": evidence["checkpoint_epoch"],
                "checkpoint_best_score": evidence["checkpoint_best_score"],
                "parity_max_absolute_difference": parity[
                    "max_absolute_difference"
                ],
            },
        )
    return {"model_id": logged.model_id, "evidence": evidence, "parity": parity}


def validate_model_package(
    model_path: str | Path,
    *,
    checkpoint_path: str | Path,
    resolved_config_path: str | Path,
    normalization_path: str | Path,
    sample_volume: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Require packaged and direct inference to agree on one raw volume."""
    config, _ = _load_typed_config(resolved_config_path)
    args = to_runtime_namespace(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    direct_model = build_model(args, args.training_pipeline, device)
    load_model_checkpoint(direct_model, checkpoint_path, args.training_pipeline)
    direct_model.to(device).eval()
    mean, std = load_normalization(normalization_path)
    direct = predict_probabilities(
        direct_model,
        sample_volume,
        mean=mean,
        std=std,
        roi_shape=args.ROI_shape,
        n_classes=args.n_classes,
        batch_size=args.predict_batch_size,
        device=device,
        use_amp=not args.if_fullprecision,
    )
    packaged = np.asarray(mlflow.pyfunc.load_model(str(model_path)).predict(sample_volume))
    if direct.shape != packaged.shape:
        raise RuntimeError("packaged model changed prediction shape")
    if not np.isfinite(packaged).all():
        raise RuntimeError("packaged model returned non-finite probabilities")
    if not np.array_equal(np.argmax(direct, axis=0), np.argmax(packaged, axis=0)):
        raise RuntimeError("packaged model changed the predicted segmentation mask")
    np.testing.assert_allclose(packaged, direct, rtol=rtol, atol=atol)
    return {
        "shape": list(packaged.shape),
        "max_absolute_difference": float(np.max(np.abs(packaged - direct))),
        "rtol": float(rtol),
        "atol": float(atol),
        "masks_equal": True,
    }
