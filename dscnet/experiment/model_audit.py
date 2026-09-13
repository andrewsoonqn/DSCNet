"""One-time audit and registry promotion for an existing formal DSCNet run.

This module is an internal library surface, not a command-line backfill interface.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
import numpy as np
from omegaconf import OmegaConf
import SimpleITK as sitk
import torch

from dscnet.evaluation.inference import load_normalization, predict_probabilities
from dscnet.evaluation.metrics import cldice_score, dice_score, to_minivess_binary_mask
from dscnet.experiment.config import ExperimentConfig, to_runtime_namespace, validate_config
from dscnet.experiment.modeling import save_model_package, validate_model_package_fresh
from dscnet.experiment.tracking import (
    RunRecorder,
    collect_git_provenance,
    sha256_file,
)
from dscnet.models.factory import build_model
from dscnet.training.checkpoints import load_model_checkpoint


def _typed_config(path: Path):
    loaded = OmegaConf.load(path)
    resolved = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), loaded)
    OmegaConf.resolve(resolved)
    config = OmegaConf.to_object(resolved)
    if not isinstance(config, ExperimentConfig):
        raise TypeError("audit config did not produce ExperimentConfig")
    validate_config(config)
    return config, resolved


def _manifest_paths(path: str | Path) -> list[Path]:
    paths = [Path(line) for line in Path(path).read_text().splitlines() if line]
    if not paths or any(not item.is_file() for item in paths):
        raise RuntimeError(f"audit manifest is empty or missing files: {path}")
    return paths


def _verify_declared_artifact(
    run_dir: Path, artifact_manifest: dict[str, Any], relative_path: str
) -> Path:
    matches = [
        item
        for item in artifact_manifest.get("artifacts", [])
        if item.get("path") == relative_path
    ]
    if len(matches) != 1:
        raise RuntimeError(f"artifact manifest lacks one {relative_path!r} record")
    path = run_dir / relative_path
    record = matches[0]
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"declared audit input is missing or unsafe: {relative_path}")
    if path.stat().st_size != record.get("size") or sha256_file(path) != record.get(
        "sha256"
    ):
        raise RuntimeError(f"declared audit input changed: {relative_path}")
    return path


def _split_evidence(image_manifest: str | Path, label_manifest: str | Path):
    images = _manifest_paths(image_manifest)
    labels = _manifest_paths(label_manifest)
    if len(images) != len(labels):
        raise RuntimeError("audit image and label manifest counts differ")
    result = []
    for image, label in zip(images, labels):
        if image.name != label.name:
            raise RuntimeError(f"audit image/label names differ: {image} / {label}")
        result.append(
            {
                "name": image.name,
                "image_sha256": sha256_file(image),
                "label_sha256": sha256_file(label),
            }
        )
    return result


def _prediction_metrics(predictions: list[np.ndarray], labels: list[Path]) -> dict[str, float]:
    if len(predictions) != len(labels) or not labels:
        raise RuntimeError("prediction and label counts differ")
    dice_values = []
    cldice_values = []
    true_positives = false_positives = false_negatives = 0
    for prediction, label_path in zip(predictions, labels):
        target = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
        prediction = to_minivess_binary_mask(
            prediction, name=f"prediction {label_path.name}"
        )
        target = to_minivess_binary_mask(target, name=f"target {label_path.name}")
        if prediction.shape != target.shape:
            raise RuntimeError(f"prediction shape differs for {label_path.name}")
        dice_values.append(dice_score(prediction, target))
        cldice_values.append(cldice_score(prediction, target))
        true_positives += int(np.logical_and(prediction == 1, target == 1).sum())
        false_positives += int(np.logical_and(prediction == 1, target == 0).sum())
        false_negatives += int(np.logical_and(prediction == 0, target == 1).sum())
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    dice_denominator = 2 * true_positives + false_positives + false_negatives
    return {
        "dice": float(np.mean(dice_values)),
        "cldice": float(np.mean(cldice_values)),
        "dice_micro": (
            2 * true_positives / dice_denominator if dice_denominator else 1.0
        ),
        "precision": (
            true_positives / precision_denominator if precision_denominator else 1.0
        ),
        "recall": (
            true_positives / recall_denominator if recall_denominator else 1.0
        ),
        "false_positives": float(false_positives),
        "false_negatives": float(false_negatives),
    }


def _evaluate_split(
    *,
    image_manifest: str | Path,
    label_manifest: str | Path,
    direct_model,
    packaged_model,
    args,
    mean: float,
    std: float,
    device: torch.device,
    rtol: float,
    atol: float,
) -> tuple[dict[str, float], float]:
    images = _manifest_paths(image_manifest)
    labels = _manifest_paths(label_manifest)
    if len(images) != len(labels):
        raise RuntimeError("audit image and label manifest counts differ")
    masks = []
    maximum_difference = 0.0
    for image_path in images:
        raw = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
        direct = predict_probabilities(
            direct_model,
            raw,
            mean=mean,
            std=std,
            roi_shape=args.ROI_shape,
            n_classes=args.n_classes,
            batch_size=args.predict_batch_size,
            device=device,
            use_amp=not args.if_fullprecision,
        )
        packaged = np.asarray(packaged_model.predict(raw.astype(np.float32, copy=False)))
        expected_shape = (args.n_classes, *raw.shape)
        if packaged.shape != expected_shape:
            raise RuntimeError(
                f"packaged model returned {packaged.shape}; expected {expected_shape}"
            )
        if not np.isfinite(packaged).all():
            raise RuntimeError(f"packaged model returned non-finite values for {image_path}")
        if np.any(packaged < -1e-6) or np.any(packaged > 1 + 1e-6):
            raise RuntimeError(f"packaged model returned invalid probabilities for {image_path}")
        if not np.allclose(packaged.sum(axis=0), 1.0, rtol=1e-4, atol=1e-5):
            raise RuntimeError(f"packaged class probabilities do not sum to one for {image_path}")
        np.testing.assert_allclose(packaged, direct, rtol=rtol, atol=atol)
        direct_mask = np.argmax(direct, axis=0)
        packaged_mask = np.argmax(packaged, axis=0)
        if not np.array_equal(packaged_mask, direct_mask):
            raise RuntimeError(f"packaged model changed mask for {image_path.name}")
        maximum_difference = max(
            maximum_difference, float(np.max(np.abs(packaged - direct)))
        )
        masks.append(packaged_mask)
    return _prediction_metrics(masks, labels), maximum_difference


def _best_validation_metrics(client, run_id: str) -> tuple[int, dict[str, float]]:
    """Read current metric names or the legacy names used by the baseline run."""
    dice_name = "validation.dice_macro"
    dice_history = client.get_metric_history(run_id, dice_name)
    legacy = not dice_history
    if legacy:
        dice_name = "validation.dice"
        dice_history = client.get_metric_history(run_id, dice_name)
    if not dice_history:
        raise RuntimeError("training run has no validation Dice history")
    best = max(dice_history, key=lambda metric: (metric.value, -metric.step))
    metrics = {"dice": float(best.value)}
    names = {
        "cldice": "cldice" if legacy else "cldice_macro",
        "precision": "precision" if legacy else "precision_micro",
        "recall": "recall" if legacy else "recall_micro",
        "false_positives": "false_positives",
        "false_negatives": "false_negatives",
    }
    if not legacy:
        names["dice_micro"] = "dice_micro"
    for result_name, stored_name in names.items():
        history = client.get_metric_history(run_id, f"validation.{stored_name}")
        matches = [metric for metric in history if metric.step == best.step]
        if len(matches) != 1:
            raise RuntimeError(
                f"training run lacks one validation.{stored_name} at step {best.step}"
            )
        metrics[result_name] = float(matches[0].value)
    return int(best.step), metrics


def _create_version(client, *, logged_model, training_run_id: str):
    name = "dscnet-standard"
    try:
        client.create_registered_model(
            name,
            description=(
                "Standard DSCNet MiniVess segmentation models. The champion alias is "
                "assigned only after checkpoint/package parity and validation/test audit."
            ),
        )
    except MlflowException as error:
        if error.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
        existing = client.get_registered_model(name)
        if existing.name != name:
            raise RuntimeError("existing registered model identity is inconsistent")
    versions = list(client.search_model_versions(f"name = '{name}'"))
    if versions:
        raise RuntimeError("dscnet-standard already has a model version; refusing v1 overwrite")
    version = client.create_model_version(
        name=name,
        source=logged_model.artifact_location,
        run_id=training_run_id,
        model_id=logged_model.model_id,
        tags={
            "dscnet.audit": "passed",
            "dscnet.logged_model_id": logged_model.model_id,
            "dscnet.test_evaluation": "reported-not-selected",
            "dscnet.training_run_id": training_run_id,
        },
        description=(
            "Provisional baseline selected by validation performance; untouched test "
            "metrics are reported in the linked audit and were not used for selection."
        ),
    )
    if str(version.version) != "1":
        raise RuntimeError("first dscnet-standard model version is not version 1")
    verified = client.get_model_version(name, "1")
    if (
        verified.source != logged_model.artifact_location
        or verified.run_id != training_run_id
        or verified.tags.get("dscnet.logged_model_id") != logged_model.model_id
    ):
        raise RuntimeError("registered model version does not match audited source")
    return version


def _promote_completed_audit(
    client,
    *,
    audit_run_id: str,
    logged_model_id: str,
    training_run_id: str,
) -> None:
    finished_audit = client.get_run(audit_run_id)
    if finished_audit.info.status != "FINISHED":
        raise RuntimeError("model audit run did not finish successfully")
    ready_model = client.get_logged_model(logged_model_id)
    if str(ready_model.status) != "READY":
        raise RuntimeError("audited Logged Model is not READY")
    client.set_logged_model_tags(logged_model_id, {"dscnet.audit": "passed"})
    _create_version(
        client,
        logged_model=ready_model,
        training_run_id=training_run_id,
    )
    # Alias assignment is intentionally the final external operation.
    client.set_registered_model_alias("dscnet-standard", "champion", "1")


def audit_and_register_existing_run(
    *,
    run_dir: str | Path,
    training_mlflow_run_id: str,
    project_root: str | Path,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Package, audit, register, and promote one completed standard training run."""
    run_dir = Path(run_dir).resolve()
    project_root = Path(project_root).resolve()
    control = run_dir / "control"
    artifact_manifest = json.loads((control / "artifacts.json").read_text())
    config_path = _verify_declared_artifact(
        run_dir, artifact_manifest, "control/resolved-config.yaml"
    )
    manifest_path = _verify_declared_artifact(
        run_dir, artifact_manifest, "control/run-manifest.json"
    )
    dataset_path = _verify_declared_artifact(
        run_dir, artifact_manifest, "control/dataset-manifest.json"
    )
    normalization = _verify_declared_artifact(
        run_dir, artifact_manifest, "control/Mean_Std.npy"
    )
    run_result_path = _verify_declared_artifact(
        run_dir, artifact_manifest, "outputs/run-result.json"
    )
    environment_path = _verify_declared_artifact(
        run_dir, artifact_manifest, "control/environment-lock.json"
    )
    for name in ("Image_Va.txt", "Label_Va.txt", "Image_Te.txt", "Label_Te.txt"):
        _verify_declared_artifact(run_dir, artifact_manifest, f"control/{name}")

    config, resolved = _typed_config(config_path)
    args = to_runtime_namespace(config)
    if config.action != "train" or not config.runtime.formal:
        raise RuntimeError("only a completed formal training run can be audited")
    if args.training_pipeline != "standard":
        raise RuntimeError("the initial registry audit accepts only the standard pipeline")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("run_id") != run_dir.name:
        raise RuntimeError("controller run manifest does not match run directory")
    run_result = json.loads(run_result_path.read_text())
    if run_result.get("run_id") != training_mlflow_run_id:
        raise RuntimeError("requested MLflow run differs from controller completion record")
    if run_result.get("experiment_digest") != manifest.get("experiment_digest"):
        raise RuntimeError("controller completion identity differs from run manifest")
    local_checkpoint = _verify_declared_artifact(
        run_dir,
        artifact_manifest,
        f"outputs/weights/{manifest.get('best_checkpoint_name')}",
    )
    if manifest.get("normalization", {}).get("sha256") != sha256_file(normalization):
        raise RuntimeError("normalization does not match controller evidence")

    tracking_uri = config.runtime.mlflow.tracking_uri
    artifact_root = config.runtime.mlflow.artifact_root
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    training_run = client.get_run(training_mlflow_run_id)
    if training_run.info.status != "FINISHED":
        raise RuntimeError("training MLflow run is not finished")
    if training_run.data.tags.get("dscnet.action") != "train":
        raise RuntimeError("MLflow source is not a training run")
    if training_run.data.tags.get("dscnet.experiment_digest") != manifest.get(
        "experiment_digest"
    ):
        raise RuntimeError("MLflow and controller experiment identities differ")

    training_dataset = json.loads(dataset_path.read_text())
    validation_evidence = _split_evidence(args.Image_Va_txt, args.Label_Va_txt)
    if validation_evidence != training_dataset.get("splits", {}).get("val"):
        raise RuntimeError("current validation data differs from controller evidence")
    test_evidence = _split_evidence(args.Image_Te_txt, args.Label_Te_txt)
    audit_dataset = {
        "training_run_dataset": training_dataset,
        "validation": validation_evidence,
        "untouched_test": test_evidence,
    }
    provenance = collect_git_provenance(project_root)
    if provenance["dirty"]:
        raise RuntimeError("model audit requires a clean packaging source")
    environment = json.loads(environment_path.read_text())
    package_provenance = {
        "training_run_id": training_mlflow_run_id,
        "controller_run_id": run_dir.name,
        "experiment_digest": manifest["experiment_digest"],
        "training_git_commit": training_run.data.tags.get("git.commit"),
        "packaging_git_commit": provenance["commit"],
        "uv_lock_sha256": environment["uv_lock_sha256"],
    }

    with RunRecorder(
        tracking_uri=tracking_uri,
        experiment_name=config.runtime.mlflow.experiment_name,
        artifact_root=artifact_root,
        run_name=f"{args.run_label}-model-audit",
        resolved_config=resolved,
        repo_root=project_root,
        dataset_manifest=audit_dataset,
        experiment_digest_value=manifest["experiment_digest"],
        parent_run_id=training_mlflow_run_id,
        extra_tags={"dscnet.action": "model_audit"},
        formal=True,
        git_provenance=provenance,
    ) as recorder:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            downloaded = temporary / "training-run-artifacts"
            downloaded.mkdir()
            checkpoint = Path(
                client.download_artifacts(
                    training_mlflow_run_id,
                    f"checkpoints/{manifest['best_checkpoint_name']}",
                    str(downloaded),
                )
            )
            mlflow_dataset = Path(
                client.download_artifacts(
                    training_mlflow_run_id,
                    "records/dataset-manifest.json",
                    str(downloaded),
                )
            )
            mlflow_config = Path(
                client.download_artifacts(
                    training_mlflow_run_id,
                    "records/resolved-config.yaml",
                    str(downloaded),
                )
            )
            if sha256_file(checkpoint) != sha256_file(local_checkpoint):
                raise RuntimeError("MLflow best checkpoint differs from controller artifact")
            if json.loads(mlflow_dataset.read_text()) != training_dataset:
                raise RuntimeError("MLflow dataset evidence differs from controller evidence")
            if OmegaConf.to_container(OmegaConf.load(mlflow_config), resolve=True) != OmegaConf.to_container(
                OmegaConf.load(config_path), resolve=True
            ):
                raise RuntimeError("MLflow configuration differs from controller evidence")

            package_path = temporary / "model"
            evidence = save_model_package(
                package_path,
                checkpoint_path=checkpoint,
                resolved_config_path=config_path,
                normalization_path=normalization,
                provenance=package_provenance,
                project_root=project_root,
            )
            first_validation = _manifest_paths(args.Image_Va_txt)[0]
            smoke = validate_model_package_fresh(
                package_path,
                checkpoint_path=checkpoint,
                resolved_config_path=config_path,
                normalization_path=normalization,
                sample_volume=sitk.GetArrayFromImage(
                    sitk.ReadImage(str(first_validation))
                ).astype(np.float32, copy=False),
                rtol=rtol,
                atol=atol,
            )
            logged = recorder.log_model_directory(
                package_path,
                name="dscnet-standard",
                tags={
                    "dscnet.pipeline": "standard",
                    "dscnet.audit": "pending",
                    "dscnet.training_run_id": training_mlflow_run_id,
                    "dscnet.checkpoint_sha256": evidence["checkpoint_sha256"],
                },
                params={
                    "checkpoint_epoch": evidence["checkpoint_epoch"],
                    "checkpoint_best_score": evidence["checkpoint_best_score"],
                },
                source_run_id=training_mlflow_run_id,
            )
            packaged_model = mlflow.pyfunc.load_model(logged.artifact_location)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            direct_model = build_model(args, args.training_pipeline, device)
            load_model_checkpoint(direct_model, checkpoint, args.training_pipeline)
            direct_model.to(device).eval()

        mean, std = load_normalization(normalization)
        validation_metrics, val_difference = _evaluate_split(
            image_manifest=args.Image_Va_txt,
            label_manifest=args.Label_Va_txt,
            direct_model=direct_model,
            packaged_model=packaged_model,
            args=args,
            mean=mean,
            std=std,
            device=device,
            rtol=rtol,
            atol=atol,
        )
        test_metrics, test_difference = _evaluate_split(
            image_manifest=args.Image_Te_txt,
            label_manifest=args.Label_Te_txt,
            direct_model=direct_model,
            packaged_model=packaged_model,
            args=args,
            mean=mean,
            std=std,
            device=device,
            rtol=rtol,
            atol=atol,
        )
        best_step, expected_validation = _best_validation_metrics(
            client, training_mlflow_run_id
        )
        for name, expected in expected_validation.items():
            if not np.isclose(validation_metrics[name], expected, rtol=rtol, atol=atol):
                raise RuntimeError(
                    f"best-checkpoint validation {name} differs from training step {best_step}"
                )
        final_metrics = {
            **{f"validation.{key}": value for key, value in validation_metrics.items()},
            **{f"test.{key}": value for key, value in test_metrics.items()},
        }
        # The micro Dice value is additional audit evidence, not a stable run metric.
        stable_final_metrics = {
            key: value for key, value in final_metrics.items() if not key.endswith(".dice_micro")
        }
        recorder.log_final_metrics(stable_final_metrics)
        report: dict[str, Any] = {
            "status": "passed",
            "training_run_id": training_mlflow_run_id,
            "controller_run_id": run_dir.name,
            "audit_run_id": recorder.run_id,
            "logged_model_id": logged.model_id,
            "checkpoint": evidence,
            "smoke_parity": smoke,
            "full_parity": {
                "validation_max_absolute_difference": val_difference,
                "test_max_absolute_difference": test_difference,
                "masks_equal": True,
            },
            "best_validation_step": best_step,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "test_use": "reported-not-selected",
            "registry_promotion": {
                "eligible": True,
                "name": "dscnet-standard",
                "version": "1",
                "alias": "champion",
                "performed_only_after_audit_run_finishes": True,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "model-audit.json"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            recorder.log_artifact(report_path, "model")

    _promote_completed_audit(
        client,
        audit_run_id=report["audit_run_id"],
        logged_model_id=logged.model_id,
        training_run_id=training_mlflow_run_id,
    )
    return report
