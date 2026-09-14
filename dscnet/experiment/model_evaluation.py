"""Run-ID-scoped audit of an existing DSCNet Logged Model.

This module evaluates and records evidence. It never creates registry versions or
assigns aliases.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import mlflow
import numpy as np
from omegaconf import OmegaConf
import SimpleITK as sitk
import torch

from dscnet.evaluation.inference import load_normalization
from dscnet.experiment.config import to_runtime_namespace
from dscnet.experiment.model_audit import (
    _best_validation_metrics,
    _evaluate_split,
    _split_evidence,
    _typed_config,
    _verify_declared_artifact,
)
from dscnet.experiment.tracking import RunRecorder, sha256_file
from dscnet.models.factory import build_model
from dscnet.training.checkpoints import load_model_checkpoint

AUDIT_PROTOCOL_VERSION = 1


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def audit_key(
    controller_run_id: str, logged_model_id: str, checkpoint_sha256: str
) -> str:
    material = {
        "protocol_version": AUDIT_PROTOCOL_VERSION,
        "controller_run_id": controller_run_id,
        "logged_model_id": logged_model_id,
        "checkpoint_sha256": checkpoint_sha256,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def start_final_test(
    audit_dir: str | Path,
    *,
    audit_key_value: str,
    controller_run_id: str,
    logged_model_id: str,
) -> Path:
    """Persist the non-repeatable boundary before any test volume is read."""
    path = Path(audit_dir) / "TEST_STARTED.json"
    if path.exists():
        raise RuntimeError("final test was already started; manual reconciliation required")
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "audit_key": audit_key_value,
            "controller_run_id": controller_run_id,
            "logged_model_id": logged_model_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return path


def audit_logged_model(
    *,
    run_dir: str | Path,
    audit_dir: str | Path,
    audit_key_value: str,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Audit one completed formal run without registry or alias mutation."""
    run_dir = Path(run_dir).resolve()
    audit_dir = Path(audit_dir).resolve()
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
    validation_result_path = _verify_declared_artifact(
        run_dir, artifact_manifest, "outputs/validation-result.json"
    )
    environment_path = _verify_declared_artifact(
        run_dir, artifact_manifest, "control/environment-lock.json"
    )
    git_path = _verify_declared_artifact(
        run_dir, artifact_manifest, "control/git.json"
    )
    for name in ("Image_Va.txt", "Label_Va.txt", "Image_Te.txt", "Label_Te.txt"):
        _verify_declared_artifact(run_dir, artifact_manifest, f"control/{name}")

    config, resolved = _typed_config(config_path)
    args = to_runtime_namespace(config)
    if config.action != "train" or not config.runtime.formal:
        raise RuntimeError("only a completed formal training run can be audited")
    manifest = json.loads(manifest_path.read_text())
    run_result = json.loads(run_result_path.read_text())
    validation_result = json.loads(validation_result_path.read_text())
    if manifest.get("run_id") != run_dir.name:
        raise RuntimeError("controller run manifest does not match run directory")
    if run_result.get("experiment_digest") != manifest.get("experiment_digest"):
        raise RuntimeError("controller completion identity differs from run manifest")
    if (
        validation_result.get("schema_version") != 1
        or validation_result.get("status") != "completed"
        or validation_result.get("controller_run_id") != run_dir.name
        or validation_result.get("training_run_id") != run_result.get("run_id")
        or validation_result.get("experiment_digest") != manifest.get("experiment_digest")
        or validation_result.get("logged_model_id") != run_result.get("logged_model_id")
    ):
        raise RuntimeError("validation result differs from controller completion evidence")

    training_run_id = validation_result["training_run_id"]
    logged_model_id = validation_result["logged_model_id"]
    local_checkpoint = _verify_declared_artifact(
        run_dir,
        artifact_manifest,
        f"outputs/weights/{manifest.get('best_checkpoint_name')}",
    )
    checkpoint_sha256 = sha256_file(local_checkpoint)
    if audit_key(run_dir.name, logged_model_id, checkpoint_sha256) != audit_key_value:
        raise RuntimeError("audit key differs from the frozen model identity")
    if validation_result.get("checkpoint") != {
        "name": manifest.get("best_checkpoint_name"),
        "sha256": checkpoint_sha256,
    }:
        raise RuntimeError("validation result checkpoint identity differs")
    if manifest.get("normalization", {}).get("sha256") != sha256_file(normalization):
        raise RuntimeError("normalization does not match controller evidence")

    tracking_uri = config.runtime.mlflow.tracking_uri
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    training_run = client.get_run(training_run_id)
    logged_model = client.get_logged_model(logged_model_id)
    if training_run.info.status != "FINISHED":
        raise RuntimeError("training MLflow run is not finished")
    if training_run.data.tags.get("dscnet.action") != "train":
        raise RuntimeError("MLflow source is not a training run")
    if training_run.data.tags.get("dscnet.experiment_digest") != manifest.get(
        "experiment_digest"
    ):
        raise RuntimeError("MLflow and controller experiment identities differ")
    if (
        str(logged_model.status) != "READY"
        or logged_model.source_run_id != training_run_id
        or logged_model.tags.get("dscnet.experiment_digest")
        != manifest.get("experiment_digest")
        or logged_model.tags.get("dscnet.checkpoint_sha256") != checkpoint_sha256
    ):
        raise RuntimeError("Logged Model identity or state differs from training evidence")

    training_dataset = json.loads(dataset_path.read_text())
    validation_evidence = _split_evidence(args.Image_Va_txt, args.Label_Va_txt)
    if validation_evidence != training_dataset.get("splits", {}).get("val"):
        raise RuntimeError("current validation data differs from controller evidence")
    environment = json.loads(environment_path.read_text())
    provenance = json.loads(git_path.read_text())
    provenance.setdefault("patch", "")
    if provenance.get("dirty"):
        raise RuntimeError("model audit requires clean recorded source")

    audit_dataset: dict[str, Any] = {
        "schema_version": 1,
        "training_run_dataset": training_dataset,
        "validation": validation_evidence,
        "test": "reserved-not-read",
    }
    with RunRecorder(
        tracking_uri=tracking_uri,
        experiment_name=config.runtime.mlflow.experiment_name,
        artifact_root=config.runtime.mlflow.artifact_root,
        run_name=f"{args.run_label}-model-audit",
        resolved_config=resolved,
        repo_root=run_dir / "source",
        dataset_manifest=audit_dataset,
        experiment_digest_value=manifest["experiment_digest"],
        parent_run_id=training_run_id,
        extra_tags={
            "dscnet.action": "model_audit",
            "dscnet.audit_key": audit_key_value,
            "dscnet.logged_model_id": logged_model_id,
        },
        formal=True,
        git_provenance=provenance,
    ) as recorder:
        with tempfile.TemporaryDirectory() as directory:
            downloaded = Path(directory) / "training-run-artifacts"
            downloaded.mkdir()
            mlflow_checkpoint = Path(
                client.download_artifacts(
                    training_run_id,
                    f"checkpoints/{manifest['best_checkpoint_name']}",
                    str(downloaded),
                )
            )
            mlflow_dataset = Path(
                client.download_artifacts(
                    training_run_id,
                    "records/dataset-manifest.json",
                    str(downloaded),
                )
            )
            mlflow_config = Path(
                client.download_artifacts(
                    training_run_id,
                    "records/resolved-config.yaml",
                    str(downloaded),
                )
            )
            if sha256_file(mlflow_checkpoint) != checkpoint_sha256:
                raise RuntimeError("MLflow best checkpoint differs from controller artifact")
            if json.loads(mlflow_dataset.read_text()) != training_dataset:
                raise RuntimeError("MLflow dataset evidence differs from controller evidence")
            if OmegaConf.to_container(
                OmegaConf.load(mlflow_config), resolve=True
            ) != OmegaConf.to_container(OmegaConf.load(config_path), resolve=True):
                raise RuntimeError("MLflow configuration differs from controller evidence")

        if not torch.cuda.is_available():
            raise RuntimeError("formal model audit requires CUDA")
        device = torch.device("cuda")
        packaged_model = mlflow.pyfunc.load_model(logged_model.artifact_location)
        direct_model = build_model(args, args.training_pipeline, device)
        load_model_checkpoint(direct_model, local_checkpoint, args.training_pipeline)
        direct_model.to(device).eval()
        mean, std = load_normalization(normalization)
        validation_metrics, validation_difference = _evaluate_split(
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
        best_step, expected_validation = _best_validation_metrics(client, training_run_id)
        for name, expected in expected_validation.items():
            if not np.isclose(validation_metrics[name], expected, rtol=rtol, atol=atol):
                raise RuntimeError(
                    f"best-checkpoint validation {name} differs from training step {best_step}"
                )

        start_final_test(
            audit_dir,
            audit_key_value=audit_key_value,
            controller_run_id=run_dir.name,
            logged_model_id=logged_model_id,
        )
        test_evidence = _split_evidence(args.Image_Te_txt, args.Label_Te_txt)
        audit_dataset["test"] = test_evidence
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
        stable_metrics = {
            **{
                f"validation.{key}": value
                for key, value in validation_metrics.items()
                if key != "dice_micro"
            },
            **{
                f"test.{key}": value
                for key, value in test_metrics.items()
                if key != "dice_micro"
            },
        }
        recorder.log_final_metrics(stable_metrics)
        test_manifest_digest = hashlib.sha256(
            json.dumps(test_evidence, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        audit_dataset_path = audit_dir / "audit-dataset.json"
        _atomic_json(audit_dataset_path, audit_dataset)
        recorder.log_artifact(audit_dataset_path, "model-audit")
        report: dict[str, Any] = {
            "schema_version": 1,
            "protocol_version": AUDIT_PROTOCOL_VERSION,
            "status": "passed",
            "audit_key": audit_key_value,
            "controller_run_id": run_dir.name,
            "training_run_id": training_run_id,
            "audit_run_id": recorder.run_id,
            "logged_model_id": logged_model_id,
            "checkpoint": validation_result["checkpoint"],
            "environment": environment,
            "full_parity": {
                "validation_max_absolute_difference": validation_difference,
                "test_max_absolute_difference": test_difference,
                "masks_equal": True,
            },
            "best_validation_step": best_step,
            "validation_metrics": validation_metrics,
            "test_manifest_digest": test_manifest_digest,
            "test_dataset": test_evidence,
            "test_metrics": test_metrics,
            "test_use": "final-held-out-evaluation-not-used-for-selection",
            "authorization": "explicit-final-test",
            "registry_mutated": False,
            "champion_mutated": False,
        }
        report_path = audit_dir / "model-audit.json"
        _atomic_json(report_path, report)
        recorder.log_artifact(report_path, "model-audit")
    _atomic_json(
        audit_dir / "COMPLETED.json",
        {
            "schema_version": 1,
            "audit_key": audit_key_value,
            "report_sha256": sha256_file(audit_dir / "model-audit.json"),
        },
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--audit-key", required=True)
    options = parser.parse_args(argv)
    report = audit_logged_model(
        run_dir=options.run_dir,
        audit_dir=options.audit_dir,
        audit_key_value=options.audit_key,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
