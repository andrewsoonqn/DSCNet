"""Execute a resolved Hydra experiment inside its MLflow run record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from omegaconf import OmegaConf

from S0_Main import Process
from S4_Experiment_Config import load_experiment_config, to_legacy_namespace
from S4_Experiment_Tracking import (
    RunRecorder,
    build_consumed_split_manifest,
    build_dataset_manifest,
    collect_git_provenance,
    experiment_digest,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


def _tracking_uri(uri: str, repo_root: Path) -> str:
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return uri
    database = Path(uri.removeprefix(prefix))
    if not database.is_absolute():
        database = repo_root / database
    return f"sqlite:///{database}"


def _artifact_root(path: str, repo_root: Path) -> Path:
    root = Path(path)
    return root if root.is_absolute() else repo_root / root


def _identity_config_yaml(resolved) -> str:
    identity = OmegaConf.create(OmegaConf.to_container(resolved, resolve=True))
    identity.training.if_retrain = True
    identity.training.start_train_epoch = 1
    return OmegaConf.to_yaml(identity, resolve=True, sort_keys=True)


def _dataset_manifest(config, args):
    if config.action == "prepare":
        data_root = Path(config.data.data_dir)
        if not data_root.is_absolute():
            data_root = REPO_ROOT / data_root
        return build_dataset_manifest(data_root)
    return build_consumed_split_manifest(args, config.action)


def _log_declared_artifacts(recorder: RunRecorder, args, action: str) -> None:
    if action == "prepare":
        for name in (
            "Image_Tr_txt",
            "Image_Va_txt",
            "Image_Te_txt",
            "Label_Tr_txt",
            "Label_Va_txt",
            "Label_Te_txt",
            "Meanstd_path",
        ):
            path = Path(getattr(args, name))
            if path.is_file():
                recorder.log_artifact(path, "prepared")
        return

    for path in sorted(Path(args.Dir_Log).glob("*")):
        if path.is_file():
            recorder.log_artifact(path, "logs")
    if action == "train":
        for name in (args.model_name, args.model_name_max):
            path = Path(args.Dir_Weights) / name
            if path.is_file():
                recorder.log_artifact(path, "checkpoints")


def run_configured_experiment(
    config_name: str = "config",
    overrides: Sequence[str] | None = None,
    parent_run_id: str | None = None,
) -> dict[str, str]:
    config, resolved = load_experiment_config(config_name, overrides)
    args = to_legacy_namespace(config)
    dataset_manifest = _dataset_manifest(config, args)
    provenance = collect_git_provenance(REPO_ROOT)
    resolved_text = _identity_config_yaml(resolved)
    lock_identifier = sha256_file(REPO_ROOT / "requirements.txt")
    digest = experiment_digest(
        provenance["commit"],
        resolved_text,
        dataset_manifest["digest"],
        lock_identifier,
    )
    args.config_digest = digest

    mlflow_config = config.runtime.mlflow
    with RunRecorder(
        tracking_uri=_tracking_uri(mlflow_config.tracking_uri, REPO_ROOT),
        experiment_name=mlflow_config.experiment_name,
        artifact_root=_artifact_root(mlflow_config.artifact_root, REPO_ROOT),
        run_name=f"{config.data.run_label}-{config.action}",
        resolved_config=resolved,
        repo_root=REPO_ROOT,
        dataset_manifest=dataset_manifest,
        experiment_digest_value=digest,
        parent_run_id=parent_run_id,
        extra_tags={"dscnet.action": config.action},
        formal=config.runtime.formal,
    ) as recorder:
        args.tracker = recorder
        result = Process(args)
        if result and config.action in {"train", "evaluate"}:
            prefix = "validation" if config.action == "train" else "test"
            recorder.log_final_metrics(
                {f"{prefix}.{name}": value for name, value in result.items()}
            )
        _log_declared_artifacts(recorder, args, config.action)
        run_id = recorder.run_id

    return {"run_id": str(run_id), "experiment_digest": digest}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--parent-run-id")
    parser.add_argument("overrides", nargs="*")
    options = parser.parse_args(argv)
    print(
        json.dumps(
            run_configured_experiment(
                options.config_name, options.overrides, options.parent_run_id
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
