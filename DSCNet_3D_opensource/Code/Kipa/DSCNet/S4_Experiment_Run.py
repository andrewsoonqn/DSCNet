"""Execute a resolved Hydra experiment inside its MLflow run record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from omegaconf import OmegaConf

from S0_Main import Process
from S4_Experiment_Config import (
    ExperimentConfig,
    load_experiment_config,
    to_legacy_namespace,
    validate_config,
)
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


def _canonical_digest(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _control_evidence(resolved) -> tuple[dict, dict, str, str]:
    control = Path(os.environ["DSCNET_CONTROL_DIR"]).resolve()
    dataset = json.loads((control / "dataset-manifest.json").read_text())
    provenance = json.loads((control / "git.json").read_text())
    patch = control / "dirty.patch"
    provenance["patch"] = patch.read_text() if patch.is_file() else ""
    source = json.loads((control / "source-manifest.json").read_text())
    environment = json.loads((control / "environment-lock.json").read_text())
    run_manifest = json.loads((control / "run-manifest.json").read_text())
    run_id = run_manifest["run_id"]
    for key in ("normalization", "evaluation_checkpoint"):
        evidence = run_manifest.get(key)
        if not evidence:
            continue
        path = (
            Path(resolved["data"]["Dir_Weights"]) / evidence["name"]
            if key == "evaluation_checkpoint"
            else control / evidence["name"]
        )
        if (
            not path.is_file()
            or path.stat().st_size != evidence["size"]
            or sha256_file(path) != evidence["sha256"]
        ):
            raise RuntimeError(f"staged {key} bytes do not match runtime evidence")
    normalized_config = _identity_config_yaml(resolved).replace(run_id, "RUN_PLACEHOLDER")
    identity_git = {key: value for key, value in provenance.items() if key != "patch"}
    digest = _canonical_digest(
        {
            "config": normalized_config,
            "source": source["digest"],
            "git": identity_git,
            "dataset": dataset,
            "environment_lock": environment,
            "normalization": run_manifest.get("normalization"),
            "evaluation_checkpoint": run_manifest.get("evaluation_checkpoint"),
            "controller_version": run_manifest["controller_version"],
        }
    )
    expected = os.environ.get("DSCNET_EXPERIMENT_DIGEST", "")
    if digest != expected or digest != run_manifest["experiment_digest"]:
        raise RuntimeError("staged experiment digest does not match runtime evidence")
    return dataset, provenance, environment["requirements_sha256"], digest


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


def _write_json_from_environment(variable: str, value: dict[str, Any]) -> None:
    destination = os.environ.get(variable)
    if not destination:
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _execute(config, resolved, parent_run_id: str | None = None) -> dict[str, str]:
    args = to_legacy_namespace(config)
    if os.environ.get("DSCNET_CONTROL_DIR"):
        dataset_manifest, provenance, lock_identifier, digest = _control_evidence(resolved)
    else:
        dataset_manifest = _dataset_manifest(config, args)
        provenance = collect_git_provenance(REPO_ROOT)
        lock_identifier = sha256_file(REPO_ROOT / "requirements.txt")
        digest = experiment_digest(
            provenance["commit"],
            _identity_config_yaml(resolved),
            dataset_manifest["digest"],
            lock_identifier,
        )
    args.config_digest = digest
    mlflow_config = config.runtime.mlflow
    final_metrics: dict[str, float] = {}
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
        git_provenance=provenance,
    ) as recorder:
        args.tracker = recorder
        result = Process(args)
        if result and config.action in {"train", "evaluate"}:
            prefix = "validation" if config.action == "train" else "test"
            final_metrics = {f"{prefix}.{name}": value for name, value in result.items()}
            recorder.log_final_metrics(final_metrics)
        _log_declared_artifacts(recorder, args, config.action)
        run_id = recorder.run_id
    output = {"run_id": str(run_id), "experiment_digest": digest}
    _write_json_from_environment("DSCNET_RUN_RESULT_PATH", output)
    if final_metrics:
        _write_json_from_environment("DSCNET_FINAL_METRICS_PATH", final_metrics)
    return output


def run_configured_experiment(
    config_name: str = "config",
    overrides: Sequence[str] | None = None,
    parent_run_id: str | None = None,
) -> dict[str, str]:
    config, resolved = load_experiment_config(config_name, overrides)
    return _execute(config, resolved, parent_run_id)


def run_resolved_experiment(
    path: str | Path, parent_run_id: str | None = None
) -> dict[str, str]:
    loaded = OmegaConf.load(path)
    resolved = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), loaded)
    OmegaConf.resolve(resolved)
    config = OmegaConf.to_object(resolved)
    if not isinstance(config, ExperimentConfig):
        raise TypeError("resolved config did not produce ExperimentConfig")
    validate_config(config)
    return _execute(config, resolved, parent_run_id)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--resolved-config")
    parser.add_argument("--parent-run-id")
    parser.add_argument("overrides", nargs="*")
    options = parser.parse_args(argv)
    if options.resolved_config:
        if options.overrides:
            parser.error("overrides cannot be combined with --resolved-config")
        result = run_resolved_experiment(options.resolved_config, options.parent_run_id)
    else:
        result = run_configured_experiment(
            options.config_name, options.overrides, options.parent_run_id
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
