"""MLflow run recording and reproducibility evidence for DSCNet."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import mlflow
from mlflow.entities import LoggedModelStatus
from mlflow.exceptions import MlflowException
from omegaconf import DictConfig, OmegaConf


STABLE_METRICS = {
    "train.loss",
    "validation.loss",
    "validation.dice",
    "validation.cldice",
    "validation.precision",
    "validation.recall",
    "validation.false_positives",
    "validation.false_negatives",
    "training.learning_rate",
    "test.dice",
    "test.cldice",
    "test.precision",
    "test.recall",
    "test.false_positives",
    "test.false_negatives",
}
SOURCE_SUFFIXES = {
    ".lock",
    ".py",
    ".sbatch",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_pair_manifest(image_manifest: Path, label_manifest: Path) -> dict[str, Any]:
    images = [Path(line) for line in image_manifest.read_text().splitlines() if line]
    labels = [Path(line) for line in label_manifest.read_text().splitlines() if line]
    if not images or len(images) != len(labels):
        raise ValueError(f"manifest pair is empty or differs in length: {image_manifest}")
    pairs = []
    for image, label in zip(images, labels):
        if image.name != label.name or not image.is_file() or not label.is_file():
            raise ValueError(f"manifest pair is missing or mismatched: {image} / {label}")
        pairs.append(
            {
                "name": image.name,
                "image": str(image.resolve()),
                "label": str(label.resolve()),
                "image_sha256": sha256_file(image),
                "label_sha256": sha256_file(label),
            }
        )
    return {
        "image_manifest": str(image_manifest.resolve()),
        "label_manifest": str(label_manifest.resolve()),
        "image_manifest_sha256": sha256_file(image_manifest),
        "label_manifest_sha256": sha256_file(label_manifest),
        "pairs": pairs,
    }


def build_consumed_split_manifest(args, action: str) -> dict[str, Any]:
    """Hash only the ordered split manifests consumed by the action."""
    if action == "train":
        splits = {
            "train": (args.Image_Tr_txt, args.Label_Tr_txt),
            "val": (args.Image_Va_txt, args.Label_Va_txt),
        }
    elif action == "evaluate":
        splits = {"test": (args.Image_Te_txt, args.Label_Te_txt)}
    else:
        raise ValueError(f"no consumed text manifests for action: {action}")
    manifest: dict[str, Any] = {"splits": {}}
    for split, (images, labels) in splits.items():
        manifest["splits"][split] = _ordered_pair_manifest(
            Path(images), Path(labels)
        )
    canonical = json.dumps(manifest["splits"], sort_keys=True, separators=(",", ":"))
    manifest["digest"] = hashlib.sha256(canonical.encode()).hexdigest()
    return manifest


def build_dataset_manifest(data_dir: str | Path) -> dict[str, Any]:
    """Hash every paired MiniVess file before preparation creates text manifests."""
    root = Path(data_dir).resolve()
    manifest: dict[str, Any] = {"root": str(root), "splits": {}}
    for split in ("train", "val", "test"):
        image_dir = root / split / "image"
        label_dir = root / split / "label"
        images = {path.name: path for path in image_dir.glob("*.nii*") if path.is_file()}
        labels = {path.name: path for path in label_dir.glob("*.nii*") if path.is_file()}
        if not images or images.keys() != labels.keys():
            raise ValueError(f"{split} image/label files are empty or do not match")
        manifest["splits"][split] = [
            {
                "name": name,
                "image_sha256": sha256_file(images[name]),
                "label_sha256": sha256_file(labels[name]),
            }
            for name in sorted(images)
        ]
    canonical = json.dumps(manifest["splits"], sort_keys=True, separators=(",", ":"))
    manifest["digest"] = hashlib.sha256(canonical.encode()).hexdigest()
    return manifest


def _git(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _untracked_source_patch(root: Path, relative_path: str) -> str:
    path = root / relative_path
    if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
        return ""
    result = subprocess.run(
        ["git", "diff", "--no-index", "--binary", "--", "/dev/null", relative_path],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip() or f"could not capture {relative_path}")
    return result.stdout


def collect_git_provenance(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all").stdout
    untracked = [
        path
        for path in _git(
            root, "ls-files", "--others", "--exclude-standard", "-z"
        ).stdout.split("\0")
        if path
        and (root / path).is_file()
        and (root / path).suffix.lower() in SOURCE_SUFFIXES
    ]
    patch = _git(root, "diff", "--binary", "HEAD").stdout
    patch += "".join(_untracked_source_patch(root, path) for path in untracked)
    return {
        "commit": _git(root, "rev-parse", "HEAD").stdout.strip(),
        "branch": _git(root, "branch", "--show-current").stdout.strip(),
        "dirty": bool(status),
        "status": status.splitlines(),
        "untracked_source": [
            {"path": path, "sha256": sha256_file(root / path)} for path in untracked
        ],
        "patch": patch,
    }


def collect_environment() -> dict[str, Any]:
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    record: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
    }
    try:
        import torch

        record["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        record["torch"] = None
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    ) if shutil_which("nvidia-smi") else None
    record["nvidia_smi"] = gpu.stdout.splitlines() if gpu and gpu.returncode == 0 else []
    record["slurm"] = {
        name.lower(): os.environ[name]
        for name in (
            "SLURM_JOB_ID",
            "SLURM_JOB_PARTITION",
            "SLURMD_NODENAME",
            "SLURM_JOB_GPUS",
            "SLURM_CPUS_PER_TASK",
            "SLURM_MEM_PER_NODE",
        )
        if name in os.environ
    }
    return record


def shutil_which(command: str) -> str | None:
    from shutil import which

    return which(command)


def flatten_config(config: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    raw = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else config
    flattened: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, (list, tuple)):
            flattened[prefix] = json.dumps(value, separators=(",", ":"))
        elif value is None:
            flattened[prefix] = "null"
        else:
            flattened[prefix] = value

    visit("", raw)
    return flattened


def experiment_digest(
    git_commit: str,
    resolved_config: str,
    dataset_digest: str,
    environment_lock: str,
) -> str:
    material = "\0".join(
        (git_commit, resolved_config, dataset_digest, environment_lock)
    ).encode()
    return hashlib.sha256(material).hexdigest()


class RunRecorder:
    """Record one training or evaluation run without process-global MLflow state."""

    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str,
        artifact_root: str | Path,
        run_name: str,
        resolved_config: DictConfig,
        repo_root: str | Path,
        dataset_manifest: Mapping[str, Any],
        experiment_digest_value: str | None = None,
        parent_run_id: str | None = None,
        extra_tags: Mapping[str, str] | None = None,
        formal: bool = False,
        git_provenance: Mapping[str, Any] | None = None,
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.artifact_root = Path(artifact_root).resolve()
        self.run_name = run_name
        self.resolved_config = resolved_config
        self.repo_root = Path(repo_root).resolve()
        self.dataset_manifest = dict(dataset_manifest)
        self.experiment_digest = experiment_digest_value
        self.parent_run_id = parent_run_id
        self.extra_tags = dict(extra_tags or {})
        self.formal = formal
        self.run_id: str | None = None
        self.experiment_id: str | None = None
        self.git_provenance: dict[str, Any] | None = (
            dict(git_provenance) if git_provenance is not None else None
        )
        self.environment: dict[str, Any] | None = None
        self.client: mlflow.MlflowClient | None = None

    def _experiment_id(self) -> str:
        assert self.client is not None
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            try:
                experiment_id = self.client.create_experiment(
                    self.experiment_name, artifact_location=self.artifact_root.as_uri()
                )
            except MlflowException:
                experiment = self.client.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    raise
                experiment_id = experiment.experiment_id
        else:
            experiment_id = experiment.experiment_id
        current = self.client.get_experiment(experiment_id)
        if Path(current.artifact_location.removeprefix("file://")).resolve() != self.artifact_root:
            raise RuntimeError("existing MLflow experiment uses a different artifact root")
        return experiment_id

    def __enter__(self) -> "RunRecorder":
        if self.tracking_uri.startswith("sqlite:///"):
            database = Path(self.tracking_uri.removeprefix("sqlite:///"))
            database.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
        if self.git_provenance is None:
            self.git_provenance = collect_git_provenance(self.repo_root)
        if self.formal and self.git_provenance["dirty"]:
            raise RuntimeError("formal MLflow runs require a clean Git tree")
        self.environment = collect_environment()
        tags = {
            "mlflow.runName": self.run_name,
            "git.commit": self.git_provenance["commit"],
            "git.dirty": str(self.git_provenance["dirty"]).lower(),
            **self.extra_tags,
        }
        if self.experiment_digest:
            tags["dscnet.experiment_digest"] = self.experiment_digest
        if self.parent_run_id:
            tags["dscnet.parent_run_id"] = self.parent_run_id
        self.experiment_id = self._experiment_id()
        run = self.client.create_run(self.experiment_id, tags=tags)
        self.run_id = run.info.run_id
        try:
            for name, value in flatten_config(self.resolved_config).items():
                self.client.log_param(self.run_id, name, value)
            self._log_run_records()
        except Exception:
            try:
                self.client.set_terminated(self.run_id, status="FAILED")
            except Exception:
                pass
            raise
        return self

    def _log_run_records(self) -> None:
        assert self.client is not None and self.run_id is not None
        assert self.git_provenance is not None and self.environment is not None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "resolved-config.yaml").write_text(
                OmegaConf.to_yaml(self.resolved_config, resolve=True, sort_keys=True)
            )
            (root / "git.json").write_text(
                json.dumps(
                    {key: value for key, value in self.git_provenance.items() if key != "patch"},
                    indent=2,
                    sort_keys=True,
                )
            )
            (root / "environment.json").write_text(
                json.dumps(self.environment, indent=2, sort_keys=True)
            )
            (root / "dataset-manifest.json").write_text(
                json.dumps(self.dataset_manifest, indent=2, sort_keys=True)
            )
            if self.git_provenance["patch"]:
                (root / "dirty.patch").write_text(self.git_provenance["patch"])
            for path in root.iterdir():
                self.client.log_artifact(self.run_id, str(path), artifact_path="records")

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        if name not in STABLE_METRICS:
            raise ValueError(f"unregistered metric name: {name}")
        assert self.client is not None and self.run_id is not None
        self.client.log_metric(
            self.run_id,
            name,
            float(value),
            timestamp=int(time.time() * 1000),
            step=0 if step is None else step,
        )

    def log_artifact(self, path: str | Path, artifact_path: str) -> None:
        assert self.client is not None and self.run_id is not None
        self.client.log_artifact(self.run_id, str(path), artifact_path=artifact_path)

    def log_model_directory(
        self,
        local_dir: str | Path,
        *,
        name: str,
        tags: Mapping[str, Any],
        params: Mapping[str, Any],
        source_run_id: str | None = None,
    ):
        """Create and upload one MLflow Logged Model without fluent run state."""
        assert self.client is not None and self.run_id is not None
        assert self.experiment_id is not None
        logged = self.client.create_logged_model(
            experiment_id=self.experiment_id,
            name=name,
            source_run_id=source_run_id or self.run_id,
            tags={key: str(value) for key, value in tags.items()},
            params={key: str(value) for key, value in params.items()},
            model_type="pyfunc",
        )
        try:
            self.client.log_model_artifacts(logged.model_id, str(local_dir))
            logged = self.client.finalize_logged_model(
                logged.model_id, LoggedModelStatus.READY
            )
        except Exception:
            try:
                self.client.finalize_logged_model(
                    logged.model_id, LoggedModelStatus.FAILED
                )
            except Exception:
                pass
            raise
        self.client.set_tag(self.run_id, "dscnet.logged_model_id", logged.model_id)
        return logged

    def log_final_metrics(self, metrics: Mapping[str, float]) -> None:
        unknown = set(metrics) - STABLE_METRICS
        if unknown:
            raise ValueError(f"unregistered metric names: {sorted(unknown)}")
        for name, value in metrics.items():
            self.log_metric(name, value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final-metrics.json"
            path.write_text(json.dumps(dict(metrics), indent=2, sort_keys=True))
            self.log_artifact(path, artifact_path="metrics")

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self.client is not None and self.run_id is not None:
            try:
                self.client.set_terminated(
                    self.run_id, status="FAILED" if exc_type else "FINISHED"
                )
            except Exception:
                if exc_type is None:
                    raise
        return False
