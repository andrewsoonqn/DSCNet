#!/usr/bin/env python3
"""Deterministic local controller for DSCNet experiments on xlogin1."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "DSCNet_3D_opensource" / "Code" / "Kipa" / "DSCNet"
sys.path.insert(0, str(CODE_DIR))

from omegaconf import OmegaConf

from S4_Experiment_Config import (
    load_experiment_config,
    resolved_yaml,
    to_legacy_namespace,
    validate_config,
)
from S4_Experiment_Tracking import build_dataset_manifest, collect_git_provenance, sha256_file

CONTROLLER_VERSION = "1"
PI_EXTENSION_VERSION = "1"
ALLOWED_HOST = "xlogin1"
ALLOWED_REMOTE_CHECKOUT = "/home/a/andrewsq/dev/urop/dscnet"
ALLOWED_REMOTE_ROOT = "/home/a/andrewsq/data/urop/experiments"
ALLOWED_PARTITION = "gpu-long"
ALLOWED_GPU_TYPE = "a100-40"
TRUSTED_ACCOUNTS: frozenset[str] = frozenset({"allusers"})
MAX_GPUS = 1
MAX_MEMORY_GB = 32
MAX_CPUS = 8
MAX_TIME_HOURS = 48
MAX_ARTIFACT_MANIFEST_BYTES = 1024 * 1024
MAX_AUTO_FILE_BYTES = 8 * 1024**3
MAX_AUTO_TOTAL_BYTES = 12 * 1024**3
MAX_ERROR_DETAIL_CHARS = 4096
SSH_TIMEOUT_SECONDS = 120
TRANSFER_TIMEOUT_SECONDS = 1800
MLFLOW_UI_PORT = 5000
MAX_UI_TIMEOUT_MINUTES = 480
MAX_TUNNEL_SECONDS = MAX_UI_TIMEOUT_MINUTES * 60
RUN_ID = re.compile(r"^[0-9a-f]{16}(?:-a[1-9][0-9]*)?$")
JOB_ID = re.compile(r"^[0-9]+$")
EXECUTION_SUFFIXES = {".py", ".yaml", ".yml", ".sh", ".sbatch", ".toml"}


class ExpctlError(RuntimeError):
    pass


@dataclass
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class SubprocessTransport:
    @staticmethod
    def _run(argv: Sequence[str], timeout: int) -> CommandResult:
        try:
            result = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise ExpctlError(f"{argv[0]} timed out after {timeout} seconds") from error
        return CommandResult(result.returncode, result.stdout, result.stderr)

    def ssh(self, host: str, argv: Sequence[str]) -> CommandResult:
        return self._run(
            ["ssh", host, "--", shlex.join(argv)], SSH_TIMEOUT_SECONDS
        )

    def sync_to(self, paths: Sequence[Path], host: str, remote_dir: str) -> CommandResult:
        return self._run(
            ["rsync", "-a", "--protect-args", *map(str, paths), f"{host}:{remote_dir}/"],
            TRANSFER_TIMEOUT_SECONDS,
        )

    def fetch_file(
        self, host: str, remote_path: str, local_path: Path, max_size: int
    ) -> CommandResult:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        return self._run(
            [
                "rsync",
                "-a",
                "--protect-args",
                f"--max-size={max_size}",
                f"{host}:{remote_path}",
                str(local_path),
            ],
            TRANSFER_TIMEOUT_SECONDS,
        )

    def fetch_files(
        self, host: str, remote_root: str, relative_paths: Sequence[str], destination: Path
    ) -> CommandResult:
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False) as stream:
            stream.write("\n".join(relative_paths) + "\n")
            files_from = stream.name
        try:
            result = self._run(
                [
                    "rsync",
                    "-a",
                    "--protect-args",
                    f"--files-from={files_from}",
                    f"--max-size={MAX_AUTO_FILE_BYTES}",
                    f"{host}:{remote_root}/",
                    str(destination) + "/",
                ],
                TRANSFER_TIMEOUT_SECONDS,
            )
        finally:
            Path(files_from).unlink(missing_ok=True)
        return CommandResult(result.returncode, result.stdout, result.stderr)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _canonical_digest(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _bounded_detail(value: str) -> str:
    value = re.sub(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\\\))", "", value)
    value = "".join(
        character
        for character in value
        if character in "\n\t" or 32 <= ord(character) < 127 or ord(character) >= 160
    ).strip()
    if len(value) > MAX_ERROR_DETAIL_CHARS:
        value = "[truncated]\n" + value[-MAX_ERROR_DETAIL_CHARS:]
    return value or "no detail"


def _checked(result: CommandResult, operation: str) -> str:
    if result.returncode != 0:
        detail = _bounded_detail(result.stderr or result.stdout)
        raise ExpctlError(f"{operation} failed: {detail}")
    return result.stdout.strip()


def _experiment_name(path: str | Path) -> str:
    candidate = Path(path).resolve()
    parent = (REPO_ROOT / "configs" / "experiment").resolve()
    if candidate.parent != parent or candidate.suffix not in {".yaml", ".yml"}:
        raise ExpctlError("experiment must be a YAML file in configs/experiment")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", candidate.stem):
        raise ExpctlError("experiment filename contains unsupported characters")
    if not candidate.is_file():
        raise ExpctlError(f"experiment config does not exist: {candidate}")
    return candidate.stem


def _execution_dirty_paths(root: Path) -> list[str]:
    changed = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "-z"],
        check=True,
        capture_output=True,
    ).stdout.decode(errors="surrogateescape")
    paths = []
    for entry in changed.split("\0"):
        if not entry:
            continue
        path = entry[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        candidate = Path(path)
        if (
            candidate.suffix.lower() in EXECUTION_SUFFIXES
            or path == "requirements.txt"
            or path.startswith("configs/")
            or path.startswith("tools/")
            or path.startswith("DSCNet_3D_opensource/")
        ):
            paths.append(path)
    return sorted(set(paths))


def _source_provenance(formal: bool) -> dict[str, Any]:
    provenance = collect_git_provenance(REPO_ROOT)
    execution_dirty = _execution_dirty_paths(REPO_ROOT)
    if formal and execution_dirty:
        raise ExpctlError(
            "formal submission rejected because execution source is dirty: "
            + ", ".join(execution_dirty)
        )
    if formal:
        return {
            "branch": provenance["branch"],
            "commit": provenance["commit"],
            "dirty": False,
            "status": "",
            "patch": "",
            "untracked_source": [],
            "ambient_unrelated_changes_ignored": bool(provenance["dirty"]),
        }
    return provenance


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    return info


def _source_manifest(source: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix()
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_symlink():
            entries.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "target": os.readlink(path),
                    "mode": mode,
                }
            )
        elif path.is_file():
            entries.append(
                {
                    "path": relative,
                    "kind": "file",
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "mode": mode,
                }
            )
    return {"schema_version": 1, "entries": entries, "digest": _canonical_digest(entries)}


def _build_source_snapshot(destination: Path, formal: bool, provenance: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        source = temporary / "source"
        source.mkdir()
        archive = temporary / "head.tar"
        with archive.open("wb") as stream:
            subprocess.run(
                ["git", "-C", str(REPO_ROOT), "archive", "--format=tar", "HEAD"],
                check=True,
                stdout=stream,
            )
        with tarfile.open(archive) as bundle:
            bundle.extractall(source, filter="data")
        if not formal and provenance.get("patch"):
            applied = subprocess.run(
                ["git", "apply", "--binary", "--unsafe-paths", "-"],
                cwd=source,
                input=provenance["patch"],
                text=True,
                capture_output=True,
            )
            if applied.returncode != 0:
                raise ExpctlError(
                    f"could not apply dirty source patch: {_bounded_detail(applied.stderr)}"
                )
        manifest = _source_manifest(source)
        raw_tar = temporary / "source.tar"
        with tarfile.open(raw_tar, "w") as bundle:
            for path in sorted(source.rglob("*")):
                bundle.add(
                    path,
                    arcname=path.relative_to(source).as_posix(),
                    recursive=False,
                    filter=_tar_filter,
                )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with raw_tar.open("rb") as raw, destination.open("wb") as target:
            with gzip.GzipFile(fileobj=target, mode="wb", mtime=0, filename="") as zipped:
                shutil.copyfileobj(raw, zipped)
    return destination, manifest


def _action_splits(action: str) -> tuple[str, ...]:
    if action == "train":
        return ("train", "val")
    if action == "evaluate":
        return ("test",)
    return ("train", "val", "test")


def _action_dataset_manifest(full: dict[str, Any], action: str) -> dict[str, Any]:
    splits = {name: full["splits"][name] for name in _action_splits(action)}
    return {"schema_version": 1, "splits": splits, "digest": _canonical_digest(splits)}


def _write_remote_text_manifests(
    control: Path, full: dict[str, Any], remote_dataset_root: str
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    prefixes = {"train": "Tr", "val": "Va", "test": "Te"}
    for split, prefix in prefixes.items():
        records = full["splits"][split]
        for kind, directory in (("Image", "image"), ("Label", "label")):
            path = control / f"{kind}_{prefix}.txt"
            lines = [
                str(PurePosixPath(remote_dataset_root) / split / directory / item["name"])
                for item in records
            ]
            path.write_text("\n".join(lines) + "\n")
            mapping[f"data.{kind}_{prefix}_txt"] = str(
                PurePosixPath("CONTROL_PLACEHOLDER") / path.name
            )
    return mapping


def _environment_lock() -> dict[str, str]:
    requirements = REPO_ROOT / "requirements.txt"
    return {
        "kind": "requirements-sha256",
        "requirements_sha256": sha256_file(requirements),
    }


def _policy(config: Any) -> None:
    runtime = config.runtime
    slurm = runtime.slurm
    failures = []
    if runtime.kind != "slurm":
        failures.append("runtime.kind must be slurm")
    if slurm.host != ALLOWED_HOST:
        failures.append("host is not allowlisted")
    if runtime.remote_checkout != ALLOWED_REMOTE_CHECKOUT:
        failures.append("remote checkout is not allowlisted")
    if runtime.remote_experiment_root != ALLOWED_REMOTE_ROOT:
        failures.append("remote experiment root is not allowlisted")
    if (
        not slurm.account
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", slurm.account)
        or slurm.account not in TRUSTED_ACCOUNTS
    ):
        failures.append("Slurm account is missing, unsafe, or not allowlisted")
    if slurm.partition != ALLOWED_PARTITION:
        failures.append("partition is not allowlisted")
    if slurm.gpu_type != ALLOWED_GPU_TYPE or slurm.gpus < 1 or slurm.gpus > MAX_GPUS:
        failures.append("GPU request exceeds the allowlist")
    if slurm.memory_gb < 1 or slurm.memory_gb > MAX_MEMORY_GB:
        failures.append("memory request exceeds the allowlist")
    if slurm.cpus < 1 or slurm.cpus > MAX_CPUS:
        failures.append("CPU request exceeds the allowlist")
    if slurm.time_hours < 1 or slurm.time_hours > MAX_TIME_HOURS:
        failures.append("time request exceeds the allowlist")
    if slurm.max_concurrent_runs != 1:
        failures.append("SQLite mode requires max_concurrent_runs=1")
    expected_python = f"{ALLOWED_REMOTE_CHECKOUT}/DSCNetEnv/bin/python"
    if slurm.python_path != expected_python:
        failures.append("Python path is not allowlisted")
    if failures:
        raise ExpctlError("; ".join(failures))


def _port_is_open(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _process_has_token(pid: int, token: str) -> bool:
    if not isinstance(pid, int) or pid <= 1 or not re.fullmatch(r"[0-9a-f]{32}", token):
        return False
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "_ui-tunnel" in result.stdout and token in result.stdout


class LocalTunnelManager:
    def start(
        self,
        state_root: Path,
        local_port: int,
        remote_port: int,
        duration_seconds: int,
    ) -> dict[str, Any]:
        if _port_is_open(local_port):
            raise ExpctlError(f"local port {local_port} is already in use")
        state_root.mkdir(parents=True, exist_ok=True)
        token = secrets.token_hex(16)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_ui-tunnel",
            "--host",
            ALLOWED_HOST,
            "--local-port",
            str(local_port),
            "--remote-port",
            str(remote_port),
            "--duration-seconds",
            str(duration_seconds),
            "--token",
            token,
        ]
        log_path = state_root / "ui-tunnel.log"
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        for _ in range(100):
            if process.poll() is not None:
                raise ExpctlError("SSH tunnel exited before becoming ready")
            if _port_is_open(local_port):
                return {
                    "tunnel_pid": process.pid,
                    "tunnel_token": token,
                    "local_port": local_port,
                    "remote_port": remote_port,
                }
            time.sleep(0.1)
        record = {"tunnel_pid": process.pid, "tunnel_token": token}
        if not self.stop(record):
            raise ExpctlError("SSH tunnel did not become ready and did not stop")
        raise ExpctlError("SSH tunnel did not become ready")

    def is_alive(self, record: dict[str, Any]) -> bool:
        return _process_has_token(
            record.get("tunnel_pid"), record.get("tunnel_token", "")
        )

    def stop(self, record: dict[str, Any]) -> bool:
        if not self.is_alive(record):
            return False
        pid = record["tunnel_pid"]
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        for _ in range(50):
            if not self.is_alive(record):
                return True
            time.sleep(0.1)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        for _ in range(50):
            if not self.is_alive(record):
                return True
            time.sleep(0.1)
        return False


def _run_tunnel_worker(args: argparse.Namespace) -> int:
    if args.host != ALLOWED_HOST:
        raise ExpctlError("tunnel host is not allowlisted")
    if args.remote_port != MLFLOW_UI_PORT:
        raise ExpctlError("remote tunnel port is not allowlisted")
    if not 1024 <= args.local_port <= 65535:
        raise ExpctlError("local tunnel port must be between 1024 and 65535")
    if not 1 <= args.duration_seconds <= MAX_TUNNEL_SECONDS:
        raise ExpctlError("tunnel duration is outside the allowlist")
    if not re.fullmatch(r"[0-9a-f]{32}", args.token):
        raise ExpctlError("tunnel token is invalid")
    command = [
        "ssh",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-N",
        "-L",
        f"127.0.0.1:{args.local_port}:127.0.0.1:{args.remote_port}",
        ALLOWED_HOST,
    ]
    process = subprocess.Popen(command)
    try:
        return process.wait(timeout=args.duration_seconds)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        return 0


class ExperimentController:
    def __init__(
        self,
        state_root: Path | None = None,
        transport: Any | None = None,
        tunnel_manager: Any | None = None,
    ):
        self.state_root = Path(
            state_root
            or os.environ.get("EXPCTL_STATE_ROOT", "~/.local/state/dscnet/expctl")
        ).expanduser()
        self.transport = transport or SubprocessTransport()
        self.tunnel_manager = tunnel_manager or LocalTunnelManager()

    def _load_config(self, experiment: str | Path, extra: Sequence[str] = ()):
        name = _experiment_name(experiment)
        overrides = [f"+experiment={name}", "runtime=slurm", *extra]
        typed, resolved = load_experiment_config(overrides=overrides)
        _policy(typed)
        return name, typed, resolved

    def _stage(self, experiment: str | Path, extra: Sequence[str] = ()) -> dict[str, Any]:
        name, config, resolved = self._load_config(experiment, extra)
        provenance = _source_provenance(config.runtime.formal)
        full_dataset = build_dataset_manifest(config.data.data_dir)
        dataset = _action_dataset_manifest(full_dataset, config.action)
        environment = _environment_lock()
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            archive, source_manifest = _build_source_snapshot(
                temporary / "source.tar.gz", config.runtime.formal, provenance
            )
            provenance["snapshot_digest"] = source_manifest["digest"]
            remote_root = PurePosixPath(config.runtime.remote_experiment_root)
            pending_control = temporary / "control"
            pending_control.mkdir()
            text_paths = _write_remote_text_manifests(
                pending_control,
                full_dataset,
                "/home/a/andrewsq/data/urop/minivess-half",
            )
            legacy_paths = to_legacy_namespace(config)
            normalization = None
            if config.action != "prepare":
                mean_std = Path(legacy_paths.Meanstd_path)
                if not mean_std.is_absolute():
                    mean_std = REPO_ROOT / mean_std
                if not mean_std.is_file():
                    raise ExpctlError(
                        "training and evaluation require a prepared Mean_Std.npy"
                    )
                shutil.copy2(mean_std, pending_control / "Mean_Std.npy")
                normalization = {
                    "name": "Mean_Std.npy",
                    "sha256": sha256_file(mean_std),
                    "size": mean_std.stat().st_size,
                }
            evaluation_checkpoint = None
            if config.action == "evaluate":
                weights = Path(legacy_paths.Dir_Weights)
                if not weights.is_absolute():
                    weights = REPO_ROOT / weights
                candidates = [
                    weights / legacy_paths.model_name_max,
                    weights / legacy_paths.model_name,
                ]
                checkpoint = next((path for path in candidates if path.is_file()), None)
                if checkpoint is None:
                    raise ExpctlError(
                        "evaluation requires an explicit best or latest checkpoint"
                    )
                shutil.copy2(checkpoint, pending_control / "evaluation-checkpoint")
                evaluation_checkpoint = {
                    "name": checkpoint.name,
                    "sha256": sha256_file(checkpoint),
                    "size": checkpoint.stat().st_size,
                }
            remote_control_placeholder = str(remote_root / "runs" / "RUN_PLACEHOLDER" / "control")
            remote_outputs_placeholder = str(remote_root / "runs" / "RUN_PLACEHOLDER" / "outputs")
            OmegaConf.update(resolved, "data.data_dir", "/home/a/andrewsq/data/urop/minivess-half")
            for key, value in text_paths.items():
                OmegaConf.update(resolved, key, value.replace("CONTROL_PLACEHOLDER", remote_control_placeholder))
            OmegaConf.update(resolved, "data.root_dir", remote_outputs_placeholder)
            OmegaConf.update(resolved, "data.Dir_Txt", remote_control_placeholder)
            OmegaConf.update(resolved, "data.Dir_Save", remote_outputs_placeholder + "/results")
            OmegaConf.update(resolved, "data.Meanstd_path", remote_control_placeholder + "/Mean_Std.npy")
            OmegaConf.update(resolved, "data.Dir_Weights", remote_outputs_placeholder + "/weights/")
            OmegaConf.update(resolved, "data.Dir_Log", str(remote_root / "runs" / "RUN_PLACEHOLDER" / "logs") + "/")
            OmegaConf.update(resolved, "data.save_path", remote_outputs_placeholder + "/predictions/")
            OmegaConf.update(resolved, "data.save_path_max", remote_outputs_placeholder + "/predictions-best/")
            validate_config(OmegaConf.to_object(resolved))
            identity_template = resolved_yaml(resolved)
            identity_git = {
                key: value for key, value in provenance.items() if key != "patch"
            }
            digest = _canonical_digest(
                {
                    "config": identity_template,
                    "source": source_manifest["digest"],
                    "git": identity_git,
                    "dataset": dataset,
                    "environment_lock": environment,
                    "normalization": normalization,
                    "evaluation_checkpoint": evaluation_checkpoint,
                    "controller_version": CONTROLLER_VERSION,
                }
            )
            run_id = digest[:16]
            remote_run = str(remote_root / "runs" / run_id)
            run_dir = self.state_root / "runs" / run_id
            existing_manifest = run_dir / "control" / "run-manifest.json"
            if existing_manifest.is_file():
                return self._record(run_id)[1]
            resolved_text = identity_template.replace("RUN_PLACEHOLDER", run_id)
            control = run_dir / "control"
            control.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive, control / "source.tar.gz")
            for path in pending_control.iterdir():
                shutil.copy2(path, control / path.name)
            (control / "resolved-config.yaml").write_text(resolved_text)
            (control / "dataset-manifest.json").write_text(
                json.dumps(dataset, indent=2, sort_keys=True) + "\n"
            )
            git_record = dict(provenance)
            patch = git_record.pop("patch", "")
            (control / "git.json").write_text(json.dumps(git_record, indent=2, sort_keys=True) + "\n")
            if patch:
                (control / "dirty.patch").write_text(patch)
            (control / "environment-lock.json").write_text(
                json.dumps(environment, indent=2, sort_keys=True) + "\n"
            )
            (control / "source-manifest.json").write_text(
                json.dumps(source_manifest, indent=2, sort_keys=True) + "\n"
            )
            manifest = {
                "schema_version": 1,
                "controller_version": CONTROLLER_VERSION,
                "pi_extension_version": PI_EXTENSION_VERSION,
                "run_id": run_id,
                "attempt": 1,
                "experiment": name,
                "action": config.action,
                "experiment_digest": digest,
                "dataset_digest": dataset["digest"],
                "source_digest": source_manifest["digest"],
                "best_checkpoint_name": legacy_paths.model_name_max,
                "normalization": normalization,
                "evaluation_checkpoint": evaluation_checkpoint,
                "archive_sha256": sha256_file(control / "source.tar.gz"),
                "host": config.runtime.slurm.host,
                "account": config.runtime.slurm.account,
                "remote_root": config.runtime.remote_experiment_root,
                "remote_run_dir": remote_run,
                "job_id": None,
                "state": "staged",
            }
            _atomic_json(control / "run-manifest.json", manifest)
            job = self._job_script(config, manifest, resolved, remote_run)
            (control / "job.sbatch").write_text(job)
            return manifest

    def _job_script(self, config: Any, manifest: dict[str, Any], resolved: Any, remote_run: str) -> str:
        slurm = config.runtime.slurm
        source = f"{remote_run}/source"
        control = f"{remote_run}/control"
        outputs = f"{remote_run}/outputs"
        command = " ".join(
            shlex.quote(value)
            for value in [
                slurm.python_path,
                f"{source}/DSCNet_3D_opensource/Code/Kipa/DSCNet/S4_Experiment_Run.py",
                "--resolved-config",
                f"{control}/resolved-config.yaml",
            ]
        )
        helper = f"{source}/tools/expctl_remote.py"
        return f"""#!/usr/bin/env bash
#SBATCH --job-name=dscnet-expctl
#SBATCH --output={remote_run}/logs/slurm-%j.out
#SBATCH --error={remote_run}/logs/slurm-%j.err
#SBATCH --account={slurm.account}
#SBATCH --partition={slurm.partition}
#SBATCH --gpus={slurm.gpu_type}:{slurm.gpus}
#SBATCH --mem={slurm.memory_gb}G
#SBATCH --cpus-per-task={slurm.cpus}
#SBATCH --time={slurm.time_hours}:00:00
set -uo pipefail
mkdir -p {shlex.quote(remote_run + '/logs')} {shlex.quote(outputs + '/weights')} {shlex.quote(outputs + '/predictions')}
export DSCNET_CONTROL_DIR={shlex.quote(control)}
export DSCNET_EXPERIMENT_DIGEST={shlex.quote(manifest['experiment_digest'])}
export DSCNET_RUN_RESULT_PATH={shlex.quote(outputs + '/run-result.json')}
export DSCNET_FINAL_METRICS_PATH={shlex.quote(outputs + '/final-metrics.json')}
{command} 2>&1 | tee {shlex.quote(remote_run + '/logs/pipeline.log')}
pipeline_status=${{PIPESTATUS[0]}}
{shlex.quote(slurm.python_path)} {shlex.quote(helper)} finalize --run-dir {shlex.quote(remote_run)} --best-checkpoint {shlex.quote(to_legacy_namespace(config).model_name_max)}
finalize_status=$?
if [[ "$pipeline_status" -ne 0 ]]; then
  exit "$pipeline_status"
fi
exit "$finalize_status"
"""

    def verify(self, experiment: str | Path, extra: Sequence[str] = ()) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as directory:
            verifier = ExperimentController(Path(directory), self.transport)
            manifest = verifier._stage(experiment, extra)
        return {**manifest, "state": "verified"}

    def submit(
        self, experiment: str | Path, extra: Sequence[str] = (), retry: bool = False
    ) -> dict[str, Any]:
        self.state_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.state_root / ".lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            staged = self._stage(experiment, extra)
            base_id = staged["run_id"]
            if retry:
                attempt = 2
                while (self.state_root / "runs" / f"{base_id}-a{attempt}").exists():
                    attempt += 1
                new_id = f"{base_id}-a{attempt}"
                old = self.state_root / "runs" / base_id
                new = self.state_root / "runs" / new_id
                shutil.copytree(old, new)
                staged = dict(staged)
                staged["run_id"] = new_id
                staged["attempt"] = attempt
                staged["remote_run_dir"] = staged["remote_run_dir"].replace(base_id, new_id)
                control = new / "control"
                for name in (
                    "resolved-config.yaml",
                    "Image_Tr.txt",
                    "Label_Tr.txt",
                    "Image_Va.txt",
                    "Label_Va.txt",
                    "Image_Te.txt",
                    "Label_Te.txt",
                    "job.sbatch",
                ):
                    path = control / name
                    if path.is_file():
                        path.write_text(path.read_text().replace(base_id, new_id))
                staged["job_id"] = None
                staged["state"] = "staged"
                staged.pop("last_status", None)
                staged.pop("error", None)
                _atomic_json(control / "run-manifest.json", staged)
            run_id = staged["run_id"]
            local_control = self.state_root / "runs" / run_id / "control"
            _, existing = self._record(run_id)
            staged = existing
            if existing.get("job_id") and not retry:
                return existing
            if existing.get("state") == "submission_rejected" and not retry:
                raise ExpctlError(
                    "identical submission was previously rejected; use --retry for a new attempt"
                )
            remote_run = staged["remote_run_dir"]
            host = staged["host"]
            if existing.get("state") == "submission_unknown" and not retry:
                helper = f"{remote_run}/source/tools/expctl_remote.py"
                recovery = self.transport.ssh(
                    host,
                    [
                        "python3",
                        helper,
                        "submit",
                        "--run-dir",
                        remote_run,
                        "--remote-root",
                        staged["remote_root"],
                        "--account",
                        staged["account"],
                    ],
                )
                if recovery.returncode != 0:
                    detail = _bounded_detail(recovery.stderr or recovery.stdout)
                    raise ExpctlError(
                        f"Slurm submission recovery failed: {detail or 'no detail'}"
                    )
                recovered = json.loads(recovery.stdout)
                job_id = str(recovered["job_id"])
                if not JOB_ID.fullmatch(job_id):
                    raise ExpctlError("remote submit helper returned an invalid job ID")
                staged.update(job_id=job_id, state="submitted")
                _atomic_json(local_control / "run-manifest.json", staged)
                return staged
            _checked(
                self.transport.ssh(
                    host,
                    [
                        "mkdir",
                        "-p",
                        f"{remote_run}/control",
                        f"{remote_run}/source",
                        f"{remote_run}/logs",
                        f"{remote_run}/outputs/weights",
                    ],
                ),
                "remote staging directory creation",
            )
            _checked(
                self.transport.sync_to(list(local_control.iterdir()), host, f"{remote_run}/control"),
                "source synchronization",
            )
            remote_archive = f"{remote_run}/control/source.tar.gz"
            remote_hash = _checked(
                self.transport.ssh(host, ["sha256sum", remote_archive]),
                "remote source digest verification",
            ).split()[0]
            if remote_hash != staged["archive_sha256"]:
                raise ExpctlError("remote source archive digest does not match local bytes")
            _checked(
                self.transport.ssh(
                    host, ["tar", "-xzf", remote_archive, "-C", f"{remote_run}/source"]
                ),
                "remote source extraction",
            )
            helper = f"{remote_run}/source/tools/expctl_remote.py"
            _checked(
                self.transport.ssh(
                    host,
                    [
                        "python3",
                        helper,
                        "verify-source",
                        "--source-root",
                        f"{remote_run}/source",
                        "--manifest",
                        f"{remote_run}/control/source-manifest.json",
                    ],
                ),
                "remote extracted-source verification",
            )
            normalization = staged.get("normalization")
            if normalization:
                remote_normalization = f"{remote_run}/control/{normalization['name']}"
                normalization_hash = _checked(
                    self.transport.ssh(host, ["sha256sum", remote_normalization]),
                    "remote normalization verification",
                ).split()[0]
                if normalization_hash != normalization["sha256"]:
                    raise ExpctlError("remote normalization digest does not match")
            checkpoint = staged.get("evaluation_checkpoint")
            if checkpoint:
                remote_checkpoint = f"{remote_run}/control/evaluation-checkpoint"
                checkpoint_hash = _checked(
                    self.transport.ssh(host, ["sha256sum", remote_checkpoint]),
                    "remote evaluation-checkpoint verification",
                ).split()[0]
                if checkpoint_hash != checkpoint["sha256"]:
                    raise ExpctlError("remote evaluation checkpoint digest does not match")
                _checked(
                    self.transport.ssh(
                        host,
                        [
                            "cp",
                            "--",
                            remote_checkpoint,
                            f"{remote_run}/outputs/weights/{checkpoint['name']}",
                        ],
                    ),
                    "evaluation-checkpoint staging",
                )
            splits = ",".join(_action_splits(staged["action"]))
            _checked(
                self.transport.ssh(
                    host,
                    [
                        "python3",
                        helper,
                        "verify-data",
                        "--dataset-root",
                        "/home/a/andrewsq/data/urop/minivess-half",
                        "--splits",
                        splits,
                        "--expected-digest",
                        staged["dataset_digest"],
                    ],
                ),
                "remote dataset verification",
            )
            try:
                submission = self.transport.ssh(
                    host,
                    [
                        "python3",
                        helper,
                        "submit",
                        "--run-dir",
                        remote_run,
                        "--remote-root",
                        staged["remote_root"],
                        "--account",
                        staged["account"],
                    ],
                )
                if submission.returncode != 0:
                    detail = _bounded_detail(submission.stderr or submission.stdout)
                    staged["state"] = (
                        "submission_rejected"
                        if "submission rejected:" in detail
                        else "submission_unknown"
                    )
                    raise ExpctlError(f"Slurm submission failed: {detail or 'no detail'}")
                submitted = json.loads(submission.stdout)
                job_id = str(submitted["job_id"])
                if not JOB_ID.fullmatch(job_id):
                    raise ExpctlError("remote submit helper returned an invalid job ID")
                staged.update(job_id=job_id, state="submitted")
            except Exception as error:
                if staged.get("state") not in {
                    "submission_rejected",
                    "submission_unknown",
                }:
                    staged["state"] = "submission_unknown"
                staged["error"] = _bounded_detail(str(error))
                _atomic_json(local_control / "run-manifest.json", staged)
                raise
            _atomic_json(local_control / "run-manifest.json", staged)
            return staged

    def _ui_config(self):
        config, _ = load_experiment_config(overrides=["runtime=slurm"])
        _policy(config)
        ui = config.runtime.mlflow.ui
        if not ui.enabled:
            raise ExpctlError("MLflow UI is disabled")
        if ui.location != "login":
            raise ExpctlError("only the allowlisted login-node MLflow UI is supported")
        if not 1 <= ui.timeout_minutes <= MAX_UI_TIMEOUT_MINUTES:
            raise ExpctlError("MLflow UI timeout is outside the allowlist")
        if config.runtime.mlflow.tracking_uri != (
            "sqlite:////home/a/andrewsq/data/urop/experiments/mlflow.db"
        ):
            raise ExpctlError("MLflow tracking URI is not allowlisted")
        if Path(config.runtime.mlflow.artifact_root) != Path(
            "/home/a/andrewsq/data/urop/experiments/mlflow-artifacts"
        ):
            raise ExpctlError("MLflow artifact root is not allowlisted")
        return config

    def _remote_ui(self, command: str, config: Any) -> dict[str, Any]:
        helper = f"{ALLOWED_REMOTE_CHECKOUT}/tools/expctl_remote.py"
        argv = [
            "python3",
            helper,
            f"ui-{command}",
            "--experiment-root",
            ALLOWED_REMOTE_ROOT,
        ]
        if command == "start":
            argv.extend(
                [
                    "--tracking-uri",
                    config.runtime.mlflow.tracking_uri,
                    "--artifact-root",
                    config.runtime.mlflow.artifact_root,
                    "--python",
                    config.runtime.slurm.python_path,
                    "--port",
                    str(MLFLOW_UI_PORT),
                    "--timeout-minutes",
                    str(config.runtime.mlflow.ui.timeout_minutes),
                ]
            )
        output = _checked(
            self.transport.ssh(ALLOWED_HOST, argv), f"remote MLflow UI {command}"
        )
        try:
            record = json.loads(output)
        except json.JSONDecodeError as error:
            raise ExpctlError("remote MLflow UI returned invalid JSON") from error
        if not isinstance(record, dict) or record.get("state") not in {
            "running",
            "off",
        }:
            raise ExpctlError("remote MLflow UI returned an invalid state")
        if command == "start" and (
            record.get("authentication") != "basic"
            or record.get("username") != "dscnet-ui"
            or not re.fullmatch(r"[A-Za-z0-9_-]{40,64}", record.get("password", ""))
        ):
            raise ExpctlError("remote MLflow UI returned invalid authentication details")
        return record

    def _local_ui_record(self) -> dict[str, Any] | None:
        path = self.state_root / "ui.json"
        if not path.is_file():
            return None
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            path.unlink(missing_ok=True)
            return None
        if not isinstance(record, dict):
            path.unlink(missing_ok=True)
            return None
        return record

    @staticmethod
    def _remote_ui_deadline(remote: dict[str, Any], timeout_minutes: int) -> float:
        deadline = remote.get("deadline_epoch")
        now = time.time()
        if (
            not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
            or deadline <= now
            or deadline > now + timeout_minutes * 60 + 30
        ):
            raise ExpctlError("remote MLflow UI returned an invalid deadline")
        return float(deadline)

    def ui_start(self, local_port: int = MLFLOW_UI_PORT) -> dict[str, Any]:
        if not 1024 <= local_port <= 65535:
            raise ExpctlError("local UI port must be between 1024 and 65535")
        config = self._ui_config()
        self.state_root.mkdir(parents=True, exist_ok=True)
        with (self.state_root / "ui.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            remote = self._remote_ui("start", config)
            try:
                deadline = self._remote_ui_deadline(
                    remote, config.runtime.mlflow.ui.timeout_minutes
                )
            except Exception:
                if remote.get("started"):
                    self._remote_ui("stop", config)
                raise
            local = self._local_ui_record()
            if local and self.tunnel_manager.is_alive(local):
                local_deadline = local.get("tunnel_deadline_epoch", 0)
                same_lifetime = (
                    isinstance(local_deadline, (int, float))
                    and abs(local_deadline - deadline) <= 2
                )
                if local.get("local_port") == local_port and same_lifetime:
                    return {
                        "state": "running",
                        "url": f"http://127.0.0.1:{local_port}",
                        "local_port": local_port,
                        "remote_deadline_epoch": deadline,
                        "authentication": "basic",
                        "username": remote["username"],
                        "password": remote["password"],
                    }
                if not self.tunnel_manager.stop(local):
                    if remote.get("started"):
                        self._remote_ui("stop", config)
                    raise ExpctlError("existing SSH tunnel did not stop")
            (self.state_root / "ui.json").unlink(missing_ok=True)
            tunnel = None
            try:
                remaining_seconds = deadline - time.time()
                if remaining_seconds < 1:
                    raise ExpctlError("remote MLflow UI is too close to its timeout")
                duration_seconds = int(remaining_seconds)
                tunnel = self.tunnel_manager.start(
                    self.state_root,
                    local_port,
                    MLFLOW_UI_PORT,
                    duration_seconds,
                )
                local_record = {
                    **tunnel,
                    "schema_version": 1,
                    "remote_deadline_epoch": deadline,
                    "tunnel_deadline_epoch": deadline,
                }
                _atomic_json(self.state_root / "ui.json", local_record)
            except Exception as error:
                tunnel_cleanup_failed = bool(
                    tunnel and not self.tunnel_manager.stop(tunnel)
                )
                if remote.get("started"):
                    self._remote_ui("stop", config)
                if tunnel_cleanup_failed:
                    raise ExpctlError(
                        "SSH tunnel state write failed and the tunnel did not stop"
                    ) from error
                raise
            return {
                "state": "running",
                "url": f"http://127.0.0.1:{local_port}",
                "local_port": local_port,
                "remote_deadline_epoch": deadline,
                "authentication": "basic",
                "username": remote["username"],
                "password": remote["password"],
            }

    def ui_status(self) -> dict[str, Any]:
        config = self._ui_config()
        self.state_root.mkdir(parents=True, exist_ok=True)
        with (self.state_root / "ui.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            remote = self._remote_ui("status", config)
            local = self._local_ui_record()
            local_deadline = local.get("tunnel_deadline_epoch", 0) if local else 0
            local_valid = (
                isinstance(local_deadline, (int, float))
                and math.isfinite(local_deadline)
                and local_deadline > time.time()
                and isinstance(local.get("local_port"), int)
                and 1024 <= local["local_port"] <= 65535
            ) if local else False
            tunnel_running = bool(
                local_valid and local and self.tunnel_manager.is_alive(local)
            )
            if remote["state"] == "off":
                if tunnel_running and not self.tunnel_manager.stop(local):
                    raise ExpctlError("SSH tunnel did not stop")
                (self.state_root / "ui.json").unlink(missing_ok=True)
                return {"state": "off", "remote": remote}
            deadline = self._remote_ui_deadline(
                remote, config.runtime.mlflow.ui.timeout_minutes
            )
            if not tunnel_running:
                if (
                    local
                    and self.tunnel_manager.is_alive(local)
                    and not self.tunnel_manager.stop(local)
                ):
                    raise ExpctlError("expired SSH tunnel did not stop")
                (self.state_root / "ui.json").unlink(missing_ok=True)
                return {"state": "remote_only", "remote": remote}
            local_port = local["local_port"]
            return {
                "state": "running",
                "url": f"http://127.0.0.1:{local_port}",
                "local_port": local_port,
                "remote_deadline_epoch": deadline,
            }

    def ui_stop(self) -> dict[str, Any]:
        config = self._ui_config()
        self.state_root.mkdir(parents=True, exist_ok=True)
        with (self.state_root / "ui.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            local = self._local_ui_record()
            tunnel_was_running = bool(local and self.tunnel_manager.is_alive(local))
            tunnel_stopped = bool(local and self.tunnel_manager.stop(local))
            if tunnel_was_running and not tunnel_stopped:
                raise ExpctlError("SSH tunnel did not stop")
            (self.state_root / "ui.json").unlink(missing_ok=True)
            remote = self._remote_ui("stop", config)
            return {
                "state": "off",
                "tunnel_stopped": tunnel_stopped,
                "remote": remote,
            }

    def _record(self, run_id: str) -> tuple[Path, dict[str, Any]]:
        if not RUN_ID.fullmatch(run_id):
            raise ExpctlError("invalid run ID")
        path = self.state_root / "runs" / run_id / "control" / "run-manifest.json"
        if not path.is_file():
            raise ExpctlError(f"unknown run ID: {run_id}")
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as error:
            raise ExpctlError("recorded run manifest is invalid") from error
        expected_remote_dir = f"{ALLOWED_REMOTE_ROOT}/runs/{run_id}"
        if not isinstance(record, dict) or record.get("run_id") != run_id:
            raise ExpctlError("recorded run identity does not match its path")
        if record.get("host") != ALLOWED_HOST:
            raise ExpctlError("recorded run host is not allowlisted")
        if record.get("account") not in TRUSTED_ACCOUNTS:
            raise ExpctlError("recorded Slurm account is not allowlisted")
        if record.get("remote_root") != ALLOWED_REMOTE_ROOT:
            raise ExpctlError("recorded remote root is not allowlisted")
        if record.get("remote_run_dir") != expected_remote_dir:
            raise ExpctlError("recorded remote run directory is invalid")
        return path, record

    def _job_operation(self, run_id: str, operation: str, extra: Sequence[str] = ()) -> dict[str, Any]:
        path, record = self._record(run_id)
        job_id = str(record.get("job_id") or "")
        if not JOB_ID.fullmatch(job_id):
            raise ExpctlError("run has no recorded Slurm job ID")
        helper = f"{record['remote_run_dir']}/source/tools/expctl_remote.py"
        argv = ["python3", helper, operation]
        if operation in {"status", "cancel"}:
            argv.extend(["--job-id", job_id])
        else:
            argv.extend(["--run-dir", record["remote_run_dir"], *extra])
        result = json.loads(
            _checked(self.transport.ssh(record["host"], argv), f"remote {operation}")
        )
        if operation == "status":
            record["last_status"] = result
            _atomic_json(path, record)
        return result

    def status(self, run_id: str) -> dict[str, Any]:
        return self._job_operation(run_id, "status")

    def logs(self, run_id: str, lines: int = 200) -> dict[str, Any]:
        if not 1 <= lines <= 1000:
            raise ExpctlError("lines must be between 1 and 1000")
        return self._job_operation(run_id, "logs", ["--lines", str(lines)])

    def cancel(self, run_id: str) -> dict[str, Any]:
        return self._job_operation(run_id, "cancel")

    def fetch(self, run_id: str) -> dict[str, Any]:
        _, record = self._record(run_id)
        local = self.state_root / "runs" / run_id / "fetched"
        artifacts_json = local / "control" / "artifacts.json"
        remote_manifest = f"{record['remote_run_dir']}/control/artifacts.json"
        size_text = _checked(
            self.transport.ssh(
                record["host"], ["stat", "--format=%s", remote_manifest]
            ),
            "artifact manifest size check",
        )
        if not size_text.isdigit() or int(size_text) > MAX_ARTIFACT_MANIFEST_BYTES:
            raise ExpctlError("remote artifact manifest exceeds the size limit")
        _checked(
            self.transport.fetch_file(
                record["host"],
                remote_manifest,
                artifacts_json,
                MAX_ARTIFACT_MANIFEST_BYTES,
            ),
            "artifact manifest retrieval",
        )
        if artifacts_json.stat().st_size > MAX_ARTIFACT_MANIFEST_BYTES:
            raise ExpctlError("remote artifact manifest exceeds the size limit")
        manifest = json.loads(artifacts_json.read_text())
        allowed_exact = {
            "control/resolved-config.yaml",
            "control/dataset-manifest.json",
            "control/source-manifest.json",
            "control/run-manifest.json",
            "control/Image_Tr.txt",
            "control/Label_Tr.txt",
            "control/Image_Va.txt",
            "control/Label_Va.txt",
            "control/Image_Te.txt",
            "control/Label_Te.txt",
            "control/Mean_Std.npy",
            "outputs/final-metrics.json",
            "outputs/run-result.json",
        }
        paths = []
        seen = set()
        total_size = 0
        for artifact in manifest.get("artifacts", []):
            path = artifact.get("path", "")
            pure = PurePosixPath(path)
            allowed = path in allowed_exact or (
                len(pure.parts) == 2 and pure.parts[0] == "logs"
            ) or (
                len(pure.parts) == 3
                and pure.parts[:2] == ("outputs", "weights")
                and pure.name == record["best_checkpoint_name"]
            )
            if not allowed or pure.is_absolute() or ".." in pure.parts:
                raise ExpctlError(f"remote artifact manifest contains undeclared path: {path}")
            if not re.fullmatch(r"[0-9a-f]{64}", artifact.get("sha256", "")):
                raise ExpctlError(f"remote artifact checksum is invalid: {path}")
            size = artifact.get("size")
            if path in seen or not isinstance(size, int) or size < 0:
                raise ExpctlError(f"remote artifact metadata is invalid: {path}")
            if size > MAX_AUTO_FILE_BYTES:
                raise ExpctlError(f"remote artifact exceeds the per-file size limit: {path}")
            total_size += size
            if total_size > MAX_AUTO_TOTAL_BYTES:
                raise ExpctlError("declared artifacts exceed the aggregate size limit")
            seen.add(path)
            paths.append(path)
        required = {
            "control/resolved-config.yaml",
            "control/dataset-manifest.json",
            "control/source-manifest.json",
            "control/run-manifest.json",
        }
        if not required.issubset(seen):
            raise ExpctlError("remote artifact manifest is missing required run records")
        helper = f"{record['remote_run_dir']}/source/tools/expctl_remote.py"
        remote_verification = json.loads(
            _checked(
                self.transport.ssh(
                    record["host"],
                    [
                        "python3",
                        helper,
                        "verify-artifacts",
                        "--run-dir",
                        record["remote_run_dir"],
                    ],
                ),
                "remote artifact size and checksum verification",
            )
        )
        if remote_verification.get("total_size") != total_size:
            raise ExpctlError("remote artifact total differs from the declared total")
        _checked(
            self.transport.fetch_files(
                record["host"], record["remote_run_dir"], paths, local
            ),
            "declared artifact retrieval",
        )
        failures = []
        for artifact in manifest["artifacts"]:
            path = local / artifact["path"]
            if (
                path.is_symlink()
                or not path.is_file()
                or not path.resolve().is_relative_to(local.resolve())
                or path.stat().st_size != artifact["size"]
                or sha256_file(path) != artifact["sha256"]
            ):
                failures.append(artifact["path"])
        if failures:
            raise ExpctlError("artifact checksum verification failed: " + ", ".join(failures))
        result = {"run_id": run_id, "destination": str(local), "artifacts": paths}
        _atomic_json(local / "fetch.json", result)
        return result


def _json_print(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="expctl")
    parser.add_argument("--state-root", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("verify", "submit"):
        child = subparsers.add_parser(command)
        child.add_argument("experiment")
        child.add_argument("--set", action="append", default=[])
        if command == "submit":
            child.add_argument("--retry", action="store_true")
    for command in ("status", "cancel", "fetch"):
        child = subparsers.add_parser(command)
        child.add_argument("run_id")
    log_parser = subparsers.add_parser("logs")
    log_parser.add_argument("run_id")
    log_parser.add_argument("--lines", type=int, default=200)
    ui_parser = subparsers.add_parser("ui")
    ui_subparsers = ui_parser.add_subparsers(dest="ui_command", required=True)
    ui_start_parser = ui_subparsers.add_parser("start")
    ui_start_parser.add_argument("--local-port", type=int, default=MLFLOW_UI_PORT)
    ui_subparsers.add_parser("status")
    ui_subparsers.add_parser("stop")
    return parser


def _tunnel_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("_worker")
    parser.add_argument("--host", required=True)
    parser.add_argument("--local-port", required=True, type=int)
    parser.add_argument("--remote-port", required=True, type=int)
    parser.add_argument("--duration-seconds", required=True, type=int)
    parser.add_argument("--token", required=True)
    return parser


def main() -> int:
    if sys.argv[1:2] == ["_ui-tunnel"]:
        try:
            return _run_tunnel_worker(_tunnel_worker_parser().parse_args())
        except Exception as error:
            print(_bounded_detail(str(error)), file=sys.stderr)
            return 1
    args = build_parser().parse_args()
    controller = ExperimentController(args.state_root)
    try:
        if args.command == "verify":
            result = controller.verify(args.experiment, args.set)
        elif args.command == "submit":
            result = controller.submit(args.experiment, args.set, args.retry)
        elif args.command == "logs":
            result = controller.logs(args.run_id, args.lines)
        elif args.command == "ui":
            if args.ui_command == "start":
                result = controller.ui_start(args.local_port)
            else:
                result = getattr(controller, f"ui_{args.ui_command}")()
        else:
            result = getattr(controller, args.command)(args.run_id)
    except Exception as error:
        print(
            json.dumps({"command": args.command, "error": _bounded_detail(str(error))}),
            file=sys.stderr,
        )
        return 1
    _json_print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
