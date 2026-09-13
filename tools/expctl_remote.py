#!/usr/bin/env python3
"""Fixed remote operations used by the local expctl controller."""

from __future__ import annotations

import argparse
import configparser
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
import time

JOB_ID = re.compile(r"^(\d+)(?:;[A-Za-z0-9_.-]+)?$")
MAX_ARTIFACT_MANIFEST_BYTES = 1024 * 1024
MAX_AUTO_FILE_BYTES = 8 * 1024**3
MAX_AUTO_TOTAL_BYTES = 12 * 1024**3
ALLOWED_EXPERIMENT_ROOT = Path("/home/a/andrewsq/data/urop/experiments")
ALLOWED_REMOTE_CHECKOUT = Path("/home/a/andrewsq/dev/urop/dscnet")
ENVIRONMENT_ROOT = Path("/home/a/andrewsq/data/urop/environments")
ENVIRONMENT_UV = Path("/home/a/andrewsq/.local/bin/uv")
ENVIRONMENT_BASE_PYTHON = Path("/usr/bin/python3")
ENVIRONMENT_PARTITION = "normal"
ENVIRONMENT_CPUS = 2
ENVIRONMENT_MEMORY = "16G"
ENVIRONMENT_TIME = "1:00:00"
TRUSTED_ACCOUNTS = frozenset({"allusers"})
ALLOWED_MLFLOW_URI = "sqlite:////home/a/andrewsq/data/urop/experiments/mlflow.db"
ALLOWED_ARTIFACT_ROOT = Path("/home/a/andrewsq/data/urop/experiments/mlflow-artifacts")
MLFLOW_UI_PORT = 5000
MAX_UI_TIMEOUT_MINUTES = 480
MLFLOW_UI_USERNAME = "dscnet-ui"
ENVIRONMENT_TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)


def _job_id(value: str) -> str:
    match = JOB_ID.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError("job ID must be numeric")
    return match.group(1)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _canonical_digest(value) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_digest(root: Path, splits: tuple[str, ...]) -> str:
    records: dict[str, list[dict[str, str]]] = {}
    for split in splits:
        image_dir = root / split / "image"
        label_dir = root / split / "label"
        images = {path.name: path for path in image_dir.glob("*.nii*") if path.is_file()}
        labels = {path.name: path for path in label_dir.glob("*.nii*") if path.is_file()}
        if not images or images.keys() != labels.keys():
            raise RuntimeError(f"remote {split} image/label pairs are empty or mismatched")
        records[split] = [
            {
                "name": name,
                "image_sha256": _sha256(images[name]),
                "label_sha256": _sha256(labels[name]),
            }
            for name in sorted(images)
        ]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_source(args: argparse.Namespace) -> dict:
    root = Path(args.source_root).resolve()
    manifest = json.loads(Path(args.manifest).read_text())
    entries = manifest.get("entries", [])
    if _canonical_digest(entries) != manifest.get("digest"):
        raise RuntimeError("source manifest digest is invalid")
    expected = {item["path"]: item for item in entries}
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if expected.keys() != actual.keys():
        raise RuntimeError("extracted source paths do not match the snapshot manifest")
    for relative, record in expected.items():
        path = actual[relative]
        mode_matches = stat.S_IMODE(path.lstat().st_mode) == record["mode"]
        if record["kind"] == "symlink":
            valid = (
                mode_matches
                and path.is_symlink()
                and os.readlink(path) == record["target"]
            )
        else:
            valid = (
                mode_matches
                and not path.is_symlink()
                and path.stat().st_size == record["size"]
                and _sha256(path) == record["sha256"]
            )
        if not valid:
            raise RuntimeError(f"extracted source differs from manifest: {relative}")
    return {"status": "verified", "source_digest": manifest["digest"]}


def verify_data(args: argparse.Namespace) -> dict:
    splits = tuple(args.splits.split(","))
    actual = _dataset_digest(Path(args.dataset_root), splits)
    if actual != args.expected_digest:
        raise RuntimeError(
            f"remote dataset digest mismatch: expected {args.expected_digest}, got {actual}"
        )
    return {"status": "verified", "dataset_digest": actual, "splits": splits}


def _environment_source(value: str) -> Path:
    supplied = Path(value).absolute()
    source = supplied.resolve()
    if supplied != source or source.is_symlink() or not source.is_dir():
        raise RuntimeError("environment source path is unsafe")
    allowed_snapshot = (
        source.name == "source"
        and source.parent.parent.parent == ALLOWED_EXPERIMENT_ROOT.resolve()
        and source.parent.parent.name == "runs"
        and bool(re.fullmatch(r"[0-9a-f]{16}(?:-a[1-9][0-9]*)?", source.parent.name))
    )
    if source != ALLOWED_REMOTE_CHECKOUT.resolve() and not allowed_snapshot:
        raise RuntimeError("environment source is outside the allowlist")
    return source


def _environment_identity(source: Path, expected: str) -> tuple[str, Path]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError("uv.lock SHA-256 is invalid")
    lock = source / "uv.lock"
    if lock.is_symlink() or not lock.is_file() or _sha256(lock) != expected:
        raise RuntimeError("source uv.lock does not match the requested environment")
    return expected, ENVIRONMENT_ROOT / expected


def _environment_control(environment_id: str) -> Path:
    if ENVIRONMENT_ROOT.is_symlink() or (
        ENVIRONMENT_ROOT.exists() and not ENVIRONMENT_ROOT.is_dir()
    ):
        raise RuntimeError("environment root path is unsafe")
    ENVIRONMENT_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    control_root = ENVIRONMENT_ROOT / ".control"
    if control_root.is_symlink() or (
        control_root.exists() and not control_root.is_dir()
    ):
        raise RuntimeError("environment control root path is unsafe")
    control_root.mkdir(mode=0o700, exist_ok=True)
    control_root.chmod(0o700)
    control = control_root / environment_id
    if control.is_symlink() or (control.exists() and not control.is_dir()):
        raise RuntimeError("environment control path is unsafe")
    control.mkdir(mode=0o700, exist_ok=True)
    control.chmod(0o700)
    return control


def _private_json(path: Path) -> dict | None:
    try:
        if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) & 0o077:
            return None
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _valid_environment_ready(environment_id: str, environment_path: Path) -> dict | None:
    control_root = ENVIRONMENT_ROOT / ".control"
    control = control_root / environment_id
    if (
        ENVIRONMENT_ROOT.is_symlink()
        or control_root.is_symlink()
        or control.is_symlink()
        or environment_path.is_symlink()
        or not environment_path.is_dir()
    ):
        return None
    python = environment_path / "bin" / "python"
    if not python.is_file() or not os.access(python, os.X_OK):
        return None
    ready_path = control / "ready.json"
    ready = _private_json(ready_path)
    published = _private_json(environment_path / ".dscnet-ready.json")
    required = {
        "schema_version": 1,
        "kind": "uv-lock",
        "uv_lock_sha256": environment_id,
        "environment_id": environment_id,
        "environment_path": str(environment_path),
        "system": "Linux",
        "machine": "x86_64",
        "package_check": "passed",
    }
    if not ready or not published or any(ready.get(k) != v for k, v in required.items()):
        return None
    if published != ready or not re.fullmatch(r"3\.12(?:\.\d+)?", ready.get("python_version", "")):
        return None
    return ready


def environment_validate(args: argparse.Namespace) -> dict:
    environment_id = args.environment_id
    if not re.fullmatch(r"[0-9a-f]{64}", environment_id):
        raise RuntimeError("environment ID is invalid")
    environment_path = ENVIRONMENT_ROOT / environment_id
    ready = _valid_environment_ready(environment_id, environment_path)
    if not ready:
        raise RuntimeError("cluster environment is not ready")
    return {
        "status": "READY",
        "environment_id": environment_id,
        "environment_path": str(environment_path),
        "ready": ready,
    }


def _environment_job_state(job_id: str) -> str:
    queue = subprocess.run(
        ["squeue", "--noheader", "--jobs", job_id, "--format", "%T"],
        check=False, capture_output=True, text=True,
    )
    if queue.returncode != 0:
        raise RuntimeError(queue.stderr.strip() or "environment squeue query failed")
    if queue.stdout.strip():
        return queue.stdout.strip().splitlines()[0]
    accounting = subprocess.run(
        ["sacct", "--noheader", "--allocations", "--jobs", job_id,
         "--format", "State", "--parsable2"],
        check=False, capture_output=True, text=True,
    )
    if accounting.returncode != 0:
        raise RuntimeError(accounting.stderr.strip() or "environment sacct query failed")
    states = [line.split("|", 1)[0] for line in accounting.stdout.splitlines() if line]
    return states[0] if states else "UNKNOWN"


def _query_environment_marker(account: str, marker: str) -> str | None:
    queue = subprocess.run(
        ["squeue", "--noheader", "--user", os.environ["USER"], "--account", account,
         "--name", "dscnet-env", "--format", "%A|%k"],
        check=False, capture_output=True, text=True,
    )
    if queue.returncode != 0:
        raise RuntimeError(queue.stderr.strip() or "environment squeue recovery failed")
    matches = [job for job, comment in _parse_job_rows(queue.stdout) if comment == marker]
    if len(matches) > 1:
        raise RuntimeError("multiple environment setup jobs have the same marker")
    if matches:
        return matches[0]
    accounting = subprocess.run(
        ["sacct", "--noheader", "--allocations", "--user", os.environ["USER"],
         "--account", account, "--name", "dscnet-env", "--starttime", "now-1days",
         "--format", "JobIDRaw,Comment", "--parsable2"],
        check=False, capture_output=True, text=True,
    )
    if accounting.returncode != 0:
        raise RuntimeError(accounting.stderr.strip() or "environment sacct recovery failed")
    matches = [job for job, comment in _parse_job_rows(accounting.stdout) if comment == marker]
    if len(matches) > 1:
        raise RuntimeError("multiple environment setup jobs have the same marker")
    return matches[0] if matches else None


def _environment_setup_script(source: Path, environment_id: str, account: str) -> str:
    helper = source / "tools" / "expctl_remote.py"
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=dscnet-env
#SBATCH --output={ALLOWED_EXPERIMENT_ROOT}/environment-setup-{environment_id}-%j.out
#SBATCH --error={ALLOWED_EXPERIMENT_ROOT}/environment-setup-{environment_id}-%j.err
#SBATCH --account={account}
#SBATCH --partition={ENVIRONMENT_PARTITION}
#SBATCH --cpus-per-task={ENVIRONMENT_CPUS}
#SBATCH --mem={ENVIRONMENT_MEMORY}
#SBATCH --time={ENVIRONMENT_TIME}
set -euo pipefail
umask 077
export TMPDIR={ALLOWED_EXPERIMENT_ROOT}/.tmp/environment-{environment_id}-$SLURM_JOB_ID
export UV_CACHE_DIR={ALLOWED_EXPERIMENT_ROOT}/.uv-cache
mkdir -p "$TMPDIR" "$UV_CACHE_DIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
{ENVIRONMENT_BASE_PYTHON} {helper} environment-build --source-root {source} --uv-lock-sha256 {environment_id}
"""


def environment_build(args: argparse.Namespace) -> dict:
    os.umask(0o077)
    if not str(os.environ.get("SLURM_JOB_ID", "")).isdigit() or os.environ.get(
        "SLURM_JOB_PARTITION"
    ) != ENVIRONMENT_PARTITION:
        raise RuntimeError("environment builds must run in the allowlisted Slurm job")
    source = _environment_source(args.source_root)
    environment_id, final = _environment_identity(source, args.uv_lock_sha256)
    control = _environment_control(environment_id)
    stage = ENVIRONMENT_ROOT / f".{environment_id}.build-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    if stage.exists() or stage.is_symlink():
        if stage.is_dir() and not stage.is_symlink():
            import shutil
            shutil.rmtree(stage)
        else:
            stage.unlink()
    stage.mkdir(parents=True, mode=0o700)
    try:
        _environment_identity(source, environment_id)
        env = {
            **os.environ,
            "UV_PROJECT_ENVIRONMENT": str(stage),
            "TMPDIR": os.environ.get("TMPDIR", str(ALLOWED_EXPERIMENT_ROOT / ".tmp")),
            "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(ALLOWED_EXPERIMENT_ROOT / ".uv-cache")),
        }
        sync = subprocess.run(
            [str(ENVIRONMENT_UV), "sync", "--locked", "--no-install-project",
             "--python", str(ENVIRONMENT_BASE_PYTHON), "--project", str(source)],
            check=False, capture_output=True, text=True, env=env,
        )
        if sync.returncode != 0:
            raise RuntimeError(sync.stderr.strip() or "uv sync failed")
        check = subprocess.run(
            [str(ENVIRONMENT_UV), "pip", "check", "--python", str(stage / "bin/python")],
            check=False, capture_output=True, text=True, env=env,
        )
        if check.returncode != 0:
            raise RuntimeError(check.stderr.strip() or "uv pip check failed")
        probe = subprocess.run(
            [str(stage / "bin/python"), "-c",
             "import json,platform,sys; print(json.dumps({'python_version': platform.python_version(), 'system': platform.system(), 'machine': platform.machine()}))"],
            check=False, capture_output=True, text=True, env=env,
        )
        if probe.returncode != 0:
            raise RuntimeError(probe.stderr.strip() or "environment Python probe failed")
        platform_evidence = json.loads(probe.stdout)
        if not re.fullmatch(r"3\.12(?:\.\d+)?", platform_evidence.get("python_version", "")):
            raise RuntimeError("environment Python must be version 3.12")
        if platform_evidence.get("system") != "Linux" or platform_evidence.get("machine") != "x86_64":
            raise RuntimeError("environment platform must be Linux x86_64")
        ready = {
            "schema_version": 1, "kind": "uv-lock",
            "uv_lock_sha256": environment_id, "environment_id": environment_id,
            "environment_path": str(final), "package_check": "passed",
            **platform_evidence,
        }
        published_ready = stage / ".dscnet-ready.json"
        _atomic_json(published_ready, ready)
        published_ready.chmod(0o600)
        if final.exists() or final.is_symlink():
            raise RuntimeError("refusing to mutate an existing published environment")
        (control / "failure.json").unlink(missing_ok=True)
        _atomic_json(control / "ready.json", ready)
        (control / "ready.json").chmod(0o600)
        os.replace(stage, final)
        return {"status": "READY", **ready}
    except Exception as error:
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
        if not final.exists():
            (control / "ready.json").unlink(missing_ok=True)
        failure = {"status": "FAILED", "environment_id": environment_id, "error": str(error)}
        _atomic_json(control / "failure.json", failure)
        (control / "failure.json").chmod(0o600)
        raise


def environment_ensure(args: argparse.Namespace) -> dict:
    source = _environment_source(args.source_root)
    environment_id, final = _environment_identity(source, args.uv_lock_sha256)
    if args.account not in TRUSTED_ACCOUNTS:
        raise RuntimeError("environment setup account is not allowlisted")
    control = _environment_control(environment_id)
    lock_path = ENVIRONMENT_ROOT / ".control" / ".environment.lock"
    if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
        raise RuntimeError("environment lock path is unsafe")
    with lock_path.open("a+") as lock:
        lock_path.chmod(0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready = _valid_environment_ready(environment_id, final)
        if ready:
            return {"status": "READY", "environment_id": environment_id, "environment_path": str(final), "ready": ready}
        receipt_path = control / "submission.json"
        intent_path = control / "submission-intent.json"
        failure_path = control / "failure.json"
        expected_marker = f"expctl-env:{environment_id}"
        receipt = _private_json(receipt_path) if receipt_path.exists() else None
        if receipt_path.exists() and (
            not receipt
            or not JOB_ID.fullmatch(str(receipt.get("job_id", "")))
            or receipt.get("marker") != expected_marker
            or receipt.get("status") not in {"submitted", "recovered"}
        ):
            raise RuntimeError("environment setup receipt is invalid")
        if receipt:
            job_id = str(receipt["job_id"])
            state = _environment_job_state(job_id).split("+", 1)[0]
            if state == "UNKNOWN":
                if not args.retry:
                    raise RuntimeError(
                        f"environment setup job {job_id} is no longer visible without "
                        "valid ready state; retry explicitly"
                    )
            elif state not in ENVIRONMENT_TERMINAL_STATES:
                return {"status": "BUILDING", "job_id": job_id, "environment_id": environment_id, "environment_path": str(final)}
            elif not args.retry:
                raise RuntimeError(f"environment setup job {job_id} ended in {state} without valid ready state; retry explicitly")
        elif failure_path.exists() and not args.retry:
            raise RuntimeError("environment setup previously failed; retry explicitly")
        if args.retry:
            for path in (receipt_path, intent_path, failure_path):
                path.unlink(missing_ok=True)
        marker = expected_marker
        if intent_path.exists():
            intent = _private_json(intent_path)
            if (
                not intent
                or intent.get("marker") != marker
                or intent.get("status") != "before_sbatch"
                or intent.get("source_root") != str(source)
            ):
                raise RuntimeError("environment setup intent is invalid")
            recovered = _query_environment_marker(args.account, marker)
            if recovered:
                record = {"status": "recovered", "job_id": recovered, "marker": marker}
                _atomic_json(receipt_path, record)
                receipt_path.chmod(0o600)
                state = _environment_job_state(recovered).split("+", 1)[0]
                if state in ENVIRONMENT_TERMINAL_STATES:
                    raise RuntimeError(f"environment setup job {recovered} ended in {state} without valid ready state; retry explicitly")
                return {"status": "BUILDING", "job_id": recovered, "environment_id": environment_id, "environment_path": str(final)}
            raise RuntimeError("environment setup recovery pending: no matching Slurm job is visible")
        script = control / "setup.sbatch"
        script.write_text(_environment_setup_script(source, environment_id, args.account))
        script.chmod(0o600)
        _atomic_json(intent_path, {"status": "before_sbatch", "marker": marker, "source_root": str(source)})
        intent_path.chmod(0o600)
        submitted = subprocess.run(
            ["sbatch", "--parsable", "--comment", marker, str(script)],
            check=False, capture_output=True, text=True,
        )
        match = JOB_ID.fullmatch(submitted.stdout.strip())
        if submitted.returncode != 0 or match is None:
            detail = submitted.stderr.strip() or submitted.stdout.strip() or "invalid empty response"
            failure = {"status": "FAILED", "error": f"environment setup submission rejected: {detail}"}
            _atomic_json(failure_path, failure)
            failure_path.chmod(0o600)
            raise RuntimeError(failure["error"])
        job_id = match.group(1)
        record = {"status": "submitted", "job_id": job_id, "marker": marker}
        _atomic_json(receipt_path, record)
        receipt_path.chmod(0o600)
        return {"status": "BUILDING", "job_id": job_id, "environment_id": environment_id, "environment_path": str(final)}


def _parse_job_rows(output: str) -> list[tuple[str, str]]:
    rows = []
    for line in output.splitlines():
        fields = line.strip().split("|", 1)
        if len(fields) == 2 and fields[0].isdigit():
            rows.append((fields[0], fields[1]))
    return rows


def _query_marked_jobs(account: str, marker: str, include_accounting: bool):
    queue = subprocess.run(
        [
            "squeue",
            "--noheader",
            "--user",
            os.environ["USER"],
            "--account",
            account,
            "--name",
            "dscnet-expctl",
            "--format",
            "%A|%k",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if queue.returncode != 0:
        raise RuntimeError(queue.stderr.strip() or "squeue failed")
    queued = _parse_job_rows(queue.stdout)
    recovered = next((job_id for job_id, comment in queued if comment == marker), None)
    if recovered or not include_accounting:
        return recovered, queued
    accounting = subprocess.run(
        [
            "sacct",
            "--noheader",
            "--allocations",
            "--user",
            os.environ["USER"],
            "--account",
            account,
            "--name",
            "dscnet-expctl",
            "--starttime",
            "now-1days",
            "--format",
            "JobIDRaw,Comment",
            "--parsable2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if accounting.returncode != 0:
        raise RuntimeError(accounting.stderr.strip() or "sacct recovery query failed")
    recovered = next(
        (
            job_id
            for job_id, comment in _parse_job_rows(accounting.stdout)
            if comment == marker
        ),
        None,
    )
    return recovered, queued


def submit(args: argparse.Namespace) -> dict:
    run_dir = Path(args.run_dir).resolve()
    remote_root = Path(args.remote_root).resolve()
    dependency_job_id = getattr(args, "dependency_job_id", None)
    if dependency_job_id is not None and not re.fullmatch(r"[0-9]+", str(dependency_job_id)):
        raise RuntimeError("dependency job ID must be numeric")
    if remote_root not in run_dir.parents:
        raise RuntimeError("run directory is outside the configured remote root")
    receipt = run_dir / "control" / "submission.json"
    intent = run_dir / "control" / "submission-intent.json"
    marker = f"expctl:{run_dir.name}"
    lock_path = remote_root / ".submit.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.is_file():
            recorded = json.loads(receipt.read_text())
            if recorded.get("job_id"):
                if recorded.get("dependency_job_id") != dependency_job_id:
                    raise RuntimeError("submission dependency differs from the recorded receipt")
                return recorded
            raise RuntimeError(recorded.get("error", "submission rejected"))
        recovered, queued = _query_marked_jobs(
            args.account, marker, include_accounting=intent.is_file()
        )
        if recovered:
            record = {
                "status": "recovered", "job_id": recovered, "marker": marker,
                "dependency_job_id": dependency_job_id,
            }
            _atomic_json(receipt, record)
            return record
        if intent.is_file():
            raise RuntimeError(
                "submission recovery pending: no matching Slurm job is visible yet"
            )
        if queued:
            raise RuntimeError(
                "submission blocked: the SQLite backend permits one active run"
            )
        _atomic_json(intent, {
            "marker": marker, "status": "before_sbatch",
            "dependency_job_id": dependency_job_id,
        })
        sbatch = ["sbatch", "--parsable", "--comment", marker]
        if dependency_job_id is not None:
            sbatch.extend(["--dependency", f"afterok:{dependency_job_id}"])
        sbatch.append(str(run_dir / "control" / "job.sbatch"))
        result = subprocess.run(
            sbatch,
            check=False,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        match = JOB_ID.fullmatch(value)
        if result.returncode != 0 or match is None:
            detail = result.stderr.strip() or value or "invalid empty response"
            error = f"submission rejected: {detail}"
            _atomic_json(receipt, {"status": "rejected", "error": error})
            raise RuntimeError(error)
        record = {
            "status": "submitted",
            "job_id": match.group(1),
            "sbatch_response": value,
            "marker": marker,
            "dependency_job_id": dependency_job_id,
        }
        _atomic_json(receipt, record)
        return record


def status(args: argparse.Namespace) -> dict:
    queue = subprocess.run(
        ["squeue", "--noheader", "--jobs", args.job_id, "--format", "%T"],
        check=False,
        capture_output=True,
        text=True,
    )
    if queue.returncode != 0:
        raise RuntimeError(queue.stderr.strip() or "squeue failed")
    queued = queue.stdout.strip().splitlines()
    if queued:
        return {"job_id": args.job_id, "source": "squeue", "state": queued[0]}
    last_error = ""
    for attempt in range(3):
        accounting = subprocess.run(
            [
                "sacct",
                "--noheader",
                "--allocations",
                "--jobs",
                args.job_id,
                "--format",
                "State",
                "--parsable2",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        states = [line.split("|")[0] for line in accounting.stdout.splitlines() if line]
        if accounting.returncode == 0 and states:
            return {"job_id": args.job_id, "source": "sacct", "state": states[0]}
        last_error = accounting.stderr.strip()
        if attempt < 2:
            time.sleep(1)
    raise RuntimeError(last_error or "job is absent from squeue and sacct")


def cancel(args: argparse.Namespace) -> dict:
    current = status(args)
    if current["state"].split("+", 1)[0] in {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }:
        return {
            "job_id": args.job_id,
            "status": "already_terminal",
            "state": current["state"],
        }
    result = subprocess.run(
        ["scancel", args.job_id], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "scancel failed")
    return {"job_id": args.job_id, "status": "cancellation_requested"}


def _ui_paths(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    if root != ALLOWED_EXPERIMENT_ROOT:
        raise RuntimeError("MLflow UI root is not allowlisted")
    control = root / "control"
    return control / "mlflow-ui.json", control / "mlflow-ui.log"


def _ui_auth(control: Path) -> tuple[Path, str, str]:
    control.mkdir(parents=True, exist_ok=True)
    control.chmod(0o700)
    config_path = control / "mlflow-auth.ini"
    auth_database = control / "mlflow-auth.db"
    secret_path = control / "mlflow-auth.secret"
    password = ""
    if config_path.is_file() and not config_path.is_symlink():
        parser = configparser.ConfigParser()
        parser.read(config_path)
        if (
            parser.get("mlflow", "admin_username", fallback="") != MLFLOW_UI_USERNAME
            or parser.get("mlflow", "database_uri", fallback="")
            != f"sqlite:///{auth_database}"
            or parser.get("mlflow", "default_permission", fallback="")
            != "NO_PERMISSIONS"
        ):
            raise RuntimeError("MLflow UI authentication config is invalid")
        password = parser.get("mlflow", "admin_password", fallback="")
        if not re.fullmatch(r"[A-Za-z0-9_-]{40,64}", password):
            raise RuntimeError("MLflow UI authentication credential is invalid")
        config_path.chmod(0o600)
    elif config_path.exists():
        raise RuntimeError("MLflow UI authentication config path is unsafe")
    else:
        password = secrets.token_urlsafe(32)
        content = "\n".join(
            [
                "[mlflow]",
                "default_permission = NO_PERMISSIONS",
                f"database_uri = sqlite:///{auth_database}",
                f"admin_username = {MLFLOW_UI_USERNAME}",
                f"admin_password = {password}",
                "authorization_function = mlflow.server.auth:authenticate_request_basic_auth",
                "grant_default_workspace_access = false",
                "",
            ]
        )
        descriptor = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
    if auth_database.is_symlink() or (
        auth_database.exists() and not auth_database.is_file()
    ):
        raise RuntimeError("MLflow UI authentication database path is unsafe")
    auth_database.touch(mode=0o600, exist_ok=True)
    auth_database.chmod(0o600)
    if secret_path.is_file() and not secret_path.is_symlink():
        secret = secret_path.read_text().strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{60,96}", secret):
            raise RuntimeError("MLflow UI CSRF secret is invalid")
        secret_path.chmod(0o600)
    elif secret_path.exists():
        raise RuntimeError("MLflow UI CSRF secret path is unsafe")
    else:
        secret = secrets.token_urlsafe(48)
        descriptor = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(secret + "\n")
    return config_path, password, secret


def _linux_process_identity(pid: int) -> tuple[str, str] | None:
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        stat_fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return boot_id, stat_fields[19]
    except (FileNotFoundError, IndexError, PermissionError, ProcessLookupError):
        return None


def _valid_ui_record(record) -> bool:
    return (
        isinstance(record, dict)
        and record.get("schema_version") == 1
        and isinstance(record.get("pid"), int)
        and record["pid"] > 1
        and isinstance(record.get("boot_id"), str)
        and isinstance(record.get("process_start_ticks"), str)
        and record.get("bind_host") == "127.0.0.1"
        and record.get("port") == MLFLOW_UI_PORT
        and record.get("tracking_uri") == ALLOWED_MLFLOW_URI
        and record.get("artifact_root") == str(ALLOWED_ARTIFACT_ROOT)
        and record.get("authentication") == "basic"
        and bool(re.fullmatch(r"[0-9a-f]{64}", record.get("environment_id", "")))
        and isinstance(record.get("started_epoch"), (int, float))
        and isinstance(record.get("deadline_epoch"), (int, float))
        and math.isfinite(record["started_epoch"])
        and math.isfinite(record["deadline_epoch"])
        and record["deadline_epoch"] >= record["started_epoch"]
    )


def _ui_process_matches(record: dict) -> bool:
    if not _valid_ui_record(record):
        return False
    pid = record.get("pid")
    if not isinstance(pid, int) or pid <= 1:
        return False
    identity = _linux_process_identity(pid)
    return identity == (record.get("boot_id"), record.get("process_start_ticks"))


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stop_started_ui_process_group(record: dict) -> bool:
    process_group = record["pid"]
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return True
    for _ in range(50):
        if not _process_group_exists(process_group):
            return True
        time.sleep(0.1)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return True
    for _ in range(50):
        if not _process_group_exists(process_group):
            return True
        time.sleep(0.1)
    return False


def _stop_ui_process(record: dict) -> bool:
    if not _valid_ui_record(record):
        return False
    identity = _linux_process_identity(record["pid"])
    if identity is None:
        return True
    if identity != (record["boot_id"], record["process_start_ticks"]):
        return False
    return _stop_started_ui_process_group(record)


def _ui_port_is_open() -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", MLFLOW_UI_PORT)) == 0


def _ui_status_record(state_path: Path) -> dict:
    if not state_path.is_file():
        return {"state": "off"}
    try:
        record = json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        state_path.unlink(missing_ok=True)
        if _ui_port_is_open():
            raise RuntimeError("MLflow UI state is invalid while its port is in use")
        return {"state": "off", "reason": "invalid_state"}
    if not _valid_ui_record(record):
        state_path.unlink(missing_ok=True)
        if _ui_port_is_open():
            raise RuntimeError("MLflow UI state is invalid while its port is in use")
        return {"state": "off", "reason": "invalid_state"}
    if not _ui_process_matches(record):
        state_path.unlink(missing_ok=True)
        return {"state": "off", "reason": "process_exited"}
    if time.time() >= record["deadline_epoch"]:
        if not _stop_ui_process(record):
            raise RuntimeError("timed-out MLflow UI process did not stop")
        state_path.unlink(missing_ok=True)
        return {"state": "off", "reason": "timeout"}
    return {**record, "state": "running"}


def _ui_start(args: argparse.Namespace) -> dict:
    state_path, log_path = _ui_paths(Path(args.experiment_root))
    if args.tracking_uri != ALLOWED_MLFLOW_URI:
        raise RuntimeError("MLflow tracking URI is not allowlisted")
    if Path(args.artifact_root).resolve() != ALLOWED_ARTIFACT_ROOT:
        raise RuntimeError("MLflow artifact root is not allowlisted")
    if not re.fullmatch(r"[0-9a-f]{64}", args.environment_id):
        raise RuntimeError("MLflow environment ID is invalid")
    environment_path = ENVIRONMENT_ROOT / args.environment_id
    if not _valid_environment_ready(args.environment_id, environment_path):
        raise RuntimeError(
            "current uv.lock cluster environment is not ready; submit an experiment "
            "or run environment-ensure first"
        )
    python = environment_path / "bin" / "python"
    if args.port != MLFLOW_UI_PORT:
        raise RuntimeError("MLflow UI port is not allowlisted")
    if not 1 <= args.timeout_minutes <= MAX_UI_TIMEOUT_MINUTES:
        raise RuntimeError("MLflow UI timeout is outside the allowlist")
    auth_config, password, csrf_secret = _ui_auth(state_path.parent)
    current = _ui_status_record(state_path)
    if current["state"] == "running" and current["environment_id"] == args.environment_id:
        return {
            **current,
            "started": False,
            "authentication": "basic",
            "username": MLFLOW_UI_USERNAME,
            "password": password,
        }
    if current["state"] == "running":
        if not _stop_ui_process(current):
            raise RuntimeError("old-environment MLflow UI process did not stop")
        state_path.unlink(missing_ok=True)
    with socket.socket() as probe:
        probe.settimeout(0.2)
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError("MLflow UI port is already in use")
    ALLOWED_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=10s",
        f"{args.timeout_minutes}m",
        str(python),
        "-m",
        "mlflow",
        "ui",
        "--app-name",
        "basic-auth",
        "--backend-store-uri",
        ALLOWED_MLFLOW_URI,
        "--default-artifact-root",
        ALLOWED_ARTIFACT_ROOT.as_uri(),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--workers",
        "1",
    ]
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **os.environ,
                "MLFLOW_AUTH_CONFIG_PATH": str(auth_config),
                "MLFLOW_FLASK_SERVER_SECRET_KEY": csrf_secret,
            },
        )
    identity = None
    for _ in range(50):
        identity = _linux_process_identity(process.pid)
        if identity is not None:
            break
        if process.poll() is not None:
            break
        time.sleep(0.1)
    if identity is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        raise RuntimeError("MLflow UI process did not start")
    now = time.time()
    record = {
        "schema_version": 1,
        "pid": process.pid,
        "boot_id": identity[0],
        "process_start_ticks": identity[1],
        "bind_host": "127.0.0.1",
        "port": args.port,
        "tracking_uri": ALLOWED_MLFLOW_URI,
        "artifact_root": str(ALLOWED_ARTIFACT_ROOT),
        "authentication": "basic",
        "environment_id": args.environment_id,
        "environment_path": str(environment_path),
        "started_epoch": now,
        "deadline_epoch": now + args.timeout_minutes * 60,
    }
    try:
        _atomic_json(state_path, record)
    except Exception as error:
        if not _stop_ui_process(record):
            raise RuntimeError(
                "MLflow UI state write failed and the process did not stop"
            ) from error
        raise
    for _ in range(150):
        if process.poll() is not None:
            if not _stop_started_ui_process_group(record):
                raise RuntimeError("failed MLflow UI left a process group running")
            state_path.unlink(missing_ok=True)
            raise RuntimeError("MLflow UI exited before becoming ready")
        with socket.socket() as probe:
            probe.settimeout(0.1)
            if probe.connect_ex(("127.0.0.1", args.port)) == 0:
                return {
                    **record,
                    "state": "running",
                    "started": True,
                    "username": MLFLOW_UI_USERNAME,
                    "password": password,
                }
        time.sleep(0.1)
    if not _stop_ui_process(record):
        raise RuntimeError("unready MLflow UI process did not stop")
    state_path.unlink(missing_ok=True)
    raise RuntimeError("MLflow UI did not become ready")


def _ui_status(args: argparse.Namespace) -> dict:
    state_path, _ = _ui_paths(Path(args.experiment_root))
    return _ui_status_record(state_path)


def _ui_stop(args: argparse.Namespace) -> dict:
    state_path, _ = _ui_paths(Path(args.experiment_root))
    current = _ui_status_record(state_path)
    stopped = current["state"] == "running" and _stop_ui_process(current)
    if current["state"] == "running" and not stopped:
        raise RuntimeError("MLflow UI process did not stop")
    state_path.unlink(missing_ok=True)
    return {"state": "off", "stopped": stopped}


def _with_ui_lock(args: argparse.Namespace, operation) -> dict:
    state_path, _ = _ui_paths(Path(args.experiment_root))
    lock_path = state_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return operation(args)


def ui_start(args: argparse.Namespace) -> dict:
    return _with_ui_lock(args, _ui_start)


def ui_status(args: argparse.Namespace) -> dict:
    return _with_ui_lock(args, _ui_status)


def ui_stop(args: argparse.Namespace) -> dict:
    return _with_ui_lock(args, _ui_stop)


MAX_LOG_FILES = 16
MAX_LOG_BYTES = 48 * 1024


def _bounded_log_tail(path: Path, line_limit: int, byte_limit: int) -> tuple[list[str], bool]:
    size = path.stat().st_size
    read_limit = max(0, byte_limit - 1)
    start = max(0, size - read_limit)
    with path.open("rb") as stream:
        stream.seek(start)
        data = stream.read(read_limit)
    lines = data.splitlines()
    selected = lines[-line_limit:]
    return [line.decode("utf-8", errors="ignore") for line in selected], (
        start > 0 or len(lines) > line_limit
    )


def logs(args: argparse.Namespace) -> dict:
    run_dir = Path(args.run_dir).resolve()
    candidates = sorted((run_dir / "logs").glob("*.log")) + sorted(
        (run_dir / "logs").glob("slurm-*.out")
    )
    output: dict[str, list[str]] = {}
    remaining = MAX_LOG_BYTES
    truncated = len(candidates) > MAX_LOG_FILES
    for path in candidates[:MAX_LOG_FILES]:
        if path.is_symlink() or not path.resolve().is_relative_to(run_dir):
            continue
        lines, file_truncated = _bounded_log_tail(path, args.lines, remaining)
        output[path.name] = lines
        remaining -= sum(len(line.encode("utf-8")) + 1 for line in lines)
        truncated = truncated or file_truncated
        if remaining <= 0:
            truncated = True
            break
    return {"run_id": run_dir.name, "logs": output, "truncated": truncated}


def verify_artifacts(args: argparse.Namespace) -> dict:
    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "control" / "artifacts.json"
    if manifest_path.stat().st_size > MAX_ARTIFACT_MANIFEST_BYTES:
        raise RuntimeError("artifact manifest exceeds the size limit")
    manifest = json.loads(manifest_path.read_text())
    total = 0
    seen = set()
    for artifact in manifest.get("artifacts", []):
        relative = artifact.get("path", "")
        path = run_dir / relative
        if (
            relative in seen
            or path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(run_dir)
        ):
            raise RuntimeError(f"artifact path is unsafe or missing: {relative}")
        size = path.stat().st_size
        if size != artifact.get("size") or _sha256(path) != artifact.get("sha256"):
            raise RuntimeError(f"artifact changed after finalization: {relative}")
        if size > MAX_AUTO_FILE_BYTES:
            raise RuntimeError(f"artifact exceeds the per-file size limit: {relative}")
        total += size
        if total > MAX_AUTO_TOTAL_BYTES:
            raise RuntimeError("artifacts exceed the aggregate size limit")
        seen.add(relative)
    return {"status": "verified", "artifact_count": len(seen), "total_size": total}


def finalize(args: argparse.Namespace) -> dict:
    run_dir = Path(args.run_dir).resolve()
    candidates = [
        run_dir / "control" / name
        for name in (
            "resolved-config.yaml",
            "dataset-manifest.json",
            "source-manifest.json",
            "source.tar.gz",
            "git.json",
            "dirty.patch",
            "environment-lock.json",
            "run-manifest.json",
            "submission.json",
            "Image_Tr.txt",
            "Label_Tr.txt",
            "Image_Va.txt",
            "Label_Va.txt",
            "Image_Te.txt",
            "Label_Te.txt",
            "Mean_Std.npy",
        )
    ] + [
        run_dir / "outputs" / "final-metrics.json",
        run_dir / "outputs" / "run-result.json",
    ]
    candidates.extend(
        path
        for path in sorted((run_dir / "logs").glob("*"))
        if not path.name.startswith("slurm-")
    )
    for checkpoint_name in (args.best_checkpoint, args.latest_checkpoint):
        if checkpoint_name:
            checkpoint = run_dir / "outputs" / "weights" / checkpoint_name
            if checkpoint not in candidates:
                candidates.append(checkpoint)
    artifacts = []
    for path in candidates:
        if (
            path.is_file()
            and not path.is_symlink()
            and path.resolve().is_relative_to(run_dir)
        ):
            artifacts.append(
                {
                    "path": str(path.relative_to(run_dir)),
                    "sha256": _sha256(path),
                    "size": path.stat().st_size,
                }
            )
    record = {"schema_version": 1, "artifacts": artifacts}
    _atomic_json(run_dir / "control" / "artifacts.json", record)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    source = subparsers.add_parser("verify-source")
    source.add_argument("--source-root", required=True)
    source.add_argument("--manifest", required=True)
    verify = subparsers.add_parser("verify-data")
    verify.add_argument("--dataset-root", required=True)
    verify.add_argument("--splits", required=True)
    verify.add_argument("--expected-digest", required=True)
    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("--run-dir", required=True)
    submit_parser.add_argument("--remote-root", required=True)
    submit_parser.add_argument("--account", required=True)
    submit_parser.add_argument("--dependency-job-id", type=_job_id)
    ensure = subparsers.add_parser("environment-ensure")
    ensure.add_argument("--source-root", required=True)
    ensure.add_argument("--uv-lock-sha256", required=True)
    ensure.add_argument("--account", required=True)
    ensure.add_argument("--retry", action="store_true")
    build = subparsers.add_parser("environment-build")
    build.add_argument("--source-root", required=True)
    build.add_argument("--uv-lock-sha256", required=True)
    environment_check = subparsers.add_parser("environment-validate")
    environment_check.add_argument("--environment-id", required=True)
    for name in ("status", "cancel"):
        child = subparsers.add_parser(name)
        child.add_argument("--job-id", required=True, type=_job_id)
    log_parser = subparsers.add_parser("logs")
    log_parser.add_argument("--run-dir", required=True)
    log_parser.add_argument("--lines", type=int, default=200, choices=range(1, 1001))
    ui_start_parser = subparsers.add_parser("ui-start")
    ui_start_parser.add_argument("--experiment-root", required=True)
    ui_start_parser.add_argument("--tracking-uri", required=True)
    ui_start_parser.add_argument("--artifact-root", required=True)
    ui_start_parser.add_argument("--environment-id", required=True)
    ui_start_parser.add_argument("--port", required=True, type=int)
    ui_start_parser.add_argument("--timeout-minutes", required=True, type=int)
    for name in ("ui-status", "ui-stop"):
        child = subparsers.add_parser(name)
        child.add_argument("--experiment-root", required=True)
    artifact_check = subparsers.add_parser("verify-artifacts")
    artifact_check.add_argument("--run-dir", required=True)
    final = subparsers.add_parser("finalize")
    final.add_argument("--run-dir", required=True)
    final.add_argument("--best-checkpoint", required=True)
    final.add_argument("--latest-checkpoint", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    handlers = {
        "verify-source": verify_source,
        "verify-data": verify_data,
        "environment-ensure": environment_ensure,
        "environment-build": environment_build,
        "environment-validate": environment_validate,
        "submit": submit,
        "status": status,
        "cancel": cancel,
        "logs": logs,
        "ui-start": ui_start,
        "ui-status": ui_status,
        "ui-stop": ui_stop,
        "verify-artifacts": verify_artifacts,
        "finalize": finalize,
    }
    try:
        result = handlers[args.command](args)
    except Exception as error:
        print(json.dumps({"error": str(error), "command": args.command}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
