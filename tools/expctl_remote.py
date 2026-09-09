#!/usr/bin/env python3
"""Fixed remote operations used by the local expctl controller."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

JOB_ID = re.compile(r"^(\d+)(?:;[A-Za-z0-9_.-]+)?$")
MAX_ARTIFACT_MANIFEST_BYTES = 1024 * 1024
MAX_AUTO_FILE_BYTES = 8 * 1024**3
MAX_AUTO_TOTAL_BYTES = 12 * 1024**3


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
                return recorded
            raise RuntimeError(recorded.get("error", "submission rejected"))
        recovered, queued = _query_marked_jobs(
            args.account, marker, include_accounting=intent.is_file()
        )
        if recovered:
            record = {"status": "recovered", "job_id": recovered, "marker": marker}
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
        _atomic_json(intent, {"marker": marker, "status": "before_sbatch"})
        result = subprocess.run(
            [
                "sbatch",
                "--parsable",
                "--comment",
                marker,
                str(run_dir / "control" / "job.sbatch"),
            ],
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
            "run-manifest.json",
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
    candidates.extend(sorted((run_dir / "logs").glob("*")))
    best_name = args.best_checkpoint
    if best_name:
        candidates.append(run_dir / "outputs" / "weights" / best_name)
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
    for name in ("status", "cancel"):
        child = subparsers.add_parser(name)
        child.add_argument("--job-id", required=True, type=_job_id)
    log_parser = subparsers.add_parser("logs")
    log_parser.add_argument("--run-dir", required=True)
    log_parser.add_argument("--lines", type=int, default=200, choices=range(1, 1001))
    artifact_check = subparsers.add_parser("verify-artifacts")
    artifact_check.add_argument("--run-dir", required=True)
    final = subparsers.add_parser("finalize")
    final.add_argument("--run-dir", required=True)
    final.add_argument("--best-checkpoint", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    handlers = {
        "verify-source": verify_source,
        "verify-data": verify_data,
        "submit": submit,
        "status": status,
        "cancel": cancel,
        "logs": logs,
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
