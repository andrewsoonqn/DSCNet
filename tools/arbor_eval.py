#!/usr/bin/env python3
"""Narrow validation-only bridge from Arbor to expctl training runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPCTL = REPO_ROOT / "tools" / "expctl.py"
EXPERIMENT = REPO_ROOT / "configs" / "experiment" / "dscnet_standard.yaml"
NORMALIZATION_NAME = "DSCNet_3D_Meanstd.npy"
TERMINAL_FAILURES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}


class ArborEvaluationError(RuntimeError):
    pass


def _run_expctl(arguments: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(EXPCTL), *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        try:
            detail = json.loads(completed.stderr).get("error", completed.stderr.strip())
        except json.JSONDecodeError:
            detail = completed.stderr.strip()
        raise ArborEvaluationError(detail or "expctl command failed")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ArborEvaluationError("expctl returned invalid JSON") from error
    if not isinstance(result, dict):
        raise ArborEvaluationError("expctl returned a non-object result")
    return result


def evaluate(
    data_dir: str | Path,
    *,
    poll_seconds: float = 30.0,
    timeout_seconds: float = 7 * 24 * 60 * 60,
) -> dict[str, Any]:
    """Train one clean candidate and return only its frozen validation score."""
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise ArborEvaluationError("the configured local dataset directory is missing")
    normalization = data_dir / NORMALIZATION_NAME
    if not normalization.is_file():
        raise ArborEvaluationError(
            f"the configured local dataset is missing {NORMALIZATION_NAME}"
        )
    submitted = _run_expctl(
        [
            "submit",
            str(EXPERIMENT),
            "--set",
            "action=train",
            "--set",
            f"data.data_dir={data_dir}",
            "--set",
            f"data.Meanstd_path={normalization}",
            "--set",
            "runtime.formal=true",
            "--set",
            "runtime.allow_dirty=false",
            "--set",
            "runtime.mlflow.isolated_run_store=true",
        ]
    )
    run_id = submitted.get("run_id")
    if not isinstance(run_id, str):
        raise ArborEvaluationError("expctl submission returned no run ID")
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            current = _run_expctl(["status", run_id])
        except ArborEvaluationError as error:
            if "ssh timed out after" not in str(error):
                raise
            if time.monotonic() >= deadline:
                raise ArborEvaluationError(
                    f"training run {run_id} exceeded the evaluator wait budget"
                ) from error
            time.sleep(poll_seconds)
            continue
        state = str(current.get("state", "")).split("+", 1)[0].split(maxsplit=1)[0]
        if state == "COMPLETED":
            break
        if state in TERMINAL_FAILURES:
            raise ArborEvaluationError(f"training run {run_id} ended in {state}")
        if time.monotonic() >= deadline:
            raise ArborEvaluationError(
                f"training run {run_id} exceeded the evaluator wait budget"
            )
        time.sleep(poll_seconds)
    _run_expctl(["fetch", run_id])
    result = _run_expctl(["result", run_id])
    selection = result.get("selection")
    if not isinstance(selection, dict) or selection.get("metric") != "validation.dice":
        raise ArborEvaluationError("expctl returned no frozen validation selection")
    value = selection.get("value")
    if not isinstance(value, (int, float)):
        raise ArborEvaluationError("expctl returned a non-numeric validation score")
    return {
        "schema_version": 1,
        "status": "completed",
        "controller_run_id": run_id,
        "metric": "validation.dice",
        "score": float(value),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DSCNET_ARBOR_DATA_DIR"),
        help="local MiniVess mirror; may be supplied as DSCNET_ARBOR_DATA_DIR",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=7 * 24 * 60 * 60)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(argv)
    try:
        if not options.data_dir:
            raise ArborEvaluationError(
                "set --data-dir or DSCNET_ARBOR_DATA_DIR for the local MiniVess mirror"
            )
        result = evaluate(
            options.data_dir,
            poll_seconds=options.poll_seconds,
            timeout_seconds=options.timeout_seconds,
        )
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
