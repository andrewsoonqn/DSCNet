#!/usr/bin/env python3
"""Restricted asynchronous candidate submission broker for Arbor.

The MCP surface accepts only an Arbor session name and node ID. Candidate
worktrees, commands, data, credentials, and result filtering remain broker
owned.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - DSCNet controllers run on POSIX.
    fcntl = None


SCHEMA_VERSION = 1
MAX_CANDIDATES = 3
BRANCH_PREFIX = "arbor/dscnet-pilot/"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
ALLOWED_PATHS = (
    "dscnet/models/**",
    "dscnet/training/losses.py",
    "configs/model/**",
    "configs/training/default.yaml",
    "tests/models/**",
    "tests/training/test_losses.py",
)
TERMINAL_STATES = {"completed", "failed"}
ACTIVE_STATES = {"reserved", "running"}
MAX_OUTPUT_BYTES = 1024 * 1024
MAX_CHANGED_FILES = 128
MAX_CANDIDATE_BLOB_BYTES = 2 * 1024 * 1024
MAX_CANDIDATE_TOTAL_BYTES = 8 * 1024 * 1024
PREFLIGHT_TIMEOUT_SECONDS = 15 * 60
EVALUATION_TIMEOUT_SECONDS = 8 * 24 * 60 * 60


class BrokerError(RuntimeError):
    """Raised when a candidate violates the broker contract."""


@dataclass(frozen=True)
class BrokerConfig:
    repo: Path
    data_dir: Path
    project_python: Path

    @classmethod
    def resolved(
        cls, repo: str | Path, data_dir: str | Path, project_python: str | Path
    ) -> "BrokerConfig":
        config = cls(
            repo=Path(repo).expanduser().resolve(),
            data_dir=Path(data_dir).expanduser().resolve(),
            # Preserve the virtual-environment entrypoint. Resolving its symlink
            # would make Python lose the DSCNet environment's site-packages.
            project_python=Path(project_python).expanduser().absolute(),
        )
        if not (config.repo / ".git").exists():
            raise BrokerError("configured DSCNet repository is not a Git checkout")
        if not config.data_dir.is_dir():
            raise BrokerError("configured MiniVess mirror is unavailable")
        if not config.project_python.is_file():
            raise BrokerError("configured DSCNet Python is unavailable")
        branch = _git(config.repo, "branch", "--show-current")
        head = _git(config.repo, "rev-parse", "HEAD")
        main = _git(config.repo, "rev-parse", "main")
        if branch != "main" or head != main:
            raise BrokerError("trusted DSCNet checkout must be on main at main HEAD")
        return config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _git(cwd: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode != 0:
        raise BrokerError(completed.stderr.strip() or "Git command failed")
    return completed.stdout.strip()


def _validate_token(value: str, label: str) -> str:
    if not isinstance(value, str) or TOKEN_PATTERN.fullmatch(value) is None:
        raise BrokerError(f"invalid {label}")
    return value


def _state_root(repo: Path, run_name: str) -> Path:
    return repo / ".arbor" / "sessions" / run_name


def _load_tree(repo: Path, run_name: str) -> dict[str, Any]:
    path = _state_root(repo, run_name) / ".coordinator" / "idea_tree.json"
    try:
        tree = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise BrokerError("Arbor session does not exist") from error
    except json.JSONDecodeError as error:
        raise BrokerError("Arbor session tree is invalid") from error
    if not isinstance(tree, dict) or not isinstance(tree.get("nodes"), dict):
        raise BrokerError("Arbor session tree has an invalid schema")
    return tree


def _node_branch(tree: dict[str, Any], node_id: str) -> str:
    node = tree["nodes"].get(node_id)
    if not isinstance(node, dict):
        raise BrokerError("Arbor node does not exist")
    if node.get("status") not in {"pending", "running"}:
        raise BrokerError("Arbor node is not eligible for submission")
    branch = node.get("code_ref")
    if not isinstance(branch, str) or not branch.startswith(BRANCH_PREFIX):
        raise BrokerError("Arbor node has no approved candidate branch")
    return branch


def _registered_worktree(repo: Path, branch: str) -> Path:
    records = _git(repo, "worktree", "list", "--porcelain").split("\n\n")
    matches: list[Path] = []
    for record in records:
        fields: dict[str, str] = {}
        for line in record.splitlines():
            key, _, value = line.partition(" ")
            if value:
                fields[key] = value
        if fields.get("branch") == f"refs/heads/{branch}" and fields.get("worktree"):
            matches.append(Path(fields["worktree"]).resolve())
    if len(matches) != 1:
        raise BrokerError("candidate branch must have one registered Git worktree")
    worktree = matches[0]
    temporary_root = Path(tempfile.gettempdir()).resolve()
    try:
        worktree.relative_to(temporary_root)
    except ValueError as error:
        raise BrokerError("candidate worktree is outside the approved temporary root") from error
    if worktree == repo or not worktree.is_dir():
        raise BrokerError("candidate worktree is invalid")
    return worktree


def _changed_paths(repo: Path, base_commit: str, candidate_commit: str) -> list[str]:
    output = _git(
        repo,
        "diff",
        "--name-only",
        "--diff-filter=ACDMRTUXB",
        f"{base_commit}..{candidate_commit}",
    )
    return [line for line in output.splitlines() if line]


def _is_allowed(path: str) -> bool:
    if fnmatch.fnmatchcase(path, "configs/model/**"):
        return Path(path).suffix in {".yaml", ".yml"}
    return any(
        fnmatch.fnmatchcase(path, pattern)
        for pattern in ALLOWED_PATHS
        if pattern != "configs/model/**"
    )


def _process_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _process_identity(pid: int) -> str | None:
    proc_stat = Path(f"/proc/{pid}/stat")
    boot_id = Path("/proc/sys/kernel/random/boot_id")
    if proc_stat.is_file() and boot_id.is_file():
        try:
            closing = proc_stat.read_text().rfind(")")
            return f"{boot_id.read_text().strip()}:{proc_stat.read_text()[closing + 2:].split()[19]}"
        except (OSError, IndexError):
            return None
    try:
        completed = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)], check=False,
            capture_output=True, text=True,
        )
    except OSError:
        return None
    value = completed.stdout.strip()
    return value or None


def _worker_alive(state: dict[str, Any]) -> bool:
    pid = state.get("worker_pid")
    return _process_alive(pid) and _process_identity(pid) == state.get("worker_start")


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    public: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "node_id": state["node_id"],
        "status": state["status"],
    }
    if state["status"] == "completed":
        public.update(
            {
                "controller_run_id": state["controller_run_id"],
                "metric": "validation.dice",
                "score": state["score"],
            }
        )
    elif state["status"] in {"failed", "rejected"}:
        public.update(
            {
                "failure_stage": state.get("failure_stage", "unknown"),
                "error": state.get("error", "candidate submission failed"),
            }
        )
    return public


def _trusted_worker_env() -> dict[str, str]:
    keep = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "TMPDIR"}
    }
    keep["PYTHONNOUSERSITE"] = "1"
    keep["PYTHONDONTWRITEBYTECODE"] = "1"
    return keep


def _candidate_env(home: Path) -> dict[str, str]:
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    return {
        "HOME": str(home),
        "TMPDIR": str(home),
        "XDG_CACHE_HOME": str(home),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _set_tree_writable(root: Path, writable: bool) -> None:
    for directory, names, files in os.walk(root, topdown=False, followlinks=False):
        for name in [*names, *files]:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            mode = path.stat().st_mode
            os.chmod(path, mode | 0o200 if writable else mode & ~0o222)
    mode = root.stat().st_mode
    os.chmod(root, mode | 0o200 if writable else mode & ~0o222)


def _bounded_run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            try:
                os.killpg(process.pid, 15)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, 9)
                except ProcessLookupError:
                    pass
                process.wait()
            raise BrokerError("candidate command exceeded its timeout") from error
        output.seek(0, os.SEEK_END)
        size = output.tell()
        output.seek(max(0, size - MAX_OUTPUT_BYTES))
        payload = output.read(MAX_OUTPUT_BYTES)
    log_path.write_bytes(payload)
    text_output = payload.decode("utf-8", errors="replace")
    return subprocess.CompletedProcess(command, returncode, text_output, "")


def _preflight_command(project_python: Path, worktree: Path, home: Path) -> list[str]:
    # This trusted script only parses candidate Python. Candidate imports and
    # tests execute later, inside the cluster Landlock boundary.
    return [str(project_python), str(worktree / "tools" / "arbor_preflight.py")]


def _parse_evaluation_result(output: str) -> tuple[str, float]:
    result = json.loads(output)
    expected = {"schema_version", "status", "controller_run_id", "metric", "score"}
    if not isinstance(result, dict) or set(result) != expected:
        raise BrokerError("candidate evaluator returned an invalid schema")
    if result.get("schema_version") != 1 or result.get("status") != "completed":
        raise BrokerError("candidate evaluator did not complete")
    if result.get("metric") != "validation.dice":
        raise BrokerError("candidate evaluator returned an unauthorized metric")
    score = result.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise BrokerError("candidate evaluator returned a non-numeric score")
    if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
        raise BrokerError("candidate evaluator returned a score outside [0, 1]")
    run_id = result.get("controller_run_id")
    if not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{16}", run_id) is None:
        raise BrokerError("candidate evaluator returned an invalid run ID")
    return run_id, float(score)


class CandidateBroker:
    def __init__(self, config: BrokerConfig, *, launcher: Any = None):
        self.config = config
        self._launcher = launcher or subprocess.Popen

    def submit_candidate(self, run_name: str, node_id: str) -> dict[str, Any]:
        """Start once, then return current or completed state on repeated calls."""
        run_name = _validate_token(run_name, "run name")
        node_id = _validate_token(node_id, "node ID")
        root = _state_root(self.config.repo, run_name)
        broker_dir = root / ".broker"
        state_path = broker_dir / "submissions" / f"{node_id}.json"
        lock_path = broker_dir / ".lock"

        with _file_lock(lock_path):
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                abandoned = state.get("status") == "reserved" or (
                    state.get("status") == "running" and not _worker_alive(state)
                )
                if abandoned:
                    state.update(
                        {
                            "status": "failed",
                            "failure_stage": "worker",
                            "error": "candidate worker ended without a recorded result",
                            "finished_at": _utc_now(),
                        }
                    )
                    _atomic_json(state_path, state)
                return _public_state(state)

            self._validate_trusted_repo()
            base_commit = self._bound_base_commit(broker_dir)
            tree = _load_tree(self.config.repo, run_name)
            branch = _node_branch(tree, node_id)
            existing = list((broker_dir / "submissions").glob("*.json"))
            if len(existing) >= MAX_CANDIDATES:
                raise BrokerError("candidate budget is exhausted")
            active = []
            for path in existing:
                recorded = json.loads(path.read_text())
                status = recorded.get("status")
                abandoned = status == "reserved" or (
                    status == "running" and not _worker_alive(recorded)
                )
                if abandoned:
                    recorded.update(
                        status="failed", failure_stage="worker",
                        error="candidate worker ended without a recorded result",
                        finished_at=_utc_now(),
                    )
                    _atomic_json(path, recorded)
                elif status in ACTIVE_STATES:
                    active.append(path)
            state = {
                "schema_version": SCHEMA_VERSION,
                "run_name": run_name,
                "node_id": node_id,
                "status": "reserved",
                "branch": branch,
                "base_commit": base_commit,
                "reserved_at": _utc_now(),
            }
            _atomic_json(state_path, state)
            try:
                if active:
                    raise BrokerError("another candidate submission is active")
                worktree = _registered_worktree(self.config.repo, branch)
                candidate_commit = self._validate_candidate(worktree, branch, base_commit)
                evaluation_worktree = broker_dir / "worktrees" / node_id
                evaluation_worktree.parent.mkdir(parents=True, exist_ok=True)
                _git(
                    self.config.repo,
                    "worktree",
                    "add",
                    "--detach",
                    str(evaluation_worktree),
                    candidate_commit,
                )
                _set_tree_writable(evaluation_worktree, False)
                state.update(
                    candidate_commit=candidate_commit,
                    evaluation_worktree=str(evaluation_worktree),
                )
                _atomic_json(state_path, state)
            except Exception as error:
                state.update(
                    status="rejected",
                    failure_stage="validation",
                    error=_sanitize_error(str(error), [self.config.repo, self.config.data_dir]),
                    finished_at=_utc_now(),
                )
                _atomic_json(state_path, state)
                raise
            log_path = broker_dir / "submissions" / f"{node_id}.worker.log"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_worker",
                "--state",
                str(state_path),
                "--lock",
                str(lock_path),
                "--worktree",
                str(evaluation_worktree),
                "--candidate-commit",
                candidate_commit,
                "--data-dir",
                str(self.config.data_dir),
                "--project-python",
                str(self.config.project_python),
                "--repo",
                str(self.config.repo),
            ]
            try:
                with log_path.open("ab") as log:
                    worker = self._launcher(
                        command,
                        cwd=self.config.repo,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env=_trusted_worker_env(),
                    )
            except Exception as error:
                state.update(
                    {
                        "status": "failed",
                        "failure_stage": "launch",
                        "error": "candidate worker could not start",
                        "finished_at": _utc_now(),
                    }
                )
                _atomic_json(state_path, state)
                raise BrokerError("candidate worker could not start") from error

            state.update(
                {
                    "status": "running",
                    "worker_pid": worker.pid,
                    "worker_start": _process_identity(worker.pid),
                    "started_at": _utc_now(),
                }
            )
            if state["worker_start"] is None:
                try:
                    os.killpg(worker.pid, 9)
                except ProcessLookupError:
                    pass
                state.update(status="failed", failure_stage="launch", error="worker identity unavailable")
                _atomic_json(state_path, state)
                raise BrokerError("candidate worker identity could not be established")
            _atomic_json(state_path, state)
            return _public_state(state)

    def _validate_trusted_repo(self) -> None:
        if _git(self.config.repo, "branch", "--show-current") != "main" or _git(
            self.config.repo, "rev-parse", "HEAD"
        ) != _git(self.config.repo, "rev-parse", "main"):
            raise BrokerError("trusted DSCNet checkout must remain on main at main HEAD")
        if _git(self.config.repo, "status", "--porcelain", "--untracked-files=all"):
            raise BrokerError("trusted DSCNet checkout must be clean")

    def _bound_base_commit(self, broker_dir: Path) -> str:
        path = broker_dir / "run.json"
        current = _git(self.config.repo, "rev-parse", "HEAD")
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            base = record.get("base_commit")
            if not isinstance(base, str) or base != current:
                raise BrokerError("trusted DSCNet base changed during this Arbor run")
            return base
        _atomic_json(
            path,
            {
                "schema_version": SCHEMA_VERSION,
                "base_commit": current,
                "max_candidates": MAX_CANDIDATES,
                "created_at": _utc_now(),
            },
        )
        return current

    def _validate_candidate(self, worktree: Path, branch: str, base_commit: str) -> str:
        if _git(worktree, "branch", "--show-current") != branch:
            raise BrokerError("candidate worktree branch does not match the Arbor node")
        if _git(worktree, "status", "--porcelain", "--untracked-files=all"):
            raise BrokerError("candidate worktree must be clean and committed")
        candidate = _git(worktree, "rev-parse", "HEAD")
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", base_commit, candidate],
            cwd=self.config.repo,
            check=False,
            capture_output=True,
            text=True,
        )
        if ancestor.returncode != 0:
            raise BrokerError("candidate does not descend from the trusted base")
        changed = _changed_paths(self.config.repo, base_commit, candidate)
        if len(changed) > MAX_CHANGED_FILES:
            raise BrokerError("candidate changes too many files")
        rejected = [path for path in changed if not _is_allowed(path)]
        if rejected:
            raise BrokerError(
                "candidate changes paths outside the allowed edit surface: "
                + ", ".join(sorted(rejected))
            )
        if not changed:
            return candidate
        entries = _git(
            self.config.repo,
            "ls-tree",
            "-r",
            candidate,
            "--",
            *changed,
        )
        total_bytes = 0
        for line in entries.splitlines():
            metadata, _, path = line.partition("\t")
            mode, kind, object_id = metadata.split()
            if kind != "blob" or mode not in {"100644", "100755"}:
                raise BrokerError(f"candidate path has an unsafe Git mode: {path}")
            size_text = _git(self.config.repo, "cat-file", "-s", object_id)
            try:
                size = int(size_text)
            except ValueError as error:
                raise BrokerError("candidate blob size is invalid") from error
            if size > MAX_CANDIDATE_BLOB_BYTES:
                raise BrokerError(f"candidate file exceeds the size limit: {path}")
            total_bytes += size
            if total_bytes > MAX_CANDIDATE_TOTAL_BYTES:
                raise BrokerError("candidate source exceeds the aggregate size limit")
            base_entry = _git(self.config.repo, "ls-tree", base_commit, "--", path)
            expected_mode = base_entry.split(None, 1)[0] if base_entry else "100644"
            if mode != expected_mode:
                raise BrokerError(f"candidate path has an unexpected Git mode: {path}")
        return candidate


def _sanitize_error(text: str, paths: Sequence[Path]) -> str:
    clean = " ".join(str(text).split())
    for path in sorted(paths, key=lambda item: len(str(item)), reverse=True):
        clean = clean.replace(str(path), "<protected-path>")
    return clean[:500] or "candidate submission failed"


def _run_worker(options: argparse.Namespace) -> int:
    state_path = Path(options.state).resolve()
    lock_path = Path(options.lock).resolve()
    worktree = Path(options.worktree).resolve()
    data_dir = Path(options.data_dir).resolve()
    # Preserve the venv entrypoint so Python loads DSCNet's site-packages.
    project_python = Path(options.project_python).absolute()
    trusted_repo = Path(options.repo).resolve()
    candidate_commit = options.candidate_commit
    logs = state_path.parent

    try:
        if _git(worktree, "rev-parse", "HEAD") != candidate_commit or _git(
            worktree, "status", "--porcelain", "--untracked-files=all"
        ):
            raise BrokerError("detached candidate changed before evaluation")
        preflight_home = logs / f"{state_path.stem}.preflight-home"
        preflight = _bounded_run(
            _preflight_command(project_python, worktree, preflight_home),
            cwd=worktree,
            env=_candidate_env(preflight_home),
            log_path=logs / f"{state_path.stem}.preflight.log",
            timeout=PREFLIGHT_TIMEOUT_SECONDS,
        )
        if preflight.returncode != 0:
            raise BrokerError("candidate syntax preflight failed")

        evaluation = _bounded_run(
            [
                str(project_python),
                str(worktree / "tools" / "arbor_eval.py"),
                "--data-dir",
                str(data_dir),
            ],
            cwd=worktree,
            env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
            log_path=logs / f"{state_path.stem}.evaluation.log",
            timeout=EVALUATION_TIMEOUT_SECONDS,
        )
        if evaluation.returncode != 0:
            try:
                detail = json.loads(evaluation.stdout).get("error", "")
            except (json.JSONDecodeError, AttributeError):
                detail = ""
            raise BrokerError(detail or "candidate formal training failed")
        run_id, score = _parse_evaluation_result(evaluation.stdout)

        with _file_lock(lock_path):
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state.update(
                {
                    "status": "completed",
                    "controller_run_id": run_id,
                    "score": score,
                    "finished_at": _utc_now(),
                }
            )
            _atomic_json(state_path, state)
        return 0
    except Exception as error:
        with _file_lock(lock_path):
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state.update(
                {
                    "status": "failed",
                    "failure_stage": (
                        "preflight"
                        if isinstance(error, BrokerError)
                        and str(error) == "candidate syntax preflight failed"
                        else "evaluation"
                    ),
                    "error": _sanitize_error(
                        str(error), [worktree, data_dir, trusted_repo]
                    ),
                    "finished_at": _utc_now(),
                }
            )
            _atomic_json(state_path, state)
        return 1
    finally:
        try:
            _set_tree_writable(worktree, True)
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=trusted_repo,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            pass


def _serve(options: argparse.Namespace) -> int:
    from mcp.server.fastmcp import FastMCP

    broker = CandidateBroker(
        BrokerConfig.resolved(options.repo, options.data_dir, options.project_python)
    )
    server = FastMCP("DSCNet Arbor candidate broker")

    @server.tool()
    def submit_candidate(run_name: str, node_id: str) -> dict[str, Any]:
        """Submit once or read status for one registered Arbor candidate node."""
        try:
            return broker.submit_candidate(run_name, node_id)
        except BrokerError as error:
            return {
                "schema_version": SCHEMA_VERSION,
                "node_id": node_id,
                "status": "rejected",
                "error": str(error),
            }

    server.run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--repo", required=True)
    serve.add_argument("--data-dir", required=True)
    serve.add_argument("--project-python", required=True)
    serve.set_defaults(handler=_serve)

    worker = subparsers.add_parser("_worker")
    worker.add_argument("--state", required=True)
    worker.add_argument("--lock", required=True)
    worker.add_argument("--worktree", required=True)
    worker.add_argument("--data-dir", required=True)
    worker.add_argument("--project-python", required=True)
    worker.add_argument("--repo", required=True)
    worker.add_argument("--candidate-commit", required=True)
    worker.set_defaults(handler=_run_worker)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(argv)
    return int(options.handler(options))


if __name__ == "__main__":
    raise SystemExit(main())
