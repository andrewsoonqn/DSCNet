import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, call, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import expctl_remote

class ExpctlRemoteTests(unittest.TestCase):
    def _completed(self, code=0, stdout="", stderr=""):
        return subprocess.CompletedProcess([], code, stdout, stderr)

    def test_submit_uses_sbatch_parsable_and_persists_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "abc"
            (run / "control").mkdir(parents=True)
            (run / "control" / "job.sbatch").write_text("#!/bin/sh\n")
            with patch(
                "expctl_remote.subprocess.run",
                side_effect=[self._completed(), self._completed(stdout="12345;cluster\n")],
            ) as command:
                arguments = argparse.Namespace(
                    run_dir=str(run), remote_root=str(root), account="test-account"
                )
                first = expctl_remote.submit(arguments)
                second = expctl_remote.submit(arguments)

        self.assertEqual(first["job_id"], "12345")
        self.assertEqual(second, first)
        self.assertEqual(command.call_count, 2)
        queue_command = command.call_args_list[0].args[0]
        self.assertIn("--user", queue_command)
        self.assertEqual(queue_command[queue_command.index("--account") + 1], "test-account")
        submit_command = command.call_args_list[1].args[0]
        self.assertEqual(submit_command[:2], ["sbatch", "--parsable"])
        self.assertEqual(
            submit_command[submit_command.index("--comment") + 1], "expctl:abc"
        )

    def test_missing_receipt_recovers_job_by_unique_slurm_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "abc"
            (run / "control").mkdir(parents=True)
            (run / "control" / "job.sbatch").write_text("#!/bin/sh\n")
            (run / "control" / "submission-intent.json").write_text("{}")
            with patch.dict(os.environ, {"USER": "tester"}), patch(
                "expctl_remote.subprocess.run",
                return_value=self._completed(stdout="24680|expctl:abc\n"),
            ) as command:
                record = expctl_remote.submit(
                    argparse.Namespace(
                        run_dir=str(run),
                        remote_root=str(root),
                        account="test-account",
                    )
                )
        self.assertEqual(record["status"], "recovered")
        self.assertEqual(record["job_id"], "24680")
        self.assertEqual(command.call_count, 1)

    def test_other_active_job_blocks_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "abc"
            (run / "control").mkdir(parents=True)
            with patch.dict(os.environ, {"USER": "tester"}), patch(
                "expctl_remote.subprocess.run",
                return_value=self._completed(stdout="999|expctl:other\n"),
            ) as command:
                with self.assertRaisesRegex(RuntimeError, "one active run"):
                    expctl_remote.submit(
                        argparse.Namespace(
                            run_dir=str(run),
                            remote_root=str(root),
                            account="test-account",
                        )
                    )
        self.assertEqual(command.call_count, 1)

    def test_invalid_sbatch_response_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "abc"
            (run / "control").mkdir(parents=True)
            (run / "control" / "job.sbatch").write_text("#!/bin/sh\n")
            with patch(
                "expctl_remote.subprocess.run",
                side_effect=[self._completed(), self._completed(stdout="Submitted 123")],
            ):
                with self.assertRaisesRegex(RuntimeError, "submission rejected"):
                    expctl_remote.submit(
                        argparse.Namespace(
                            run_dir=str(run),
                            remote_root=str(root),
                            account="test-account",
                        )
                    )

    def test_status_falls_through_to_bounded_sacct_retries(self):
        responses = [
            self._completed(),
            self._completed(),
            self._completed(),
            self._completed(stdout="COMPLETED|\n"),
        ]
        with patch("expctl_remote.subprocess.run", side_effect=responses) as command, patch(
            "expctl_remote.time.sleep"
        ) as sleep:
            result = expctl_remote.status(argparse.Namespace(job_id="12345"))
        self.assertEqual(result["source"], "sacct")
        self.assertEqual(result["state"], "COMPLETED")
        self.assertEqual(command.call_count, 4)
        self.assertEqual(sleep.call_count, 2)

    def test_cancel_is_idempotent_for_terminal_job(self):
        with patch(
            "expctl_remote.subprocess.run",
            side_effect=[self._completed(), self._completed(stdout="COMPLETED|\n")],
        ) as command:
            result = expctl_remote.cancel(argparse.Namespace(job_id="12345"))
        self.assertEqual(result["status"], "already_terminal")
        self.assertEqual(command.call_count, 2)

    def test_verify_source_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            file = source / "code.py"
            file.write_text("value = 1\n")
            entries = [
                {
                    "path": "code.py",
                    "kind": "file",
                    "size": file.stat().st_size,
                    "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                    "mode": stat.S_IMODE(file.stat().st_mode),
                }
            ]
            digest = hashlib.sha256(
                json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            manifest = root / "source-manifest.json"
            manifest.write_text(json.dumps({"entries": entries, "digest": digest}))
            args = argparse.Namespace(source_root=str(source), manifest=str(manifest))
            self.assertEqual(
                expctl_remote.verify_source(args)["source_digest"], digest
            )
            file.write_text("value = 2\n")
            with self.assertRaisesRegex(RuntimeError, "differs from manifest"):
                expctl_remote.verify_source(args)

    def test_verify_data_hashes_only_declared_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "test" / "image" / "case.nii.gz"
            label = root / "test" / "label" / "case.nii.gz"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(b"image")
            label.write_bytes(b"label")
            records = {
                "test": [
                    {
                        "name": "case.nii.gz",
                        "image_sha256": hashlib.sha256(b"image").hexdigest(),
                        "label_sha256": hashlib.sha256(b"label").hexdigest(),
                    }
                ]
            }
            expected = hashlib.sha256(
                json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            result = expctl_remote.verify_data(
                argparse.Namespace(
                    dataset_root=str(root), splits="test", expected_digest=expected
                )
            )
        self.assertEqual(result["dataset_digest"], expected)

    def test_ready_environment_requires_private_matching_complete_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment_id = "a" * 64
            environment = root / environment_id
            python = environment / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\n")
            python.chmod(0o700)
            control = root / ".control" / environment_id
            control.mkdir(parents=True)
            ready = {
                "schema_version": 1,
                "kind": "uv-lock",
                "uv_lock_sha256": environment_id,
                "environment_id": environment_id,
                "environment_path": str(environment),
                "system": "Linux",
                "machine": "x86_64",
                "python_version": "3.12.9",
                "package_check": "passed",
            }
            for path in (control / "ready.json", environment / ".dscnet-ready.json"):
                path.write_text(json.dumps(ready))
                path.chmod(0o600)
            with patch.object(expctl_remote, "ENVIRONMENT_ROOT", root):
                self.assertEqual(
                    expctl_remote._valid_environment_ready(environment_id, environment),
                    ready,
                )
                (control / "ready.json").chmod(0o644)
                self.assertIsNone(
                    expctl_remote._valid_environment_ready(environment_id, environment)
                )
                (control / "ready.json").chmod(0o600)
                python.chmod(0o600)
                self.assertIsNone(
                    expctl_remote._valid_environment_ready(environment_id, environment)
                )
                python.chmod(0o700)
                published = environment / ".dscnet-ready.json"
                published.unlink()
                published.symlink_to(control / "ready.json")
                self.assertIsNone(
                    expctl_remote._valid_environment_ready(environment_id, environment)
                )

    def test_environment_ensure_submits_one_fixed_cpu_atomic_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            source = base / "checkout"
            source.mkdir()
            lock = source / "uv.lock"
            lock.write_bytes(b"lock")
            (source / "tools").mkdir()
            (source / "tools" / "expctl_remote.py").write_text("# helper\n")
            digest = hashlib.sha256(b"lock").hexdigest()
            environments = base / "environments"
            experiments = base / "experiments"
            experiments.mkdir()
            arguments = argparse.Namespace(
                source_root=str(source), uv_lock_sha256=digest,
                account="test-account", retry=False,
            )
            with patch.multiple(
                expctl_remote,
                ALLOWED_REMOTE_CHECKOUT=source,
                ALLOWED_EXPERIMENT_ROOT=experiments,
                ENVIRONMENT_ROOT=environments,
                TRUSTED_ACCOUNTS=frozenset({"test-account"}),
            ), patch.dict(os.environ, {"USER": "tester"}), patch(
                "expctl_remote.subprocess.run",
                side_effect=[
                    self._completed(stdout="456;cluster\n"),
                    self._completed(stdout="PENDING\n"),
                ],
            ) as command:
                first = expctl_remote.environment_ensure(arguments)
                second = expctl_remote.environment_ensure(arguments)
                script = (environments / ".control" / digest / "setup.sbatch").read_text()
        self.assertEqual(first["status"], "BUILDING")
        self.assertEqual(second, first)
        self.assertEqual(command.call_count, 2)
        self.assertIn("#SBATCH --partition=normal", script)
        self.assertIn("#SBATCH --cpus-per-task=2", script)
        self.assertIn("#SBATCH --mem=16G", script)
        self.assertIn("#SBATCH --time=1:00:00", script)
        self.assertNotIn("gpu", script.lower())
        self.assertIn("umask 077", script)
        self.assertIn("environment-build", script)
        self.assertIn("UV_CACHE_DIR", script)
        self.assertIn("trap 'rm -rf -- \"$TMPDIR\"' EXIT", script)
        self.assertNotIn("exec /usr/bin/python3", script)

    def test_failed_environment_setup_requires_explicit_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            source = base / "checkout"
            source.mkdir()
            (source / "uv.lock").write_bytes(b"lock")
            digest = hashlib.sha256(b"lock").hexdigest()
            environments = base / "environments"
            control = environments / ".control" / digest
            control.mkdir(parents=True)
            receipt = control / "submission.json"
            receipt.write_text(
                json.dumps(
                    {
                        "status": "submitted",
                        "job_id": "456",
                        "marker": f"expctl-env:{digest}",
                    }
                )
            )
            receipt.chmod(0o600)
            arguments = argparse.Namespace(
                source_root=str(source), uv_lock_sha256=digest,
                account="test-account", retry=False,
            )
            with patch.multiple(
                expctl_remote,
                ALLOWED_REMOTE_CHECKOUT=source,
                ENVIRONMENT_ROOT=environments,
                TRUSTED_ACCOUNTS=frozenset({"test-account"}),
            ), patch("expctl_remote._environment_job_state", return_value="FAILED"):
                with self.assertRaisesRegex(RuntimeError, "retry explicitly"):
                    expctl_remote.environment_ensure(arguments)
                arguments.retry = True
                with patch(
                    "expctl_remote.subprocess.run",
                    return_value=self._completed(stdout="789\n"),
                ):
                    retried = expctl_remote.environment_ensure(arguments)
        self.assertEqual(retried["status"], "BUILDING")
        self.assertEqual(retried["job_id"], "789")

    def test_retry_does_not_duplicate_less_common_nonterminal_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            source = base / "checkout"
            source.mkdir()
            (source / "uv.lock").write_bytes(b"lock")
            digest = hashlib.sha256(b"lock").hexdigest()
            environments = base / "environments"
            control = environments / ".control" / digest
            control.mkdir(parents=True)
            receipt = control / "submission.json"
            receipt.write_text(
                json.dumps(
                    {
                        "status": "submitted",
                        "job_id": "456",
                        "marker": f"expctl-env:{digest}",
                    }
                )
            )
            receipt.chmod(0o600)
            arguments = argparse.Namespace(
                source_root=str(source), uv_lock_sha256=digest,
                account="test-account", retry=True,
            )
            with patch.multiple(
                expctl_remote,
                ALLOWED_REMOTE_CHECKOUT=source,
                ENVIRONMENT_ROOT=environments,
                TRUSTED_ACCOUNTS=frozenset({"test-account"}),
            ), patch(
                "expctl_remote._environment_job_state", return_value="REQUEUE_HOLD"
            ), patch("expctl_remote.subprocess.run") as submit:
                result = expctl_remote.environment_ensure(arguments)
        self.assertEqual(result["status"], "BUILDING")
        self.assertEqual(result["job_id"], "456")
        submit.assert_not_called()

    def test_unknown_setup_job_requires_retry_and_can_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            source = base / "checkout"
            source.mkdir()
            (source / "uv.lock").write_bytes(b"lock")
            digest = hashlib.sha256(b"lock").hexdigest()
            environments = base / "environments"
            control = environments / ".control" / digest
            control.mkdir(parents=True)
            for name, payload in (
                (
                    "submission.json",
                    {
                        "status": "submitted",
                        "job_id": "456",
                        "marker": f"expctl-env:{digest}",
                    },
                ),
                ("failure.json", {"status": "failed"}),
            ):
                path = control / name
                path.write_text(json.dumps(payload))
                path.chmod(0o600)
            arguments = argparse.Namespace(
                source_root=str(source), uv_lock_sha256=digest,
                account="test-account", retry=False,
            )
            with patch.multiple(
                expctl_remote,
                ALLOWED_REMOTE_CHECKOUT=source,
                ENVIRONMENT_ROOT=environments,
                TRUSTED_ACCOUNTS=frozenset({"test-account"}),
            ), patch("expctl_remote._environment_job_state", return_value="UNKNOWN"):
                with self.assertRaisesRegex(RuntimeError, "no longer visible"):
                    expctl_remote.environment_ensure(arguments)
                arguments.retry = True
                with patch(
                    "expctl_remote.subprocess.run",
                    return_value=self._completed(stdout="789\n"),
                ):
                    retried = expctl_remote.environment_ensure(arguments)
        self.assertEqual(retried["status"], "BUILDING")
        self.assertEqual(retried["job_id"], "789")

    def test_environment_ensure_rejects_malformed_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            source = base / "checkout"
            source.mkdir()
            (source / "uv.lock").write_bytes(b"lock")
            digest = hashlib.sha256(b"lock").hexdigest()
            environments = base / "environments"
            control = environments / ".control" / digest
            control.mkdir(parents=True)
            receipt = control / "submission.json"
            receipt.write_text(json.dumps({"job_id": "../../unrelated"}))
            receipt.chmod(0o600)
            arguments = argparse.Namespace(
                source_root=str(source), uv_lock_sha256=digest,
                account="test-account", retry=False,
            )
            with patch.multiple(
                expctl_remote,
                ALLOWED_REMOTE_CHECKOUT=source,
                ENVIRONMENT_ROOT=environments,
                TRUSTED_ACCOUNTS=frozenset({"test-account"}),
            ), self.assertRaisesRegex(RuntimeError, "receipt is invalid"):
                expctl_remote.environment_ensure(arguments)

    def test_environment_build_cleans_ready_evidence_if_publish_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            source = base / "checkout"
            source.mkdir()
            (source / "uv.lock").write_bytes(b"lock")
            digest = hashlib.sha256(b"lock").hexdigest()
            environments = base / "environments"
            final = environments / digest
            real_replace = os.replace

            def command(argv, **kwargs):
                if "sync" in argv:
                    stage = Path(kwargs["env"]["UV_PROJECT_ENVIRONMENT"])
                    python = stage / "bin" / "python"
                    python.parent.mkdir(parents=True)
                    python.write_text("#!/bin/sh\n")
                    python.chmod(0o700)
                    return self._completed()
                if argv[-2:] == ["-c", argv[-1]]:
                    return self._completed(
                        stdout=json.dumps(
                            {
                                "python_version": "3.12.9",
                                "system": "Linux",
                                "machine": "x86_64",
                            }
                        )
                    )
                return self._completed()

            def replace(source_path, destination_path):
                if Path(destination_path) == final:
                    raise OSError("publish failed")
                return real_replace(source_path, destination_path)

            arguments = argparse.Namespace(
                source_root=str(source), uv_lock_sha256=digest,
            )
            with patch.multiple(
                expctl_remote,
                ALLOWED_REMOTE_CHECKOUT=source,
                ENVIRONMENT_ROOT=environments,
            ), patch.dict(
                os.environ,
                {"SLURM_JOB_ID": "123", "SLURM_JOB_PARTITION": "normal"},
            ), patch(
                "expctl_remote.subprocess.run", side_effect=command
            ), patch("expctl_remote.os.replace", side_effect=replace):
                with self.assertRaisesRegex(OSError, "publish failed"):
                    expctl_remote.environment_build(arguments)
            control = environments / ".control" / digest
            self.assertFalse(final.exists())
            self.assertFalse((control / "ready.json").exists())
            self.assertTrue((control / "failure.json").is_file())
            self.assertFalse(any(environments.glob(f".{digest}.build-*")))

    def test_environment_build_rejects_login_node_execution(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            RuntimeError, "allowlisted Slurm job"
        ):
            expctl_remote.environment_build(
                argparse.Namespace(source_root="/tmp", uv_lock_sha256="a" * 64)
            )

    def test_submit_persists_optional_afterok_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "abc"
            (run / "control").mkdir(parents=True)
            (run / "control" / "job.sbatch").write_text("#!/bin/sh\n")
            arguments = argparse.Namespace(
                run_dir=str(run), remote_root=str(root), account="test-account",
                dependency_job_id="777",
            )
            with patch(
                "expctl_remote.subprocess.run",
                side_effect=[self._completed(), self._completed(stdout="12345\n")],
            ) as command:
                record = expctl_remote.submit(arguments)
                receipt = json.loads((run / "control" / "submission.json").read_text())
        submit_command = command.call_args_list[1].args[0]
        self.assertIn("afterok:777", submit_command)
        self.assertEqual(record["dependency_job_id"], "777")
        self.assertEqual(receipt["dependency_job_id"], "777")

    def test_ui_start_rejects_environment_without_ready_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            args = argparse.Namespace(
                experiment_root=str(root), tracking_uri="sqlite:///test.db",
                artifact_root=str(root / "artifacts"), environment_id="a" * 64,
                port=5000, timeout_minutes=60,
            )
            with patch.multiple(
                expctl_remote,
                ALLOWED_EXPERIMENT_ROOT=root,
                ALLOWED_MLFLOW_URI="sqlite:///test.db",
                ALLOWED_ARTIFACT_ROOT=root / "artifacts",
                ENVIRONMENT_ROOT=root / "environments",
            ):
                with self.assertRaisesRegex(RuntimeError, "environment is not ready"):
                    expctl_remote.ui_start(args)

    def test_ui_start_is_localhost_only_bounded_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            artifact_root = root / "mlflow-artifacts"
            environment_id = "a" * 64
            python = root / environment_id / "bin" / "python"
            process = Mock(pid=1234)
            process.poll.return_value = None
            probe = MagicMock()
            probe.__enter__.return_value = probe
            probe.connect_ex.side_effect = [1, 0]
            arguments = argparse.Namespace(
                experiment_root=str(root),
                tracking_uri="sqlite:///test.db",
                artifact_root=str(artifact_root),
                environment_id=environment_id,
                port=5000,
                timeout_minutes=60,
            )
            with patch.multiple(
                expctl_remote,
                ALLOWED_EXPERIMENT_ROOT=root,
                ALLOWED_MLFLOW_URI="sqlite:///test.db",
                ALLOWED_ARTIFACT_ROOT=artifact_root,
                ENVIRONMENT_ROOT=root,
            ), patch("expctl_remote._valid_environment_ready", return_value={"status": "ready"}), patch("expctl_remote.socket.socket", return_value=probe), patch(
                "expctl_remote.subprocess.Popen", return_value=process
            ) as launch, patch(
                "expctl_remote._linux_process_identity", return_value=("boot", "ticks")
            ), patch(
                "expctl_remote._ui_process_matches", return_value=True
            ):
                started = expctl_remote.ui_start(arguments)
                repeated = expctl_remote.ui_start(arguments)
            command = launch.call_args.args[0]
        self.assertEqual(started["state"], "running")
        self.assertEqual(repeated["pid"], started["pid"])
        self.assertEqual(launch.call_count, 1)
        self.assertIn("--host", command)
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--app-name") + 1], "basic-auth")
        self.assertIn("MLFLOW_AUTH_CONFIG_PATH", launch.call_args.kwargs["env"])
        self.assertIn("MLFLOW_FLASK_SERVER_SECRET_KEY", launch.call_args.kwargs["env"])
        self.assertEqual(
            launch.call_args.kwargs["env"]["MLFLOW_SERVER_ENABLE_JOB_EXECUTION"],
            "false",
        )
        self.assertEqual(started["authentication"], "basic")
        self.assertEqual(started["username"], "dscnet-ui")
        self.assertRegex(started["password"], r"^[A-Za-z0-9_-]{40,64}$")
        self.assertIn("60m", command)

    def test_ui_start_replaces_a_running_old_lock_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            artifact_root = root / "mlflow-artifacts"
            probe = MagicMock()
            probe.__enter__.return_value = probe
            probe.connect_ex.side_effect = [1, 0, 1, 0]
            first_process = Mock(pid=1234)
            first_process.poll.return_value = None
            second_process = Mock(pid=5678)
            second_process.poll.return_value = None

            def arguments(environment_id):
                return argparse.Namespace(
                    experiment_root=str(root),
                    tracking_uri="sqlite:///test.db",
                    artifact_root=str(artifact_root),
                    environment_id=environment_id,
                    port=5000,
                    timeout_minutes=60,
                )

            with patch.multiple(
                expctl_remote,
                ALLOWED_EXPERIMENT_ROOT=root,
                ALLOWED_MLFLOW_URI="sqlite:///test.db",
                ALLOWED_ARTIFACT_ROOT=artifact_root,
                ENVIRONMENT_ROOT=root,
            ), patch(
                "expctl_remote._valid_environment_ready", return_value={"status": "ready"}
            ), patch(
                "expctl_remote.socket.socket", return_value=probe
            ), patch(
                "expctl_remote.subprocess.Popen",
                side_effect=[first_process, second_process],
            ) as launch, patch(
                "expctl_remote._linux_process_identity", return_value=("boot", "ticks")
            ), patch(
                "expctl_remote._ui_process_matches", return_value=True
            ), patch(
                "expctl_remote._stop_ui_process", return_value=True
            ) as stop:
                first = expctl_remote.ui_start(arguments("a" * 64))
                second = expctl_remote.ui_start(arguments("b" * 64))
        self.assertEqual(first["environment_id"], "a" * 64)
        self.assertEqual(second["environment_id"], "b" * 64)
        self.assertEqual(launch.call_count, 2)
        stop.assert_called_once()

    def test_ui_auth_uses_private_persistent_random_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            control = Path(directory) / "control"
            config, password, secret = expctl_remote._ui_auth(control)
            repeated_config, repeated_password, repeated_secret = expctl_remote._ui_auth(control)
            config_mode = config.stat().st_mode & 0o777
            control_mode = control.stat().st_mode & 0o777
        self.assertEqual(config, repeated_config)
        self.assertEqual(password, repeated_password)
        self.assertNotEqual(password, "password1234")
        self.assertEqual(secret, repeated_secret)
        self.assertRegex(secret, r"^[A-Za-z0-9_-]{60,96}$")
        self.assertEqual(config_mode, 0o600)
        self.assertEqual(control_mode, 0o700)

    def test_ui_auth_rejects_a_symlinked_database(self):
        with tempfile.TemporaryDirectory() as directory:
            control = Path(directory) / "control"
            control.mkdir()
            target = Path(directory) / "outside.db"
            target.write_text("outside")
            (control / "mlflow-auth.db").symlink_to(target)
            target.unlink()
            with self.assertRaisesRegex(RuntimeError, "database path is unsafe"):
                expctl_remote._ui_auth(control)

    def test_ui_state_write_failure_stops_the_new_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            artifact_root = root / "mlflow-artifacts"
            environment_id = "a" * 64
            python = root / environment_id / "bin" / "python"
            process = Mock(pid=1234)
            process.poll.return_value = None
            probe = MagicMock()
            probe.__enter__.return_value = probe
            probe.connect_ex.return_value = 1
            arguments = argparse.Namespace(
                experiment_root=str(root),
                tracking_uri="sqlite:///test.db",
                artifact_root=str(artifact_root),
                environment_id=environment_id,
                port=5000,
                timeout_minutes=60,
            )
            with patch.multiple(
                expctl_remote,
                ALLOWED_EXPERIMENT_ROOT=root,
                ALLOWED_MLFLOW_URI="sqlite:///test.db",
                ALLOWED_ARTIFACT_ROOT=artifact_root,
                ENVIRONMENT_ROOT=root,
            ), patch("expctl_remote._valid_environment_ready", return_value={"status": "ready"}), patch("expctl_remote.socket.socket", return_value=probe), patch(
                "expctl_remote.subprocess.Popen", return_value=process
            ), patch(
                "expctl_remote._linux_process_identity", return_value=("boot", "ticks")
            ), patch(
                "expctl_remote._atomic_json", side_effect=OSError("disk full")
            ), patch(
                "expctl_remote._stop_ui_process", return_value=True
            ) as stop:
                with self.assertRaisesRegex(OSError, "disk full"):
                    expctl_remote.ui_start(arguments)
        stop.assert_called_once()

    def test_ui_stop_waits_and_escalates_before_success(self):
        record = {"pid": 1234, "boot_id": "boot", "process_start_ticks": "ticks"}
        with patch("expctl_remote._valid_ui_record", return_value=True), patch(
            "expctl_remote._linux_process_identity", return_value=("boot", "ticks")
        ), patch(
            "expctl_remote._process_group_exists", side_effect=[True] * 50 + [False]
        ), patch("expctl_remote.os.killpg") as kill, patch(
            "expctl_remote.time.sleep"
        ):
            self.assertTrue(expctl_remote._stop_ui_process(record))
        self.assertEqual(
            kill.call_args_list,
            [
                call(1234, expctl_remote.signal.SIGTERM),
                call(1234, expctl_remote.signal.SIGKILL),
            ],
        )

    def test_ui_timeout_stops_only_the_recorded_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            state = root / "control" / "mlflow-ui.json"
            state.parent.mkdir(parents=True)
            state.write_text(
                json.dumps(
                    {
                        "pid": 1234,
                        "boot_id": "boot",
                        "process_start_ticks": "ticks",
                        "deadline_epoch": 0,
                    }
                )
            )
            with patch.object(expctl_remote, "ALLOWED_EXPERIMENT_ROOT", root), patch(
                "expctl_remote._valid_ui_record", return_value=True
            ), patch(
                "expctl_remote._ui_process_matches", return_value=True
            ), patch("expctl_remote._stop_ui_process", return_value=True) as stop:
                result = expctl_remote.ui_status(
                    argparse.Namespace(experiment_root=str(root))
                )
        self.assertEqual(result, {"state": "off", "reason": "timeout"})
        stop.assert_called_once()
        self.assertFalse(state.exists())

    def test_ui_status_quarantines_malformed_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            state = root / "control" / "mlflow-ui.json"
            state.parent.mkdir(parents=True)
            for malformed in ([], {"deadline_epoch": "later"}, "bad"):
                state.write_text(json.dumps(malformed))
                with self.subTest(malformed=malformed), patch.object(
                    expctl_remote, "ALLOWED_EXPERIMENT_ROOT", root
                ), patch("expctl_remote._ui_port_is_open", return_value=False):
                    result = expctl_remote.ui_status(
                        argparse.Namespace(experiment_root=str(root))
                    )
                    self.assertEqual(
                        result, {"state": "off", "reason": "invalid_state"}
                    )
                    self.assertFalse(state.exists())

    def test_ui_status_fails_closed_for_malformed_state_with_active_port(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            state = root / "control" / "mlflow-ui.json"
            state.parent.mkdir(parents=True)
            state.write_text("{truncated")
            with patch.object(
                expctl_remote, "ALLOWED_EXPERIMENT_ROOT", root
            ), patch("expctl_remote._ui_port_is_open", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "port is in use"):
                    expctl_remote.ui_status(
                        argparse.Namespace(experiment_root=str(root))
                    )

    def test_ui_status_fails_closed_for_unreadable_state_with_active_port(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            state = root / "control" / "mlflow-ui.json"
            state.parent.mkdir(parents=True)
            state.write_text("{}")
            with patch.object(
                expctl_remote, "ALLOWED_EXPERIMENT_ROOT", root
            ), patch.object(Path, "read_text", side_effect=PermissionError), patch(
                "expctl_remote._ui_port_is_open", return_value=True
            ):
                with self.assertRaisesRegex(RuntimeError, "port is in use"):
                    expctl_remote.ui_status(
                        argparse.Namespace(experiment_root=str(root))
                    )

    def test_logs_bound_files_bytes_and_newline_free_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "runs" / "abc"
            logs = run / "logs"
            logs.mkdir(parents=True)
            (logs / "pipeline.log").write_bytes(b"x" * (expctl_remote.MAX_LOG_BYTES * 2))
            for index in range(expctl_remote.MAX_LOG_FILES + 2):
                (logs / f"slurm-{index:02d}.out").write_text(f"job {index}\n")
            result = expctl_remote.logs(
                argparse.Namespace(run_dir=str(run), lines=1000)
            )
        self.assertTrue(result["truncated"])
        self.assertLessEqual(len(result["logs"]), expctl_remote.MAX_LOG_FILES)
        payload_size = sum(
            len(line.encode("utf-8")) + 1
            for lines in result["logs"].values()
            for line in lines
        )
        self.assertLessEqual(payload_size, expctl_remote.MAX_LOG_BYTES)
        self.assertLessEqual(
            len(result["logs"]["pipeline.log"][0]), expctl_remote.MAX_LOG_BYTES
        )

    def test_verify_artifacts_checks_actual_size_before_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            file = run / "logs" / "pipeline.log"
            file.parent.mkdir(parents=True)
            file.write_bytes(b"actual")
            manifest = run / "control" / "artifacts.json"
            manifest.parent.mkdir()
            manifest.write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "path": "logs/pipeline.log",
                                "size": 1,
                                "sha256": hashlib.sha256(b"actual").hexdigest(),
                            }
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(RuntimeError, "changed after finalization"):
                expctl_remote.verify_artifacts(argparse.Namespace(run_dir=str(run)))

    def test_finalize_declares_only_bounded_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            (run / "control").mkdir(parents=True)
            (run / "logs").mkdir()
            (run / "outputs" / "weights").mkdir(parents=True)
            (run / "outputs" / "predictions").mkdir(parents=True)
            (run / "control" / "resolved-config.yaml").write_text("action: train\n")
            (run / "control" / "source.tar.gz").write_bytes(b"source")
            (run / "control" / "git.json").write_text("{}\n")
            (run / "control" / "environment-lock.json").write_text("{}\n")
            (run / "control" / "submission.json").write_text('{"job_id":"123"}\n')
            (run / "control" / "Mean_Std.npy").write_bytes(b"normalization")
            (run / "logs" / "pipeline.log").write_text("done\n")
            (run / "logs" / "slurm-123.out").write_text("still mutable\n")
            (run / "outputs" / "weights" / "model_best.pth").write_bytes(b"best")
            (run / "outputs" / "weights" / "model_latest.pth").write_bytes(b"latest")
            (run / "outputs" / "predictions" / "large.nii.gz").write_bytes(b"large")
            result = expctl_remote.finalize(
                argparse.Namespace(
                    run_dir=str(run),
                    best_checkpoint="model_best.pth",
                    latest_checkpoint="model_latest.pth",
                )
            )
        paths = {item["path"] for item in result["artifacts"]}
        self.assertIn("control/Mean_Std.npy", paths)
        self.assertIn("control/source.tar.gz", paths)
        self.assertIn("control/git.json", paths)
        self.assertIn("control/environment-lock.json", paths)
        self.assertIn("control/submission.json", paths)
        self.assertIn("outputs/weights/model_best.pth", paths)
        self.assertIn("outputs/weights/model_latest.pth", paths)
        self.assertNotIn("logs/slurm-123.out", paths)
        self.assertNotIn("outputs/predictions/large.nii.gz", paths)

if __name__ == "__main__":
    unittest.main()
