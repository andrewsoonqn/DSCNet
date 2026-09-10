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

REPO_ROOT = Path(__file__).resolve().parents[1]
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

    def test_ui_start_is_localhost_only_bounded_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            artifact_root = root / "mlflow-artifacts"
            python = root / "python"
            process = Mock(pid=1234)
            process.poll.return_value = None
            probe = MagicMock()
            probe.__enter__.return_value = probe
            probe.connect_ex.side_effect = [1, 0]
            arguments = argparse.Namespace(
                experiment_root=str(root),
                tracking_uri="sqlite:///test.db",
                artifact_root=str(artifact_root),
                python=str(python),
                port=5000,
                timeout_minutes=60,
            )
            with patch.multiple(
                expctl_remote,
                ALLOWED_EXPERIMENT_ROOT=root,
                ALLOWED_MLFLOW_URI="sqlite:///test.db",
                ALLOWED_ARTIFACT_ROOT=artifact_root,
                ALLOWED_PYTHON=python,
            ), patch("expctl_remote.socket.socket", return_value=probe), patch(
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
        self.assertEqual(started["authentication"], "basic")
        self.assertEqual(started["username"], "dscnet-ui")
        self.assertRegex(started["password"], r"^[A-Za-z0-9_-]{40,64}$")
        self.assertIn("60m", command)

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
            python = root / "python"
            process = Mock(pid=1234)
            process.poll.return_value = None
            probe = MagicMock()
            probe.__enter__.return_value = probe
            probe.connect_ex.return_value = 1
            arguments = argparse.Namespace(
                experiment_root=str(root),
                tracking_uri="sqlite:///test.db",
                artifact_root=str(artifact_root),
                python=str(python),
                port=5000,
                timeout_minutes=60,
            )
            with patch.multiple(
                expctl_remote,
                ALLOWED_EXPERIMENT_ROOT=root,
                ALLOWED_MLFLOW_URI="sqlite:///test.db",
                ALLOWED_ARTIFACT_ROOT=artifact_root,
                ALLOWED_PYTHON=python,
            ), patch("expctl_remote.socket.socket", return_value=probe), patch(
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
            (run / "control" / "Mean_Std.npy").write_bytes(b"normalization")
            (run / "logs" / "pipeline.log").write_text("done\n")
            (run / "outputs" / "weights" / "model_best.pth").write_bytes(b"best")
            (run / "outputs" / "predictions" / "large.nii.gz").write_bytes(b"large")
            result = expctl_remote.finalize(
                argparse.Namespace(run_dir=str(run), best_checkpoint="model_best.pth")
            )
        paths = {item["path"] for item in result["artifacts"]}
        self.assertIn("control/Mean_Std.npy", paths)
        self.assertIn("outputs/weights/model_best.pth", paths)
        self.assertNotIn("outputs/predictions/large.nii.gz", paths)


if __name__ == "__main__":
    unittest.main()
