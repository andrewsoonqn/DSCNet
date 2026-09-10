import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, call, patch

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from expctl import (
    CommandResult,
    ExpctlError,
    ExperimentController,
    LocalTunnelManager,
    SubprocessTransport,
    _bounded_detail,
    _execution_dirty_paths,
    _run_tunnel_worker,
)

sys.path.insert(
    0, str(REPO_ROOT / "DSCNet_3D_opensource" / "Code" / "Kipa" / "DSCNet")
)
from S4_Experiment_Run import _control_evidence

EXPERIMENT = REPO_ROOT / "configs" / "experiment" / "dscnet_standard.yaml"
OVERRIDES = [
    "action=prepare",
    "runtime.formal=false",
    "runtime.allow_dirty=true",
    "runtime.slurm.account=test-account",
]


class FakeTransport:
    def __init__(self):
        self.calls = []
        self.archive_digest = None
        self.synced_digests = {}
        self.submit_count = 0
        self.fail_submit = False
        self.remote_files = {}
        self.remote_ui_running = False
        self.remote_ui_deadline = None

    def ssh(self, host, argv):
        self.calls.append(("ssh", host, tuple(argv)))
        if argv[0] == "sha256sum":
            name = Path(argv[1]).name
            digest = self.synced_digests.get(name, self.archive_digest)
            return CommandResult(0, f"{digest}  {argv[1]}\n")
        if argv[0] == "stat":
            return CommandResult(
                0, str(len(self.remote_files["control/artifacts.json"])) + "\n"
            )
        if "ui-start" in argv:
            started = not self.remote_ui_running
            self.remote_ui_running = True
            if started:
                self.remote_ui_deadline = time.time() + 3600
            return CommandResult(
                0,
                json.dumps(
                    {
                        "state": "running",
                        "started": started,
                        "deadline_epoch": self.remote_ui_deadline,
                        "authentication": "basic",
                        "username": "dscnet-ui",
                        "password": "a" * 43,
                    }
                ),
            )
        if "ui-status" in argv:
            return CommandResult(
                0,
                json.dumps(
                    {
                        "state": "running" if self.remote_ui_running else "off",
                        "deadline_epoch": self.remote_ui_deadline,
                    }
                ),
            )
        if "ui-stop" in argv:
            self.remote_ui_running = False
            return CommandResult(0, json.dumps({"state": "off", "stopped": True}))
        if "verify-data" in argv:
            return CommandResult(0, json.dumps({"status": "verified"}))
        if "submit" in argv:
            self.submit_count += 1
            if self.fail_submit:
                return CommandResult(
                    1, stderr='{"error":"submission rejected: invalid account"}'
                )
            return CommandResult(0, json.dumps({"job_id": "12345"}))
        if "verify-artifacts" in argv:
            manifest = json.loads(self.remote_files["control/artifacts.json"])
            return CommandResult(
                0,
                json.dumps(
                    {
                        "status": "verified",
                        "total_size": sum(
                            item["size"] for item in manifest["artifacts"]
                        ),
                    }
                ),
            )
        if "status" in argv:
            return CommandResult(
                0, json.dumps({"job_id": "12345", "source": "sacct", "state": "COMPLETED"})
            )
        if "logs" in argv:
            return CommandResult(0, json.dumps({"run_id": "run", "logs": {}}))
        if "cancel" in argv:
            return CommandResult(0, json.dumps({"job_id": "12345", "status": "cancellation_requested"}))
        return CommandResult(0)

    def sync_to(self, paths, host, remote_dir):
        self.calls.append(("sync", host, remote_dir, tuple(path.name for path in paths)))
        for path in paths:
            if path.is_file():
                self.synced_digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        archive = next((path for path in paths if path.name == "source.tar.gz"), None)
        if archive:
            self.archive_digest = self.synced_digests[archive.name]
        return CommandResult(0)

    def fetch_file(self, host, remote_path, local_path, max_size):
        self.calls.append(("fetch_file", host, remote_path, max_size))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.remote_files["control/artifacts.json"])
        return CommandResult(0)

    def fetch_files(self, host, remote_root, relative_paths, destination):
        self.calls.append(("fetch_files", host, remote_root, tuple(relative_paths)))
        for relative in relative_paths:
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.remote_files[relative])
        return CommandResult(0)


class FakeTunnelManager:
    def __init__(self):
        self.starts = []
        self.stops = []
        self.alive = False
        self.fail_start = False
        self.fail_stop = False

    def start(self, state_root, local_port, remote_port, duration_seconds):
        self.starts.append((state_root, local_port, remote_port, duration_seconds))
        if self.fail_start:
            raise ExpctlError("tunnel failed")
        self.alive = True
        return {
            "tunnel_pid": 1234,
            "tunnel_token": "a" * 32,
            "local_port": local_port,
            "remote_port": remote_port,
        }

    def is_alive(self, record):
        return self.alive

    def stop(self, record):
        was_alive = self.alive
        self.stops.append(record)
        if self.fail_stop:
            return False
        self.alive = False
        return was_alive


class ExpctlControllerTests(unittest.TestCase):
    def setUp(self):
        account_patch = patch(
            "expctl.TRUSTED_ACCOUNTS", frozenset({"allusers", "test-account"})
        )
        account_patch.start()
        self.addCleanup(account_patch.stop)

    def _controller(self, root, transport=None, tunnel_manager=None):
        return ExperimentController(
            root, transport or FakeTransport(), tunnel_manager=tunnel_manager
        )

    def test_formal_dirty_check_ignores_only_non_execution_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "Test"], check=True
            )
            (root / "tools").mkdir()
            (root / "docs").mkdir()
            (root / "tools" / "run.py").write_text("value = 1\n")
            (root / "docs" / "note.md").write_text("note\n")
            (root / ".gitignore").write_text("cache/\n")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "initial"], check=True
            )
            (root / "docs" / "note.md").write_text("changed note\n")
            (root / ".gitignore").write_text("cache/\nartifacts/\n")
            self.assertEqual(_execution_dirty_paths(root), [])
            (root / "tools" / "run.py").write_text("value = 2\n")
            self.assertEqual(_execution_dirty_paths(root), ["tools/run.py"])

    def test_verify_is_non_persistent_and_rejects_non_allowlisted_host(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(Path(directory))
            verified = controller.verify(EXPERIMENT, OVERRIDES)
            self.assertEqual(verified["state"], "verified")
            self.assertFalse((Path(directory) / "runs").exists())
            with self.assertRaisesRegex(ExpctlError, "host is not allowlisted"):
                controller.verify(
                    EXPERIMENT,
                    [*OVERRIDES, "runtime.slurm.host=other-host"],
                )

    def test_staged_digest_matches_the_runtime_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self._controller(root)
            record = controller._stage(EXPERIMENT, OVERRIDES)
            self.assertEqual(record["controller_version"], "1")
            self.assertEqual(record["pi_extension_version"], "1")
            control = root / "runs" / record["run_id"] / "control"
            resolved = OmegaConf.load(control / "resolved-config.yaml")
            with patch.dict(
                "os.environ",
                {
                    "DSCNET_CONTROL_DIR": str(control),
                    "DSCNET_EXPERIMENT_DIGEST": record["experiment_digest"],
                },
                clear=False,
            ):
                _, _, _, digest = _control_evidence(resolved)
        self.assertEqual(digest, record["experiment_digest"])

    def test_generated_job_uses_the_configuration_free_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._controller(root)._stage(EXPERIMENT, OVERRIDES)
            script = (
                root / "runs" / record["run_id"] / "control" / "job.sbatch"
            ).read_text()
        self.assertIn("export DSCNET_RESOLVED_CONFIG=", script)
        self.assertIn("export DSCNET_SOURCE_ROOT=", script)
        self.assertIn("/run_models.sbatch", script)
        self.assertIn("--latest-checkpoint", script)
        self.assertNotIn("S0_Main.py", script)

    def test_resource_and_path_allowlists_fail_closed(self):
        cases = {
            "Slurm account": [*OVERRIDES, "runtime.slurm.account=untrusted"],
            "memory": [*OVERRIDES, "runtime.slurm.memory_gb=33"],
            "GPU": [*OVERRIDES, "runtime.slurm.gpus=2"],
            "partition": [*OVERRIDES, "runtime.slurm.partition=other"],
            "remote experiment root": [
                *OVERRIDES,
                "runtime.remote_experiment_root=/tmp/experiments",
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(Path(directory))
            for message, overrides in cases.items():
                with self.subTest(message=message), self.assertRaisesRegex(
                    ExpctlError, message
                ):
                    controller.verify(EXPERIMENT, overrides)

    def test_submit_is_idempotent_and_uses_verified_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FakeTransport()
            controller = self._controller(Path(directory), transport)
            first = controller.submit(EXPERIMENT, OVERRIDES)
            second = controller.submit(EXPERIMENT, OVERRIDES)

        self.assertEqual(first["job_id"], "12345")
        self.assertEqual(second["job_id"], first["job_id"])
        self.assertEqual(transport.submit_count, 1)
        commands = [call[2] for call in transport.calls if call[0] == "ssh"]
        self.assertTrue(any(command[0] == "sha256sum" for command in commands))
        self.assertTrue(any("verify-source" in command for command in commands))
        self.assertTrue(any("verify-data" in command for command in commands))

    def test_explicit_retry_creates_a_separate_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FakeTransport()
            controller = self._controller(Path(directory), transport)
            first = controller.submit(EXPERIMENT, OVERRIDES)
            retried = controller.submit(EXPERIMENT, OVERRIDES, retry=True)

        self.assertEqual(retried["run_id"], first["run_id"] + "-a2")
        self.assertEqual(retried["attempt"], 2)
        self.assertEqual(transport.submit_count, 2)

    def test_ambiguous_submit_recovers_the_same_remote_receipt(self):
        class AmbiguousTransport(FakeTransport):
            def ssh(self, host, argv):
                if "submit" in argv and self.submit_count == 0:
                    self.calls.append(("ssh", host, tuple(argv)))
                    self.submit_count += 1
                    return CommandResult(255, stderr="connection closed")
                return super().ssh(host, argv)

        with tempfile.TemporaryDirectory() as directory:
            transport = AmbiguousTransport()
            controller = self._controller(Path(directory), transport)
            with self.assertRaisesRegex(ExpctlError, "connection closed"):
                controller.submit(EXPERIMENT, OVERRIDES)
            recovered = controller.submit(EXPERIMENT, OVERRIDES)
        self.assertEqual(recovered["job_id"], "12345")
        self.assertEqual(transport.submit_count, 2)
        self.assertEqual(
            sum(call[0] == "sync" for call in transport.calls),
            1,
            "ambiguous recovery must not mutate remote control evidence",
        )

    def test_normalization_bytes_change_experiment_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mean_std = root / "Mean_Std.npy"
            mean_std.write_bytes(b"first")
            overrides = [
                *OVERRIDES,
                "action=train",
                f"data.Meanstd_path={mean_std}",
            ]
            controller = self._controller(root / "state", FakeTransport())
            first = controller.submit(EXPERIMENT, overrides)
            mean_std.write_bytes(b"second")
            second = controller.submit(EXPERIMENT, overrides)
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertNotEqual(
            first["normalization"]["sha256"], second["normalization"]["sha256"]
        )

    def test_resumed_training_stages_latest_checkpoint_in_a_new_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights"
            weights.mkdir()
            mean_std = root / "Mean_Std.npy"
            mean_std.write_bytes(b"normalization")
            common = [
                *OVERRIDES,
                "action=train",
                f"data.Meanstd_path={mean_std}",
                f"data.Dir_Weights={weights}",
                "data.model_name=latest.ckpt",
            ]
            transport = FakeTransport()
            controller = self._controller(root / "state", transport)
            initial = controller.submit(EXPERIMENT, common)
            (weights / "latest.ckpt").write_bytes(b"checkpoint-state")
            resumed = controller.submit(
                EXPERIMENT,
                [*common, "training.if_retrain=false", "training.start_train_epoch=2"],
                retry=True,
            )
            control = root / "state" / "runs" / resumed["run_id"] / "control"
            resolved = OmegaConf.load(control / "resolved-config.yaml")
            staged_checkpoint = (control / "training-checkpoint").read_bytes()
            resumed_retrain = resolved.training.if_retrain
            resumed_epoch = resolved.training.start_train_epoch

        self.assertEqual(initial["experiment_digest"], resumed["experiment_digest"])
        self.assertEqual(resumed["run_id"], initial["run_id"] + "-a2")
        self.assertEqual(staged_checkpoint, b"checkpoint-state")
        self.assertFalse(resumed_retrain)
        self.assertEqual(resumed_epoch, 2)
        self.assertEqual(
            resumed["training_checkpoint"]["sha256"],
            hashlib.sha256(b"checkpoint-state").hexdigest(),
        )
        commands = [call[2] for call in transport.calls if call[0] == "ssh"]
        self.assertTrue(
            any(
                command[0] == "cp" and "training-checkpoint" in command[2]
                for command in commands
            )
        )

    def test_evaluation_stages_and_binds_an_explicit_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights"
            weights.mkdir()
            checkpoint = weights / "evaluation-best.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            mean_std = root / "Mean_Std.npy"
            mean_std.write_bytes(b"normalization")
            overrides = [
                *OVERRIDES,
                "action=evaluate",
                f"data.Meanstd_path={mean_std}",
                f"data.Dir_Weights={weights}",
                "data.model_name_max=evaluation-best.ckpt",
            ]
            transport = FakeTransport()
            controller = self._controller(root / "state", transport)
            record = controller.submit(EXPERIMENT, overrides)
            staged = root / "state" / "runs" / record["run_id"] / "control" / "evaluation-checkpoint"
            staged_bytes = staged.read_bytes()
        self.assertEqual(staged_bytes, b"checkpoint")
        self.assertEqual(
            record["evaluation_checkpoint"]["sha256"],
            hashlib.sha256(b"checkpoint").hexdigest(),
        )
        commands = [call[2] for call in transport.calls if call[0] == "ssh"]
        self.assertTrue(any(command[0] == "cp" for command in commands))

    def test_remote_digest_mismatch_fails_before_submission(self):
        class MismatchTransport(FakeTransport):
            def sync_to(self, paths, host, remote_dir):
                super().sync_to(paths, host, remote_dir)
                self.archive_digest = "0" * 64
                self.synced_digests["source.tar.gz"] = "0" * 64
                return CommandResult(0)

        with tempfile.TemporaryDirectory() as directory:
            transport = MismatchTransport()
            controller = self._controller(Path(directory), transport)
            with self.assertRaisesRegex(ExpctlError, "digest does not match"):
                controller.submit(EXPERIMENT, OVERRIDES)
        self.assertEqual(transport.submit_count, 0)

    def test_submission_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FakeTransport()
            transport.fail_submit = True
            root = Path(directory)
            controller = self._controller(root, transport)
            with self.assertRaisesRegex(ExpctlError, "invalid account"):
                controller.submit(EXPERIMENT, OVERRIDES)
            with self.assertRaisesRegex(ExpctlError, "use --retry"):
                controller.submit(EXPERIMENT, OVERRIDES)
            records = list((root / "runs").glob("*/control/run-manifest.json"))
            record = json.loads(records[0].read_text())
        self.assertEqual(record["state"], "submission_rejected")
        self.assertIsNone(record["job_id"])

    def test_status_logs_and_cancel_use_only_recorded_job(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FakeTransport()
            controller = self._controller(Path(directory), transport)
            record = controller.submit(EXPERIMENT, OVERRIDES)
            self.assertEqual(controller.status(record["run_id"])["state"], "COMPLETED")
            self.assertEqual(controller.logs(record["run_id"])["logs"], {})
            self.assertEqual(
                controller.cancel(record["run_id"])["status"], "cancellation_requested"
            )
            with self.assertRaisesRegex(ExpctlError, "unknown run ID"):
                controller.cancel("0" * 16)

    def test_submit_rejects_tampered_existing_routing(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FakeTransport()
            root = Path(directory)
            controller = self._controller(root, transport)
            record = controller._stage(EXPERIMENT, OVERRIDES)
            path = root / "runs" / record["run_id"] / "control" / "run-manifest.json"
            original = json.loads(path.read_text())
            cases = {
                "run_id": "f" * 16,
                "host": "attacker.example",
                "account": "untrusted",
                "remote_root": "/tmp/escape",
                "remote_run_dir": "/tmp/escape",
            }
            for field, value in cases.items():
                tampered = dict(original)
                tampered[field] = value
                path.write_text(json.dumps(tampered))
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ExpctlError, "recorded"):
                        controller.submit(EXPERIMENT, OVERRIDES)
                    self.assertEqual(transport.calls, [])
                path.write_text(json.dumps(original))

    def test_run_operations_reject_tampered_local_routing(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FakeTransport()
            root = Path(directory)
            controller = self._controller(root, transport)
            record = controller.submit(EXPERIMENT, OVERRIDES)
            path = root / "runs" / record["run_id"] / "control" / "run-manifest.json"
            original = json.loads(path.read_text())
            cases = {
                "run_id": "f" * 16,
                "host": "attacker.example",
                "remote_root": "/tmp/escape",
                "remote_run_dir": "/tmp/escape",
            }
            for field, value in cases.items():
                tampered = dict(original)
                tampered[field] = value
                path.write_text(json.dumps(tampered))
                calls_before = len(transport.calls)
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ExpctlError, "recorded"):
                        controller.cancel(record["run_id"])
                    self.assertEqual(len(transport.calls), calls_before)
            path.write_text("not json")
            with self.assertRaisesRegex(ExpctlError, "manifest is invalid"):
                controller.status(record["run_id"])

    def test_ui_lifecycle_starts_only_one_bounded_localhost_tunnel(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FakeTransport()
            tunnel = FakeTunnelManager()
            root = Path(directory)
            controller = self._controller(root, transport, tunnel)
            started = controller.ui_start(5050)
            repeated = controller.ui_start(5050)
            status = controller.ui_status()
            stopped = controller.ui_stop()
        self.assertEqual(started["url"], "http://127.0.0.1:5050")
        self.assertEqual(repeated, started)
        self.assertEqual(status["state"], "running")
        self.assertEqual(stopped["state"], "off")
        self.assertEqual(len(tunnel.starts), 1)
        self.assertEqual(tunnel.starts[0][1:3], (5050, 5000))
        self.assertGreaterEqual(tunnel.starts[0][3], 3598)
        self.assertLessEqual(tunnel.starts[0][3], 3600)
        self.assertEqual(len(tunnel.stops), 1)
        ui_calls = [call for call in transport.calls if "ui-" in " ".join(call[2])]
        self.assertTrue(ui_calls)
        self.assertTrue(all(call[1] == "xlogin1" for call in ui_calls))
        self.assertTrue(
            all("127.0.0.1" not in " ".join(call[2]) for call in ui_calls)
        )

    def test_ui_tunnel_failure_stops_only_a_new_remote_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FakeTransport()
            tunnel = FakeTunnelManager()
            tunnel.fail_start = True
            controller = self._controller(Path(directory), transport, tunnel)
            with self.assertRaisesRegex(ExpctlError, "tunnel failed"):
                controller.ui_start()
        self.assertFalse(transport.remote_ui_running)
        self.assertTrue(any("ui-stop" in call[2] for call in transport.calls))

    def test_ui_state_write_failure_cleans_up_tunnel_and_new_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FakeTransport()
            tunnel = FakeTunnelManager()
            controller = self._controller(Path(directory), transport, tunnel)
            with patch("expctl._atomic_json", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    controller.ui_start()
        self.assertFalse(tunnel.alive)
        self.assertFalse(transport.remote_ui_running)
        self.assertEqual(len(tunnel.stops), 1)

    def test_ui_replaces_a_tunnel_with_the_wrong_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FakeTransport()
            transport.remote_ui_running = True
            transport.remote_ui_deadline = time.time() + 30
            tunnel = FakeTunnelManager()
            tunnel.alive = True
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            (root / "ui.json").write_text(
                json.dumps(
                    {
                        "tunnel_pid": 1234,
                        "tunnel_token": "a" * 32,
                        "local_port": 5000,
                        "remote_port": 5000,
                        "tunnel_deadline_epoch": time.time() + 5,
                    }
                )
            )
            controller = self._controller(root, transport, tunnel)
            result = controller.ui_start()
        self.assertEqual(result["state"], "running")
        self.assertEqual(len(tunnel.stops), 1)
        self.assertEqual(len(tunnel.starts), 1)
        self.assertGreaterEqual(tunnel.starts[0][3], 28)
        self.assertLessEqual(tunnel.starts[0][3], 30)

    def test_failed_tunnel_replacement_stops_a_new_remote_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FakeTransport()
            tunnel = FakeTunnelManager()
            tunnel.alive = True
            tunnel.fail_stop = True
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            (root / "ui.json").write_text(
                json.dumps(
                    {
                        "tunnel_pid": 1234,
                        "tunnel_token": "a" * 32,
                        "local_port": 5000,
                        "remote_port": 5000,
                        "tunnel_deadline_epoch": time.time() + 5,
                    }
                )
            )
            controller = self._controller(root, transport, tunnel)
            with self.assertRaisesRegex(ExpctlError, "did not stop"):
                controller.ui_start()
        self.assertFalse(transport.remote_ui_running)
        self.assertTrue(any("ui-stop" in call[2] for call in transport.calls))

    def test_ui_status_reports_remote_reader_without_a_local_tunnel(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FakeTransport()
            transport.remote_ui_running = True
            transport.remote_ui_deadline = time.time() + 3600
            tunnel = FakeTunnelManager()
            controller = self._controller(Path(directory), transport, tunnel)
            status = controller.ui_status()
        self.assertEqual(status["state"], "remote_only")

    def test_local_tunnel_stop_waits_and_escalates_before_success(self):
        record = {"tunnel_pid": 1234, "tunnel_token": "a" * 32}
        with patch(
            "expctl._process_has_token", side_effect=[True] * 51 + [False]
        ), patch("expctl.os.killpg") as kill, patch("expctl.time.sleep"):
            self.assertTrue(LocalTunnelManager().stop(record))
        self.assertEqual(
            kill.call_args_list,
            [call(1234, signal.SIGTERM), call(1234, signal.SIGKILL)],
        )

    def test_tunnel_worker_forwards_only_loopback_and_ends_on_timeout(self):
        arguments = argparse.Namespace(
            host="xlogin1",
            local_port=5050,
            remote_port=5000,
            duration_seconds=60,
            token="a" * 32,
        )
        process = Mock()
        process.wait.side_effect = [subprocess.TimeoutExpired("ssh", 60), 0]
        with patch("expctl.subprocess.Popen", return_value=process) as launch:
            self.assertEqual(_run_tunnel_worker(arguments), 0)
        command = launch.call_args.args[0]
        self.assertIn("127.0.0.1:5050:127.0.0.1:5000", command)
        self.assertIn("ExitOnForwardFailure=yes", command)
        process.terminate.assert_called_once()

    def test_rsync_commands_support_the_system_rsync(self):
        transport = SubprocessTransport()
        result = CommandResult(0)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            transport, "_run", return_value=result
        ) as run:
            source = Path(directory) / "source"
            source.write_text("data")
            transport.sync_to([source], "xlogin1", "/remote/control")
            transport.fetch_file(
                "xlogin1", "/remote/file", Path(directory) / "file", 1024
            )
            transport.fetch_files(
                "xlogin1", "/remote", ["control/file"], Path(directory) / "tree"
            )
        for invocation in run.call_args_list:
            command = invocation.args[0]
            self.assertEqual(command[:2], ["rsync", "-a"])
            self.assertNotIn("--protect-args", command)

    def test_subprocess_transport_times_out_with_a_bounded_error(self):
        transport = SubprocessTransport()
        with patch(
            "expctl.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["ssh"], 120),
        ):
            with self.assertRaisesRegex(ExpctlError, "ssh timed out after 120 seconds"):
                transport.ssh("xlogin1", ["true"])
        detail = _bounded_detail("\x1b[31m" + "x" * 5000 + "\x1b[0m")
        self.assertNotIn("\x1b", detail)
        self.assertLessEqual(len(detail), 4096 + len("[truncated]\n"))

    def test_fetch_accepts_declared_artifacts_and_verifies_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FakeTransport()
            controller = self._controller(Path(directory), transport)
            record = controller.submit(EXPERIMENT, OVERRIDES)
            payloads = {
                "control/resolved-config.yaml": b"action: prepare\n",
                "control/dataset-manifest.json": b"{}\n",
                "control/source-manifest.json": b"{}\n",
                "control/source.tar.gz": b"source archive",
                "control/git.json": b"{}\n",
                "control/environment-lock.json": b"{}\n",
                "control/run-manifest.json": b"{}\n",
                "control/submission.json": b'{"job_id":"12345"}\n',
                "logs/pipeline.log": b"complete\n",
                "outputs/final-metrics.json": b"{}\n",
            }
            artifact_manifest = {
                "schema_version": 1,
                "artifacts": [
                    {
                        "path": path,
                        "sha256": hashlib.sha256(value).hexdigest(),
                        "size": len(value),
                    }
                    for path, value in payloads.items()
                ],
            }
            transport.remote_files = {
                **payloads,
                "control/artifacts.json": json.dumps(artifact_manifest).encode(),
            }
            stale = (
                Path(directory)
                / "runs"
                / record["run_id"]
                / "fetched"
                / "logs"
                / "slurm-old.out"
            )
            stale.parent.mkdir(parents=True)
            stale.write_text("stale")
            fetched = controller.fetch(record["run_id"])
            stale_remained = stale.exists()
            destination_exists = Path(fetched["destination"]).is_dir()
        self.assertEqual(set(fetched["artifacts"]), set(payloads))
        self.assertFalse(stale_remained)
        self.assertTrue(destination_exists)

    def test_fetch_rejects_oversized_declared_artifact_before_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FakeTransport()
            controller = self._controller(Path(directory), transport)
            record = controller.submit(EXPERIMENT, OVERRIDES)
            artifacts = [
                {"path": path, "sha256": "0" * 64, "size": 1}
                for path in (
                    "control/resolved-config.yaml",
                    "control/dataset-manifest.json",
                    "control/source-manifest.json",
                    "control/run-manifest.json",
                )
            ]
            artifacts.append(
                {
                    "path": "outputs/final-metrics.json",
                    "sha256": "0" * 64,
                    "size": 9 * 1024**3,
                }
            )
            transport.remote_files = {
                "control/artifacts.json": json.dumps({"artifacts": artifacts}).encode()
            }
            with self.assertRaisesRegex(ExpctlError, "per-file size limit"):
                controller.fetch(record["run_id"])
        self.assertFalse(any(call[0] == "fetch_files" for call in transport.calls))

    def test_fetch_rejects_undeclared_remote_path(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FakeTransport()
            controller = self._controller(Path(directory), transport)
            record = controller.submit(EXPERIMENT, OVERRIDES)
            transport.remote_files = {
                "control/artifacts.json": json.dumps(
                    {
                        "artifacts": [
                            {"path": "../../secret", "sha256": "0" * 64, "size": 1}
                        ]
                    }
                ).encode()
            }
            with self.assertRaisesRegex(ExpctlError, "undeclared path"):
                controller.fetch(record["run_id"])


if __name__ == "__main__":
    unittest.main()
