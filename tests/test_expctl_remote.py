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
from unittest.mock import patch

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
