import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import arbor_eval


class ArborEvaluationTests(unittest.TestCase):
    def _completed(self, value, code=0):
        return subprocess.CompletedProcess(
            [], code, stdout=json.dumps(value), stderr=""
        )

    def test_adapter_submits_formal_training_and_returns_only_validation_score(self):
        responses = [
            self._completed({"run_id": "a" * 16}),
            self._completed({"state": "PENDING"}),
            self._completed({"state": "COMPLETED"}),
            self._completed({"destination": "/fetched"}),
            self._completed(
                {
                    "selection": {
                        "metric": "validation.dice",
                        "value": 0.86,
                    },
                    "logged_model_id": "m-hidden",
                }
            ),
        ]
        with tempfile.TemporaryDirectory() as directory, patch(
            "arbor_eval.subprocess.run", side_effect=responses
        ) as command, patch("arbor_eval.time.sleep"):
            normalization = Path(directory) / arbor_eval.NORMALIZATION_NAME
            normalization.write_bytes(b"training normalization")
            result = arbor_eval.evaluate(directory, poll_seconds=0)

        self.assertEqual(
            result,
            {
                "schema_version": 1,
                "status": "completed",
                "controller_run_id": "a" * 16,
                "metric": "validation.dice",
                "score": 0.86,
            },
        )
        submit = command.call_args_list[0].args[0]
        self.assertIn("action=train", submit)
        self.assertIn(f"data.Meanstd_path={normalization.resolve()}", submit)
        self.assertIn("runtime.formal=true", submit)
        self.assertIn("runtime.allow_dirty=false", submit)
        self.assertIn("runtime.mlflow.isolated_run_store=true", submit)
        invoked_operations = [call.args[0][2] for call in command.call_args_list]
        self.assertEqual(
            invoked_operations, ["submit", "status", "status", "fetch", "result"]
        )
        self.assertNotIn("audit", invoked_operations)
        self.assertNotIn("logged_model_id", result)

    def test_failed_training_never_fetches_or_reads_results(self):
        responses = [
            self._completed({"run_id": "a" * 16}),
            self._completed({"state": "FAILED"}),
        ]
        with tempfile.TemporaryDirectory() as directory, patch(
            "arbor_eval.subprocess.run", side_effect=responses
        ) as command:
            (Path(directory) / arbor_eval.NORMALIZATION_NAME).write_bytes(
                b"training normalization"
            )
            with self.assertRaisesRegex(arbor_eval.ArborEvaluationError, "FAILED"):
                arbor_eval.evaluate(directory)
        invoked_operations = [call.args[0][2] for call in command.call_args_list]
        self.assertEqual(invoked_operations, ["submit", "status"])

    def test_cancelled_training_with_slurm_actor_suffix_is_terminal(self):
        responses = [
            self._completed({"run_id": "a" * 16}),
            self._completed({"state": "CANCELLED by 54098"}),
        ]
        with tempfile.TemporaryDirectory() as directory, patch(
            "arbor_eval.subprocess.run", side_effect=responses
        ) as command:
            (Path(directory) / arbor_eval.NORMALIZATION_NAME).write_bytes(
                b"training normalization"
            )
            with self.assertRaisesRegex(arbor_eval.ArborEvaluationError, "CANCELLED"):
                arbor_eval.evaluate(directory)
        invoked_operations = [call.args[0][2] for call in command.call_args_list]
        self.assertEqual(invoked_operations, ["submit", "status"])

    def test_adapter_requires_normalization_beside_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                arbor_eval.ArborEvaluationError, arbor_eval.NORMALIZATION_NAME
            ):
                arbor_eval.evaluate(directory)

    def test_adapter_has_no_audit_or_test_operation(self):
        source = Path(arbor_eval.__file__).read_text()
        self.assertNotIn('_run_expctl(["audit"', source)
        self.assertNotIn("action=evaluate", source)
        self.assertNotIn("test.dice", source)


if __name__ == "__main__":
    unittest.main()
