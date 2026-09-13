import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS

from dscnet.experiment.model_audit import (
    _best_validation_metrics,
    _create_version,
    _promote_completed_audit,
    _verify_declared_artifact,
)
from dscnet.experiment.tracking import sha256_file


class ModelAuditTests(unittest.TestCase):
    def test_declared_audit_input_must_match_size_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "control" / "record.json"
            path.parent.mkdir()
            path.write_text("trusted")
            manifest = {
                "artifacts": [
                    {
                        "path": "control/record.json",
                        "size": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                ]
            }
            self.assertEqual(
                _verify_declared_artifact(root, manifest, "control/record.json"),
                path,
            )
            path.write_text("tampered")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                _verify_declared_artifact(root, manifest, "control/record.json")

    def test_best_validation_metrics_use_the_best_dice_step(self):
        client = MagicMock()
        histories = {
            "validation.dice": [
                SimpleNamespace(step=70, value=0.86),
                SimpleNamespace(step=100, value=0.82),
            ],
            "validation.cldice": [SimpleNamespace(step=70, value=0.84)],
            "validation.precision": [SimpleNamespace(step=70, value=0.61)],
            "validation.recall": [SimpleNamespace(step=70, value=0.92)],
            "validation.false_positives": [SimpleNamespace(step=70, value=10)],
            "validation.false_negatives": [SimpleNamespace(step=70, value=4)],
        }
        client.get_metric_history.side_effect = lambda run_id, name: histories.get(name, [])

        step, metrics = _best_validation_metrics(client, "run")

        self.assertEqual(step, 70)
        self.assertEqual(metrics["dice"], 0.86)
        self.assertEqual(metrics["cldice"], 0.84)
        self.assertNotIn("dice_micro", metrics)

    def test_registry_version_is_verified_without_assigning_alias(self):
        client = MagicMock()
        logged = SimpleNamespace(model_id="m-1")
        created = SimpleNamespace(version="1", model_id="m-1", run_id="run-1")
        client.search_model_versions.return_value = []
        client.create_model_version.return_value = created
        client.get_model_version.return_value = created

        _create_version(client, logged_model=logged, training_run_id="run-1")

        client.get_model_version.assert_called_once_with("dscnet-standard", "1")
        client.set_registered_model_alias.assert_not_called()

    def test_existing_registry_version_is_never_overwritten(self):
        client = MagicMock()
        client.create_registered_model.side_effect = MlflowException(
            "exists", RESOURCE_ALREADY_EXISTS
        )
        client.get_registered_model.return_value = SimpleNamespace(name="dscnet-standard")
        client.search_model_versions.return_value = [SimpleNamespace(version="1")]

        with self.assertRaisesRegex(RuntimeError, "refusing v1 overwrite"):
            _create_version(
                client,
                logged_model=SimpleNamespace(model_id="m-new"),
                training_run_id="run-new",
            )

        client.create_model_version.assert_not_called()
        client.set_registered_model_alias.assert_not_called()

    def test_non_existence_registry_error_is_not_hidden(self):
        client = MagicMock()
        client.create_registered_model.side_effect = MlflowException("denied")
        with self.assertRaisesRegex(MlflowException, "denied"):
            _create_version(
                client,
                logged_model=SimpleNamespace(model_id="m-new"),
                training_run_id="run-new",
            )
        client.get_registered_model.assert_not_called()

    def test_champion_alias_is_the_final_operation(self):
        client = MagicMock()
        ready = SimpleNamespace(model_id="m-1", status="READY")
        client.get_run.return_value = SimpleNamespace(
            info=SimpleNamespace(status="FINISHED")
        )
        client.get_logged_model.return_value = ready
        client.search_model_versions.return_value = []
        created = SimpleNamespace(version="1", model_id="m-1", run_id="run-1")
        client.create_model_version.return_value = created
        client.get_model_version.return_value = created
        _promote_completed_audit(
            client,
            audit_run_id="audit-1",
            logged_model_id="m-1",
            training_run_id="run-1",
        )

        self.assertEqual(
            client.mock_calls[-1],
            call.set_registered_model_alias("dscnet-standard", "champion", "1"),
        )

    def test_promotion_failures_never_assign_alias(self):
        for failing_method in (
            "set_logged_model_tags",
            "create_registered_model",
            "create_model_version",
            "get_model_version",
        ):
            with self.subTest(failing_method=failing_method):
                client = MagicMock()
                client.get_run.return_value = SimpleNamespace(
                    info=SimpleNamespace(status="FINISHED")
                )
                client.get_logged_model.return_value = SimpleNamespace(
                    model_id="m-1", status="READY"
                )
                client.search_model_versions.return_value = []
                created = SimpleNamespace(version="1", model_id="m-1", run_id="run-1")
                client.create_model_version.return_value = created
                client.get_model_version.return_value = created
                getattr(client, failing_method).side_effect = RuntimeError("failure")
                with self.assertRaisesRegex(RuntimeError, "failure"):
                    _promote_completed_audit(
                        client,
                        audit_run_id="audit-1",
                        logged_model_id="m-1",
                        training_run_id="run-1",
                    )
                client.set_registered_model_alias.assert_not_called()

    def test_failed_audit_or_model_never_assigns_alias(self):
        for audit_status, model_status in [("FAILED", "READY"), ("FINISHED", "FAILED")]:
            with self.subTest(audit=audit_status, model=model_status):
                client = MagicMock()
                client.get_run.return_value = SimpleNamespace(
                    info=SimpleNamespace(status=audit_status)
                )
                client.get_logged_model.return_value = SimpleNamespace(
                    model_id="m-1", status=model_status
                )
                with self.assertRaises(RuntimeError):
                    _promote_completed_audit(
                        client,
                        audit_run_id="audit-1",
                        logged_model_id="m-1",
                        training_run_id="run-1",
                    )
                client.set_registered_model_alias.assert_not_called()


if __name__ == "__main__":
    unittest.main()
