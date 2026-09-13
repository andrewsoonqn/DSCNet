import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

from mlflow.entities import LoggedModelStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS
from mlflow.tracking import MlflowClient

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
        logged = SimpleNamespace(model_id="m-1", artifact_location="/models/m-1")
        created = SimpleNamespace(
            version="1",
            source="/models/m-1",
            run_id="run-1",
            tags={"dscnet.logged_model_id": "m-1"},
        )
        client.search_model_versions.return_value = []
        client.create_model_version.return_value = created
        client.get_model_version.return_value = created

        _create_version(client, logged_model=logged, training_run_id="run-1")

        client.create_model_version.assert_called_once_with(
            name="dscnet-standard",
            source="/models/m-1",
            run_id="run-1",
            model_id="m-1",
            tags={
                "dscnet.audit": "passed",
                "dscnet.logged_model_id": "m-1",
                "dscnet.test_evaluation": "reported-not-selected",
                "dscnet.training_run_id": "run-1",
            },
            description=(
                "Provisional baseline selected by validation performance; untouched test "
                "metrics are reported in the linked audit and were not used for selection."
            ),
        )
        client.get_model_version.assert_called_once_with("dscnet-standard", "1")
        client.set_registered_model_alias.assert_not_called()

    def test_sqlite_registry_promotes_logged_model_artifact_location(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uri = f"sqlite:///{root / 'mlflow.db'}"
            client = MlflowClient(tracking_uri=uri, registry_uri=uri)
            experiment_id = client.create_experiment(
                "model-audit", artifact_location=str(root / "artifacts")
            )
            run_id = client.create_run(experiment_id).info.run_id
            logged = client.create_logged_model(
                experiment_id=experiment_id,
                name="audit-candidate",
                source_run_id=run_id,
                model_type="pyfunc",
            )
            client.finalize_logged_model(
                logged.model_id, status=LoggedModelStatus.READY
            )

            version = _create_version(
                client, logged_model=logged, training_run_id=run_id
            )
            stored = client.get_model_version("dscnet-standard", "1")

            self.assertEqual(str(version.version), "1")
            self.assertEqual(stored.source, logged.artifact_location)
            self.assertEqual(stored.run_id, run_id)
            self.assertEqual(stored.tags["dscnet.logged_model_id"], logged.model_id)

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
                logged_model=SimpleNamespace(
                    model_id="m-new", artifact_location="/models/m-new"
                ),
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
                logged_model=SimpleNamespace(
                    model_id="m-new", artifact_location="/models/m-new"
                ),
                training_run_id="run-new",
            )
        client.get_registered_model.assert_not_called()

    def test_champion_alias_is_the_final_operation(self):
        client = MagicMock()
        ready = SimpleNamespace(
            model_id="m-1", status="READY", artifact_location="/models/m-1"
        )
        client.get_run.return_value = SimpleNamespace(
            info=SimpleNamespace(status="FINISHED")
        )
        client.get_logged_model.return_value = ready
        client.search_model_versions.return_value = []
        created = SimpleNamespace(
            version="1",
            source="/models/m-1",
            run_id="run-1",
            tags={"dscnet.logged_model_id": "m-1"},
        )
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
                    model_id="m-1",
                    status="READY",
                    artifact_location="/models/m-1",
                )
                client.search_model_versions.return_value = []
                created = SimpleNamespace(
                    version="1",
                    source="/models/m-1",
                    run_id="run-1",
                    tags={"dscnet.logged_model_id": "m-1"},
                )
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

    def test_persisted_version_mismatches_never_assign_alias(self):
        mismatches = (
            {"source": "/wrong", "run_id": "run-1", "logged_id": "m-1"},
            {"source": "/models/m-1", "run_id": "wrong", "logged_id": "m-1"},
            {"source": "/models/m-1", "run_id": "run-1", "logged_id": "wrong"},
        )
        for mismatch in mismatches:
            with self.subTest(mismatch=mismatch):
                client = MagicMock()
                client.get_run.return_value = SimpleNamespace(
                    info=SimpleNamespace(status="FINISHED")
                )
                client.get_logged_model.return_value = SimpleNamespace(
                    model_id="m-1",
                    status="READY",
                    artifact_location="/models/m-1",
                )
                client.search_model_versions.return_value = []
                client.create_model_version.return_value = SimpleNamespace(version="1")
                client.get_model_version.return_value = SimpleNamespace(
                    version="1",
                    source=mismatch["source"],
                    run_id=mismatch["run_id"],
                    tags={"dscnet.logged_model_id": mismatch["logged_id"]},
                )
                with self.assertRaisesRegex(RuntimeError, "audited source"):
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
