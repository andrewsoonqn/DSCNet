import inspect
import json
from pathlib import Path
import tempfile
import unittest

from dscnet.experiment.model_evaluation import (
    AUDIT_PROTOCOL_VERSION,
    audit_key,
    audit_logged_model,
    start_final_test,
)


class ModelEvaluationTests(unittest.TestCase):
    def test_audit_key_binds_protocol_run_and_logged_model(self):
        initial = audit_key("controller-1", "m-one", "a" * 64)
        self.assertEqual(initial, audit_key("controller-1", "m-one", "a" * 64))
        self.assertNotEqual(initial, audit_key("controller-2", "m-one", "a" * 64))
        self.assertNotEqual(initial, audit_key("controller-1", "m-two", "a" * 64))
        self.assertNotEqual(initial, audit_key("controller-1", "m-one", "b" * 64))
        self.assertEqual(AUDIT_PROTOCOL_VERSION, 1)

    def test_final_test_marker_is_written_before_reuse_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = start_final_test(
                directory,
                audit_key_value="audit-key",
                controller_run_id="controller-1",
                logged_model_id="m-one",
            )
            evidence = json.loads(marker.read_text())
            self.assertEqual(evidence["audit_key"], "audit-key")
            with self.assertRaisesRegex(RuntimeError, "already started"):
                start_final_test(
                    directory,
                    audit_key_value="audit-key",
                    controller_run_id="controller-1",
                    logged_model_id="m-one",
                )

    def test_reusable_audit_has_no_registry_or_alias_mutation(self):
        source = inspect.getsource(audit_logged_model)
        for forbidden in (
            "create_registered_model",
            "create_model_version",
            "set_registered_model_alias",
            "set_logged_model_tags",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
