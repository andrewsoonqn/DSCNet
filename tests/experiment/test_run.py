import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlflow import MlflowClient

from dscnet.experiment.config import load_experiment_config
from dscnet.experiment.run import (
    _control_evidence,
    _identity_config_yaml,
    run_configured_experiment,
)


class ExperimentRunTests(unittest.TestCase):
    def _dataset(self, root):
        for split in ("train", "val", "test"):
            image_dir = root / split / "image"
            label_dir = root / split / "label"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            (image_dir / "sample.nii.gz").write_bytes(f"{split}-image".encode())
            (label_dir / "sample.nii.gz").write_bytes(f"{split}-label".encode())

    def test_runtime_verifies_the_checkpoint_file_evaluation_consumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "control"
            weights = root / "weights"
            control.mkdir()
            weights.mkdir()
            checkpoint = weights / "best.pth"
            checkpoint.write_bytes(b"expected")
            evidence = {
                "name": checkpoint.name,
                "size": checkpoint.stat().st_size,
                "sha256": hashlib.sha256(b"expected").hexdigest(),
            }
            (control / "dataset-manifest.json").write_text("{}")
            (control / "git.json").write_text("{}")
            (control / "source-manifest.json").write_text("{}")
            (control / "environment-lock.json").write_text("{}")
            (control / "run-manifest.json").write_text(
                json.dumps({"run_id": "abc", "evaluation_checkpoint": evidence})
            )
            checkpoint.write_bytes(b"tampered")
            with patch.dict("os.environ", {"DSCNET_CONTROL_DIR": str(control)}):
                with self.assertRaisesRegex(RuntimeError, "evaluation_checkpoint bytes"):
                    _control_evidence({"data": {"Dir_Weights": str(weights)}})

    def test_resume_controls_do_not_change_experiment_identity(self):
        _, initial = load_experiment_config()
        _, resumed = load_experiment_config(
            overrides=["training.if_retrain=false", "training.start_train_epoch=2"]
        )
        self.assertEqual(
            _identity_config_yaml(initial), _identity_config_yaml(resumed)
        )

    def test_configured_runner_attaches_digest_and_tracker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            self._dataset(dataset)
            overrides = [
                f"runtime.mlflow.tracking_uri=sqlite:///{root / 'mlflow.db'}",
                f"runtime.mlflow.artifact_root={root / 'artifacts'}",
                f"data.Dir_Log={root / 'logs'}",
                f"data.Dir_Weights={root / 'weights'}",
            ]
            for split, prefix in (("train", "Tr"), ("val", "Va"), ("test", "Te")):
                image_manifest = root / f"{split}-images.txt"
                label_manifest = root / f"{split}-labels.txt"
                image_manifest.write_text(
                    f"{dataset / split / 'image' / 'sample.nii.gz'}\n"
                )
                label_manifest.write_text(
                    f"{dataset / split / 'label' / 'sample.nii.gz'}\n"
                )
                overrides.extend(
                    [
                        f"data.Image_{prefix}_txt={image_manifest}",
                        f"data.Label_{prefix}_txt={label_manifest}",
                    ]
                )
            with patch("dscnet.experiment.run.Process", return_value=None) as process:
                result = run_configured_experiment(overrides=overrides)

            args = process.call_args.args[0]
            self.assertEqual(args.config_digest, result["experiment_digest"])
            self.assertEqual(args.tracker.run_id, result["run_id"])
            run = MlflowClient(
                tracking_uri=f"sqlite:///{root / 'mlflow.db'}"
            ).get_run(result["run_id"])
            self.assertEqual(
                run.data.tags["dscnet.experiment_digest"], result["experiment_digest"]
            )


if __name__ == "__main__":
    unittest.main()
