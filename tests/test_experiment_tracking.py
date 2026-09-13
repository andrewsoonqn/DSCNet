import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from types import SimpleNamespace
from unittest.mock import patch

import mlflow
from mlflow import MlflowClient

REPO_ROOT = Path(__file__).parents[1]

from dscnet.experiment.config import load_experiment_config
from dscnet.experiment.tracking import (
    RunRecorder,
    build_consumed_split_manifest,
    build_dataset_manifest,
    collect_git_provenance,
    experiment_digest,
)
from dscnet.experiment.run import (
    _control_evidence,
    _identity_config_yaml,
    run_configured_experiment,
)

class ExperimentTrackingTests(unittest.TestCase):
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

    def _dataset(self, root):
        for split in ("train", "val", "test"):
            image_dir = root / split / "image"
            label_dir = root / split / "label"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            (image_dir / "sample.nii.gz").write_bytes(f"{split}-image".encode())
            (label_dir / "sample.nii.gz").write_bytes(f"{split}-label".encode())

    def test_dataset_manifest_covers_every_pair_and_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._dataset(root)
            first = build_dataset_manifest(root)
            second = build_dataset_manifest(root)

        self.assertEqual(first, second)
        self.assertEqual(set(first["splits"]), {"train", "val", "test"})
        self.assertEqual(sum(map(len, first["splits"].values())), 3)
        self.assertEqual(len(first["digest"]), 64)

    def test_consumed_manifest_binds_ordered_text_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._dataset(root)
            values = {}
            for split, prefix in (("train", "Tr"), ("val", "Va"), ("test", "Te")):
                image = root / split / "image" / "sample.nii.gz"
                label = root / split / "label" / "sample.nii.gz"
                image_manifest = root / f"{split}-images.txt"
                label_manifest = root / f"{split}-labels.txt"
                image_manifest.write_text(f"{image}\n")
                label_manifest.write_text(f"{label}\n")
                values[f"Image_{prefix}_txt"] = str(image_manifest)
                values[f"Label_{prefix}_txt"] = str(label_manifest)
            args = SimpleNamespace(**values)
            first = build_consumed_split_manifest(args, "train")
            Path(args.Image_Tr_txt).write_text(f"{root / 'train' / 'image' / 'sample.nii.gz'}\n\n")
            second = build_consumed_split_manifest(args, "train")

        self.assertNotEqual(
            first["splits"]["train"]["image_manifest_sha256"],
            second["splits"]["train"]["image_manifest_sha256"],
        )
        self.assertNotEqual(first["digest"], second["digest"])
        self.assertEqual(set(first["splits"]), {"train", "val"})

    def test_evaluation_identity_requires_only_test_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._dataset(root)
            image_manifest = root / "test-images.txt"
            label_manifest = root / "test-labels.txt"
            image_manifest.write_text(f"{root / 'test' / 'image' / 'sample.nii.gz'}\n")
            label_manifest.write_text(f"{root / 'test' / 'label' / 'sample.nii.gz'}\n")
            manifest = build_consumed_split_manifest(
                SimpleNamespace(
                    Image_Te_txt=str(image_manifest),
                    Label_Te_txt=str(label_manifest),
                ),
                "evaluate",
            )

        self.assertEqual(set(manifest["splits"]), {"test"})

    def test_dataset_manifest_rejects_unpaired_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._dataset(root)
            (root / "val" / "image" / "unpaired.nii.gz").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "do not match"):
                build_dataset_manifest(root)

    def test_resume_controls_do_not_change_experiment_identity(self):
        _, initial = load_experiment_config()
        _, resumed = load_experiment_config(
            overrides=["training.if_retrain=false", "training.start_train_epoch=2"]
        )
        self.assertEqual(
            _identity_config_yaml(initial), _identity_config_yaml(resumed)
        )

    def test_digest_binds_every_declared_input(self):
        base = experiment_digest("commit", "config", "dataset", "environment")
        self.assertNotEqual(
            base, experiment_digest("commit", "changed", "dataset", "environment")
        )
        self.assertEqual(len(base), 64)

    def test_local_smoke_run_records_replay_evidence_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            store = root / "mlruns"
            self._dataset(dataset)
            manifest = build_dataset_manifest(dataset)
            _, resolved = load_experiment_config(
                overrides=[f"data.data_dir={dataset}", "training.n_epochs=51"]
            )
            tracking_uri = f"sqlite:///{store / 'mlflow.db'}"
            artifact_root = store / "artifacts"

            with RunRecorder(
                tracking_uri=tracking_uri,
                experiment_name="smoke",
                artifact_root=artifact_root,
                run_name="local-smoke",
                experiment_digest_value="digest-value",
                resolved_config=resolved,
                repo_root=REPO_ROOT,
                dataset_manifest=manifest,
            ) as recorder:
                checkpoint = root / "best.pt"
                checkpoint.write_bytes(b"checkpoint")
                log = root / "train.log"
                log.write_text("complete\n")
                recorder.log_metric("train.loss", 0.5, step=1)
                recorder.log_metric("validation.dice", 0.75, step=1)
                recorder.log_final_metrics({"validation.dice": 0.75})
                recorder.log_artifact(checkpoint, "checkpoints")
                recorder.log_artifact(log, "logs")
                run_id = recorder.run_id

            client = MlflowClient(tracking_uri=tracking_uri)
            run = client.get_run(run_id)
            self.assertEqual(run.info.status, "FINISHED")
            self.assertEqual(run.data.tags["dscnet.experiment_digest"], "digest-value")
            self.assertEqual(run.data.metrics["validation.dice"], 0.75)
            self.assertEqual(run.data.params["model.kernel_size"], "5")
            names = {
                item.path
                for item in client.list_artifacts(run_id, path="records")
            }
            self.assertTrue(
                {
                    "records/resolved-config.yaml",
                    "records/git.json",
                    "records/environment.json",
                    "records/dataset-manifest.json",
                }.issubset(names)
            )
            final_path = client.download_artifacts(run_id, "metrics/final-metrics.json")
            self.assertEqual(json.loads(Path(final_path).read_text())["validation.dice"], 0.75)
            self.assertEqual(
                client.list_artifacts(run_id, "checkpoints")[0].path,
                "checkpoints/best.pt",
            )
            self.assertEqual(client.list_artifacts(run_id, "logs")[0].path, "logs/train.log")

    def test_evaluation_run_can_link_to_training_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            self._dataset(dataset)
            manifest = build_dataset_manifest(dataset)
            _, resolved = load_experiment_config(
                overrides=[f"data.data_dir={dataset}", "training.n_epochs=51"]
            )
            tracking_uri = f"sqlite:///{root / 'mlflow.db'}"
            common = {
                "tracking_uri": tracking_uri,
                "experiment_name": "links",
                "artifact_root": root / "artifacts",
                "resolved_config": resolved,
                "repo_root": REPO_ROOT,
                "dataset_manifest": manifest,
            }
            with RunRecorder(run_name="training", **common) as training:
                training_run_id = training.run_id
            with RunRecorder(
                run_name="evaluation",
                parent_run_id=training_run_id,
                **common,
            ) as evaluation:
                evaluation_run_id = evaluation.run_id

            client = MlflowClient(tracking_uri=tracking_uri)
            training_run = client.get_run(training_run_id)
            evaluation_run = client.get_run(evaluation_run_id)
            self.assertNotEqual(training_run_id, evaluation_run_id)
            self.assertEqual(training_run.info.status, "FINISHED")
            self.assertEqual(
                evaluation_run.data.tags["dscnet.parent_run_id"], training_run_id
            )

    def test_untracked_source_content_is_captured_in_dirty_patch(self):
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
            (root / "tracked.txt").write_text("tracked\n")
            subprocess.run(["git", "-C", str(root), "add", "tracked.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "initial"], check=True
            )
            (root / "new-config.yaml").write_text("important: true\n")
            provenance = collect_git_provenance(root)
            (root / "new-config.yaml").unlink()
            apply_check = subprocess.run(
                ["git", "-C", str(root), "apply", "--check", "-"],
                input=provenance["patch"],
                text=True,
                capture_output=True,
            )

        self.assertEqual(apply_check.returncode, 0, apply_check.stderr)
        self.assertTrue(provenance["dirty"])
        self.assertIn("important: true", provenance["patch"])
        self.assertEqual(provenance["untracked_source"][0]["path"], "new-config.yaml")

    def test_recorder_marks_setup_failure_as_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            self._dataset(dataset)
            _, resolved = load_experiment_config(
                overrides=[f"data.data_dir={dataset}", "training.n_epochs=51"]
            )
            tracking_uri = f"sqlite:///{root / 'mlflow.db'}"
            recorder = RunRecorder(
                tracking_uri=tracking_uri,
                experiment_name="failures",
                artifact_root=root / "artifacts",
                run_name="broken",
                resolved_config=resolved,
                repo_root=REPO_ROOT,
                dataset_manifest=build_dataset_manifest(dataset),
            )
            with patch.object(
                recorder, "_log_run_records", side_effect=RuntimeError("broken")
            ):
                with self.assertRaisesRegex(RuntimeError, "broken"):
                    recorder.__enter__()
            run = MlflowClient(tracking_uri=tracking_uri).get_run(recorder.run_id)
            self.assertEqual(run.info.status, "FAILED")

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

    def test_unknown_metric_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            self._dataset(dataset)
            _, resolved = load_experiment_config(
                overrides=[f"data.data_dir={dataset}", "training.n_epochs=51"]
            )
            with RunRecorder(
                tracking_uri=f"sqlite:///{root / 'mlflow.db'}",
                experiment_name="metrics",
                artifact_root=root / "artifacts",
                run_name="metric-check",
                resolved_config=resolved,
                repo_root=REPO_ROOT,
                dataset_manifest=build_dataset_manifest(dataset),
            ) as recorder:
                with self.assertRaisesRegex(ValueError, "unregistered metric"):
                    recorder.log_metric("dice", 1.0)

if __name__ == "__main__":
    unittest.main()
