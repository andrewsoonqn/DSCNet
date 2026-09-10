import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hydra.errors import ConfigCompositionException

MODULE_DIR = (
    Path(__file__).parents[1]
    / "DSCNet_3D_opensource"
    / "Code"
    / "Kipa"
    / "DSCNet"
)
sys.path.insert(0, str(MODULE_DIR))

import S0_Main
from S4_Experiment_Config import (
    load_experiment_config,
    resolved_yaml,
    to_runtime_namespace,
)


class ExperimentConfigTests(unittest.TestCase):
    def test_unknown_keys_fail_during_composition(self):
        for override in (
            "unknown=true",
            "model.kernl_size=5",
            "data.unknown=true",
            "training.unknown=true",
            "runtime.unknown=true",
        ):
            with self.subTest(override=override):
                with self.assertRaises(ConfigCompositionException):
                    load_experiment_config(overrides=[override])

    def test_fixed_kernel_rejects_unsupported_value(self):
        with self.assertRaisesRegex(ValueError, "kernel_size is fixed"):
            load_experiment_config(overrides=["model.kernel_size=7"])

    def test_roi_must_match_unet_downsampling_depth(self):
        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            load_experiment_config(overrides=["training.ROI_shape=[66,64,64]"])

    def test_positive_training_values_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            load_experiment_config(overrides=["training.batch_size=0"])

    def test_formal_runs_reject_dirty_policy(self):
        with self.assertRaisesRegex(ValueError, "formal runs cannot allow"):
            load_experiment_config(
                overrides=["runtime.formal=true", "runtime.allow_dirty=true"]
            )

    def test_mlflow_ui_location_and_timeout_are_bounded(self):
        for location in ("login", "slurm"):
            config, _ = load_experiment_config(
                overrides=[f"runtime.mlflow.ui.location={location}"]
            )
            self.assertEqual(config.runtime.mlflow.ui.location, location)
        with self.assertRaisesRegex(ValueError, "location must be login or slurm"):
            load_experiment_config(overrides=["runtime.mlflow.ui.location=public"])
        for timeout in (1, 480):
            config, _ = load_experiment_config(
                overrides=[f"runtime.mlflow.ui.timeout_minutes={timeout}"]
            )
            self.assertEqual(config.runtime.mlflow.ui.timeout_minutes, timeout)
        for timeout in (0, 481):
            with self.subTest(timeout=timeout):
                with self.assertRaisesRegex(ValueError, "timeout_minutes"):
                    load_experiment_config(
                        overrides=[f"runtime.mlflow.ui.timeout_minutes={timeout}"]
                    )

    def test_named_optimized_experiment_composes(self):
        config, _ = load_experiment_config("experiment/dscnet_optimized")
        self.assertEqual(config.model.training_pipeline, "optimized")

    def test_optimized_pipeline_rejects_ignored_depth(self):
        with self.assertRaisesRegex(ValueError, "fixed unet_layers"):
            load_experiment_config(
                "experiment/dscnet_optimized", overrides=["model.unet_layers=3"]
            )

    def test_resolved_yaml_contains_every_effective_group(self):
        config, resolved = load_experiment_config()
        text = resolved_yaml(resolved)
        self.assertIn("kernel_size: 5", text)
        self.assertIn("ROI_shape:", text)
        self.assertIn("seed: 2026", text)
        self.assertIn("deterministic: true", text)
        args = to_runtime_namespace(config)
        self.assertEqual(args.kernel_size, 5)
        self.assertEqual(args.seed, 2026)
        self.assertTrue(args.deterministic)

    def test_invalid_training_schedule_fails(self):
        with self.assertRaisesRegex(ValueError, "start_train_epoch"):
            load_experiment_config(overrides=["training.start_train_epoch=101"])
        with self.assertRaisesRegex(ValueError, "rlr_factor"):
            load_experiment_config(overrides=["training.rlr_factor=1.0"])

    def test_action_artifacts_fail_before_output_directories_are_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                action="train",
                config_digest="test-digest",
                Meanstd_path=str(root / "missing-meanstd.npy"),
                Image_Tr_txt=str(root / "missing-train-images.txt"),
                Label_Tr_txt=str(root / "missing-train-labels.txt"),
                Image_Va_txt=str(root / "missing-val-images.txt"),
                Label_Va_txt=str(root / "missing-val-labels.txt"),
            )
            with patch.object(S0_Main, "Create_files") as create_files:
                with self.assertRaisesRegex(FileNotFoundError, "training requires"):
                    S0_Main.Process(args)
            create_files.assert_not_called()

    def test_evaluation_requires_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            required = []
            for name in ("mean.npy", "images.txt", "labels.txt"):
                path = root / name
                path.touch()
                required.append(str(path))
            args = SimpleNamespace(
                action="evaluate",
                Meanstd_path=required[0],
                Image_Te_txt=required[1],
                Label_Te_txt=required[2],
                Dir_Weights=str(root),
                model_name="latest.pt",
                model_name_max="best.pt",
            )
            with self.assertRaisesRegex(FileNotFoundError, "one checkpoint"):
                S0_Main.validate_action_artifacts(args)

    def test_resumed_training_requires_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            required = []
            for name in ("mean.npy", "train-images.txt", "train-labels.txt", "val-images.txt", "val-labels.txt"):
                path = root / name
                path.touch()
                required.append(str(path))
            args = SimpleNamespace(
                action="train",
                Meanstd_path=required[0],
                Image_Tr_txt=required[1],
                Label_Tr_txt=required[2],
                Image_Va_txt=required[3],
                Label_Va_txt=required[4],
                if_retrain=False,
                Dir_Weights=str(root),
                model_name="latest.pt",
            )
            with self.assertRaisesRegex(FileNotFoundError, "resumed training"):
                S0_Main.validate_action_artifacts(args)

    def test_prepare_requires_all_split_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                action="prepare",
                Tr_Image_dir=str(root / "train-image"),
                Tr_Label_dir=str(root / "train-label"),
                Va_Image_dir=str(root / "val-image"),
                Va_Label_dir=str(root / "val-label"),
                Te_Image_dir=str(root / "test-image"),
                Te_Label_dir=str(root / "test-label"),
            )
            with self.assertRaisesRegex(FileNotFoundError, "dataset directories"):
                S0_Main.validate_action_artifacts(args)


if __name__ == "__main__":
    unittest.main()
