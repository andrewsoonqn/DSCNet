import random
import sys
import tempfile
import types
import unittest
from pathlib import Path

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from dscnet import workflow

class ActionDispatchTests(unittest.TestCase):
    def _prepared_args(self, root, action):
        paths = {}
        for name in (
            "Meanstd_path",
            "Image_Tr_txt",
            "Label_Tr_txt",
            "Image_Va_txt",
            "Label_Va_txt",
            "Image_Te_txt",
            "Label_Te_txt",
        ):
            path = root / name
            path.touch()
            paths[name] = str(path)
        checkpoint = root / "DSCNet_trial_max"
        checkpoint.touch()
        return SimpleNamespace(
            action=action,
            training_pipeline="standard",
            Dir_Weights=str(root),
            model_name="DSCNet_trial",
            model_name_max=checkpoint.name,
            if_retrain=True,
            seed=2026,
            deterministic=True,
            config_digest="test-digest",
            **paths,
        )

    def _pipeline(self):
        pipeline = types.ModuleType("dscnet.training.standard")
        pipeline.Train = Mock()
        pipeline.Evaluate = Mock()
        return pipeline

    def test_same_seed_reproduces_the_deterministic_smoke_sequence(self):
        def sample():
            return (
                random.random(),
                np.random.random(),
                torch.rand(4),
            )

        workflow.apply_reproducibility(2026, True)
        first = sample()
        workflow.apply_reproducibility(2026, True)
        second = sample()
        self.assertEqual(first[:2], second[:2])
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)

    def test_process_rejects_configuration_without_a_hydra_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._prepared_args(Path(directory), "train")
            del args.config_digest
            with self.assertRaisesRegex(RuntimeError, "Hydra experiment digest"):
                workflow.Process(args)

    def test_train_does_not_evaluate_test_set(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._prepared_args(Path(directory), "train")
            pipeline = self._pipeline()
            with patch.object(workflow, "Create_files"), patch.object(
                workflow, "apply_reproducibility"
            ) as reproducibility, patch.dict(
                sys.modules, {"dscnet.training.standard": pipeline}
            ):
                workflow.Process(args)

            reproducibility.assert_called_once_with(2026, True)
            pipeline.Train.assert_called_once_with(args)
            pipeline.Evaluate.assert_not_called()

    def test_evaluate_does_not_train_or_prepare(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._prepared_args(Path(directory), "evaluate")
            pipeline = self._pipeline()
            forbidden_preparation = types.ModuleType("dscnet.data.normalization")
            forbidden_preparation.Getmeanstd = Mock(
                side_effect=AssertionError("evaluation attempted preprocessing")
            )
            with patch.object(workflow, "Create_files"), patch.object(
                workflow, "apply_reproducibility"
            ), patch.dict(
                sys.modules,
                {
                    "dscnet.training.standard": pipeline,
                    "dscnet.data.normalization": forbidden_preparation,
                },
            ):
                workflow.Process(args)

            pipeline.Evaluate.assert_called_once_with(args)
            pipeline.Train.assert_not_called()
            forbidden_preparation.Getmeanstd.assert_not_called()

    def test_evaluate_requires_frozen_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._prepared_args(Path(directory), "evaluate")
            Path(args.Meanstd_path).unlink()
            with patch.object(workflow, "Create_files"), patch.object(
                workflow, "apply_reproducibility"
            ):
                with self.assertRaisesRegex(FileNotFoundError, "evaluation requires"):
                    workflow.Process(args)

if __name__ == "__main__":
    unittest.main()
