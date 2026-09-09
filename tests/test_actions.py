import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

MODULE_DIR = (
    Path(__file__).parents[1]
    / "DSCNet_3D_opensource"
    / "Code"
    / "Kipa"
    / "DSCNet"
)
sys.path.insert(0, str(MODULE_DIR))

import S0_Main


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
        return SimpleNamespace(
            action=action,
            training_pipeline="standard",
            **paths,
        )

    def _pipeline(self):
        pipeline = types.ModuleType("S3_Train_Process")
        pipeline.Train = Mock()
        pipeline.Evaluate = Mock()
        return pipeline

    def test_train_does_not_evaluate_test_set(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._prepared_args(Path(directory), "train")
            pipeline = self._pipeline()
            with patch.object(S0_Main, "Create_files"), patch.dict(
                sys.modules, {"S3_Train_Process": pipeline}
            ):
                S0_Main.Process(args)

            pipeline.Train.assert_called_once_with(args)
            pipeline.Evaluate.assert_not_called()

    def test_evaluate_does_not_train_or_prepare(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._prepared_args(Path(directory), "evaluate")
            pipeline = self._pipeline()
            forbidden_preparation = types.ModuleType("S1_Pre_Getmeanstd")
            forbidden_preparation.Getmeanstd = Mock(
                side_effect=AssertionError("evaluation attempted preprocessing")
            )
            with patch.object(S0_Main, "Create_files"), patch.dict(
                sys.modules,
                {
                    "S3_Train_Process": pipeline,
                    "S1_Pre_Getmeanstd": forbidden_preparation,
                },
            ):
                S0_Main.Process(args)

            pipeline.Evaluate.assert_called_once_with(args)
            pipeline.Train.assert_not_called()
            forbidden_preparation.Getmeanstd.assert_not_called()

    def test_evaluate_requires_frozen_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._prepared_args(Path(directory), "evaluate")
            Path(args.Meanstd_path).unlink()
            with patch.object(S0_Main, "Create_files"):
                with self.assertRaisesRegex(FileNotFoundError, "evaluation requires"):
                    S0_Main.Process(args)


if __name__ == "__main__":
    unittest.main()
