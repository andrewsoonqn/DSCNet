import sys
import tempfile
import unittest
from pathlib import Path

import torch

MODULE_DIR = (
    Path(__file__).parents[1]
    / "DSCNet_3D_opensource"
    / "Code"
    / "Kipa"
    / "DSCNet"
)
sys.path.insert(0, str(MODULE_DIR))

from S3_Checkpoint import load_model_checkpoint, save_model_checkpoint


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_current_checkpoint_round_trip(self):
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model_checkpoint(source, path, pipeline="standard")
            load_model_checkpoint(target, path, pipeline="standard")

        for source_parameter, target_parameter in zip(
            source.parameters(), target.parameters()
        ):
            torch.testing.assert_close(source_parameter, target_parameter)

    def test_rejects_legacy_bare_state_dict(self):
        model = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(model.state_dict(), path)
            with self.assertRaisesRegex(RuntimeError, "incompatible checkpoint"):
                load_model_checkpoint(model, path, pipeline="standard")

    def test_rejects_checkpoint_from_other_pipeline(self):
        model = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimized.pt"
            save_model_checkpoint(model, path, pipeline="optimized")
            with self.assertRaisesRegex(RuntimeError, "incompatible checkpoint"):
                load_model_checkpoint(model, path, pipeline="standard")


if __name__ == "__main__":
    unittest.main()
