import json
import tempfile
import unittest
from pathlib import Path

import torch

from dscnet.models.optimized import DSCNet
from dscnet.training.checkpoints import save_model_checkpoint


FIXTURE = Path(__file__).parent / "fixtures" / "optimized_state_dict_keys.json"


class OptimizedModelContractTests(unittest.TestCase):
    def test_model_keys_and_checkpoint_metadata_match_legacy_contract(self):
        model = DSCNet(1, 2, 9, 1.0, True, "cpu", 4, 4, epochs=1)
        expected_keys = json.loads(FIXTURE.read_text())
        self.assertEqual(list(model.state_dict()), expected_keys)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimized.pt"
            save_model_checkpoint(model, path, pipeline="optimized")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        self.assertEqual(
            list(checkpoint),
            [
                "format_version",
                "sampler_implementation",
                "pipeline",
                "model_state_dict",
            ],
        )
        self.assertEqual(checkpoint["format_version"], 2)
        self.assertEqual(checkpoint["sampler_implementation"], "grid_sample_v1")
        self.assertEqual(checkpoint["pipeline"], "optimized")
        self.assertEqual(list(checkpoint["model_state_dict"]), expected_keys)


if __name__ == "__main__":
    unittest.main()
