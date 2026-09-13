import json
from pathlib import Path
import unittest

import torch

from dscnet.experiment.config import load_experiment_config, to_runtime_namespace
from dscnet.models.factory import build_model


FIXTURES = Path(__file__).parent / "fixtures"


class ModelFactoryTests(unittest.TestCase):
    def _args(self, experiment):
        config, _ = load_experiment_config(f"experiment/{experiment}")
        return to_runtime_namespace(config)

    def test_standard_factory_preserves_checkpoint_parameter_contract(self):
        model = build_model(self._args("dscnet_standard"), "standard", torch.device("cpu"))
        expected = json.loads((FIXTURES / "standard_state_dict_keys.json").read_text())
        self.assertEqual(list(model.state_dict()), expected)

    def test_optimized_factory_preserves_checkpoint_parameter_contract(self):
        model = build_model(self._args("dscnet_optimized"), "optimized", torch.device("cpu"))
        expected = json.loads((FIXTURES / "optimized_state_dict_keys.json").read_text())
        self.assertEqual(list(model.state_dict()), expected)

    def test_unknown_pipeline_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported training pipeline"):
            build_model(self._args("dscnet_standard"), "unknown", torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
