import json
import tempfile
import unittest
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
import numpy as np
from omegaconf import OmegaConf
import torch

from dscnet.experiment.config import load_experiment_config, to_runtime_namespace
from dscnet.experiment.modeling import (
    architecture_text,
    model_signature,
    save_model_package,
    validate_model_package_fresh,
)
from dscnet.models.factory import build_model
from dscnet.training.checkpoints import save_model_checkpoint


REPO_ROOT = Path(__file__).parents[2]


class ModelPackagingTests(unittest.TestCase):
    def _small_standard(self):
        config, resolved = load_experiment_config(
            "experiment/dscnet_standard",
            [
                "model.n_basic_layer=4",
                "model.dim=4",
                "model.unet_layers=3",
                "training.ROI_shape=[8,8,8]",
                "training.predict_batch_size=1",
            ],
        )
        return config, resolved, to_runtime_namespace(config)

    def test_signature_is_one_volume_to_class_probabilities(self):
        signature = model_signature(2)
        self.assertEqual(signature.inputs.inputs[0].shape, (-1, -1, -1))
        self.assertEqual(signature.outputs.inputs[0].shape, (2, -1, -1, -1))

    def test_package_round_trip_preserves_probabilities_and_human_architecture(self):
        _, resolved, args = self._small_standard()
        model = build_model(args, "standard", torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            save_model_checkpoint(model, checkpoint, "standard")
            normalization = root / "normalization.npy"
            np.save(normalization, np.array([0.5, 2.0], dtype=np.float32))
            config_path = root / "resolved.yaml"
            config_path.write_text(OmegaConf.to_yaml(resolved, resolve=True, sort_keys=True))
            package = root / "model"
            evidence = save_model_package(
                package,
                checkpoint_path=checkpoint,
                resolved_config_path=config_path,
                normalization_path=normalization,
                provenance={"training_run_id": "run-1", "git_commit": "abc"},
                project_root=REPO_ROOT,
            )
            parity = validate_model_package_fresh(
                package,
                checkpoint_path=checkpoint,
                resolved_config_path=config_path,
                normalization_path=normalization,
                sample_volume=np.zeros((8, 8, 8), dtype=np.float32),
            )
            loaded = mlflow.pyfunc.load_model(str(package))
            prediction = np.asarray(loaded.predict(np.zeros((8, 8, 8), dtype=np.float32)))
            with self.assertRaises(MlflowException):
                loaded.predict(np.zeros((8, 8, 8), dtype=np.float64))
            architecture_path = next(package.rglob("architecture.txt"))
            architecture = architecture_path.read_text()

        self.assertEqual(prediction.shape, (2, 8, 8, 8))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue(parity["masks_equal"])
        self.assertTrue(parity["fresh_process"])
        self.assertIn("/code/dscnet/experiment/modeling.py", parity["bundled_module_origin"])
        self.assertEqual(evidence["pipeline"], "standard")
        self.assertEqual(evidence["training_run_id"], "run-1")
        self.assertIn('"total_parameters"', architecture)
        self.assertIn("DSCNet(", architecture)

    def test_architecture_text_uses_actual_model(self):
        _, _, args = self._small_standard()
        model = build_model(args, "standard", torch.device("cpu"))
        summary = architecture_text(model, args)
        header = json.loads(summary.split("\n\n", 1)[0])
        self.assertEqual(header["pipeline"], "standard")
        self.assertEqual(header["total_parameters"], sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
