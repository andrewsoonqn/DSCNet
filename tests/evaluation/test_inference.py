import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dscnet.evaluation.inference import load_normalization, predict_probabilities


class FixedProbabilityModel(torch.nn.Module):
    def forward(self, inputs):
        foreground = torch.sigmoid(inputs)
        return torch.cat((1 - foreground, foreground), dim=1)


class InferenceTests(unittest.TestCase):
    def test_raw_volume_is_normalized_and_returns_finite_probabilities(self):
        volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
        result = predict_probabilities(
            FixedProbabilityModel(),
            volume,
            mean=13,
            std=2,
            roi_shape=(4, 4, 4),
            n_classes=2,
            batch_size=2,
            device=torch.device("cpu"),
        )

        expected_foreground = 1 / (1 + np.exp(-((volume.astype(np.float32) - 13) / 2)))
        self.assertEqual(result.shape, (2, 3, 3, 3))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_allclose(result[1], expected_foreground, rtol=1e-6)
        np.testing.assert_allclose(result.sum(axis=0), 1.0, atol=1e-6)

    def test_invalid_volume_and_normalization_are_rejected(self):
        common = {
            "model": FixedProbabilityModel(),
            "mean": 0,
            "std": 1,
            "roi_shape": (2, 2, 2),
            "n_classes": 2,
            "batch_size": 1,
            "device": torch.device("cpu"),
        }
        with self.assertRaisesRegex(ValueError, "three-dimensional"):
            predict_probabilities(volume=np.zeros((2, 2)), **common)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            predict_probabilities(volume=np.full((2, 2, 2), np.nan), **common)
        for invalid_std in (0, -1):
            with self.subTest(std=invalid_std), self.assertRaisesRegex(
                ValueError, "positive"
            ):
                predict_probabilities(
                    volume=np.zeros((2, 2, 2)),
                    **{**common, "std": invalid_std},
                )

    def test_normalization_file_has_exact_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normalization.npy"
            np.save(path, np.array([2.0, 4.0], dtype=np.float32))
            self.assertEqual(load_normalization(path), (2.0, 4.0))
            np.save(path, np.array([2.0], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "exactly"):
                load_normalization(path)


if __name__ == "__main__":
    unittest.main()
