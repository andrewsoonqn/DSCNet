import unittest
from pathlib import Path

import numpy as np
import torch

from dscnet.evaluation.sliding_window import axis_starts, sliding_window_logits

class TwoClassIdentity(torch.nn.Module):
    def forward(self, image):
        return torch.cat((image, image * 2), dim=1)

class NonFiniteModel(torch.nn.Module):
    def forward(self, image):
        return torch.full(
            (image.shape[0], 2, *image.shape[2:]),
            float("nan"),
            device=image.device,
        )

class SlidingWindowTests(unittest.TestCase):
    def test_axis_starts_snap_to_awkward_far_edge(self):
        self.assertEqual(axis_starts(9, 4), (0, 2, 4, 5))
        coverage = np.zeros(9, dtype=bool)
        for start in axis_starts(9, 4):
            coverage[start : start + 4] = True
        self.assertTrue(coverage.all())

    def test_awkward_volume_is_finite_and_exact_at_every_voxel(self):
        image = np.arange(5 * 7 * 9, dtype=np.float32).reshape(5, 7, 9) / 100
        logits = sliding_window_logits(
            TwoClassIdentity(),
            image,
            (4, 4, 4),
            n_classes=2,
            batch_size=3,
            device=torch.device("cpu"),
        )
        self.assertEqual(logits.shape, (2, 5, 7, 9))
        self.assertTrue(np.isfinite(logits).all())
        np.testing.assert_allclose(logits[0], image, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(logits[1], image * 2, rtol=1e-5, atol=1e-6)

    def test_volume_smaller_than_roi_is_padded_then_cropped(self):
        image = np.ones((2, 3, 4), dtype=np.float32)
        logits = sliding_window_logits(
            TwoClassIdentity(),
            image,
            (4, 4, 4),
            n_classes=2,
            batch_size=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(logits.shape, (2, 2, 3, 4))
        np.testing.assert_allclose(logits[0], image)

    def test_non_finite_model_output_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            sliding_window_logits(
                NonFiniteModel(),
                np.ones((4, 4, 4), dtype=np.float32),
                (4, 4, 4),
                n_classes=2,
                batch_size=1,
                device=torch.device("cpu"),
            )

if __name__ == "__main__":
    unittest.main()
