import math
import sys
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

from S3_Loss import categorical_entropy, entropy_loss


class CategoricalEntropyTests(unittest.TestCase):
    def test_uniform_two_class_distribution_is_log_two(self):
        probabilities = torch.full((2, 2, 3, 4, 5), 0.5)
        actual = categorical_entropy(probabilities)
        self.assertAlmostEqual(actual.item(), math.log(2), places=6)

    def test_deterministic_distribution_has_near_zero_entropy(self):
        probabilities = torch.zeros((1, 2, 2, 3, 4))
        probabilities[:, 0] = 1.0
        actual = categorical_entropy(probabilities)
        self.assertLess(actual.item(), 1e-6)

    def test_entropy_is_independent_of_spatial_shape(self):
        compact = torch.tensor([[[[[0.25]]], [[[0.75]]]]])
        tiled = compact.expand(3, 2, 4, 5, 6)
        torch.testing.assert_close(
            categorical_entropy(compact), categorical_entropy(tiled)
        )

    def test_entropy_module_uses_the_same_class_axis(self):
        probabilities = torch.full((1, 2, 2, 3, 7), 0.5)
        module = entropy_loss()
        self.assertAlmostEqual(
            module(probabilities, epoch=1).item(), math.log(2), places=6
        )


if __name__ == "__main__":
    unittest.main()
