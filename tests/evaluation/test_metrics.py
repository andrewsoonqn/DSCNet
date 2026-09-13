import unittest
from pathlib import Path

import numpy as np

from dscnet.evaluation.metrics import cldice_score, dice_score, to_minivess_binary_mask

class BinaryMaskTests(unittest.TestCase):
    def test_accepts_three_dimensional_integer_mask(self):
        mask = np.zeros((2, 3, 4), dtype=np.uint8)
        mask[1, 1, 1] = 1
        actual = to_minivess_binary_mask(mask)
        self.assertEqual(actual.dtype, np.bool_)
        np.testing.assert_array_equal(actual, mask)

    def test_rejects_unexpected_label(self):
        with self.assertRaisesRegex(ValueError, r"found \[0, 2\]"):
            to_minivess_binary_mask(np.array([0, 2]), "label")

class DiceScoreTests(unittest.TestCase):
    def test_perfect_mask_scores_one(self):
        mask = np.zeros((9, 9), dtype=bool)
        mask[4, 2:7] = True
        self.assertEqual(dice_score(mask, mask), 1.0)

    def test_two_empty_masks_score_one(self):
        empty = np.zeros((9, 9), dtype=bool)
        self.assertEqual(dice_score(empty, empty), 1.0)

    def test_disjoint_masks_score_zero(self):
        target = np.zeros((9, 9), dtype=bool)
        prediction = np.zeros_like(target)
        target[2, 2:7] = True
        prediction[6, 2:7] = True
        self.assertEqual(dice_score(prediction, target), 0.0)

    def test_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            dice_score(np.zeros((2, 3)), np.zeros((3, 2)))

class ClDiceScoreTests(unittest.TestCase):
    def test_perfect_vessel_scores_one(self):
        vessel = np.zeros((9, 9), dtype=bool)
        vessel[4, 2:7] = True
        self.assertEqual(cldice_score(vessel, vessel), 1.0)

    def test_two_empty_masks_score_one(self):
        empty = np.zeros((9, 9), dtype=bool)
        self.assertEqual(cldice_score(empty, empty), 1.0)

    def test_one_empty_mask_scores_zero(self):
        empty = np.zeros((9, 9), dtype=bool)
        vessel = empty.copy()
        vessel[4, 2:7] = True
        self.assertEqual(cldice_score(empty, vessel), 0.0)

    def test_disjoint_vessels_score_zero(self):
        target = np.zeros((9, 9), dtype=bool)
        prediction = np.zeros_like(target)
        target[2, 2:7] = True
        prediction[6, 2:7] = True
        self.assertEqual(cldice_score(prediction, target), 0.0)

    def test_broken_vessel_is_penalized(self):
        target = np.zeros((9, 9), dtype=bool)
        prediction = np.zeros_like(target)
        target[4, 1:8] = True
        prediction[4, 1:4] = True
        prediction[4, 5:8] = True
        score = cldice_score(prediction, target)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_false_branch_is_penalized(self):
        target = np.zeros((9, 9), dtype=bool)
        prediction = np.zeros_like(target)
        target[4, 1:8] = True
        prediction[4, 1:8] = True
        prediction[2:5, 4] = True
        score = cldice_score(prediction, target)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

if __name__ == "__main__":
    unittest.main()
