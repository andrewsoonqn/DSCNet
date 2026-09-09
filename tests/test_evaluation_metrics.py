import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

MODULE_DIR = (
    Path(__file__).parents[1]
    / "DSCNet_3D_opensource"
    / "Code"
    / "Kipa"
    / "DSCNet"
)
sys.path.insert(0, str(MODULE_DIR))

from S3_Evaluation_Metrics import summarize_predictions


class EvaluationSummaryTests(unittest.TestCase):
    def test_summary_reports_topology_and_confusion_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_dir = root / "predictions"
            prediction_dir.mkdir()
            target = np.zeros((3, 3, 3), dtype=np.uint8)
            target[1, 1, 0:3] = 1
            prediction = target.copy()
            prediction[0, 0, 0] = 1
            prediction[1, 1, 2] = 0
            label_path = root / "sample.nii.gz"
            sitk.WriteImage(sitk.GetImageFromArray(target), str(label_path))
            sitk.WriteImage(
                sitk.GetImageFromArray(prediction),
                str(prediction_dir / label_path.name),
            )
            manifest = root / "labels.txt"
            manifest.write_text(f"{label_path}\n")

            metrics = summarize_predictions(
                manifest,
                prediction_dir,
                lambda prediction_path, target_path: self.fail(
                    "equal shapes must not call mismatch loader"
                ),
            )

        self.assertEqual(metrics["false_positives"], 1.0)
        self.assertEqual(metrics["false_negatives"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertLess(metrics["dice"], 1.0)
        self.assertLess(metrics["cldice"], 1.0)


if __name__ == "__main__":
    unittest.main()
