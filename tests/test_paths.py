import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

MODULE_DIR = (
    Path(__file__).parents[1]
    / "DSCNet_3D_opensource"
    / "Code"
    / "Kipa"
    / "DSCNet"
)
sys.path.insert(0, str(MODULE_DIR))

from S0_Main import resolve_paths


class PathResolutionTests(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            "root_dir": "/tmp/dscnet-root",
            "data_dir": "/tmp/minivess",
            "run_label": "trial",
            "Meanstd_name": None,
            "save_path": None,
            "save_path_max": None,
            "model_name": None,
            "model_name_max": None,
            "log_name": None,
        }
        for name in (
            "Tr_Image_dir",
            "Va_Image_dir",
            "Te_Image_dir",
            "Tr_Label_dir",
            "Va_Label_dir",
            "Te_Label_dir",
            "Dir_Txt",
            "Dir_Log",
            "Dir_Save",
            "Dir_Weights",
            "Image_Tr_txt",
            "Image_Va_txt",
            "Image_Te_txt",
            "Label_Tr_txt",
            "Label_Va_txt",
            "Label_Te_txt",
        ):
            values[name] = None
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_root_and_data_paths_do_not_need_trailing_separators(self):
        args = resolve_paths(self._args())

        self.assertEqual(args.Tr_Image_dir, "/tmp/minivess/train/image")
        self.assertEqual(args.Label_Te_txt, "/tmp/dscnet-root/Txt/Txt_trial/Label_Te.txt")
        self.assertEqual(args.Meanstd_path, "/tmp/dscnet-root/trial_Meanstd.npy")
        self.assertEqual(args.save_path_max, "/tmp/dscnet-root/Results/trial/DSCNet_max")

    def test_explicit_paths_are_preserved(self):
        args = resolve_paths(
            self._args(
                Tr_Image_dir="/data/custom-images",
                Dir_Txt="/runs/manifests",
                Image_Tr_txt="/frozen/train-images.txt",
                Meanstd_name="/frozen/train-stats.npy",
            )
        )

        self.assertEqual(args.Tr_Image_dir, "/data/custom-images")
        self.assertEqual(args.Image_Tr_txt, "/frozen/train-images.txt")
        self.assertEqual(args.Meanstd_path, "/frozen/train-stats.npy")


if __name__ == "__main__":
    unittest.main()
