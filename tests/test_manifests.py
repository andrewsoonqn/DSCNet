import sys
import tempfile
import unittest
from pathlib import Path

MODULE_DIR = (
    Path(__file__).parents[1]
    / "DSCNet_3D_opensource"
    / "Code"
    / "Kipa"
    / "DSCNet"
)
sys.path.insert(0, str(MODULE_DIR))

from S2_Pre_Generate_Txt import Generate_Paired_Txt, Generate_Txt, Get_file_list


class ManifestTests(unittest.TestCase):
    def test_lists_only_nifti_files_in_natural_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["10.nii.gz", "2.nii.gz", "1.nii.gz", ".DS_Store"]:
                (root / name).touch()

            files, count = Get_file_list(root)

            self.assertEqual(count, 3)
            self.assertEqual([path.name for path in files], ["1.nii.gz", "2.nii.gz", "10.nii.gz"])

    def test_writes_manifest_without_a_trailing_blank_line(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            (images / "2.nii.gz").touch()
            (images / "1.nii.gz").touch()
            manifest = root / "manifests" / "images.txt"

            Generate_Txt(images, manifest)

            self.assertEqual(
                manifest.read_text(),
                f"{images / '1.nii.gz'}\n{images / '2.nii.gz'}",
            )

    def test_paired_manifests_follow_the_same_natural_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            for name in ["10.nii.gz", "2.nii.gz", "1.nii.gz"]:
                (images / name).touch()
                (labels / name).touch()

            image_txt = root / "image.txt"
            label_txt = root / "label.txt"
            Generate_Paired_Txt(images, labels, image_txt, label_txt)

            self.assertEqual(
                [Path(line).name for line in image_txt.read_text().splitlines()],
                ["1.nii.gz", "2.nii.gz", "10.nii.gz"],
            )
            self.assertEqual(
                [Path(line).name for line in label_txt.read_text().splitlines()],
                ["1.nii.gz", "2.nii.gz", "10.nii.gz"],
            )

    def test_rejects_mismatched_image_and_label_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            (images / "1.nii.gz").touch()
            (labels / "2.nii.gz").touch()

            with self.assertRaisesRegex(ValueError, "filenames do not match"):
                Generate_Paired_Txt(
                    images, labels, root / "image.txt", root / "label.txt"
                )

    def test_rejects_empty_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()

            with self.assertRaisesRegex(ValueError, "split is empty"):
                Generate_Paired_Txt(
                    images, labels, root / "image.txt", root / "label.txt"
                )


if __name__ == "__main__":
    unittest.main()
