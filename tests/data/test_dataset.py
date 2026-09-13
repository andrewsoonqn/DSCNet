import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from dscnet.data.dataset import Dataloader


class DatasetDtypeTests(unittest.TestCase):
    def test_every_augmentation_output_is_float32_and_contiguous(self):
        image = np.arange(8, dtype=np.int16).reshape(2, 2, 2)
        label = np.zeros((2, 2, 2), dtype=np.uint8)
        label[0, 0, 0] = 1

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_list = root / "images.txt"
            label_list = root / "labels.txt"
            image_list.write_text("image.nii.gz\n")
            label_list.write_text("label.nii.gz\n")
            args = SimpleNamespace(
                Image_Tr_txt=str(image_list),
                Label_Tr_txt=str(label_list),
                Meanstd_path=str(root / "Mean_Std.npy"),
                ROI_shape=(2, 2, 2),
                sample_count=1,
            )
            loader = Dataloader(args)

            def numpy_output(image_value, label_value):
                image_output = image_value.astype(np.float64).swapaxes(-1, -2)
                label_output = label_value.astype(np.float64).swapaxes(-1, -2)
                self.assertFalse(image_output.flags.c_contiguous)
                self.assertFalse(label_output.flags.c_contiguous)
                return {"image": image_output, "label": label_output}

            def tensor_output(image_value, label_value):
                image_output = torch.as_tensor(
                    image_value, dtype=torch.float64
                ).transpose(-1, -2)
                label_output = torch.as_tensor(
                    label_value, dtype=torch.float64
                ).transpose(-1, -2)
                self.assertFalse(image_output.is_contiguous())
                self.assertFalse(label_output.is_contiguous())
                return {"image": image_output, "label": label_output}

            for output_factory in (numpy_output, tensor_output):
                with self.subTest(output=output_factory.__name__):
                    def transform(image_value, label_value, _args):
                        self.assertEqual(image_value.dtype, np.float32)
                        self.assertEqual(label_value.dtype, np.float32)
                        return output_factory(image_value, label_value)

                    with patch(
                        "dscnet.data.dataset.sitk.ReadImage", side_effect=lambda path: path
                    ), patch(
                        "dscnet.data.dataset.sitk.GetArrayFromImage",
                        side_effect=[image.copy(), label.copy()],
                    ), patch(
                        "dscnet.data.dataset.np.load",
                        return_value=np.array([0.0, 1.0], dtype=np.float64),
                    ), patch(
                        "dscnet.data.dataset.transform_img_lab",
                        side_effect=transform,
                    ):
                        image_result, label_result = loader[0]

                    self.assertEqual(image_result.shape, (1, 2, 2, 2))
                    self.assertEqual(label_result.shape, (2, 2, 2, 2))
                    self.assertEqual(image_result.dtype, np.float32)
                    self.assertEqual(label_result.dtype, np.float32)
                    self.assertTrue(image_result.flags.c_contiguous)
                    self.assertTrue(label_result.flags.c_contiguous)
                    np.testing.assert_array_equal(
                        label_result.sum(axis=0), np.ones((2, 2, 2))
                    )
                    self.assertEqual(set(np.unique(label_result)), {0.0, 1.0})
                    self.assertEqual(np.count_nonzero(label_result[1]), 1)


if __name__ == "__main__":
    unittest.main()
