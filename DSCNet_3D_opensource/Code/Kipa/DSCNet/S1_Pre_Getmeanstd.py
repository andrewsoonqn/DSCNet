"""Compute normalization statistics from the training images."""

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def Getmeanstd(image_path, output_path):
    image_dir = Path(image_path)
    image_files = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.name.endswith((".nii", ".nii.gz"))
    )
    if not image_files:
        raise ValueError(f"No NIfTI images found in {image_dir}")

    voxel_count = 0
    voxel_sum = 0.0
    squared_voxel_sum = 0.0
    for image_file in image_files:
        image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_file))).astype(
            np.float32
        )
        voxel_count += image.size
        voxel_sum += np.sum(image, dtype=np.float64)
        squared_voxel_sum += np.sum(np.square(image), dtype=np.float64)

    mean = voxel_sum / voxel_count
    variance = max(squared_voxel_sum / voxel_count - mean**2, 0.0)
    std = np.sqrt(variance)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, [mean, std])
    print("1 Finish Getmeanstd:", output_path)
    print("Mean and std are:", mean, std)
