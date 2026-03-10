import tifffile as tiff
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir",
                        default="",
                        help="the address of the tiff image directory")
    
    parser.add_argument("--save_dir",
                        default="",
                        help="the address of the nifti image directory")
    
    parser.add_argument("--volume_name",
                        default="",
                        help="the name of the nifti volume")

    args, unknown = parser.parse_known_args()
    folder = args.dir
    files = sorted([f for f in os.listdir(folder) if f.endswith((".tif"))])

    slices = []
    for file in files:
        img = tiff.imread(os.path.join(folder, file))
        slices.append(img)

    # Stack into volume (Z, H, W)
    volume = np.stack(slices, axis=0).astype(np.float32)

    # Optional: reorder to (H, W, Z)
    volume = np.transpose(volume, (1, 2, 0))

    affine = np.eye(4)

    nifti_img = nib.Nifti1Image(volume, affine)
    save_path = os.path.join(args.save_dir, args.volume_name)
    nib.save(nifti_img, save_path)