import argparse
import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_dir",
                        default="",
                        help="the address of the nii predict label directory")
    
    parser.add_argument("--gt_dir",
                        default="",
                        help="the address of the nii groundtruth label directory")
    
    parser.add_argument("--save_dir",
                        default="",
                        help="the address of the directory for saving png files")

    args, unknown = parser.parse_known_args()
    pred_folder = args.pred_dir
    gt_folder = args.gt_dir
    save_folder = args.save_dir
    files = [f for f in os.listdir(pred_folder) if f.endswith((".nii", ".nii.gz"))]

    for file in files:
        # Load the NIfTI file
        pred_path = os.path.join(pred_folder, file)
        gt_path = os.path.join(gt_folder, file)
        print(pred_path, gt_path)

        # Load Pred
        pred_label = nib.load(pred_path)
        pred_data = pred_label.get_fdata().astype(bool)
        affine = pred_label.affine

        # Load Groundtruth
        gt_label = nib.load(gt_path)
        gt_data = gt_label.get_fdata().astype(bool)

        # Compare
        new_data = np.zeros(pred_data.shape, dtype=np.uint8)
        new_data[pred_data & ~gt_data] = 1   # Pred=1, GT=0
        new_data[~pred_data & gt_data] = 2   # Pred=0, GT=1
        new_data[pred_data & gt_data]  = 3   # Pred=1, GT=1

        # intersect_mask = np.logical_and(pred_data, gt_data)
        # new_data = pred_data
        # new_data = np.where(gt_data, 2, new_data)
        # new_data = np.where(intersect_mask, 3, new_data)

        # Save the new file
        new_label = nib.Nifti1Image(new_data, affine)
        save_path = os.path.join(save_folder, file)
        nib.save(new_label, save_path)