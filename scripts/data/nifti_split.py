import nibabel as nib
import numpy as np
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir",
                        default="",
                        help="the address of the nifti image directory")
    
    parser.add_argument("--save_dir",
                        default="",
                        help="the save address of the split nifti image files")

    args, unknown = parser.parse_known_args()
    folder = args.dir
    files = sorted([f for f in os.listdir(folder) if f.endswith((".nii", ".nii.gz"))])

    count = 0
    for file in files:
        load_path = os.path.join(folder, file)
        img = nib.load(load_path)
        data = img.get_fdata()
        affine = img.affine

        print(data.shape)

        X, Y, Z = data.shape

        x_mid = X // 2
        y_mid = Y // 2
        z_mid = Z // 2

        subvolumes = []
        sub_affines = []

        for xi, x_slice in enumerate([(0, x_mid), (x_mid, X)]):
            for yi, y_slice in enumerate([(0, y_mid), (y_mid, Y)]):
                for zi, z_slice in enumerate([(0, z_mid), (z_mid, Z)]):

                    # Extract subvolume
                    sub = data[
                        x_slice[0]:x_slice[1],
                        y_slice[0]:y_slice[1],
                        z_slice[0]:z_slice[1]
                    ]

                    # Compute new affine
                    offset = np.array([x_slice[0], y_slice[0], z_slice[0], 1])
                    new_affine = affine.copy()
                    new_affine[:3, 3] = (affine @ offset)[:3]

                    subvolumes.append(sub)
                    sub_affines.append(new_affine)

        for i, (sub, sub_affine) in enumerate(zip(subvolumes, sub_affines)):
            sub_img = nib.Nifti1Image(sub, sub_affine)
            save_path = os.path.join(args.save_dir, f"{i + count*8}.nii.gz")
            nib.save(sub_img, save_path)
        
        count += 1