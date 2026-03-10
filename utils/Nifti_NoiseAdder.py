import argparse
import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir",
                        default="",
                        help="the address of the nii image directory")
    
    parser.add_argument("--save_dir",
                        default="",
                        help="the address of the directory for saving png files")

    args, unknown = parser.parse_known_args()
    folder = args.dir
    save_folder = args.save_dir
    files = [f for f in os.listdir(folder) if f.endswith((".nii", ".nii.gz"))]

    mean = 0 
    std = 0
    length = 0
    strength = 1 # Adjust this value to control the noise level

    # for file in files:
    #     path = os.path.join(folder, file)
    #     image = sitk.ReadImage(path)
    #     image = sitk.GetArrayFromImage(image).astype(np.float32)
    #     length += image.size
    #     mean += np.sum(image)
    #     # print(mean, length)
    # mean = mean / length
    # print("Mean: " + str(mean))

    # for file in files:
    #     path = os.path.join(folder, file)
    #     image = sitk.ReadImage(path)
    #     image = sitk.GetArrayFromImage(image).astype(np.float32)
    #     std += np.sum(np.square((image - mean)))
    #     # print(std)
    # std = np.sqrt(std / length)
    # print("Standard Deviation: " + str(std))

    for file in files:
        # Load the NIfTI file
        path = os.path.join(folder, file)
        print(path)
        img = nib.load(path)
        data = img.get_fdata()
        affine = img.affine

        mean = np.mean(data)
        std = np.std(data)
        print("Mean: " + str(mean) + ", Std: " + str(std))

        # Add Gaussian noise
        std_dev = strength * std  
        noise = np.random.normal(mean, std_dev, size=data.shape)

        # Add noise to the data
        noisy_data = data + noise

        # Save the new file
        noisy_img = nib.Nifti1Image(noisy_data, affine)
        save_path = os.path.join(save_folder, file)
        nib.save(noisy_img, save_path)