import argparse
import nibabel as nib
import matplotlib.pyplot as plt
import os

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
    
    # Load NIfTI files
    for file in files:
        path = os.path.join(folder, file)
        img = nib.load(path)
        data = img.get_fdata()

        # Save slice
        print(str(path) +  " - Dimension: " + str(data.shape))
        plt.imshow(data[:, :], cmap="gray")
        plt.axis('off')
        #plt.show()

        save_path = os.path.join(save_folder, file)
        #plt.imsave(f"{save_path}.png", data[:, :], cmap='gray')
        plt.savefig(f"{save_path}.png", bbox_inches="tight", pad_inches=0)
