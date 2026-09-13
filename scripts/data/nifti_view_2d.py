import argparse
import nibabel as nib
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir",
                        default="",
                        help="the address of the nii image directory")

    parser.add_argument("--load_entire_dir",
                        default=False,
                        help="input True to load entire directory at once")
    
    args, unknown = parser.parse_known_args()
    folder = args.dir
    files = [f for f in os.listdir(folder) if f.endswith((".nii", ".nii.gz"))]
    
    # Load NIfTI files
    for file in files:
        path = os.path.join(folder, file)
        img = nib.load(path)
        data = img.get_fdata()
        
        # Show slice
        print(str(path) +  " - Dimension: " + str(data.shape))
        plt.imshow(data[:, :], cmap="gray")
        plt.axis('off')
        plt.show()
        if (args.load_entire_dir != True):  
            break