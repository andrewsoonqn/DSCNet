from tifffile import imread
import matplotlib.pyplot as plt
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir",
                        default="",
                        help="the address of the tiff image directory")

    parser.add_argument("--load_entire_dir",
                        default=False,
                        help="input True to load entire directory at once")
    
    args, unknown = parser.parse_known_args()
    folder = args.dir
    files = [f for f in os.listdir(folder) if f.endswith((".tif"))]
    
    count = 0
    # Load TIFF files
    for file in files:
        path = os.path.join(folder, file)
        
        img = imread(path)

        # Show slice
        print(str(path) +  " - Dtype: " + str(img.dtype) + 
              ", Min: " + str(img.min()) + 
              ", Max: " + str(img.max()) + 
              ", Dimension: " + str(img.shape))
        plt.imshow(img, cmap="gray")
        plt.colorbar()
        plt.show()

        count += 1

        if (count >= 100 and args.load_entire_dir != True):  
            break