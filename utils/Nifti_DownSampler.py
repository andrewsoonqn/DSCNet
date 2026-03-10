import nibabel as nib
import numpy as np
import argparse 
import os

def block_reduce(volume, factor):
    sx, sy, sz = factor
    X, Y, Z = volume.shape
    volume = volume[:X - (X % sx),
                    :Y - (Y % sy),
                    :Z - (Z % sz)]
    
    out = volume.reshape(
        volume.shape[0] // sx, sx,
        volume.shape[1] // sy, sy,
        volume.shape[2] // sz, sz
    ).mean(axis=(1, 3, 5))
    return out

def block_reduce_labels(volume, factor):
    sx, sy, sz = factor
    X, Y, Z = volume.shape
    x0, y0, z0 = (X//sx)*sx, (Y//sy)*sy, (Z//sz)*sz
    volume = volume[:x0, :y0, :z0]

    # reshape to group blocks
    grouped_arr = volume.reshape(x0//sx, sx, y0//sy, sy, z0//sz, sz)
    
    # move block voxels into one axis: (Xb, Yb, Zb, sx*sy*sz)
    grouped_arr = grouped_arr.transpose(0,2,4,1,3,5).reshape(x0//sx, y0//sy, z0//sz, sx*sy*sz)
    
    # compute mode via bincount per-block
    out = np.zeros((x0//sx, y0//sy, z0//sz), dtype=volume.dtype)
    it = np.nditer(out, flags=['multi_index'], op_flags=['writeonly'])
    while not it.finished:
        block = grouped_arr[it.multi_index]
        vals, counts = np.unique(block, return_counts=True) # bincount
        it[0] = vals[np.argmax(counts)]
        it.iternext()
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir",
                        default="",
                        help="the address of the nii image directory")
    
    parser.add_argument("--save_dir",
                        default="",
                        help="the address of the directory for saving downsampled files")
    
    parser.add_argument("--is_label",
                        default=False,
                        type=bool,
                        help="whether to use mode when downsampling labels (default is mean for raw image)")

    args, unknown = parser.parse_known_args()
    folder = args.dir
    save_folder = args.save_dir
    files = [f for f in os.listdir(folder) if f.endswith((".nii", ".nii.gz"))]

    for file in files:
        # Load the NIfTI file
        path = os.path.join(folder, file)
        print(path)
        img = nib.load(path)
        data = img.get_fdata()
        affine = img.affine

        factor = (2, 2, 2)  # downsample by 2 in each dimension -> "binned 8"
        if args.is_label:
            down = block_reduce_labels(data, factor)
        else:
            down = block_reduce(data, factor)

        new_affine = affine.copy()
        new_affine[:3, :3] *= factor

        down_img = nib.Nifti1Image(down, new_affine)
        save_path = os.path.join(save_folder, file)
        nib.save(down_img, save_path)