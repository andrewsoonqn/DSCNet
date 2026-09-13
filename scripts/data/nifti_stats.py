import argparse
import numpy as np
import SimpleITK as sitk
import os
from skimage.morphology import skeletonize, ball, dilation
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_score, recall_score, accuracy_score

def load_with_upsample(pred_nifti_path, ref_nifti_path):
    """
    Load a 3D prediction array with upsampling back to original resolution if necessary.

    Args:
        pred_nifti_path (str): (low-resolution) prediction (Z, Y, X)
        ref_nifti_path (str): path to reference full-resolution NIfTI groundtruth

    Returns:
        numpy array: upsampled prediction array with referenced shape
    """

    # Load reference image parameters (for metadata + size reference)
    ref_img = sitk.ReadImage(ref_nifti_path)
    ref_size = ref_img.GetSize() 
    ref_spacing = ref_img.GetSpacing()
    ref_origin = ref_img.GetOrigin()
    ref_direction = ref_img.GetDirection()

    # Load prediction image parameters and set parameters (in case not set correctly when created)
    pred_img = sitk.ReadImage(pred_nifti_path)
    pred_size = pred_img.GetSize()
    pred_img.SetOrigin(ref_origin)
    pred_img.SetDirection(ref_direction)
    new_spacing = tuple(ref_spacing[i] * (ref_size[i] / pred_size[i]) for i in range(3)) # setting scale for upsampling
    pred_img.SetSpacing(new_spacing)

    # Resample back to original resolution (using nearest neighbor)
    resample = sitk.ResampleImageFilter()
    resample.SetSize(ref_size)
    resample.SetOutputSpacing(ref_spacing)
    resample.SetOutputOrigin(ref_origin)
    resample.SetOutputDirection(ref_direction)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)

    upsampled_img = resample.Execute(pred_img)

    # Convert back to numpy (SimpleITK uses (Z, Y, X) order)
    upsampled_arr = sitk.GetArrayFromImage(upsampled_img)
    groundtruth_arr = sitk.GetArrayFromImage(ref_img)

    return upsampled_arr, groundtruth_arr

def Dice(label_dir, pred_dir):

    i = 0
    files = sorted([f for f in os.listdir(label_dir) if f.endswith((".nii", ".nii.gz"))])

    dice_vein = np.zeros(shape=(len(files)), dtype=np.float32)
    dice_artery = np.zeros(shape=(len(files)), dtype=np.float32)

    print("Dice:")
    for file in files:
        predict_path = os.path.join(pred_dir, file)
        groundtruth_path = os.path.join(label_dir, file)
        
        predict = sitk.ReadImage(predict_path)
        groundtruth = sitk.ReadImage(groundtruth_path)

        if predict.GetSize() == groundtruth.GetSize():
            predict = sitk.GetArrayFromImage(predict)
            groundtruth = sitk.GetArrayFromImage(groundtruth)
        else:
            predict, groundtruth = load_with_upsample(predict_path, groundtruth_path)

        groundtruth = np.where(groundtruth == 2, 0, groundtruth)
        groundtruth = np.where(groundtruth == 3, 2, groundtruth)
        groundtruth = np.where(groundtruth == 4, 0, groundtruth)

        predict_vein = np.where(predict == 1, 1, 0).flatten()
        predict_artery = np.where(predict == 2, 1, 0).flatten()
        groundtruth_vein = np.where(groundtruth == 1, 1, 0).flatten()
        groundtruth_artery = np.where(groundtruth == 2, 1, 0).flatten()

        tmp = predict_vein + groundtruth_vein
        a = np.sum(np.where(tmp == 2, 1, 0))
        b = np.sum(predict_vein)
        c = np.sum(groundtruth_vein)
        dice_vein[i] = (2 * a) / (b + c)

        tmp = predict_artery + groundtruth_artery
        a = np.sum(np.where(tmp == 2, 1, 0))
        b = np.sum(predict_artery)
        c = np.sum(groundtruth_artery)
        dice_artery[i] = (2 * a) / (b + c)
        print(file, dice_vein[i], dice_artery[i])
        i += 1

    return dice_vein, dice_artery

def clDice(label_dir, pred_dir, radius=1):
    
    i = 0
    files = sorted([f for f in os.listdir(label_dir) if f.endswith((".nii", ".nii.gz"))])

    cl_Dice = np.zeros(shape=(len(files)), dtype=np.float32)

    print("clDice:")
    for file in files:
        predict_path = os.path.join(pred_dir, file)
        groundtruth_path = os.path.join(label_dir, file)
        
        predict = sitk.ReadImage(predict_path)
        groundtruth = sitk.ReadImage(groundtruth_path)

        if predict.GetSize() == groundtruth.GetSize():
            predict = sitk.GetArrayFromImage(predict)
            groundtruth = sitk.GetArrayFromImage(groundtruth)
        else:
            predict, groundtruth = load_with_upsample(predict_path, groundtruth_path)


        predict = predict.astype(bool)
        groundtruth = groundtruth.astype(bool)

        skel_predict = skeletonize(predict)
        skel_groundtruth = skeletonize(groundtruth)


        if radius > 0:
            selem = ball(radius) if predict.ndim == 3 else None
            predict_dil = dilation(predict, selem)
            groundtruth_dil = dilation(groundtruth, selem)
        else:
            predict_dil = predict
            groundtruth_dil = groundtruth

        #intersection = np.logical_and(skel_predict, skel_groundtruth).sum()
        #size_predict = skel_predict.sum()
        #size_groundtruth = skel_groundtruth.sum()

        # Topology-aware coverage
        tpc = np.logical_and(skel_groundtruth, predict_dil).sum()
        tpp = np.logical_and(skel_predict, groundtruth_dil).sum()

        cl_Dice[i] = (2 * tpc) / (tpc + tpp + skel_groundtruth.sum())

        #if size_predict + size_groundtruth == 0: 
        #    cl_Dice[i] = 1.0
        #else:
        #    cl_Dice[i] = 2.0 * intersection / (size_predict + size_groundtruth)
        print(file, cl_Dice[i])
        i += 1
    
    return cl_Dice

def hausdorff_dist(label_dir, pred_dir, spacing):
    i = 0
    files = sorted([f for f in os.listdir(label_dir) if f.endswith((".nii", ".nii.gz"))])

    h_dist = np.zeros(shape=(len(files)), dtype=np.float32)

    print("Hausdorff Distance:")
    for file in files:
        predict_path = os.path.join(pred_dir, file)
        groundtruth_path = os.path.join(label_dir, file)
        
        predict = sitk.ReadImage(predict_path)
        groundtruth = sitk.ReadImage(groundtruth_path)

        if predict.GetSize() == groundtruth.GetSize():
            predict = sitk.GetArrayFromImage(predict)
            groundtruth = sitk.GetArrayFromImage(groundtruth)
        else:
            predict, groundtruth = load_with_upsample(predict_path, groundtruth_path)

        p_pts = np.argwhere(predict > 0)
        g_pts = np.argwhere(groundtruth > 0)

        if spacing is not None:
            p_pts = p_pts * spacing
            g_pts = g_pts * spacing

        if len(p_pts) == 0 or len(g_pts) == 0:
            h_dist[i] = np.inf

        dists = cdist(p_pts, g_pts)

        d_pg = np.min(dists, axis=1)
        d_gp = np.min(dists, axis=0)

        hd95_pg = np.percentile(d_pg, 95)
        hd95_gp = np.percentile(d_gp, 95)

        h_dist[i] = max(hd95_pg, hd95_gp)
        i += 1

    return h_dist

def precision_recall_accuracy_score(label_dir, pred_dir):

    i = 0
    files = sorted([f for f in os.listdir(label_dir) if f.endswith((".nii", ".nii.gz"))])

    precision = np.zeros(shape=(len(files)), dtype=np.float32)
    recall = np.zeros(shape=(len(files)), dtype=np.float32)
    accuracy = np.zeros(shape=(len(files)), dtype=np.float32)

    print("Precision, Recall, Accuracy:")
    for file in files:
        predict_path = os.path.join(pred_dir, file)
        groundtruth_path = os.path.join(label_dir, file)
        
        predict = sitk.ReadImage(predict_path)
        groundtruth = sitk.ReadImage(groundtruth_path)

        if predict.GetSize() == groundtruth.GetSize():
            predict = sitk.GetArrayFromImage(predict)
            groundtruth = sitk.GetArrayFromImage(groundtruth)
        else:
            predict, groundtruth = load_with_upsample(predict_path, groundtruth_path)

        predict_flat = predict.flatten()
        groundtruth_flat = groundtruth.flatten()

        p = precision_score(groundtruth_flat, predict_flat, average="binary")
        r = recall_score(groundtruth_flat, predict_flat, average="binary")
        a = accuracy_score(groundtruth_flat, predict_flat)
        precision[i] = p
        recall[i] = r
        accuracy[i] = a
        print(file, precision[i], recall[i], accuracy[i])
        i += 1
    
    return precision, recall, accuracy
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--label_dir",
                        default="",
                        help="the address of the nii groundtruth label directory")
    
    parser.add_argument("--pred_dir",
                        default="",
                        help="the address of the nii predicted result directory")
    
    parser.add_argument("--spacing",
                        default=1.0,
                        type=float,
                        help="the address of the nii predicted result directory")
    
    args, unknown = parser.parse_known_args()

    Dice(args.label_dir, args.pred_dir)
    clDice(args.label_dir, args.pred_dir)
    #hausdorff_dist(args.label_dir, args.pred_dir, args.spacing)
    precision_recall_accuracy_score(args.label_dir, args.pred_dir)


    