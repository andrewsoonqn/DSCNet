# -*- coding: utf-8 -*-
import os
import torch
import logging
import numpy as np
from os.path import join
import SimpleITK as sitk
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.morphology import skeletonize, ball, disk, dilation
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torchinfo import summary
from monai.data.utils import dense_patch_slices
from monai.losses import DiceCELoss, DiceLoss

from S3_DSCNet_Optimized import DSCNet
from S3_Dataloader import Dataloader
from S3_Loss import cross_loss, dice_cross_loss, entropy_regularization_cross_loss, entropy_loss

import warnings

warnings.filterwarnings("ignore")


# Use <AverageMeter> to calculate the mean in the process
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# One epoch in training process
def train_epoch(model, loader, optimizer, criterion, epoch, n_epochs, logger, args):
    losses = AverageMeter()
    c_losses = AverageMeter()
    e_losses = AverageMeter()

    model.train()
    for batch_idx, (image, label) in enumerate(loader):
        if torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()
        optimizer.zero_grad()
        model.zero_grad()

        output = model(image)
        loss= criterion(label, output, epoch)
        
        # Separate components for monitoring only
        cross_loss_fn = cross_loss()
        c_loss = cross_loss_fn(label, output)
        entropy_fn = entropy_loss(beta_start=args.beta, beta_end=args.min_beta, decay_power=args.beta_decay_power, total_steps=args.n_epochs)
        entropy = entropy_fn(output, epoch)

        losses.update(loss.data, label.size(0))
        c_losses.update(c_loss.data, label.size(0))
        e_losses.update(entropy.data, label.size(0))

        loss.backward()
        optimizer.step()

        res = "\t".join(
            [
                "Epoch: [%d/%d]" % (epoch, n_epochs),
                "Iter: [%d/%d]" % (batch_idx + 1, len(loader)),
                "Lr: [%.7f]" % (optimizer.param_groups[0]["lr"]),
                "Loss %f" % (losses.avg),
            ]
        )
        print(res)
        if batch_idx + 1 == len(loader):
            logger.info(f"Total Loss: {losses.avg:.6f} | Beta: {entropy_fn.beta:.6f} | Entropy: {e_losses.avg:.6f} | CE: {c_losses.avg:.6f} | Ratio : {(entropy_fn.beta*e_losses.avg/c_losses.avg):.6f}x")
    return losses.avg


# Generate the log
def Get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

# Generate the log for model stats
def Get_logger_model(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def Close_logger(logger):
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

# Train process
def Train_net(net, args, device, map_kernel_tensor):
    dice_mean, dice_save, dice_v, dice_a = 0, 0, 0, 0

    # Determine if trained parameters exist
    if not args.if_retrain and os.path.exists(
        os.path.join(args.Dir_Weights, args.model_name)
    ):
        net.load_state_dict(torch.load(os.path.join(args.Dir_Weights, args.model_name)))
        print(os.path.join(args.Dir_Weights, args.model_name))
    if torch.cuda.is_available():
        net = net.cuda()

    # Load dataset
    train_dataset = Dataloader(args)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    ) #num_workers=8, persistent_workers=True (slower)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.95)) # try weight_decay = 1e-5
    
    def poly_decay(epoch):
        decay = (1 - epoch / args.n_epochs) ** args.poly_decay_power
        lr = args.min_lr + (args.lr - args.min_lr) * decay
        lr_factor = lr / args.lr
        return lr_factor
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_decay)

    #criterion = cross_loss() # try new dice_cross_loss function
    # criterion = DiceCELoss(
    #     to_onehot_y=False,     # ground truth is already converted to one-hot with to_categorical
    #     softmax=True,          # foreground + background are mutually exclusive, probabilities must sum to 1
    #     sigmoid=False,         # never use alongside softmax
    #     lambda_dice=1.0,       # equal weighting to start — tune if loss imbalance observed
    #     lambda_ce=1.0,
    #     include_background=False,  # exclude background channel from dice computation (else double computation)
    # )
    # criterion = dice_cross_loss(lambda_dice=0.2, lambda_ce=1.0)
    criterion = entropy_regularization_cross_loss(beta_start=args.beta, beta_end=args.min_beta, decay_power=args.beta_decay_power, total_steps=args.n_epochs)

    dt = datetime.today()
    log_name = (
        str(dt.date())
        + "_"
        + str(dt.time().hour)
        + "."
        + str(dt.time().minute)
        + "."
        + str(dt.time().second)
        + "_"
        + args.log_name
    )
    logger = Get_logger(args.Dir_Log + log_name)
    logger.info("start training!")

    # Main train process
    for epoch in range(args.start_train_epoch, args.n_epochs + 1):
        loss = train_epoch(
            net, train_dataloader, optimizer, criterion, epoch, args.n_epochs, logger, args
        )
        torch.save(net.state_dict(), os.path.join(args.Dir_Weights, args.model_name))
        scheduler.step()

        if epoch >= args.start_verify_epoch  and (epoch % args.verify_gap) == 0:
            net.load_state_dict(
                torch.load(os.path.join(args.Dir_Weights, args.model_name))
            )
            new_predict(net, args.Image_Va_txt, args.Va_Meanstd_name, args.save_path, args, device, map_kernel_tensor)
            dice_v, dice_a = Dice(args.Label_Va_txt, args.save_path)
            dice_v = np.mean(dice_v)
            dice_mean = dice_v
            if dice_mean > dice_save:
                dice_save = dice_mean
                torch.save(
                    net.state_dict(),
                    os.path.join(args.Dir_Weights, args.model_name_max),
                )
        logger.info(
            "Epoch:[{}/{}] lr={:.7f} loss={:.5f} dice_mean={:.4f} saved_dice={:.4f}".format(
                epoch,
                args.n_epochs,
                optimizer.param_groups[0]["lr"],
                loss,
                dice_mean,
                dice_save,
            )
        )
    logger.info("finish training!")
    duration = (datetime.today()-dt).total_seconds()
    logger.info("Train Time (s): " + str(duration))
    Close_logger(logger)


def read_file_from_txt(txt_path):
    files = []
    for line in open(txt_path, "r"):
        files.append(line.strip())
    return files


def reshape_img(image, z, y, x):
    out = np.zeros([z, y, x], dtype=np.float32)
    out[0 : image.shape[0], 0 : image.shape[1], 0 : image.shape[2]] = image
    return out


def generate_map_kernel(ROI_shape):
    a = np.zeros(shape=ROI_shape)
    a = np.where(a == 0)
    map_kernel = 1.0 / (
        (a[0] - ROI_shape[0] // 2) ** 4
        + (a[1] - ROI_shape[1] // 2) ** 4
        + (a[2] - ROI_shape[2] // 2) ** 4
        + 1
    )
    map_kernel = np.reshape(map_kernel, newshape=(1, 1,) + ROI_shape)
    return map_kernel


# new predict process
def new_predict(model, image_dir, meanstd_filename, save_path, args, device, map_kernel_tensor):
    print("Predict test data")
    model.eval()
 
    file = read_file_from_txt(image_dir)
    file_num = len(file)
 
    stride = (args.ROI_shape[0] // 2, args.ROI_shape[1] // 2, args.ROI_shape[2] // 2)
    map_kernel = map_kernel_tensor.data.cpu().numpy()
 
    for t in range(file_num):
        image_path = file[t]
        print(image_path)

        # Load image and preserve SimpleITK metadata
        image = sitk.ReadImage(image_path)
        orig_spacing   = image.GetSpacing()
        orig_origin    = image.GetOrigin()
        orig_direction = image.GetDirection()
        image = sitk.GetArrayFromImage(image).astype(np.float32)
 
        # Normalise
        name = image_path[image_path.rfind("/") + 1:]
        mean, std = np.load(args.root_dir + meanstd_filename)
        image = (image - mean) / std
 
        # Pad if any dimension is smaller than ROI
        z_old, y_old, x_old = image.shape
        z = max(z_old, args.ROI_shape[0])
        y = max(y_old, args.ROI_shape[1])
        x = max(x_old, args.ROI_shape[2])

        if (z, y, x) != (z_old, y_old, x_old):
            image = reshape_img(image, z, y, x)
 
        # Accumulation buffers
        predict = np.zeros([1, args.n_classes, z, y, x], dtype=np.float32)
        n_map = np.zeros([1, args.n_classes, z, y, x], dtype=np.float32)
 
        # Add batch and channel dims for slicing (1, 1, Z, Y, X)
        image = image[np.newaxis, np.newaxis, :, :, :]
 
        # Build snap-to-edge patch positions
        slices = dense_patch_slices(image_size=(z, y, x), patch_size=args.ROI_shape, scan_interval=stride)
 
        # Inference loop
        # with torch.no_grad():
        #     for slice_z, slice_y, slice_x in slices:
        #         patch = image[:, :, slice_z, slice_y, slice_x] # (1, 1, sz, sy, sx)
        #         patch_tensor = torch.from_numpy(patch).to(device)
 
        #         output = model(patch_tensor) # (1, n_classes, sz, sy, sx)
        #         output_weighted = (output * map_kernel_tensor).data.cpu().numpy()
 
        #         predict[:, :, slice_z, slice_y, slice_x] += output_weighted
        #         n_map[:, :, slice_z, slice_y, slice_x] += map_kernel

        # patches_size = 0
        # patches = np.zeros([args.num_batch, args.n_channels, args.ROI_shape[0], args.ROI_shape[1], args.ROI_shape[2]])

        # with torch.no_grad():
        #     for slice_z, slice_y, slice_x in slices:
        #         patch = image[:, :, slice_z, slice_y, slice_x] # (1, n_channels, sz, sy, sx)
        #         patches[patches_size] = patch # (num_batch, n_channels, sz, sy, sx)
        #         patches_size += 1
        #         if patches_size == args.num_batch:
        #             patches_tensor = torch.from_numpy(patches).to(device)
        #             output = model(patches_tensor) # (num_batch, n_classes, sz, sy, sx)
        #             output_weighted = (output * map_kernel_tensor).data.cpu().numpy() # (num_batch, n_classes, sz, sy, sx)
 
        #             predict[:, :, slice_z, slice_y, slice_x] += output_weighted
        #             n_map[:, :, slice_z, slice_y, slice_x] += map_kernel

        with torch.no_grad():
            batch_patches = []   # accumulates patch tensors for current batch
            batch_slices  = []   # tracks which slices each patch came from
 
            def run_batch():
                if not batch_patches:
                    return
                # Stack into (N, 1, sz, sy, sx) and move to GPU in one transfer
                batch = torch.cat(batch_patches, dim=0).to(device, non_blocking=True) # (num_batch, n_channels, sz, sy, sx)
                output = model(batch) # (num_batch, n_classes, sz, sy, sx)
                output_weighted = (output * map_kernel_tensor).data.cpu().numpy() # (num_batch, n_classes, sz, sy, sx)
 
                # Scatter each patch result back into the accumulation arrays
                for patch_index, (slice_z, slice_y, slice_x) in enumerate(batch_slices):
                    predict[:, :, slice_z, slice_y, slice_x] += output_weighted[patch_index]
                    n_map[:, :, slice_z, slice_y, slice_x] += map_kernel
 
                batch_patches.clear()
                batch_slices.clear()
 
            for slice_z, slice_y, slice_x in slices:
                patch = image[:, :, slice_z, slice_y, slice_x] # (1, n_channels, sz, sy, sx)
                batch_patches.append(torch.from_numpy(patch.copy()))
                batch_slices.append((slice_z, slice_y, slice_x))
 
                if len(batch_patches) == args.predict_batch_size:
                    run_batch()
                    
            run_batch()
 
        # Blend, argmax, cast, crop
        predict = predict / n_map
        predict = np.argmax(predict[0], axis=0)
        predict = predict.astype(np.uint16)
        out = predict[0:z_old, 0:y_old, 0:x_old]
 
        # Restore SimpleITK metadata and save
        out = sitk.GetImageFromArray(out)
        out.SetSpacing(orig_spacing)
        out.SetOrigin(orig_origin)
        out.SetDirection(orig_direction)
        sitk.WriteImage(out, join(save_path, name))
    print("finish!")

# Predict process (takes in map_kernel rather than map_kernel_tensor)
# def predict(model, image_dir, meanstd_filename, save_path, args, map_kernel):
#     print("Predict test data")
#     model.eval()
#     file = read_file_from_txt(image_dir)
#     file_num = len(file)

#     for t in range(file_num):
#         image_path = file[t]
#         print(image_path)

#         image = sitk.ReadImage(image_path)
#         orig_spacing = image.GetSpacing()
#         orig_origin = image.GetOrigin()
#         orig_direction = image.GetDirection()
#         image = sitk.GetArrayFromImage(image)
#         image = image.astype(np.float32)

#         name = image_path[image_path.rfind("/") + 1 :]
#         mean, std = np.load(args.root_dir + meanstd_filename)
#         image = (image - mean) / std
#         z_old, y_old, x_old = image.shape
#         z = max(z_old, args.ROI_shape[0])
#         y = max(y_old, args.ROI_shape[1])
#         x = max(x_old, args.ROI_shape[2])

#         if (z, y, x) != (z_old, y_old, x_old):
#             image = reshape_img(image, z, y, x)

#         predict = np.zeros([1, args.n_classes, z, y, x], dtype=np.float32)
#         n_map = np.zeros([1, args.n_classes, z, y, x], dtype=np.float32)

#         """
#         Our prediction is carried out using sliding patches, 
#         and for each patch a corresponding result is predicted, 
#         and for the part where the patches overlap, 
#         we use weight <map_kernel> balance, 
#         and we agree that the closer to the center of the patch, the higher the weight
#         """

#         shape = args.ROI_shape
 
#         # print(np.max(map_kernal))
#         image = image[np.newaxis, np.newaxis, :, :, :]
#         stride_x = shape[0] // 2
#         stride_y = shape[1] // 2
#         stride_z = shape[2] // 2
#         for i in range(z // stride_x - 1):
#             for j in range(y // stride_y - 1):
#                 for k in range(x // stride_z - 1):
#                     image_i = image[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
#                               k * stride_z:k * stride_z + shape[2]]
#                     image_i = torch.from_numpy(image_i)
#                     if torch.cuda.is_available():
#                         image_i = image_i.cuda()
#                     with torch.no_grad():
#                         output = model(image_i)
#                     output = output.data.cpu().numpy()

#                     predict[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
#                     k * stride_z:k * stride_z + shape[2]] += output * map_kernel

#                     n_map[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
#                     k * stride_z:k * stride_z + shape[2]] += map_kernel

#                 image_i = image[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
#                           x - shape[2]:x]
#                 image_i = torch.from_numpy(image_i)
#                 if torch.cuda.is_available():
#                     image_i = image_i.cuda()
#                 with torch.no_grad():
#                     output = model(image_i)
#                 output = output.data.cpu().numpy()
#                 predict[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
#                 x - shape[2]:x] += output * map_kernel

#                 n_map[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
#                 x - shape[2]:x] += map_kernel

#             for k in range(x // stride_z - 1):
#                 image_i = image[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
#                           k * stride_z:k * stride_z + shape[2]]
#                 image_i = torch.from_numpy(image_i)
#                 if torch.cuda.is_available():
#                     image_i = image_i.cuda()
#                 with torch.no_grad():
#                     output = model(image_i)
#                 output = output.data.cpu().numpy()
#                 predict[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
#                 k * stride_z:k * stride_z + shape[2]] += output * map_kernel

#                 n_map[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
#                 k * stride_z:k * stride_z + shape[2]] += map_kernel

#             image_i = image[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x]
#             image_i = torch.from_numpy(image_i)
#             if torch.cuda.is_available():
#                 image_i = image_i.cuda()
#             with torch.no_grad():
#                 output = model(image_i)
#             output = output.data.cpu().numpy()

#             predict[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x] += output * map_kernel
#             n_map[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x] += map_kernel

#         for j in range(y // stride_y - 1):
#             for k in range((x - shape[2]) // stride_z):
#                 image_i = image[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
#                           k * stride_z:k * stride_z + shape[2]]
#                 image_i = torch.from_numpy(image_i)
#                 if torch.cuda.is_available():
#                     image_i = image_i.cuda()
#                 with torch.no_grad():
#                     output = model(image_i)
#                 output = output.data.cpu().numpy()

#                 predict[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
#                 k * stride_z:k * stride_z + shape[2]] += output * map_kernel

#                 n_map[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
#                 k * stride_z:k * stride_z + shape[2]] += map_kernel

#             image_i = image[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
#                       x - shape[2]:x]
#             image_i = torch.from_numpy(image_i)
#             if torch.cuda.is_available():
#                 image_i = image_i.cuda()
#             with torch.no_grad():
#                 output = model(image_i)
#             output = output.data.cpu().numpy()

#             predict[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
#             x - shape[2]:x] += output * map_kernel

#             n_map[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
#             x - shape[2]:x] += map_kernel

#         for k in range(x // stride_z - 1):
#             image_i = image[:, :, z - shape[0]:z, y - shape[1]:y,
#                       k * stride_z:k * stride_z + shape[2]]
#             image_i = torch.from_numpy(image_i)
#             if torch.cuda.is_available():
#                 image_i = image_i.cuda()
#             with torch.no_grad():
#                 output = model(image_i)
#             output = output.data.cpu().numpy()

#             predict[:, :, z - shape[0]:z, y - shape[1]:y,
#             k * stride_z:k * stride_z + shape[2]] += output * map_kernel

#             n_map[:, :, z - shape[0]:z, y - shape[1]:y,
#             k * stride_z:k * stride_z + shape[2]] += map_kernel

#         image_i = image[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x]
#         image_i = torch.from_numpy(image_i)
#         if torch.cuda.is_available():
#             image_i = image_i.cuda()
#         with torch.no_grad():
#             output = model(image_i)
#         output = output.data.cpu().numpy()

#         predict[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x] += output * map_kernel
#         n_map[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x] += map_kernel

#         predict = predict / n_map
#         predict = np.argmax(predict[0], axis=0)
#         predict = predict.astype(np.uint16)
#         out = predict[0:z_old, 0:y_old, 0:x_old]
#         out = sitk.GetImageFromArray(out)
#         out.SetSpacing(orig_spacing)
#         out.SetOrigin(orig_origin)
#         out.SetDirection(orig_direction)
#         sitk.WriteImage(out, join(save_path, name))
#     print("finish!")


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
    file = read_file_from_txt(label_dir)
    file_num = len(file)
    i = 0
    dice_vein = np.zeros(shape=(file_num), dtype=np.float32)
    dice_artery = np.zeros(shape=(file_num), dtype=np.float32)

    print("Dice:")
    for t in range(file_num):
        image_path = file[t]
        name = image_path[image_path.rfind('/') + 1:]
        predict = sitk.ReadImage(join(pred_dir, name))
        groundtruth = sitk.ReadImage(image_path)

        if predict.GetSize() == groundtruth.GetSize():
            predict = sitk.GetArrayFromImage(predict)
            groundtruth = sitk.GetArrayFromImage(groundtruth)
        else:
            predict, groundtruth = load_with_upsample(join(pred_dir, name), image_path)

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
        print(name, dice_vein[i], dice_artery[i])
        i += 1

    return dice_vein, dice_artery


def clDice(label_dir, pred_dir, radius=1):
    file = read_file_from_txt(label_dir)
    file_num = len(file)
    i = 0
    cl_Dice = np.zeros(shape=(file_num), dtype=np.float32)

    print("clDice:")
    for t in range(file_num):
        image_path = file[t]
        name = image_path[image_path.rfind('/') + 1:]
        predict = sitk.ReadImage(join(pred_dir, name))
        groundtruth = sitk.ReadImage(image_path)

        if predict.GetSize() == groundtruth.GetSize():
            predict = sitk.GetArrayFromImage(predict)
            groundtruth = sitk.GetArrayFromImage(groundtruth)
        else:
            predict, groundtruth = load_with_upsample(join(pred_dir, name), image_path)

        predict = predict.astype(bool)
        groundtruth = groundtruth.astype(bool)
        skel_pred = skeletonize(predict)
        skel_gt   = skeletonize(groundtruth)

        if skel_pred.sum() == 0 or skel_gt.sum() == 0:
            cl_Dice[i] = 0.0
            print(name, cl_Dice[i])
            i += 1
            continue

        #selem = ball(radius) if predict.ndim == 3 else disk(radius)
        #pred_dil = dilation(predict, selem)
        #gt_dil   = dilation(groundtruth, selem)

        tprec = np.logical_and(skel_pred, groundtruth).sum()  / skel_pred.sum()
        tsens = np.logical_and(skel_gt, predict).sum() / skel_gt.sum()

        cl_Dice[i] = 2 * tprec * tsens / (tprec + tsens)

        print(name, cl_Dice[i])
        i += 1
    
    return cl_Dice


def precision_recall_accuracy_score(label_dir, pred_dir):
    file = read_file_from_txt(label_dir)
    file_num = len(file)
    i = 0
    precision = np.zeros(shape=(file_num), dtype=np.float32)
    recall = np.zeros(shape=(file_num), dtype=np.float32)
    accuracy = np.zeros(shape=(file_num), dtype=np.float32)

    print("Precision, Recall, Accuracy:")
    for t in range(file_num):
        image_path = file[t]
        name = image_path[image_path.rfind('/') + 1:]
        predict = sitk.ReadImage(join(pred_dir, name))
        groundtruth = sitk.ReadImage(image_path)

        if predict.GetSize() == groundtruth.GetSize():
            predict = sitk.GetArrayFromImage(predict)
            groundtruth = sitk.GetArrayFromImage(groundtruth)
        else:
            predict, groundtruth = load_with_upsample(join(pred_dir, name), image_path)

        predict_flat = predict.flatten()
        groundtruth_flat = groundtruth.flatten()

        p = precision_score(groundtruth_flat, predict_flat, average="binary")
        r = recall_score(groundtruth_flat, predict_flat, average="binary")
        a = accuracy_score(groundtruth_flat, predict_flat)
        precision[i] = p
        recall[i] = r
        accuracy[i] = a
        print(name, precision[i], recall[i], accuracy[i])
        i += 1
    
    return precision, recall, accuracy


def Create_files(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.save_path_max):
        os.mkdir(args.save_path_max)


def Predict_Network(net, args, device, map_kernel_tensor):
    if torch.cuda.is_available():
        net = net.cuda()
    try:
        net.load_state_dict(
            torch.load(os.path.join(args.Dir_Weights, args.model_name_max))
        )
        print(os.path.join(args.Dir_Weights, args.model_name_max))
    except:
        print(
            "Warning 100: No parameters in weights_max, here use parameters in weights"
        )
        net.load_state_dict(torch.load(os.path.join(args.Dir_Weights, args.model_name)))
        print(os.path.join(args.Dir_Weights, args.model_name))

    dt = datetime.today()
    log_name = (
        str(dt.date())
        + "_"
        + str(dt.time().hour)
        + "."
        + str(dt.time().minute)
        + "."
        + str(dt.time().second)
        + "_"
        + args.log_name
    )
    logger = Get_logger(args.Dir_Log + log_name)

    logger.info("Start Prediction!")
    new_predict(net, args.Image_Te_txt, args.Te_Meanstd_name, args.save_path_max, args, device, map_kernel_tensor) # Added torch.no_grad()

    # Calculate Metrics
    dice = Dice(args.Label_Te_txt, args.save_path_max)
    dice_mean = np.mean(dice[0])
    dice_std = np.std(dice[0])
    cldice = clDice(args.Label_Te_txt, args.save_path_max)
    cldice_mean = np.mean(cldice)
    cldice_std = np.std(cldice)
    precision, recall, accuracy = precision_recall_accuracy_score(args.Label_Te_txt, args.save_path_max)
    precision_mean = np.mean(precision)
    precision_std = np.std(precision)
    recall_mean = np.mean(recall)
    recall_std = np.std(recall)
    accuracy_mean = np.mean(accuracy)
    accuracy_std = np.std(accuracy)

    # Log Metrics
    logger.info("Dice: " + np.array2string(dice[0], separator=","))
    logger.info("Dice mean: " + str(dice_mean))
    logger.info("Dice std: " + str(dice_std))
    logger.info("clDice: " + np.array2string(cldice, separator=","))
    logger.info("clDice mean: " + str(cldice_mean))
    logger.info("clDice std: " + str(cldice_std))
    logger.info("Precision: " + np.array2string(precision, separator=","))
    logger.info("Precision mean: " + str(precision_mean))
    logger.info("Precision std: " + str(precision_std))
    logger.info("Recall: " + np.array2string(recall, separator=","))
    logger.info("Recall mean: " + str(recall_mean))
    logger.info("Recall std: " + str(recall_std))
    logger.info("Accuracy: " + np.array2string(accuracy, separator=","))
    logger.info("Accuracy mean: " + str(accuracy_mean))
    logger.info("Accuracy std: " + str(accuracy_std))
    logger.info("Finish!")
    duration = (datetime.today()-dt).total_seconds()
    logger.info("Test Time (s): " + str(duration))
    Close_logger(logger)


def Train(args):
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_id 
    #removed above, replace with use of `export CUDA_VISIBLE_DEVICES=<num>` and `export OMP_NUM_THREADS=<num>` before running S0_Main.py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    
    net = DSCNet(
        n_channels=args.n_channels,
        n_classes=args.n_classes,
        kernel_size=args.kernel_size,
        extend_scope=args.extend_scope,
        if_offset=args.if_offset,
        device=device,
        number=args.n_basic_layer,
        dim=args.dim,
        epochs=args.n_epochs
    )
    Create_files(args)

    model_stats = summary(net, input_size=(args.batch_size, args.n_channels, args.ROI_shape[0], args.ROI_shape[1], args.ROI_shape[2]), device=device)
    dt = datetime.today()
    log_name = (
        str(dt.date())
        + "_"
        + str(dt.time().hour)
        + "."
        + str(dt.time().minute)
        + "."
        + str(dt.time().second)
        + "_"
        + args.log_name
    )
    logger = Get_logger_model(args.Dir_Log + log_name + "_model")
    for line in str(model_stats).splitlines():
         logger.info(line)
    Close_logger(logger)

    # map_kernel for repeated use during prediction (generate once based on ROI_shape)
    map_kernel = generate_map_kernel(args.ROI_shape)
    map_kernel_tensor = torch.from_numpy(map_kernel).to(device)

    if not args.if_onlytest:
        Train_net(net, args, device, map_kernel_tensor)
        Predict_Network(net, args, device, map_kernel_tensor)
    else:
        Predict_Network(net, args, device, map_kernel_tensor)



