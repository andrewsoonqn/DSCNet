# -*- coding: utf-8 -*-
import os
import torch
import logging
import numpy as np
from os.path import join
from pathlib import Path
import SimpleITK as sitk
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torchinfo import summary
from monai.losses import DiceCELoss, DiceLoss

from dscnet.training.checkpoints import (
    load_model_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from dscnet.models.optimized import DSCNet
from dscnet.data.dataset import Dataloader
from dscnet.evaluation.summary import log_summary, summarize_predictions
from dscnet.training.losses import cross_loss, dice_cross_loss, entropy_regularization_cross_loss, entropy_loss
from dscnet.evaluation.metrics import cldice_score, dice_score, to_minivess_binary_mask
from dscnet.evaluation.sliding_window import sliding_window_logits

import warnings

warnings.filterwarnings("ignore")


PIPELINE_NAME = "optimized"


def _checkpoint_path(args, name):
    return os.path.join(args.Dir_Weights, name)


def _load_checkpoint(net, args, name):
    path = _checkpoint_path(args, name)
    load_model_checkpoint(net, path, PIPELINE_NAME)
    print(path)


def _save_training_checkpoint(net, args, name, optimizer, scheduler, epoch, best_score):
    save_training_checkpoint(
        net,
        _checkpoint_path(args, name),
        PIPELINE_NAME,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=epoch,
        best_score=best_score,
        config_digest=args.config_digest,
        loop_state={},
    )


def _resume_training(net, args, optimizer, scheduler):
    state = load_training_checkpoint(
        net,
        _checkpoint_path(args, args.model_name),
        PIPELINE_NAME,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_config_digest=args.config_digest,
    )
    return state["epoch"] + 1, state["best_score"]


def _log_metric(args, name, value, step):
    tracker = getattr(args, "tracker", None)
    if tracker is not None:
        tracker.log_metric(name, value, step=step)


def _load_evaluation_checkpoint(net, args):
    best_path = _checkpoint_path(args, args.model_name_max)
    name = args.model_name_max if os.path.isfile(best_path) else args.model_name
    _load_checkpoint(net, args, name)


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
    dice_mean, dice_save = 0, 0
    validation_metrics = None
    start_epoch = args.start_train_epoch

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
    if not args.if_retrain:
        start_epoch, dice_save = _resume_training(net, args, optimizer, scheduler)

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
    for epoch in range(start_epoch, args.n_epochs + 1):
        loss = train_epoch(
            net, train_dataloader, optimizer, criterion, epoch, args.n_epochs, logger, args
        )
        _log_metric(args, "train.loss", loss, epoch)
        _log_metric(
            args, "training.learning_rate", optimizer.param_groups[0]["lr"], epoch
        )
        scheduler.step()
        is_best = False

        if epoch >= args.start_verify_epoch and (epoch % args.verify_gap) == 0:
            new_predict(net, args.Image_Va_txt, args.Meanstd_path, args.save_path, args, device, map_kernel_tensor)
            validation_metrics = summarize_predictions(
                args.Label_Va_txt, args.save_path, load_with_upsample
            )
            dice_mean = validation_metrics["dice"]
            log_summary(args, "validation", validation_metrics, step=epoch)
            if dice_mean > dice_save:
                dice_save = dice_mean
                is_best = True
        _save_training_checkpoint(
            net, args, args.model_name, optimizer, scheduler, epoch, dice_save
        )
        if is_best:
            _save_training_checkpoint(
                net,
                args,
                args.model_name_max,
                optimizer,
                scheduler,
                epoch,
                dice_save,
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
    return validation_metrics


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
def new_predict(
    model, image_dir, meanstd_path, save_path, args, device, map_kernel_tensor
):
    """Run the shared edge-safe inference path for the optimized model."""
    del map_kernel_tensor
    print("Predict test data")
    mean, std = np.load(meanstd_path)
    if not np.isfinite(std) or std == 0:
        raise ValueError("normalization standard deviation must be finite and non-zero")
    for image_path in read_file_from_txt(image_dir):
        print(image_path)
        source = sitk.ReadImage(image_path)
        normalized = (sitk.GetArrayFromImage(source).astype(np.float32) - mean) / std
        logits = sliding_window_logits(
            model,
            normalized,
            args.ROI_shape,
            args.n_classes,
            args.predict_batch_size,
            device,
        )
        prediction = np.argmax(logits, axis=0).astype(np.uint16)
        output = sitk.GetImageFromArray(prediction)
        output.CopyInformation(source)
        sitk.WriteImage(output, join(save_path, Path(image_path).name))
    print("finish!")


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
    files = read_file_from_txt(label_dir)
    scores = np.zeros(len(files), dtype=np.float32)

    print("Dice:")
    for index, image_path in enumerate(files):
        name = os.path.basename(image_path)
        prediction_image = sitk.ReadImage(join(pred_dir, name))
        target_image = sitk.ReadImage(image_path)

        if prediction_image.GetSize() == target_image.GetSize():
            prediction = sitk.GetArrayFromImage(prediction_image)
            target = sitk.GetArrayFromImage(target_image)
        else:
            prediction, target = load_with_upsample(
                join(pred_dir, name), image_path
            )

        scores[index] = dice_score(prediction, target)
        print(name, scores[index])

    return scores


def clDice(label_dir, pred_dir):
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

        cl_Dice[i] = cldice_score(predict, groundtruth)

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

        predict_flat = to_minivess_binary_mask(
            predict, name=f"prediction {name}"
        ).flatten()
        groundtruth_flat = to_minivess_binary_mask(
            groundtruth, name=f"target {name}"
        ).flatten()

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
    _load_evaluation_checkpoint(net, args)

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
    new_predict(net, args.Image_Te_txt, args.Meanstd_path, args.save_path_max, args, device, map_kernel_tensor) # Added torch.no_grad()

    test_metrics = summarize_predictions(
        args.Label_Te_txt, args.save_path_max, load_with_upsample
    )
    log_summary(args, "test", test_metrics)
    # Calculate Metrics
    dice = Dice(args.Label_Te_txt, args.save_path_max)
    dice_mean = np.mean(dice)
    dice_std = np.std(dice)
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
    logger.info("Dice: " + np.array2string(dice, separator=","))
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
    return test_metrics


def _build_model(args):
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
        epochs=args.n_epochs,
    )
    map_kernel = generate_map_kernel(args.ROI_shape)
    map_kernel_tensor = torch.from_numpy(map_kernel).to(device)
    return net, device, map_kernel_tensor


def _log_model_summary(net, args, device):
    model_stats = summary(
        net,
        input_size=(args.batch_size, args.n_channels, *args.ROI_shape),
        device=device,
    )
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


def Train(args):
    net, device, map_kernel_tensor = _build_model(args)
    Create_files(args)
    _log_model_summary(net, args, device)
    return Train_net(net, args, device, map_kernel_tensor)


def Evaluate(args):
    net, device, map_kernel_tensor = _build_model(args)
    Create_files(args)
    return Predict_Network(net, args, device, map_kernel_tensor)



