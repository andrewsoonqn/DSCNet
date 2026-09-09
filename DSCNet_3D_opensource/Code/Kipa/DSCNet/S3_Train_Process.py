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
from sklearn.metrics import precision_score, recall_score, accuracy_score
# from torchinfo import summary

from S3_Checkpoint import (
    load_model_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from S3_DSCNet import DSCNet
from S3_Dataloader import Dataloader
from S3_Evaluation_Metrics import log_summary, summarize_predictions
from S3_Loss import cross_loss
from S3_Metrics import cldice_score, dice_score, to_minivess_binary_mask

import warnings

warnings.filterwarnings("ignore")


PIPELINE_NAME = "standard"


def _checkpoint_path(args, name):
    return os.path.join(args.Dir_Weights, name)


def _load_checkpoint(net, args, name):
    path = _checkpoint_path(args, name)
    load_model_checkpoint(net, path, PIPELINE_NAME)
    print(path)


def _save_training_checkpoint(
    net, args, name, optimizer, scheduler, scaler, epoch, best_score, loop_state
):
    save_training_checkpoint(
        net,
        _checkpoint_path(args, name),
        PIPELINE_NAME,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        best_score=best_score,
        config_digest=args.config_digest,
        loop_state=loop_state,
    )


def _resume_training(net, args, optimizer, scheduler, scaler):
    state = load_training_checkpoint(
        net,
        _checkpoint_path(args, args.model_name),
        PIPELINE_NAME,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_config_digest=args.config_digest,
    )
    loop_state = state["loop_state"]
    return (
        state["epoch"] + 1,
        state["best_score"],
        loop_state.get("dice_max", state["best_score"]),
        loop_state.get("early_stopping_counter", 0),
    )


def _log_metric(args, name, value, step):
    tracker = getattr(args, "tracker", None)
    if tracker is not None:
        tracker.log_metric(name, value, step=step)


def _early_stopping_reached(args, counter):
    return args.use_earlystop and counter >= args.earlystop_patience


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
def train_epoch(model, loader, optimizer, criterion, epoch, n_epochs):
    losses = AverageMeter()

    model.train()
    for batch_idx, (image, label) in enumerate(loader):
        if torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()
        optimizer.zero_grad()
        model.zero_grad()

        output = model(image)
        loss = criterion(label, output)
        losses.update(loss.data, label.size(0))

        loss.backward()
        optimizer.step()

        res = "\t".join(
            [
                "Epoch: [%d/%d]" % (epoch, n_epochs),
                "Iter: [%d/%d]" % (batch_idx + 1, len(loader)),
                "Lr: [%f]" % (optimizer.param_groups[0]["lr"]),
                "Loss %f" % (losses.avg),
            ]
        )
        print(res)
    return losses.avg


# One epoch in training process (with AMP implementation)
def train_epoch_amp(model, loader, optimizer, criterion, scaler, epoch, n_epochs):
    losses = AverageMeter()

    model.train()
    for batch_idx, (image, label) in enumerate(loader):
        if torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()
        optimizer.zero_grad()
        model.zero_grad()

        with torch.amp.autocast(device_type="cuda"):
            output = model(image)
            loss = criterion(label, output)

        losses.update(loss.data, label.size(0))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        res = "\t".join(
            [
                "Epoch: [%d/%d]" % (epoch, n_epochs),
                "Iter: [%d/%d]" % (batch_idx + 1, len(loader)),
                "Lr: [%f]" % (optimizer.param_groups[0]["lr"]),
                "Loss %f" % (losses.avg),
            ]
        )
        print(res)
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


def Close_logger(logger):
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


# Train process
def Train_net(net, args):
    dice_mean, dice_save, dice_max = 0, 0, 0
    validation_metrics = None
    counter = 0
    start_epoch = args.start_train_epoch

    if torch.cuda.is_available():
        net = net.cuda()

    # Load dataset
    train_dataset = Dataloader(args)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )  # num_workers=8, persistent_workers=True (slower)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, betas=(0.9, 0.95)
    )  # try weight_decay
    # It is possible to choose whether to use a dynamic learning rate,
    # which was not used in our original experiment, but you can choose to use
    scheduler = (
        ReduceLROnPlateau(  # original values: mode="min", factor=0.8, patience=50
            optimizer,
            mode="max",
            factor=args.rlr_factor,
            threshold=args.rlr_threshold,
            patience=args.rlr_patience,
            cooldown=args.rlr_cooldown,
            min_lr=0.000001,
        )
    )

    criterion = cross_loss()
    if not args.if_retrain:
        start_epoch, dice_save, dice_max, counter = _resume_training(
            net, args, optimizer, scheduler, scaler=None
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
    logger = Get_logger(args.Dir_Log + log_name)
    logger.info("start training!")

    # Early stopping mechanism
    min_delta = args.earlystop_threshold
    patience = args.earlystop_patience
    if _early_stopping_reached(args, counter):
        logger.info("checkpoint already reached the early-stopping condition")
        Close_logger(logger)
        return validation_metrics

    # Main train process
    for epoch in range(start_epoch, args.n_epochs + 1):
        loss = train_epoch(
            net, train_dataloader, optimizer, criterion, epoch, args.n_epochs
        )
        _log_metric(args, "train.loss", loss, epoch)
        _log_metric(
            args, "training.learning_rate", optimizer.param_groups[0]["lr"], epoch
        )
        is_best = False

        if epoch >= args.start_verify_epoch:
            # The validation set is selected according to the task
            predict(net, args.Image_Va_txt, args.Meanstd_path, args.save_path, args)
            validation_metrics = summarize_predictions(
                args.Label_Va_txt, args.save_path, load_with_upsample
            )
            dice_mean = validation_metrics["dice"]
            log_summary(args, "validation", validation_metrics, step=epoch)
            if dice_mean > dice_save:
                dice_save = dice_mean
                is_best = True
            if dice_mean > dice_max + min_delta:
                dice_max = dice_mean
                counter = 0
            else:
                counter += 1
            if args.use_rlrop:
                scheduler.step(dice_mean)  # for ReduceLROnPlateau
        loop_state = {
            "dice_max": dice_max,
            "early_stopping_counter": counter,
        }
        _save_training_checkpoint(
            net,
            args,
            args.model_name,
            optimizer,
            scheduler,
            None,
            epoch,
            dice_save,
            loop_state,
        )
        if is_best:
            _save_training_checkpoint(
                net,
                args,
                args.model_name_max,
                optimizer,
                scheduler,
                None,
                epoch,
                dice_save,
                loop_state,
            )
        logger.info(
            "Epoch:[{}/{}]  lr={:.6f}  loss={:.5f}  counter={} dice_mean={:.4f} "
            "max_dice={:.4f} saved_dice={:.4f}".format(
                epoch,
                args.n_epochs,
                optimizer.param_groups[0]["lr"],
                loss,
                counter,
                dice_mean,
                dice_max,
                dice_save,
            )
        )
        if _early_stopping_reached(args, counter):
            logger.info("Early stopping triggered!")
            break
    logger.info("finish training!")
    Close_logger(logger)
    return validation_metrics


# Train process with AMP
def Train_net_amp(net, args):
    dice_mean, dice_save, dice_max = 0, 0, 0
    validation_metrics = None
    counter = 0
    start_epoch = args.start_train_epoch

    if torch.cuda.is_available():
        net = net.cuda()

    # Load dataset
    train_dataset = Dataloader(args)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    scaler = torch.amp.GradScaler(device="cuda")  # for AMP implementation

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.95))
    # It is possible to choose whether to use a dynamic learning rate,
    # which was not used in our original experiment, but you can choose to use
    scheduler = (
        ReduceLROnPlateau(  # original values: mode="min", factor=0.8, patience=50
            optimizer,
            mode="max",
            factor=args.rlr_factor,
            threshold=args.rlr_threshold,
            patience=args.rlr_patience,
            cooldown=args.rlr_cooldown,
            min_lr=0.000001,
        )
    )
    criterion = cross_loss()
    if not args.if_retrain:
        start_epoch, dice_save, dice_max, counter = _resume_training(
            net, args, optimizer, scheduler, scaler
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
    logger = Get_logger(args.Dir_Log + log_name)
    logger.info("start training!")

    # Early stopping mechanism
    min_delta = args.earlystop_threshold
    patience = args.earlystop_patience
    if _early_stopping_reached(args, counter):
        logger.info("checkpoint already reached the early-stopping condition")
        Close_logger(logger)
        return validation_metrics

    # Main train process
    for epoch in range(start_epoch, args.n_epochs + 1):
        loss = train_epoch_amp(
            net, train_dataloader, optimizer, criterion, scaler, epoch, args.n_epochs
        )
        _log_metric(args, "train.loss", loss, epoch)
        _log_metric(
            args, "training.learning_rate", optimizer.param_groups[0]["lr"], epoch
        )
        is_best = False

        if epoch >= args.start_verify_epoch:
            # The validation set is selected according to the task
            predict_amp(
                net, args.Image_Va_txt, args.Meanstd_path, args.save_path, args
            )
            validation_metrics = summarize_predictions(
                args.Label_Va_txt, args.save_path, load_with_upsample
            )
            dice_mean = validation_metrics["dice"]
            log_summary(args, "validation", validation_metrics, step=epoch)
            if dice_mean > dice_save:
                dice_save = dice_mean
                is_best = True
            if dice_mean > dice_max + min_delta:
                dice_max = dice_mean
                counter = 0
            else:
                counter += 1
            scheduler.step(dice_mean)
        loop_state = {
            "dice_max": dice_max,
            "early_stopping_counter": counter,
        }
        _save_training_checkpoint(
            net,
            args,
            args.model_name,
            optimizer,
            scheduler,
            scaler,
            epoch,
            dice_save,
            loop_state,
        )
        if is_best:
            _save_training_checkpoint(
                net,
                args,
                args.model_name_max,
                optimizer,
                scheduler,
                scaler,
                epoch,
                dice_save,
                loop_state,
            )
        logger.info(
            "Epoch:[{}/{}]  lr={:.6f}  loss={:.5f}  dice_mean={:.4f} "
            "max_dice={:.4f} saved_dice={:.4f}".format(
                epoch,
                args.n_epochs,
                optimizer.param_groups[0]["lr"],
                loss,
                dice_mean,
                dice_max,
                dice_save,
            )
        )
        if _early_stopping_reached(args, counter):
            logger.info("Early stopping triggered!")
            break
    logger.info("finish training!")
    Close_logger(logger)
    return validation_metrics


def read_file_from_txt(txt_path):  # 从txt里读取数据
    files = []
    for line in open(txt_path, "r"):
        files.append(line.strip())
    return files


def reshape_img(image, z, y, x):
    out = np.zeros([z, y, x], dtype=np.float32)
    out[0 : image.shape[0], 0 : image.shape[1], 0 : image.shape[2]] = image[
        0 : image.shape[0], 0 : image.shape[1], 0 : image.shape[2]
    ]
    return out


# Predict process
def predict(model, image_dir, meanstd_path, save_path, args):
    print("Predict test data")
    model.eval()
    file = read_file_from_txt(image_dir)
    file_num = len(file)

    for t in range(file_num):
        image_path = file[t]
        print(image_path)

        image = sitk.ReadImage(image_path)
        orig_spacing = image.GetSpacing()
        orig_origin = image.GetOrigin()
        orig_direction = image.GetDirection()
        image = sitk.GetArrayFromImage(image)
        image = image.astype(np.float32)

        name = image_path[image_path.rfind("/") + 1 :]
        mean, std = np.load(meanstd_path)
        image = (image - mean) / std
        z, y, x = image.shape
        z_old, y_old, x_old = z, y, x

        if args.ROI_shape[0] > z:
            z = args.ROI_shape[0]
            image = reshape_img(image, z, y, x)
        if args.ROI_shape[1] > y:
            y = args.ROI_shape[1]
            image = reshape_img(image, z, y, x)
        if args.ROI_shape[2] > x:
            x = args.ROI_shape[2]
            image = reshape_img(image, z, y, x)

        predict = np.zeros([1, args.n_classes, z, y, x], dtype=np.float32)
        n_map = np.zeros([1, args.n_classes, z, y, x], dtype=np.float32)

        """
        Our prediction is carried out using sliding patches, 
        and for each patch a corresponding result is predicted, 
        and for the part where the patches overlap, 
        we use weight <map_kernel> balance, 
        and we agree that the closer to the center of the patch, the higher the weight
        """

        shape = args.ROI_shape
        a = np.zeros(shape=shape)
        a = np.where(a == 0)
        map_kernal = 1 / (
            (a[0] - shape[0] // 2) ** 4
            + (a[1] - shape[1] // 2) ** 4
            + (a[2] - shape[2] // 2) ** 4
            + 1
        )
        map_kernal = np.reshape(
            map_kernal,
            newshape=(
                1,
                1,
            )
            + shape,
        )

        # print(np.max(map_kernal))
        image = image[np.newaxis, np.newaxis, :, :, :]
        stride_x = shape[0] // 2
        stride_y = shape[1] // 2
        stride_z = shape[2] // 2
        for i in range(z // stride_x - 1):
            for j in range(y // stride_y - 1):
                for k in range(x // stride_z - 1):
                    image_i = image[
                        :,
                        :,
                        i * stride_x : i * stride_x + shape[0],
                        j * stride_y : j * stride_y + shape[1],
                        k * stride_z : k * stride_z + shape[2],
                    ]
                    image_i = torch.from_numpy(image_i)
                    if torch.cuda.is_available():
                        image_i = image_i.cuda()
                    with torch.no_grad():
                        output = model(image_i)
                    output = output.data.cpu().numpy()

                    predict[
                        :,
                        :,
                        i * stride_x : i * stride_x + shape[0],
                        j * stride_y : j * stride_y + shape[1],
                        k * stride_z : k * stride_z + shape[2],
                    ] += output * map_kernal

                    n_map[
                        :,
                        :,
                        i * stride_x : i * stride_x + shape[0],
                        j * stride_y : j * stride_y + shape[1],
                        k * stride_z : k * stride_z + shape[2],
                    ] += map_kernal

                image_i = image[
                    :,
                    :,
                    i * stride_x : i * stride_x + shape[0],
                    j * stride_y : j * stride_y + shape[1],
                    x - shape[2] : x,
                ]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                with torch.no_grad():
                    output = model(image_i)
                output = output.data.cpu().numpy()
                predict[
                    :,
                    :,
                    i * stride_x : i * stride_x + shape[0],
                    j * stride_y : j * stride_y + shape[1],
                    x - shape[2] : x,
                ] += output * map_kernal

                n_map[
                    :,
                    :,
                    i * stride_x : i * stride_x + shape[0],
                    j * stride_y : j * stride_y + shape[1],
                    x - shape[2] : x,
                ] += map_kernal

            for k in range(x // stride_z - 1):
                image_i = image[
                    :,
                    :,
                    i * stride_x : i * stride_x + shape[0],
                    y - shape[1] : y,
                    k * stride_z : k * stride_z + shape[2],
                ]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                with torch.no_grad():
                    output = model(image_i)
                output = output.data.cpu().numpy()
                predict[
                    :,
                    :,
                    i * stride_x : i * stride_x + shape[0],
                    y - shape[1] : y,
                    k * stride_z : k * stride_z + shape[2],
                ] += output * map_kernal

                n_map[
                    :,
                    :,
                    i * stride_x : i * stride_x + shape[0],
                    y - shape[1] : y,
                    k * stride_z : k * stride_z + shape[2],
                ] += map_kernal

            image_i = image[
                :,
                :,
                i * stride_x : i * stride_x + shape[0],
                y - shape[1] : y,
                x - shape[2] : x,
            ]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            with torch.no_grad():
                output = model(image_i)
            output = output.data.cpu().numpy()

            predict[
                :,
                :,
                i * stride_x : i * stride_x + shape[0],
                y - shape[1] : y,
                x - shape[2] : x,
            ] += output * map_kernal
            n_map[
                :,
                :,
                i * stride_x : i * stride_x + shape[0],
                y - shape[1] : y,
                x - shape[2] : x,
            ] += map_kernal

        for j in range(y // stride_y - 1):
            for k in range((x - shape[2]) // stride_z):
                image_i = image[
                    :,
                    :,
                    z - shape[0] : z,
                    j * stride_y : j * stride_y + shape[1],
                    k * stride_z : k * stride_z + shape[2],
                ]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                with torch.no_grad():
                    output = model(image_i)
                output = output.data.cpu().numpy()

                predict[
                    :,
                    :,
                    z - shape[0] : z,
                    j * stride_y : j * stride_y + shape[1],
                    k * stride_z : k * stride_z + shape[2],
                ] += output * map_kernal

                n_map[
                    :,
                    :,
                    z - shape[0] : z,
                    j * stride_y : j * stride_y + shape[1],
                    k * stride_z : k * stride_z + shape[2],
                ] += map_kernal

            image_i = image[
                :,
                :,
                z - shape[0] : z,
                j * stride_y : j * stride_y + shape[1],
                x - shape[2] : x,
            ]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            with torch.no_grad():
                output = model(image_i)
            output = output.data.cpu().numpy()

            predict[
                :,
                :,
                z - shape[0] : z,
                j * stride_y : j * stride_y + shape[1],
                x - shape[2] : x,
            ] += output * map_kernal

            n_map[
                :,
                :,
                z - shape[0] : z,
                j * stride_y : j * stride_y + shape[1],
                x - shape[2] : x,
            ] += map_kernal

        for k in range(x // stride_z - 1):
            image_i = image[
                :,
                :,
                z - shape[0] : z,
                y - shape[1] : y,
                k * stride_z : k * stride_z + shape[2],
            ]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            with torch.no_grad():
                output = model(image_i)
            output = output.data.cpu().numpy()

            predict[
                :,
                :,
                z - shape[0] : z,
                y - shape[1] : y,
                k * stride_z : k * stride_z + shape[2],
            ] += output * map_kernal

            n_map[
                :,
                :,
                z - shape[0] : z,
                y - shape[1] : y,
                k * stride_z : k * stride_z + shape[2],
            ] += map_kernal

        image_i = image[:, :, z - shape[0] : z, y - shape[1] : y, x - shape[2] : x]
        image_i = torch.from_numpy(image_i)
        if torch.cuda.is_available():
            image_i = image_i.cuda()
        with torch.no_grad():
            output = model(image_i)
        output = output.data.cpu().numpy()

        predict[:, :, z - shape[0] : z, y - shape[1] : y, x - shape[2] : x] += (
            output * map_kernal
        )
        n_map[:, :, z - shape[0] : z, y - shape[1] : y, x - shape[2] : x] += map_kernal

        predict = predict / n_map
        predict = np.argmax(predict[0], axis=0)
        predict = predict.astype(np.uint16)
        out = predict[0:z_old, 0:y_old, 0:x_old]
        out = sitk.GetImageFromArray(out)
        out.SetSpacing(orig_spacing)
        out.SetOrigin(orig_origin)
        out.SetDirection(orig_direction)
        sitk.WriteImage(out, join(save_path, name))
    print("finish!")


# Predict process (with AMP implementation)
def predict_amp(model, image_dir, meanstd_path, save_path, args):
    print("Predict test data")
    model.eval()
    file = read_file_from_txt(image_dir)
    file_num = len(file)

    for t in range(file_num):
        image_path = file[t]
        print(image_path)

        image = sitk.ReadImage(image_path)
        orig_spacing = image.GetSpacing()
        orig_origin = image.GetOrigin()
        orig_direction = image.GetDirection()
        image = sitk.GetArrayFromImage(image)
        image = image.astype(np.float32)

        name = image_path[image_path.rfind("/") + 1 :]
        mean, std = np.load(meanstd_path)
        image = (image - mean) / std
        z, y, x = image.shape
        z_old, y_old, x_old = z, y, x

        if args.ROI_shape[0] > z:
            z = args.ROI_shape[0]
            image = reshape_img(image, z, y, x)
        if args.ROI_shape[1] > y:
            y = args.ROI_shape[1]
            image = reshape_img(image, z, y, x)
        if args.ROI_shape[2] > x:
            x = args.ROI_shape[2]
            image = reshape_img(image, z, y, x)

        predict = np.zeros([1, args.n_classes, z, y, x], dtype=np.float32)
        n_map = np.zeros([1, args.n_classes, z, y, x], dtype=np.float32)

        """
        Our prediction is carried out using sliding patches, 
        and for each patch a corresponding result is predicted, 
        and for the part where the patches overlap, 
        we use weight <map_kernel> balance, 
        and we agree that the closer to the center of the patch, the higher the weight
        """

        shape = args.ROI_shape
        a = np.zeros(shape=shape)
        a = np.where(a == 0)
        map_kernal = 1 / (
            (a[0] - shape[0] // 2) ** 4
            + (a[1] - shape[1] // 2) ** 4
            + (a[2] - shape[2] // 2) ** 4
            + 1
        )
        map_kernal = np.reshape(
            map_kernal,
            newshape=(
                1,
                1,
            )
            + shape,
        )

        # print(np.max(map_kernal))
        image = image[np.newaxis, np.newaxis, :, :, :]
        stride_x = shape[0] // 2
        stride_y = shape[1] // 2
        stride_z = shape[2] // 2
        for i in range(z // stride_x - 1):
            for j in range(y // stride_y - 1):
                for k in range(x // stride_z - 1):
                    image_i = image[
                        :,
                        :,
                        i * stride_x : i * stride_x + shape[0],
                        j * stride_y : j * stride_y + shape[1],
                        k * stride_z : k * stride_z + shape[2],
                    ]
                    image_i = torch.from_numpy(image_i)
                    if torch.cuda.is_available():
                        image_i = image_i.cuda()
                    with torch.no_grad():
                        with torch.amp.autocast(device_type="cuda"):
                            output = model(image_i)
                    output = output.data.cpu().numpy()

                    predict[
                        :,
                        :,
                        i * stride_x : i * stride_x + shape[0],
                        j * stride_y : j * stride_y + shape[1],
                        k * stride_z : k * stride_z + shape[2],
                    ] += output * map_kernal

                    n_map[
                        :,
                        :,
                        i * stride_x : i * stride_x + shape[0],
                        j * stride_y : j * stride_y + shape[1],
                        k * stride_z : k * stride_z + shape[2],
                    ] += map_kernal

                image_i = image[
                    :,
                    :,
                    i * stride_x : i * stride_x + shape[0],
                    j * stride_y : j * stride_y + shape[1],
                    x - shape[2] : x,
                ]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda"):
                        output = model(image_i)
                output = output.data.cpu().numpy()
                predict[
                    :,
                    :,
                    i * stride_x : i * stride_x + shape[0],
                    j * stride_y : j * stride_y + shape[1],
                    x - shape[2] : x,
                ] += output * map_kernal

                n_map[
                    :,
                    :,
                    i * stride_x : i * stride_x + shape[0],
                    j * stride_y : j * stride_y + shape[1],
                    x - shape[2] : x,
                ] += map_kernal

            for k in range(x // stride_z - 1):
                image_i = image[
                    :,
                    :,
                    i * stride_x : i * stride_x + shape[0],
                    y - shape[1] : y,
                    k * stride_z : k * stride_z + shape[2],
                ]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda"):
                        output = model(image_i)
                output = output.data.cpu().numpy()
                predict[
                    :,
                    :,
                    i * stride_x : i * stride_x + shape[0],
                    y - shape[1] : y,
                    k * stride_z : k * stride_z + shape[2],
                ] += output * map_kernal

                n_map[
                    :,
                    :,
                    i * stride_x : i * stride_x + shape[0],
                    y - shape[1] : y,
                    k * stride_z : k * stride_z + shape[2],
                ] += map_kernal

            image_i = image[
                :,
                :,
                i * stride_x : i * stride_x + shape[0],
                y - shape[1] : y,
                x - shape[2] : x,
            ]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda"):
                    output = model(image_i)
            output = output.data.cpu().numpy()

            predict[
                :,
                :,
                i * stride_x : i * stride_x + shape[0],
                y - shape[1] : y,
                x - shape[2] : x,
            ] += output * map_kernal
            n_map[
                :,
                :,
                i * stride_x : i * stride_x + shape[0],
                y - shape[1] : y,
                x - shape[2] : x,
            ] += map_kernal

        for j in range(y // stride_y - 1):
            for k in range((x - shape[2]) // stride_z):
                image_i = image[
                    :,
                    :,
                    z - shape[0] : z,
                    j * stride_y : j * stride_y + shape[1],
                    k * stride_z : k * stride_z + shape[2],
                ]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda"):
                        output = model(image_i)
                output = output.data.cpu().numpy()

                predict[
                    :,
                    :,
                    z - shape[0] : z,
                    j * stride_y : j * stride_y + shape[1],
                    k * stride_z : k * stride_z + shape[2],
                ] += output * map_kernal

                n_map[
                    :,
                    :,
                    z - shape[0] : z,
                    j * stride_y : j * stride_y + shape[1],
                    k * stride_z : k * stride_z + shape[2],
                ] += map_kernal

            image_i = image[
                :,
                :,
                z - shape[0] : z,
                j * stride_y : j * stride_y + shape[1],
                x - shape[2] : x,
            ]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda"):
                    output = model(image_i)
            output = output.data.cpu().numpy()

            predict[
                :,
                :,
                z - shape[0] : z,
                j * stride_y : j * stride_y + shape[1],
                x - shape[2] : x,
            ] += output * map_kernal

            n_map[
                :,
                :,
                z - shape[0] : z,
                j * stride_y : j * stride_y + shape[1],
                x - shape[2] : x,
            ] += map_kernal

        for k in range(x // stride_z - 1):
            image_i = image[
                :,
                :,
                z - shape[0] : z,
                y - shape[1] : y,
                k * stride_z : k * stride_z + shape[2],
            ]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda"):
                    output = model(image_i)
            output = output.data.cpu().numpy()

            predict[
                :,
                :,
                z - shape[0] : z,
                y - shape[1] : y,
                k * stride_z : k * stride_z + shape[2],
            ] += output * map_kernal

            n_map[
                :,
                :,
                z - shape[0] : z,
                y - shape[1] : y,
                k * stride_z : k * stride_z + shape[2],
            ] += map_kernal

        image_i = image[:, :, z - shape[0] : z, y - shape[1] : y, x - shape[2] : x]
        image_i = torch.from_numpy(image_i)
        if torch.cuda.is_available():
            image_i = image_i.cuda()
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda"):
                output = model(image_i)
        output = output.data.cpu().numpy()

        predict[:, :, z - shape[0] : z, y - shape[1] : y, x - shape[2] : x] += (
            output * map_kernal
        )
        n_map[:, :, z - shape[0] : z, y - shape[1] : y, x - shape[2] : x] += map_kernal

        predict = predict / n_map
        predict = np.argmax(predict[0], axis=0)
        predict = predict.astype(np.uint16)
        out = predict[0:z_old, 0:y_old, 0:x_old]
        out = sitk.GetImageFromArray(out)
        out.SetSpacing(orig_spacing)
        out.SetOrigin(orig_origin)
        out.SetDirection(orig_direction)
        sitk.WriteImage(out, join(save_path, name))
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
    new_spacing = tuple(
        ref_spacing[i] * (ref_size[i] / pred_size[i]) for i in range(3)
    )  # setting scale for upsampling
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
        name = image_path[image_path.rfind("/") + 1 :]
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
        name = image_path[image_path.rfind("/") + 1 :]
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


def Predict_Network(net, args):
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
    predict(
        net, args.Image_Te_txt, args.Meanstd_path, args.save_path_max, args
    )  # Added torch.no_grad()

    test_metrics = summarize_predictions(
        args.Label_Te_txt, args.save_path_max, load_with_upsample
    )
    log_summary(args, "test", test_metrics)
    dice = Dice(args.Label_Te_txt, args.save_path_max)
    dice_mean = np.mean(dice)
    cldice = clDice(args.Label_Te_txt, args.save_path_max)
    cldice_mean = np.mean(cldice)
    precision, recall, accuracy = precision_recall_accuracy_score(
        args.Label_Te_txt, args.save_path_max
    )
    precision_mean = np.mean(precision)
    recall_mean = np.mean(recall)
    accuracy_mean = np.mean(accuracy)
    logger.info("Dice: " + np.array2string(dice, separator=","))
    logger.info("Dice mean: " + str(dice_mean))
    logger.info("clDice: " + np.array2string(cldice, separator=","))
    logger.info("clDice mean: " + str(cldice_mean))
    logger.info("Precision: " + np.array2string(precision, separator=","))
    logger.info("Precision mean: " + str(precision_mean))
    logger.info("Recall: " + np.array2string(recall, separator=","))
    logger.info("Recall mean: " + str(recall_mean))
    logger.info("Accuracy: " + np.array2string(accuracy, separator=","))
    logger.info("Accuracy mean: " + str(accuracy_mean))
    logger.info("Finish!")
    Close_logger(logger)
    return test_metrics


# AMP implementation
def Predict_Network_amp(net, args):
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
    predict_amp(
        net, args.Image_Te_txt, args.Meanstd_path, args.save_path_max, args
    )  # Added torch.no_grad()

    test_metrics = summarize_predictions(
        args.Label_Te_txt, args.save_path_max, load_with_upsample
    )
    log_summary(args, "test", test_metrics)
    dice = Dice(args.Label_Te_txt, args.save_path_max)
    dice_mean = np.mean(dice)
    cldice = clDice(args.Label_Te_txt, args.save_path_max)
    cldice_mean = np.mean(cldice)
    precision, recall, accuracy = precision_recall_accuracy_score(
        args.Label_Te_txt, args.save_path_max
    )
    precision_mean = np.mean(precision)
    recall_mean = np.mean(recall)
    accuracy_mean = np.mean(accuracy)
    logger.info("Dice: " + np.array2string(dice, separator=","))
    logger.info("Dice mean: " + str(dice_mean))
    logger.info("clDice: " + np.array2string(cldice, separator=","))
    logger.info("clDice mean: " + str(cldice_mean))
    logger.info("Precision: " + np.array2string(precision, separator=","))
    logger.info("Precision mean: " + str(precision_mean))
    logger.info("Recall: " + np.array2string(recall, separator=","))
    logger.info("Recall mean: " + str(recall_mean))
    logger.info("Accuracy: " + np.array2string(accuracy, separator=","))
    logger.info("Accuracy mean: " + str(accuracy_mean))
    logger.info("Finish!")
    Close_logger(logger)
    return test_metrics


def _build_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    return DSCNet(
        n_channels=args.n_channels,
        n_classes=args.n_classes,
        kernel_size=args.kernel_size,
        extend_scope=args.extend_scope,
        if_offset=args.if_offset,
        device=device,
        number=args.n_basic_layer,
        dim=args.dim,
        unet_layers=args.unet_layers,
    )


def Train(args):
    net = _build_model(args)
    Create_files(args)
    if args.if_fullprecision:
        return Train_net(net, args)
    return Train_net_amp(net, args)


def Evaluate(args):
    net = _build_model(args)
    Create_files(args)
    if args.if_fullprecision:
        return Predict_Network(net, args)
    return Predict_Network_amp(net, args)
