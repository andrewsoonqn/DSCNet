# -*- coding: utf-8 -*-
import os
import argparse
from pathlib import Path

"""
This code contains all the "Parameters" for the entire project -- <DSCNet>
Code Introduction: (The easiest way to run a code!)
    !!! You just need to change lines with "# todo" to get straight to run
    !!! Our code is encapsulated, but it also provides some test interfaces for debugging
    !!! If you want to change the dataset, you can change "KIPA" to other task name
    
KIPA22 [1-4] challenge (including simulataneous segmentation of arteries and veins) is used as 
a public 3D dataset to further validate our method
Challenge: https://kipa22.grand-challenge.org/

[1] He, Y. et. al. 2021. Meta grayscale adaptive network for 3D integrated renal structures segmentation. 
Medical image analysis 71, 102055.
[2] He, Y. et. al. 2020. Dense biased networks with deep priori anatomy and hard region adaptation: 
Semisupervised learning for fine renal artery segmentation. Medical Image Analysis 63, 101722.
[3] Shao, P. et. al. 2011. Laparoscopic partial nephrectomy with segmental renal artery clamping: 
technique and clinical outcomes. European urology 59, 849–855.
[4] Shao, P. et. al. 2012. Precise segmental renal artery clamping under the guidance of dual-source computed 
tomography angiography during laparoscopic partial nephrectomy. European urology 62, 1001–1008.
"""


def resolve_paths(args):
    """Resolve every derived path without requiring trailing separators."""
    root = Path(args.root_dir)
    data = Path(args.data_dir)

    defaults = {
        "Tr_Image_dir": data / "train" / "image",
        "Va_Image_dir": data / "val" / "image",
        "Te_Image_dir": data / "test" / "image",
        "Tr_Label_dir": data / "train" / "label",
        "Va_Label_dir": data / "val" / "label",
        "Te_Label_dir": data / "test" / "label",
        "Dir_Txt": root / "Txt" / f"Txt_{args.run_label}",
        "Dir_Log": root / "Log" / args.run_label,
        "Dir_Save": root / "Results" / args.run_label,
        "Dir_Weights": root / "Weights" / args.run_label,
    }
    for attribute, default in defaults.items():
        value = getattr(args, attribute)
        setattr(args, attribute, str(Path(value) if value else default))
    args.Dir_Log = os.path.join(args.Dir_Log, "")

    txt_dir = Path(args.Dir_Txt)
    manifest_defaults = {
        "Image_Tr_txt": txt_dir / "Image_Tr.txt",
        "Image_Va_txt": txt_dir / "Image_Va.txt",
        "Image_Te_txt": txt_dir / "Image_Te.txt",
        "Label_Tr_txt": txt_dir / "Label_Tr.txt",
        "Label_Va_txt": txt_dir / "Label_Va.txt",
        "Label_Te_txt": txt_dir / "Label_Te.txt",
    }
    for attribute, default in manifest_defaults.items():
        value = getattr(args, attribute)
        setattr(args, attribute, str(Path(value) if value else default))

    if args.Meanstd_name is None:
        args.Meanstd_name = f"{args.run_label}_Meanstd.npy"
    args.Meanstd_path = str(root / args.Meanstd_name)

    if args.save_path is None:
        args.save_path = str(root / "Results" / args.run_label / "DSCNet")
    if args.save_path_max is None:
        args.save_path_max = str(root / "Results" / args.run_label / "DSCNet_max")
    if args.model_name is None:
        args.model_name = f"DSCNet_{args.run_label}"
    if args.model_name_max is None:
        args.model_name_max = f"DSCNet_{args.run_label}_max"
    if args.log_name is None:
        args.log_name = f"DSCNet_{args.run_label}.log"

    return args


def Create_files(args):
    print("0 Start all process ...")
    if not os.path.exists(args.Dir_Txt):
        os.makedirs(args.Dir_Txt)
    if not os.path.exists(args.Dir_Log):
        os.makedirs(args.Dir_Log)
    if not os.path.exists(args.Dir_Save):
        os.makedirs(args.Dir_Save)
    if not os.path.exists(args.Dir_Weights):
        os.makedirs(args.Dir_Weights)


def _require_files(paths, action):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{action} requires prepared files: {', '.join(missing)}"
        )


def Process(args):
    Create_files(args)

    if args.action == "prepare":
        from S1_Pre_Getmeanstd import Getmeanstd
        from S2_Pre_Generate_Txt import Generate_Paired_Txt

        Getmeanstd(args.Tr_Image_dir, args.Meanstd_path)
        Generate_Paired_Txt(
            args.Tr_Image_dir,
            args.Tr_Label_dir,
            args.Image_Tr_txt,
            args.Label_Tr_txt,
        )
        Generate_Paired_Txt(
            args.Va_Image_dir,
            args.Va_Label_dir,
            args.Image_Va_txt,
            args.Label_Va_txt,
        )
        Generate_Paired_Txt(
            args.Te_Image_dir,
            args.Te_Label_dir,
            args.Image_Te_txt,
            args.Label_Te_txt,
        )
        return

    if args.action == "train":
        _require_files(
            [
                args.Meanstd_path,
                args.Image_Tr_txt,
                args.Label_Tr_txt,
                args.Image_Va_txt,
                args.Label_Va_txt,
            ],
            "training",
        )
        operation = "Train"
    else:
        _require_files(
            [args.Meanstd_path, args.Image_Te_txt, args.Label_Te_txt],
            "evaluation",
        )
        operation = "Evaluate"

    if args.training_pipeline == "optimized":
        import S3_Optimized_Train_Process as pipeline
    else:
        import S3_Train_Process as pipeline
    getattr(pipeline, operation)(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # "root_dir" refers to the address of the outermost code, and "***" needs to be replaced
    root_dir = "DSCNet_3D_opensource/"  # todo
    data_dir = "Data/MiniVess_Half/"  # todo
    run_label = "DSCNet_3D"  # todo

    parser.add_argument(
        "--root_dir", default=root_dir, help="the address of the outermost code"
    )
    parser.add_argument(
        "--data_dir", default=data_dir, help="the address of the data directory"
    )
    parser.add_argument(
        "--run_label", default=run_label, help="the name of the current run"
    )

    # information about the image and label
    parser.add_argument(
        "--Tr_Image_dir",
        default=None,
        help="the address of the train image",
    )
    parser.add_argument(
        "--Va_Image_dir",
        default=None,
        help="the address of the validation image",
    )
    parser.add_argument(
        "--Te_Image_dir",
        default=None,
        help="the address of the test image",
    )
    parser.add_argument(
        "--Tr_Label_dir",
        default=None,
        help="the address of the train label",
    )
    parser.add_argument(
        "--Va_Label_dir",
        default=None,
        help="the address of the validation label",
    )
    parser.add_argument(
        "--Te_Label_dir",
        default=None,
        help="the address of the test label",
    )
    parser.add_argument(
        "--Meanstd_name",
        default=None,
        help="training-set mean and standard deviation file",
    )

    # files that are needed to be used to store contents
    parser.add_argument("--Dir_Txt", default=None, help="Txt path")
    parser.add_argument("--Dir_Log", default=None, help="Log path")
    parser.add_argument("--Dir_Save", default=None, help="Save path")
    parser.add_argument("--Dir_Weights", default=None, help="Weights path")

    # Folders, dataset, etc.
    parser.add_argument(
        "--Image_Tr_txt",
        default=None,
        help="train image txt path",
    )
    parser.add_argument(
        "--Image_Va_txt",
        default=None,
        help="validation image txt path",
    )
    parser.add_argument(
        "--Image_Te_txt",
        default=None,
        help="test image txt path",
    )
    parser.add_argument(
        "--Label_Tr_txt",
        default=None,
        help="train label txt path",
    )
    parser.add_argument(
        "--Label_Va_txt",
        default=None,
        help="validation label txt path",
    )
    parser.add_argument(
        "--Label_Te_txt",
        default=None,
        help="test label txt path",
    )

    # Detailed path for saving results
    """
    Breif description:
        Due to the small proportion of the thin tubular structure, 
        the results of the model may bring huge fluctuations. 
        In order to reduce the influence of uncertain factors on the model analysis, 
        we save the <best> results on the validation dataset in the <max> folder, 
        and apply the same standard to all comparative methods to ensure fairness!!
    """
    parser.add_argument("--save_path", default=None, help="Save dir")
    parser.add_argument(
        "--save_path_max",
        default=None,
        help="Save max dir",
    )
    parser.add_argument("--model_name", default=None, help="Weights name")
    parser.add_argument("--model_name_max", default=None, help="Max Weights name")
    parser.add_argument("--log_name", default=None, help="Log name")

    # Network options
    parser.add_argument("--n_channels", default=1, type=int, help="input channels")
    parser.add_argument(
        "--n_classes", default=2, type=int, help="output channels"
    )  # test this
    parser.add_argument(
        "--kernel_size", default=5, type=int, help="odd DSConv kernel size (>= 3)"
    )
    parser.add_argument(
        "--extend_scope", default=1.75, type=float, help="DSConv offset range"
    )
    parser.add_argument(
        "--if_offset",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="enable learned DSConv offsets",
    )
    parser.add_argument(
        "--n_basic_layer", default=16, type=int, help="basic layer numbers"
    )
    parser.add_argument("--dim", default=8, type=int, help="dim numbers")
    parser.add_argument(
        "--unet_layers",
        default=4,
        type=int,
        choices=[3, 4, 5],
        help="number of U-Net levels to use in the standard pipeline",
    )
    parser.add_argument(
        "--training_pipeline",
        default="standard",
        choices=["standard", "optimized"],
        help="training implementation to use",
    )

    # Training options
    parser.add_argument("--GPU_id", default="0", help="GPU ID")  # not in use
    """
    Reference: --ROI_shape: (128, 96, 96)  3090's memory occupancy is about 16653 MiB
    """
    parser.add_argument(
        "--ROI_shape",
        default=(64, 64, 64),
        nargs=3,
        type=int,
        metavar=("DEPTH", "HEIGHT", "WIDTH"),
        help="training patch size",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument(
        "--sample_count",
        default=1,
        type=int,
        help="number of times each image is sampled per epoch",
    )
    parser.add_argument(
        "--predict_batch_size",
        default=4,
        type=int,
        help="number of inference patches processed together",
    )
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--min_lr", default=5e-6, type=float, help="minimum learning rate")
    parser.add_argument(
        "--poly_decay_power",
        default=0.9,
        type=float,
        help="polynomial learning-rate decay power",
    )
    parser.add_argument(
        "--beta",
        default=1e-2,
        type=float,
        help="initial entropy-regularization weight",
    )
    parser.add_argument(
        "--min_beta",
        default=1e-6,
        type=float,
        help="minimum entropy-regularization weight",
    )
    parser.add_argument(
        "--beta_decay_power",
        default=2.0,
        type=float,
        help="polynomial entropy-weight decay power",
    )

    parser.add_argument(
        "--use_rlrop",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="use ReduceLROnPlateau while training",
    )
    parser.add_argument(
        "--rlr_factor", default=0.5, type=float, help="ReduceLROnPlateau Factor"
    )
    parser.add_argument(
        "--rlr_threshold", default=0.002, type=float, help="ReduceLROnPlateau Threshold"
    )
    parser.add_argument(
        "--rlr_patience", default=10, type=int, help="ReduceLROnPlateau Patience"
    )
    parser.add_argument(
        "--rlr_cooldown", default=2, type=int, help="ReduceLROnPlateau Cooldown"
    )

    parser.add_argument(
        "--start_train_epoch", default=1, type=int, help="Start training epoch"
    )
    parser.add_argument(
        "--start_verify_epoch",
        default=51,
        type=int,
        help="Start verifying epoch",  # Original: 200
    )
    parser.add_argument(
        "--n_epochs", default=100, type=int, help="Epoch Num"
    )  # Original: 400
    parser.add_argument(
        "--verify_gap", default=1, type=int, help="validate every N epochs"
    )
    parser.add_argument(
        "--if_retrain",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="start training without loading an existing checkpoint",
    )
    parser.add_argument(
        "--action",
        required=True,
        choices=["prepare", "train", "evaluate"],
        help="prepare artifacts, train with validation, or evaluate the frozen test set",
    )
    parser.add_argument(
        "--if_fullprecision",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="disable automatic mixed precision",
    )
    parser.add_argument(
        "--use_earlystop",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="use early stopping while training",
    )
    parser.add_argument(
        "--earlystop_threshold",
        default=0.002,
        type=float,
        help="Early Stopping Threshold",
    )
    parser.add_argument(
        "--earlystop_patience", default=30, type=int, help="Early Stopping Patience"
    )

    args = resolve_paths(parser.parse_args())
    Process(args)
