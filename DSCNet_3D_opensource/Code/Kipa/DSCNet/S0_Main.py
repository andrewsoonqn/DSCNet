# -*- coding: utf-8 -*-
import os
import argparse

from S1_Pre_Getmeanstd import Getmeanstd
from S2_Pre_Generate_Txt import Generate_Txt
from S3_Train_Process import Train

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


def Process(args):
    # step 0: Prepare all files in this projects
    Create_files(args)

    # Step 1: Prepare image and calculate the "mean" and "std" for normalization
    Getmeanstd(args, args.Tr_Image_dir, args.Tr_Meanstd_name)
    Getmeanstd(args, args.Va_Image_dir, args.Va_Meanstd_name)
    Getmeanstd(args, args.Te_Image_dir, args.Te_Meanstd_name)

    # Step 2: Prepare ".txt" files for training and testing data
    Generate_Txt(args.Tr_Image_dir, args.Image_Tr_txt)
    Generate_Txt(args.Va_Image_dir, args.Image_Va_txt)
    Generate_Txt(args.Te_Image_dir, args.Image_Te_txt)
    Generate_Txt(args.Tr_Label_dir, args.Label_Tr_txt)
    Generate_Txt(args.Va_Label_dir, args.Label_Va_txt)
    Generate_Txt(args.Te_Label_dir, args.Label_Te_txt)

    # Step 3: Train the "Network"
    Train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # "root_dir" refers to the address of the outermost code, and "***" needs to be replaced
    root_dir = "DSCNet_3D_opensource/" # todo
    data_dir = "Data/MiniVess_Half/" # todo 
    run_label = "<label_name>" # todo

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
        "--Tr_Meanstd_name",
        default=None,
        help="Train image Mean and std for normalization",
    )
    parser.add_argument(
        "--Va_Meanstd_name",
        default=None,
        help="Validation image Mean and std for normalization",
    )
    parser.add_argument(
        "--Te_Meanstd_name",
        default=None,
        help="Test image Mean and std for normalization",
    ) 

    # files that are needed to be used to store contents
    parser.add_argument(
        "--Dir_Txt", default=None, help="Txt path"
    )
    parser.add_argument("--Dir_Log", default=None, help="Log path")
    parser.add_argument(
        "--Dir_Save", default=None, help="Save path"
    )
    parser.add_argument(
        "--Dir_Weights", default=None, help="Weights path"
    )

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
    parser.add_argument(
        "--save_path", default=None, help="Save dir"
    )
    parser.add_argument(
        "--save_path_max",
        default=None,
        help="Save max dir",
    )
    parser.add_argument("--model_name", default=None, help="Weights name")
    parser.add_argument(
        "--model_name_max", default=None, help="Max Weights name"
    )
    parser.add_argument("--log_name", default=None, help="Log name")

    # Network options
    parser.add_argument("--n_channels", default=1, type=int, help="input channels")
    parser.add_argument("--n_classes", default=2, type=int, help="output channels") # test this
    parser.add_argument(
        "--kernel_size", default=9, type=int, help="kernel size"
    )  # 9 refers to 1*9/9*1 for DSConv (This parameter is not in use - kernel fixed at 9)
    parser.add_argument(
        "--extend_scope", default=1.0, type=float, help="extend scope"
    )  # This parameter is not used
    parser.add_argument(
        "--if_offset", default=True, type=bool, help="if offset"
    )  # Whether to use the deformation or not
    parser.add_argument(
        "--n_basic_layer", default=16, type=int, help="basic layer numbers"
    )
    parser.add_argument("--dim", default=8, type=int, help="dim numbers")

    # Training options
    parser.add_argument("--GPU_id", default="0", help="GPU ID") # not in use
    """
    Reference: --ROI_shape: (128, 96, 96)  3090's memory occupancy is about 16653 MiB
    """
    parser.add_argument("--ROI_shape", default=(96, 96, 96), type=int, help="roi size") # Original: 128, 96, 96
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")

    parser.add_argument("--use_rlrop", default=False, type=bool, help="Use ReduceLROnPlateau (when training)")
    parser.add_argument("--rlr_factor", default=0.5, type=float, help="ReduceLROnPlateau Factor") 
    parser.add_argument("--rlr_threshold", default=0.002, type=float, help="ReduceLROnPlateau Threshold") 
    parser.add_argument("--rlr_patience", default=10, type=int, help="ReduceLROnPlateau Patience") 
    parser.add_argument("--rlr_cooldown", default=2, type=int, help="ReduceLROnPlateau Cooldown")

    parser.add_argument(
        "--start_train_epoch", default=1, type=int, help="Start training epoch"
    )
    parser.add_argument(
        "--start_verify_epoch", default=101, type=int, help="Start verifying epoch" # Original: 200
    )
    parser.add_argument("--n_epochs", default=200, type=int, help="Epoch Num") # Original: 400
    parser.add_argument("--if_retrain", default=True, type=bool, help="If Retrain") 
    parser.add_argument("--if_onlytest", default=False, type=bool, help="If Only Test")

    parser.add_argument("--if_fullprecision", default=True, type=bool, help="If Full Precision (disable AMP)")

    parser.add_argument("--use_earlystop", default=False, type=bool, help="Use Early Stopping (when training)")
    parser.add_argument("--earlystop_threshold", default=0.002, type=float, help="Early Stopping Threshold")
    parser.add_argument("--earlystop_patience", default=30, type=int, help="Early Stopping Patience")

    args, unknown = parser.parse_known_args()

    if args.Tr_Image_dir is None:
        args.Tr_Image_dir = args.data_dir + "train/image/"
    if args.Va_Image_dir is None:
        args.Va_Image_dir = args.data_dir + "val/image/"
    if args.Te_Image_dir is None:
        args.Te_Image_dir = args.data_dir + "test/image/"

    if args.Tr_Label_dir is None:
        args.Tr_Label_dir = args.data_dir + "train/label/"
    if args.Va_Label_dir is None:
        args.Va_Label_dir = args.data_dir + "val/label/"
    if args.Te_Label_dir is None:
        args.Te_Label_dir = args.data_dir + "test/label/"

    if args.Tr_Meanstd_name is None:
        args.Tr_Meanstd_name = args.run_label + "_Tr_Meanstd.npy"
    if args.Va_Meanstd_name is None:
        args.Va_Meanstd_name = args.run_label + "_Va_Meanstd.npy"
    if args.Te_Meanstd_name is None:
        args.Te_Meanstd_name = args.run_label + "_Te_Meanstd.npy"

    if args.Dir_Txt is None:
        args.Dir_Txt = args.root_dir + "Txt/Txt_" + args.run_label + "/"
    if args.Dir_Log is None:
        args.Dir_Log = args.root_dir + "Log/" + args.run_label + "/"
    if args.Dir_Save is None:
        args.Dir_Save = args.root_dir + "Results/" + args.run_label + "/"
    if args.Dir_Weights is None:
        args.Dir_Weights = args.root_dir + "Weights/" + args.run_label + "/"

    if args.Image_Tr_txt is None:
        args.Image_Tr_txt = args.root_dir + "Txt/Txt_" + args.run_label + "/Image_Tr.txt"
    if args.Image_Va_txt is None:
        args.Image_Va_txt = args.root_dir + "Txt/Txt_" + args.run_label + "/Image_Va.txt"
    if args.Image_Te_txt is None:
        args.Image_Te_txt = args.root_dir + "Txt/Txt_" + args.run_label + "/Image_Te.txt"

    if args.Label_Tr_txt is None:
        args.Label_Tr_txt = args.root_dir + "Txt/Txt_" + args.run_label + "/Label_Tr.txt"
    if args.Label_Va_txt is None:
        args.Label_Va_txt = args.root_dir + "Txt/Txt_" + args.run_label + "/Label_Va.txt"
    if args.Label_Te_txt is None:
        args.Label_Te_txt = args.root_dir + "Txt/Txt_" + args.run_label + "/Label_Te.txt"

    if args.save_path is None:
        args.save_path = args.root_dir + "Results/" + args.run_label + "/DSCNet/"
    if args.save_path_max is None:
        args.save_path_max = args.root_dir + "Results/" + args.run_label + "/DSCNet_max/"
    if args.model_name is None:
        args.model_name = "DSCNet_" + args.run_label
    if args.model_name_max is None:
        args.model_name_max = "DSCNet_" + args.run_label + "_max"
    if args.log_name is None:
        args.log_name = "DSCNet_" + args.run_label + ".log"

    Process(args)