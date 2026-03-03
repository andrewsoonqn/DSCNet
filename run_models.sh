#!/usr/bin/env bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python -u DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py --run_label MVR_Half_Binned_64x64x64_5Offset_Extend1p75_1
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python -u DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py --run_label MVR_Half_Binned_64x64x64_5Offset_Extend1p75_2
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python -u DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py --run_label MVR_Half_Binned_64x64x64_5Offset_Extend1p75_3