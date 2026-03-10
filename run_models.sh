#!/usr/bin/env bash

DSCNetEnv/Scripts/activate
python DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py --run_label Test_Run
deactivate
