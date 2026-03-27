#!/usr/bin/env bash

set -euo pipefail

model_variant="unet4"

if [[ $# -gt 0 ]]; then
  case "$1" in
    unet3|3)
      model_variant="unet3"
      shift
      ;;
    unet4|4)
      model_variant="unet4"
      shift
      ;;
    unet5|5)
      model_variant="unet5"
      shift
      ;;
  esac
fi

case "$model_variant" in
  unet3) unet_layers=3 ;;
  unet4) unet_layers=4 ;;
  unet5) unet_layers=5 ;;
  *)
    echo "Usage: $0 [unet3|unet4|unet5] [extra S0_Main.py args...]" >&2
    exit 1
    ;;
esac

venv_activate="DSCNetEnv/Scripts/activate"
if [[ -f "$venv_activate" ]]; then
  # shellcheck disable=SC1090
  source "$venv_activate"
fi

python3 DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py \
  --run_label "Test_Run_${model_variant}" \
  --unet_layers "$unet_layers" \
  "$@"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate
fi
