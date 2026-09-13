#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <prepare|train|evaluate> [unet3|unet4|unet5] [Hydra overrides...]" >&2
  exit 1
fi

action="$1"
shift
case "$action" in
  prepare|train|evaluate) ;;
  *)
    echo "Unknown action: $action" >&2
    exit 1
    ;;
esac

model_variant="unet4"
if [[ $# -gt 0 ]]; then
  case "$1" in
    unet3|3) model_variant="unet3"; shift ;;
    unet4|4) model_variant="unet4"; shift ;;
    unet5|5) model_variant="unet5"; shift ;;
  esac
fi

case "$model_variant" in
  unet3) unet_layers=3 ;;
  unet4) unet_layers=4 ;;
  unet5) unet_layers=5 ;;
esac

activated_venv=false
if [[ -z "${VIRTUAL_ENV:-}" && -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  activated_venv=true
fi

export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
python3 -m dscnet \
  --config-name experiment/dscnet_standard \
  "action=$action" \
  "data.run_label=Test_Run_${model_variant}" \
  "model.unet_layers=$unet_layers" \
  "$@"

if [[ "$activated_venv" == true ]]; then
  deactivate
fi
