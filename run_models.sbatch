#!/usr/bin/env bash
# Configuration-free cluster bootstrap used by expctl.
# Slurm resources, paths, and the resolved experiment are supplied by expctl's
# immutable generated job script; this file owns no experiment values.

set -euo pipefail

: "${DSCNET_PYTHON:?expctl must set DSCNET_PYTHON}"
: "${DSCNET_SOURCE_ROOT:?expctl must set DSCNET_SOURCE_ROOT}"
: "${DSCNET_RESOLVED_CONFIG:?expctl must set DSCNET_RESOLVED_CONFIG}"

cd "$DSCNET_SOURCE_ROOT"
exec "$DSCNET_PYTHON" \
  "$DSCNET_SOURCE_ROOT/DSCNet_3D_opensource/Code/Kipa/DSCNet/S4_Experiment_Run.py" \
  --resolved-config "$DSCNET_RESOLVED_CONFIG"
