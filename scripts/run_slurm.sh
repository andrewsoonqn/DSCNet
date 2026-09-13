#!/usr/bin/env bash
# Configuration-free cluster bootstrap used by expctl.
# Slurm resources, paths, and the resolved experiment are supplied by expctl's
# immutable generated job script; this file owns no experiment values.

set -euo pipefail

: "${DSCNET_PYTHON:?expctl must set DSCNET_PYTHON}"
: "${DSCNET_SOURCE_ROOT:?expctl must set DSCNET_SOURCE_ROOT}"
: "${DSCNET_RESOLVED_CONFIG:?expctl must set DSCNET_RESOLVED_CONFIG}"

cd "$DSCNET_SOURCE_ROOT"
export PYTHONPATH="$DSCNET_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$DSCNET_PYTHON" -m dscnet --resolved-config "$DSCNET_RESOLVED_CONFIG"
