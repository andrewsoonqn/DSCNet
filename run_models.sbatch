#!/bin/bash
# ─── Job identification ─────────────────────────────────────────────────
#SBATCH --job-name=sg-hist-ocr
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

# ─── Resource allocation ────────────────────────────────────────────────
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# ─── Time limit ─────────────────────────────────────────────────────────
#SBATCH --time=8:00:00

# ─── Email notifications (optional — edit or remove) ────────────────────
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrewsoonqn@gmail.com


echo "========================================================================"
echo "  SLURM Job Details"
echo "========================================================================"
echo "  Job ID          : ${SLURM_JOB_ID}"
echo "  Job Name        : ${SLURM_JOB_NAME}"
echo "  Partition       : ${SLURM_JOB_PARTITION}"
echo "  Node            : ${SLURMD_NODENAME}"
echo "  CPUs on node    : ${SLURM_CPUS_ON_NODE}"
echo "  GPUs requested  : ${SLURM_GPUS:-1}"
echo "  Memory          : ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "  Working Dir     : $(pwd)"
echo "  Start Time      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"

# ─── Project directory ──────────────────────────────────────────────────
PROJECT_DIR="." #TODO
cd "${PROJECT_DIR}"
echo "Changed to project directory: ${PROJECT_DIR}"

# ─── Create log directory if it doesn't exist ──────────────────────────
mkdir -p logs

# ─── Load cluster modules ──────────────────────────────────────────────
#  Adjust these module names to match your cluster's module system.
#  Comment out any that are not available or not needed.
if command -v module &> /dev/null; then
    echo "Loading cluster modules..."
    module purge 2>/dev/null || true
    module load cuda/11.8 2>/dev/null || echo "WARN: cuda/11.8 module not found, skipping"
    module load cudnn/8.6 2>/dev/null || echo "WARN: cudnn/8.6 module not found, skipping"
    module load python/3.11 2>/dev/null || echo "WARN: python/3.11 module not found, skipping"
    module list 2>&1
else
    echo "No module system detected, skipping module loads."
fi

# ─── Activate virtual environment ──────────────────────────────────────
#  Assumes a venv exists at .venv/ in the project directory.
#  If you use conda, replace this block with:
#    source activate your-env-name
#  or:
#    conda activate your-env-name
VENV_DIR= #TODO or use V"${VENV_DIR:-.venv}"
if [ -d "${VENV_DIR}" ]; then
    echo "Activating virtual environment: ${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
else
    echo "WARNING: Virtual environment '${VENV_DIR}' not found."
    echo "         Make sure dependencies are installed in the default Python."
fi

# ─── Verify Python and key packages ────────────────────────────────────
echo ""
echo "Python environment:"
echo "  Python   : $(which python) ($(python --version 2>&1))"
echo "  Pip      : $(which pip) ($(pip --version 2>&1 | head -1))"
echo ""

python -c "
import torch
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA ok  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU      : {torch.cuda.get_device_name(0)}')
    print(f'  GPU Mem  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'  CUDA ver : {torch.version.cuda}')
"

# python -c "
# import transformers, PIL, google.cloud.storage, xmltodict
# print(f'  HF Trans : {transformers.__version__}')
# print(f'  Pillow   : {PIL.__version__}')
# print(f'  GCS      : {google.cloud.storage.__version__}')
# # print(f'  xmltodict: {getattr(xmltodict, "__version__", "unknown")}')
# " || echo "WARNING: Some Python dependencies are missing. Install via: pip install -r requirements.txt"

echo ""

# ###############################################################################
# #                        PERFORMANCE TUNING                                   #
# ###############################################################################
#
# # Number of data-loading workers for PyTorch
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
#
# # Prevent tokenizer parallelism warnings
# export TOKENIZERS_PARALLELISM=false
#
# # HuggingFace cache directory — use a fast local disk if available
# export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
# echo "HF cache   : ${HF_HOME}"
#
# # CUDA memory management
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
#
# # Limit GPU visibility to assigned GPU only (Slurm usually handles this,
# # but this is a safety measure)
# if [ -n "${SLURM_JOB_GPUS:-}" ]; then
#     export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
#     echo "CUDA devs  : ${CUDA_VISIBLE_DEVICES}"
# fi
#
# # Disable wandb / mlflow if not needed
# export WANDB_DISABLED=true
# export MLFLOW_TRACKING_URI=""

###############################################################################
#                           NVIDIA GPU INFO                                   #
###############################################################################

echo ""
echo "========================================================================"
echo "  GPU Status (nvidia-smi)"
echo "========================================================================"
nvidia-smi || echo "WARNING: nvidia-smi not available"
echo ""

###############################################################################
#                          RUN THE PIPELINE                                   #
###############################################################################

CMD="python main.py" #TODO

echo "========================================================================"
echo "  Executing pipeline"
echo "========================================================================"
echo "  Command: ${CMD}"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
echo ""

# Record start time for duration calculation
START_SECONDS=$SECONDS

# Run the pipeline — tee to a separate per-task log for convenience
${CMD} 2>&1 | tee "logs/pipeline_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.log"
PIPELINE_EXIT_CODE=${PIPESTATUS[0]}

# Record end time
ELAPSED=$(( SECONDS - START_SECONDS ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

###############################################################################
#                        POST-RUN SUMMARY                                     #
###############################################################################

echo ""
echo "========================================================================"
echo "  Pipeline Finished"
echo "========================================================================"
echo "  Exit code: ${PIPELINE_EXIT_CODE}"
echo "  Duration : ${ELAPSED_MIN}m ${ELAPSED_SEC}s (${ELAPSED}s total)"
echo "  Ended    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"

# Print final GPU memory state
echo ""
echo "Final GPU memory state:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null \
    | while IFS=',' read -r idx name mem_used mem_total gpu_util; do
        echo "  GPU ${idx} (${name}): ${mem_used}/${mem_total} MiB used, ${gpu_util}% utilization"
    done || echo "  (nvidia-smi query unavailable)"

# Exit with the pipeline's exit code so Slurm marks the job correctly
exit ${PIPELINE_EXIT_CODE}
