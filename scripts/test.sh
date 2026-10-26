#!/bin/bash
# Generate core images for the test set and report PSNR / SSIM.
# Usage: bash scripts/test.sh <experiment_name> <dataroot> [checkpoints_dir] [results_dir] [which_epoch] [log_norm]
#   log_norm must match the one used for training (phase_minmax by default)
set -e
NAME=${1:?experiment name}
DATAROOT=${2:?dataroot}
CKPT=${3:-./checkpoints}
RESULTS=${4:-./results}
EPOCH=${5:-latest}
LOG_NORM=${6:-phase_minmax}

python test.py \
    --name "${NAME}" \
    --dataroot "${DATAROOT}" \
    --checkpoints_dir "${CKPT}" \
    --results_dir "${RESULTS}" \
    --which_epoch "${EPOCH}" \
    --log_norm "${LOG_NORM}" \
    --ngf 64 --n_downsample_global 4 --n_blocks_global 9 \
    --gpu_ids 0
