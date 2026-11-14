#!/bin/bash
# Generate core images for the test set and report PSNR / SSIM.
# Usage: bash scripts/test.sh <experiment_name> <dataroot> <n_pads: 3|4> [checkpoints_dir] [results_dir] [which_epoch]
#   n_pads must match the one used for training
set -e
NAME=${1:?experiment name}
DATAROOT=${2:?dataroot}
PADS=${3:?number of FMS pads: 3 or 4}
CKPT=${4:-./checkpoints}
RESULTS=${5:-./results}
EPOCH=${6:-latest}

python test.py \
    --name "${NAME}" \
    --dataroot "${DATAROOT}" \
    --checkpoints_dir "${CKPT}" \
    --results_dir "${RESULTS}" \
    --which_epoch "${EPOCH}" \
    --input_channels "${PADS}" \
    --ngf 64 --n_downsample_global 4 --n_blocks_global 9 \
    --gpu_ids 0
