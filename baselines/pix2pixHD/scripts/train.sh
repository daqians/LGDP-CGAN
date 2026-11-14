#!/bin/bash
# Train the pix2pixHD baseline (FMS only).
# Usage: bash scripts/train.sh <experiment_name> <dataroot> <n_pads: 3|4> [checkpoints_dir]
#   n_pads = number of FMS pads used as input channels (the paper reports 3 pads and 4 pads)
set -e
NAME=${1:?experiment name}
DATAROOT=${2:?dataroot}
PADS=${3:?number of FMS pads: 3 or 4}
CKPT=${4:-./checkpoints}

python train.py \
    --name "${NAME}" \
    --dataroot "${DATAROOT}" \
    --checkpoints_dir "${CKPT}" \
    --input_channels "${PADS}" \
    --ngf 64 --n_downsample_global 4 --n_blocks_global 9 \
    --batchSize 4 \
    --niter 150 --niter_decay 100 \
    --lr 0.0001 --beta1 0.5 \
    --lambda_feat 10.0 --lambda_ffl 1.0 --ffl_alpha 1.0 \
    --gpu_ids 0 \
    --save_epoch_freq 25
