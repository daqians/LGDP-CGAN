#!/bin/bash
# Train LGDP-CGAN.
# Usage: bash scripts/train.sh <experiment_name> <dataroot> [checkpoints_dir] [log_norm: phase_minmax (default) | train_robust]
set -e
NAME=${1:?experiment name}
DATAROOT=${2:?dataroot}
CKPT=${3:-./checkpoints}
LOG_NORM=${4:-phase_minmax}

python train.py \
    --name "${NAME}" \
    --dataroot "${DATAROOT}" \
    --checkpoints_dir "${CKPT}" \
    --log_norm "${LOG_NORM}" \
    --ngf 64 --n_downsample_global 4 --n_blocks_global 9 \
    --batchSize 4 \
    --niter 150 --niter_decay 100 \
    --lr 0.0001 --beta1 0.5 \
    --lambda_feat 10.0 --lambda_ffl 1.0 --ffl_alpha 1.0 --ffl_warmup_epochs 10 \
    --gpu_ids 0 \
    --save_epoch_freq 25
