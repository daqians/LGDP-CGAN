# LGDP-CGAN

Code for *Conditional Geological Core Image Generation via Multimodal Feature Fusion Framework*:
a conditional GAN that generates core images from Formation MicroScanner (FMS) image patches and
eight depth-aligned well-log values.

## Requirements

Python 3 with PyTorch and a CUDA GPU.

```bash
pip install -r requirements.txt
```

## Data

```
dataroot/
├── train_A/                  # FMS patches (.npy, shape 24 x 16 x 4)
├── train_B/                  # core images (.jpg), paired with train_A by sorted file name
├── train_add_modalities.csv  # columns: Files, mean_fms, Cmax, Eccentricity, POTA, THOR, URAN, IDPH, Inv
├── test_A/
├── test_B/
└── test_add_modalities.csv
```

## Train

```bash
bash scripts/train.sh <experiment_name> <dataroot> [checkpoints_dir] [log_norm]
```

`log_norm` selects the well-log normalisation (`data/log_norm.py`): `phase_minmax` (default) or `train_robust`.

## Test

```bash
bash scripts/test.sh <experiment_name> <dataroot> [checkpoints_dir] [results_dir] [which_epoch] [log_norm]
```

Use the same `log_norm` as for training.

Generated images and `metrics.csv` (PSNR / SSIM per sample and the mean) are written to
`results_dir/<experiment_name>/test_<which_epoch>/`.
