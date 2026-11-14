# LGDP-CGAN

Code for *Multimodal generation of synthetic carbonate core images from borehole microresistivity images and conventional downhole logs*:
a conditional GAN that generates core images from Formation MicroScanner (FMS) image patches and
eight depth-aligned downhole-log features. The repository also contains the Pix2Pix-HD baseline of the paper
(see [Baselines](#baselines)).

## Requirements

Python 3 with PyTorch and a CUDA GPU.

```bash
pip install -r requirements.txt
```

## Data

The processed dataset will be released on Zenodo; see the Data availability statement of the paper.
Whole core images were assigned to the training or test set at random within each Dunham texture class
(1,820 training and 2,791 test patches), so that all patches from one core image fall in the same set.

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

`log_norm` selects the well-log normalisation (`data/log_norm.py`): `train_robust` (default; statistics fitted on the training set only and reused at test time) or `phase_minmax`.

All models are trained for a fixed number of epochs, and the final checkpoint is evaluated on the test set.

## Test

```bash
bash scripts/test.sh <experiment_name> <dataroot> [checkpoints_dir] [results_dir] [which_epoch] [log_norm]
```

Use the same `log_norm` as for training.

Generated images and `metrics.csv` (PSNR / SSIM per sample and the mean) are written to
`results_dir/<experiment_name>/test_<which_epoch>/`.

## Baselines

- **Pix2Pix-HD (3 or 4 FMS pads)**: `baselines/pix2pixHD/`, adapted from the official
  [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) implementation. See its README for training and testing.
- **Pix2Pix (3 FMS pads)**: trained with the official
  [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) implementation,
  with the first three FMS pads as a three-channel input.
