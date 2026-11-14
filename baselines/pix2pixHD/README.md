# pix2pixHD baseline: FMS → core image

Baseline code for *Conditional Geological Core Image Generation via Multimodal Feature Fusion Framework*.
It generates a core image from a Formation MicroScanner (FMS) image patch alone, without well logs.
In the paper it is reported as **Pix2Pix-HD, 3 pads** and **Pix2Pix-HD, 4 pads**. The two runs use the
same code and differ only in how many of the four FMS pads are used as input channels.

The code is adapted from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) (global generator +
multi-scale discriminator), cut down to what this baseline uses.

## Requirements

Python 3 with PyTorch and a CUDA GPU.

```bash
pip install -r requirements.txt
```

## Data

```
dataroot/
├── train_A/   # FMS patches (.npy, shape 24 x 16 x 4: depth x azimuth x pad)
├── train_B/   # core images (.jpg, 512 x 512), same file names as train_A
├── test_A/
└── test_B/
```

FMS and core files are paired by file name (`xxx.npy` ↔ `xxx.jpg`).

## Model

| Part | Setting |
|---|---|
| Input | the first `n_pads` FMS pads (3 or 4) as channels, bilinearly resized from 24 x 16 to 512 x 512, min-max scaled to [-1, 1] per sample |
| Generator | pix2pixHD global generator: 64 base channels, 4 downsampling stages, 9 residual blocks, instance norm |
| Discriminator | 2-scale PatchGAN (3 layers, 64 base channels), conditioned on cat(FMS input, core image) |
| Losses | least-squares GAN + discriminator feature matching (λ = 10) + focal frequency loss (λ = 1, α = 1). No VGG loss |
| Training | 150 epochs at lr 1e-4, then 100 epochs of linear decay to 0; Adam β1 = 0.5; batch size 4; 1 GPU |
| Augmentation | random horizontal flip of the core image only |
| Checkpoint | the final epoch (`latest`) |

Unlike the LGDP-CGAN release, this baseline has **no** focal-frequency-loss warm-up and **no**
gradient clipping. The focal frequency loss can be switched off with `--no_ffl_loss`.

## Train

```bash
bash scripts/train.sh <experiment_name> <dataroot> <n_pads: 3|4> [checkpoints_dir]
```

Checkpoints (`latest_net_G.pth`, `latest_net_D.pth`, plus every 25 epochs), `loss_log.txt` and
`opt.txt` are written to `checkpoints_dir/<experiment_name>/`.

GAN training is not always stable. If `D_real` and `D_fake` in `loss_log.txt` drop below about
0.03 early in training and stay there, and the outputs are flat average colours, the discriminator
has collapsed: discard that run and train again.

## Test

```bash
bash scripts/test.sh <experiment_name> <dataroot> <n_pads: 3|4> [checkpoints_dir] [results_dir] [which_epoch]
```

Use the same `n_pads` as for training. For every test sample, the FMS input preview (first three
pads as RGB), the generated core and the real core are saved to
`results_dir/<experiment_name>/test_<which_epoch>/images/`, and `metrics.csv` lists PSNR / SSIM per
sample and the mean.

PSNR and SSIM are computed with scikit-image on the 512 x 512 RGB images scaled to [0, 1]
(`data_range = 1`; SSIM over the three colour channels).

## Files

```
train.py                    training loop
test.py                     inference + PSNR / SSIM
options.py                  command-line options and defaults
data/fms_dataset.py         paired FMS / core dataset
models/pix2pixhd_model.py   generator + discriminator, losses, optimisers, save / load
models/networks.py          global generator, multi-scale discriminator, GAN and focal frequency losses
scripts/train.sh, test.sh   the settings used in the paper
```
