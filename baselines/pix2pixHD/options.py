import argparse
import os

import torch


def _base_parser():
    p = argparse.ArgumentParser()
    # experiment
    p.add_argument('--name', type=str, required=True, help='experiment name; checkpoints are stored in checkpoints_dir/name')
    p.add_argument('--dataroot', type=str, required=True, help='folder with {phase}_A/ (FMS .npy) and {phase}_B/ (core .jpg)')
    p.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    p.add_argument('--gpu_ids', type=str, default='0', help='e.g. 0; use -1 for CPU')
    p.add_argument('--nThreads', type=int, default=2, help='data loading workers')
    # input
    p.add_argument('--input_channels', type=int, default=4, choices=[1, 2, 3, 4],
                   help='number of FMS pads used as input channels (the first N of the 4 pads): 3 or 4 in the paper')
    p.add_argument('--fineSize', type=int, default=512, help='FMS input is resized to, and core images are, fineSize x fineSize')
    # generator (pix2pixHD global generator)
    p.add_argument('--ngf', type=int, default=64, help='base channels of the generator')
    p.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling stages')
    p.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks at the bottleneck')
    return p


def _finish(opt, is_train):
    opt.isTrain = is_train
    opt.norm = 'instance'
    opt.output_nc = 3
    opt.gpu_ids = [int(i) for i in opt.gpu_ids.split(',') if int(i) >= 0]
    if opt.gpu_ids and torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu_ids[0])
    else:
        opt.gpu_ids = []

    print('------------ Options -------------')
    for k, v in sorted(vars(opt).items()):
        print('%s: %s' % (k, v))
    print('-------------- End ----------------')

    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(expr_dir, exist_ok=True)
    if is_train:
        with open(os.path.join(expr_dir, 'opt.txt'), 'wt') as f:
            for k, v in sorted(vars(opt).items()):
                f.write('%s: %s\n' % (k, v))
    return opt


def train_options():
    p = _base_parser()
    p.add_argument('--phase', type=str, default='train')
    p.add_argument('--batchSize', type=int, default=4)
    p.add_argument('--niter', type=int, default=150, help='epochs at the initial learning rate')
    p.add_argument('--niter_decay', type=int, default=100, help='epochs of linear decay to zero')
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--beta1', type=float, default=0.5)
    p.add_argument('--no_flip', action='store_true', help='disable random horizontal flip of the core image')
    # discriminator
    p.add_argument('--num_D', type=int, default=2, help='number of discriminator scales')
    p.add_argument('--n_layers_D', type=int, default=3)
    p.add_argument('--ndf', type=int, default=64)
    # losses
    p.add_argument('--lambda_feat', type=float, default=10.0, help='weight of the discriminator feature-matching loss')
    p.add_argument('--no_ffl_loss', action='store_true', help='disable the focal frequency loss (on by default)')
    p.add_argument('--lambda_ffl', type=float, default=1.0, help='weight of the focal frequency loss')
    p.add_argument('--ffl_alpha', type=float, default=1.0, help='focusing exponent of the focal frequency loss')
    # logging
    p.add_argument('--print_freq', type=int, default=100)
    p.add_argument('--display_freq', type=int, default=500, help='how often to save a training sample image')
    p.add_argument('--save_latest_freq', type=int, default=1000)
    p.add_argument('--save_epoch_freq', type=int, default=25)
    return _finish(p.parse_args(), True)


def test_options():
    p = _base_parser()
    p.add_argument('--phase', type=str, default='test')
    p.add_argument('--which_epoch', type=str, default='latest')
    p.add_argument('--results_dir', type=str, default='./results')
    return _finish(p.parse_args(), False)
