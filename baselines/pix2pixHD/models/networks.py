"""pix2pixHD global generator, multi-scale discriminator and losses (adapted from pix2pixHD)."""
import functools

import numpy as np
import torch
import torch.nn as nn


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    if norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    raise NotImplementedError('normalization layer [%s] is not found' % norm_type)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True)):
        super().__init__()
        assert padding_type == 'reflect'
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3), norm_layer(dim), activation,
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3), norm_layer(dim),
        )

    def forward(self, x):
        return x + self.conv_block(x)



# ---------------------------------------------------------------------------
# pix2pixHD global generator
# ---------------------------------------------------------------------------
class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert n_blocks >= 0
        super().__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]
        # residual blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        # upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


def define_G(input_nc, output_nc, ngf, n_downsample_global=4, n_blocks_global=9, norm='instance', gpu_ids=()):
    netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, get_norm_layer(norm))
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

# ---------------------------------------------------------------------------
# Multi-scale PatchGAN discriminator
# ---------------------------------------------------------------------------
class NLayerDiscriminator(nn.Module):
    """PatchGAN that returns the output of every layer (used for feature matching)."""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.n_layers = n_layers
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev, nf = nf, min(nf * 2, 512)
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                          norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        nf_prev, nf = nf, min(nf * 2, 512)
        sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                      norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, num_D=2):
        super().__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer)
            for j in range(n_layers + 2):
                setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        """Returns, for each scale (finest first), the list of all layer outputs."""
        result = []
        x = input
        for i in range(self.num_D):
            layers = [getattr(self, 'scale' + str(self.num_D - 1 - i) + '_layer' + str(j))
                      for j in range(self.n_layers + 2)]
            feats = [x]
            for layer in layers:
                feats.append(layer(feats[-1]))
            result.append(feats[1:])
            if i != self.num_D - 1:
                x = self.downsample(x)
        return result


def define_D(input_nc, ndf, n_layers_D, norm='instance', num_D=2, gpu_ids=()):
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, get_norm_layer(norm), num_D)
    if len(gpu_ids) > 0:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
class GANLoss(nn.Module):
    """Least-squares GAN loss summed over discriminator scales (uses the last layer of each scale)."""

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def __call__(self, preds, target_is_real):
        loss = 0
        for pred_scale in preds:
            pred = pred_scale[-1]
            target = torch.full_like(pred, 1.0 if target_is_real else 0.0)
            loss += self.loss(pred, target)
        return loss


class FocalFrequencyLoss(nn.Module):
    """mean( w * |F_real - F_fake|^2 ), w = |F_real - F_fake|^alpha, orthonormal 2-D FFT."""

    def __init__(self, alpha=1.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, fake, real):
        F_fake = torch.fft.fft2(fake, norm='ortho')
        F_real = torch.fft.fft2(real, norm='ortho')
        mag = torch.abs(F_real - F_fake)
        w = torch.pow(mag + self.eps, self.alpha)
        return (w * (mag ** 2)).mean()
