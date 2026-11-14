import os

import torch

from . import networks


class Pix2PixHDModel(torch.nn.Module):
    """pix2pixHD global generator + conditional multi-scale discriminator, losses and optimisers."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not opt.isTrain:
            torch.backends.cudnn.benchmark = True

        self.netG = networks.define_G(opt.input_channels, opt.output_nc, opt.ngf, opt.n_downsample_global,
                                      opt.n_blocks_global, opt.norm, gpu_ids=self.gpu_ids)

        if opt.isTrain:
            self.use_ffl = not opt.no_ffl_loss
            self.loss_names = ['G_GAN', 'G_GAN_Feat'] + (['G_FFL'] if self.use_ffl else []) + ['D_real', 'D_fake']
            # the discriminator sees cat(FMS input, core image)
            self.netD = networks.define_D(opt.input_channels + opt.output_nc, opt.ndf, opt.n_layers_D, opt.norm,
                                          opt.num_D, gpu_ids=self.gpu_ids)
            self.criterionGAN = networks.GANLoss()
            self.criterionFeat = torch.nn.L1Loss()
            if self.use_ffl:
                self.criterionFFL = networks.FocalFrequencyLoss(alpha=opt.ffl_alpha)
            self.old_lr = opt.lr
            self.optimizer_G = torch.optim.Adam(list(self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(list(self.netD.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        else:
            self.load_network(self.netG, 'G', opt.which_epoch)

    def _device(self, x):
        return x.cuda() if self.gpu_ids else x

    def forward(self, fms, real_image):
        """Returns ([G_GAN, G_GAN_Feat, (G_FFL,) D_real, D_fake], fake_image)."""
        fms, real_image = self._device(fms), self._device(real_image)
        fake_image = self.netG(fms)

        # discriminator
        pred_fake_pool = self.netD(torch.cat((fms, fake_image.detach()), dim=1))
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)
        pred_real = self.netD(torch.cat((fms, real_image.detach()), dim=1))
        loss_D_real = self.criterionGAN(pred_real, True)

        # generator: adversarial
        pred_fake = self.netD(torch.cat((fms, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # generator: feature matching over the intermediate discriminator layers
        loss_G_GAN_Feat = 0
        feat_weights = 4.0 / (self.opt.n_layers_D + 1)
        D_weights = 1.0 / self.opt.num_D
        for i in range(self.opt.num_D):
            for j in range(len(pred_fake[i]) - 1):
                loss_G_GAN_Feat += D_weights * feat_weights * \
                    self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        losses = [loss_G_GAN, loss_G_GAN_Feat]
        # generator: focal frequency loss
        if self.use_ffl:
            losses.append(self.criterionFFL(fake_image, real_image) * self.opt.lambda_ffl)
        return losses + [loss_D_real, loss_D_fake], fake_image

    @torch.no_grad()
    def inference(self, fms):
        return self.netG(self._device(fms))

    def save(self, which_epoch):
        for net, label in ((self.netG, 'G'), (self.netD, 'D')):
            state = {k: v.cpu() for k, v in net.state_dict().items()}
            torch.save(state, os.path.join(self.save_dir, '%s_net_%s.pth' % (which_epoch, label)))

    def load_network(self, net, label, which_epoch):
        path = os.path.join(self.save_dir, '%s_net_%s.pth' % (which_epoch, label))
        net.load_state_dict(torch.load(path, map_location='cpu'))
        print('loaded %s' % path)

    def update_learning_rate(self):
        lr = self.old_lr - self.opt.lr / self.opt.niter_decay
        for optimizer in (self.optimizer_D, self.optimizer_G):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        self.old_lr = lr
