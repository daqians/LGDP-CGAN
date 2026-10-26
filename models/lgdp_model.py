import os

import torch

from . import networks
from .generator import define_generator


class LGDPModel(torch.nn.Module):
    """Generator + conditional multi-scale discriminator, losses and optimisers."""

    loss_names = ['G_GAN', 'G_GAN_Feat', 'G_FFL', 'D_real', 'D_fake']

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not opt.isTrain:
            torch.backends.cudnn.benchmark = True

        self.netG = define_generator(fms_input_nc=opt.fms_input_nc, log_input_nc=opt.log_input_nc,
                                     output_nc=opt.output_nc, ngf=opt.ngf,
                                     n_downsample_global=opt.n_downsample_global,
                                     n_blocks_global=opt.n_blocks_global, norm=opt.norm,
                                     gate_reduction=opt.gate_reduction, gpu_ids=self.gpu_ids)

        if opt.isTrain:
            # the discriminator sees cat(FMS, core image, log values broadcast to the image size)
            netD_input_nc = opt.fms_input_nc + opt.output_nc + opt.log_input_nc
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm,
                                          opt.num_D, gpu_ids=self.gpu_ids)
            self.criterionGAN = networks.GANLoss()
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionFFL = networks.FocalFrequencyLoss(alpha=opt.ffl_alpha)
            self.old_lr = opt.lr
            self.optimizer_G = torch.optim.Adam(list(self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(list(self.netD.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        else:
            self.load_network(self.netG, 'G', opt.which_epoch)

    def _device(self, x):
        return x.cuda() if self.gpu_ids else x

    def _d_input(self, fms, log, image):
        B, _, H, W = image.shape
        log_spatial = log.view(B, self.opt.log_input_nc, 1, 1).expand(B, self.opt.log_input_nc, H, W)
        return torch.cat((fms, image, log_spatial), dim=1)

    def forward(self, fms, log, real_image):
        """Returns ([G_GAN, G_GAN_Feat, G_FFL, D_real, D_fake], fake_image)."""
        fms, log, real_image = self._device(fms), self._device(log), self._device(real_image)
        fake_image = self.netG(fms, log)

        # discriminator
        pred_fake_pool = self.netD(self._d_input(fms, log, fake_image.detach()))
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)
        pred_real = self.netD(self._d_input(fms, log, real_image.detach()))
        loss_D_real = self.criterionGAN(pred_real, True)

        # generator: adversarial
        pred_fake = self.netD(self._d_input(fms, log, fake_image))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # generator: feature matching over the intermediate discriminator layers
        loss_G_GAN_Feat = 0
        feat_weights = 4.0 / (self.opt.n_layers_D + 1)
        D_weights = 1.0 / self.opt.num_D
        for i in range(self.opt.num_D):
            for j in range(len(pred_fake[i]) - 1):
                loss_G_GAN_Feat += D_weights * feat_weights * \
                    self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        # generator: focal frequency loss
        loss_G_FFL = self.criterionFFL(fake_image, real_image) * self.opt.lambda_ffl

        return [loss_G_GAN, loss_G_GAN_Feat, loss_G_FFL, loss_D_real, loss_D_fake], fake_image

    @torch.no_grad()
    def inference(self, fms, log):
        return self.netG(self._device(fms), self._device(log))

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
