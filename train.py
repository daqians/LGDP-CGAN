"""Train LGDP-CGAN (FMS patch + 8 well-log values -> core image)."""
import math
import os
import time

import numpy as np
import torch

import util
from data.fms_log_dataset import create_dataloader
from models.lgdp_model import LGDPModel
from options import train_options


def main():
    opt = train_options()
    opt.print_freq = opt.print_freq * opt.batchSize // math.gcd(opt.print_freq, opt.batchSize)

    dataset, loader = create_dataloader(opt, shuffle=True)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    model = LGDPModel(opt)
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    img_dir = os.path.join(expr_dir, 'web', 'images')
    os.makedirs(img_dir, exist_ok=True)
    log_name = os.path.join(expr_dir, 'loss_log.txt')
    with open(log_name, 'a') as f:
        f.write('================ Training Loss (%s) ================\n' % time.strftime('%c'))

    total_steps = 0
    for epoch in range(1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        # the FFL weight ramps linearly from 0 to 1 over the first ffl_warmup_epochs epochs
        ffl_weight = min(1.0, (epoch - 1) / max(opt.ffl_warmup_epochs, 1))
        for data in loader:
            if total_steps % opt.print_freq == 0:
                iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            losses, generated = model(data['fms'], data['log'], data['image'])
            loss_dict = dict(zip(model.loss_names, losses))
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_FFL'] * ffl_weight

            model.optimizer_G.zero_grad()
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(model.netG.parameters(), max_norm=1.0)
            model.optimizer_G.step()

            model.optimizer_D.zero_grad()
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(model.netD.parameters(), max_norm=1.0)
            model.optimizer_D.step()

            if total_steps % opt.print_freq == 0:
                errors = {k: v.item() for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.print_freq
                message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, epoch_iter, t)
                message += ''.join('%s: %.3f ' % (k, v) for k, v in errors.items())
                print(message)
                with open(log_name, 'a') as f:
                    f.write(message + '\n')
                if any(np.isnan(v) or np.isinf(v) for v in errors.values()):
                    print('NaN/Inf loss, stopping')
                    return

            if total_steps % opt.display_freq == 0:
                for label, im in (('input_fms', data['fms'][0].repeat(3, 1, 1)),
                                  ('synthesized_core', generated.data[0]), ('real_core', data['image'][0])):
                    util.save_image(util.tensor2im(im), os.path.join(img_dir, 'epoch%.3d_%s.jpg' % (epoch, label)))

            if total_steps % opt.save_latest_freq == 0:
                model.save('latest')

            if epoch_iter >= dataset_size:
                break

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        if epoch % opt.save_epoch_freq == 0:
            model.save('latest')
            model.save(epoch)
        if epoch > opt.niter:
            model.update_learning_rate()

    model.save('latest')


if __name__ == '__main__':
    main()
