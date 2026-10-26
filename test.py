"""Generate core images for the test set and report PSNR / SSIM."""
import csv
import os

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import util
from data.fms_log_dataset import create_dataloader
from models.lgdp_model import LGDPModel
from options import test_options


def to_unit_range(t):
    """(3, H, W) tensor in [-1, 1] -> (H, W, 3) float array in [0, 1]."""
    return np.clip(((t + 1.0) / 2.0).permute(1, 2, 0).detach().cpu().numpy(), 0, 1)


def main():
    opt = test_options()
    opt.batchSize = 1
    opt.nThreads = 1

    _, loader = create_dataloader(opt, shuffle=False)
    model = LGDPModel(opt)
    model.eval()

    out_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    img_dir = os.path.join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    for i, data in enumerate(loader):
        generated = model.inference(data['fms'], data['log'])[0]
        real = data['image'][0]
        gen_np, real_np = to_unit_range(generated), to_unit_range(real)
        p = psnr(real_np, gen_np, data_range=1.0)
        s = ssim(real_np, gen_np, data_range=1.0, channel_axis=2)
        name = os.path.splitext(os.path.basename(data['path'][0]))[0]
        rows.append((name, p, s))
        print('[%d/%d] %s  PSNR %.4f  SSIM %.4f' % (i + 1, len(loader), name, p, s))
        for label, im in (('input_fms', data['fms'][0].repeat(3, 1, 1)),
                          ('synthesized_core', generated), ('original_core', real)):
            util.save_image(util.tensor2im(im), os.path.join(img_dir, '%s_%s.jpg' % (name, label)))

    mean_psnr = float(np.mean([r[1] for r in rows]))
    mean_ssim = float(np.mean([r[2] for r in rows]))
    with open(os.path.join(out_dir, 'metrics.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample', 'psnr', 'ssim'])
        w.writerows(rows)
        w.writerow(['MEAN', mean_psnr, mean_ssim])
    print('%d samples  mean PSNR %.4f dB  mean SSIM %.4f' % (len(rows), mean_psnr, mean_ssim))
    print('results saved to %s' % out_dir)


if __name__ == '__main__':
    main()
