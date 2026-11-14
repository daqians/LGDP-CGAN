"""Paired dataset: FMS patch (.npy, 24 x 16 x 4) -> core image (.jpg, 512 x 512)."""
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


def _list_files(folder, exts):
    assert os.path.isdir(folder), '%s is not a valid directory' % folder
    files = []
    for root, _, fnames in sorted(os.walk(folder)):
        for fname in fnames:
            if fname.endswith(exts):
                files.append(os.path.join(root, fname))
    return sorted(files)


class FMSDataset(torch.utils.data.Dataset):
    """
    A: the first `input_channels` FMS pads, bilinearly resized to fineSize x fineSize and min-max scaled
       to [-1, 1] per sample (over all channels together).
    B: core image in [-1, 1]; random horizontal flip during training (the FMS input is not flipped).
    A and B are paired by sorted file name.
    """

    def __init__(self, opt):
        self.opt = opt
        self.A_paths = _list_files(os.path.join(opt.dataroot, opt.phase + '_A'), ('.npy',))
        self.B_paths = _list_files(os.path.join(opt.dataroot, opt.phase + '_B'),
                                   ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG'))
        assert len(self.A_paths) == len(self.B_paths), \
            '%d FMS files but %d core images' % (len(self.A_paths), len(self.B_paths))
        for a, b in zip(self.A_paths, self.B_paths):
            sa, sb = os.path.splitext(os.path.basename(a))[0], os.path.splitext(os.path.basename(b))[0]
            assert sa == sb, 'unpaired files: %s vs %s' % (a, b)
        to_tensor = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.to_tensor = transforms.Compose(to_tensor)
        print('%s: %d samples, %d FMS pads as input' % (opt.phase, len(self.A_paths), opt.input_channels))

    def load_fms(self, path):
        a = np.load(path)
        if a.ndim == 2:
            a = a[:, :, np.newaxis]
        a = a[:, :, :self.opt.input_channels]
        a = torch.from_numpy(a).float().permute(2, 0, 1)  # (C, 24, 16)
        size = self.opt.fineSize
        if a.shape[1:] != (size, size):
            a = F.interpolate(a.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False).squeeze(0)
        a_min, a_max = a.min(), a.max()
        a = (a - a_min) / (a_max - a_min) if a_max > a_min else torch.zeros_like(a)
        return a * 2.0 - 1.0

    def __getitem__(self, index):
        fms = self.load_fms(self.A_paths[index])
        core = Image.open(self.B_paths[index]).convert('RGB')
        # same random calls as pix2pixHD's get_params (crop position is unused: no cropping)
        random.randint(0, 0)
        random.randint(0, 0)
        flip = random.random() > 0.5
        if self.opt.isTrain and not getattr(self.opt, 'no_flip', False) and flip:
            core = core.transpose(Image.FLIP_LEFT_RIGHT)
        return {'fms': fms, 'image': self.to_tensor(core), 'path': self.A_paths[index]}

    def __len__(self):
        return len(self.A_paths)


def create_dataloader(opt, shuffle):
    dataset = FMSDataset(opt)
    loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=shuffle,
                                         num_workers=int(opt.nThreads))
    return dataset, loader
