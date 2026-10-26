"""FMS patch (.npy) + 8 well-log values (CSV) -> core image (.jpg) dataset.

Expected layout of opt.dataroot:
    {phase}_A/*.npy                 FMS patches, shape (24, 16, 4) = depth x samples per pad x pads
    {phase}_B/*.jpg                 core images (paired with {phase}_A by sorted file name)
    {phase}_add_modalities.csv      column 'Files' (= .npy file name) + 8 log columns
"""
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from data.log_norm import build_log_normalizer

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG')


def _list_files(folder, extensions):
    assert os.path.isdir(folder), '%s is not a valid directory' % folder
    paths = []
    for root, _, fnames in sorted(os.walk(folder)):
        for fname in fnames:
            if fname.endswith(extensions):
                paths.append(os.path.join(root, fname))
    return sorted(paths)


class FMSLogDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.A_paths = _list_files(os.path.join(opt.dataroot, opt.phase + '_A'), ('.npy',))
        self.B_paths = _list_files(os.path.join(opt.dataroot, opt.phase + '_B'), IMG_EXTENSIONS)
        assert len(self.A_paths) == len(self.B_paths), 'FMS and core image counts differ'
        self._load_logs(os.path.join(opt.dataroot, '%s_add_modalities.csv' % opt.phase))
        self.flip = opt.isTrain and not getattr(opt, 'no_flip', True)
        print('dataset: %d samples, log features: %s, log normalisation: %s'
              % (len(self.A_paths), self.log_columns, self.opt.log_norm))

    # ---- well logs -------------------------------------------------------
    def _load_logs(self, csv_path):
        df = pd.read_csv(csv_path)
        self.log_columns = [c for c in df.columns if c != 'Files']
        assert len(self.log_columns) == self.opt.log_input_nc, 'expected %d log columns' % self.opt.log_input_nc
        self.log_dict = {str(row['Files']).strip(): row[self.log_columns].values.astype(np.float32)
                         for _, row in df.iterrows()}
        self.normalize_logs = build_log_normalizer(self.opt, self.log_columns,
                                                   np.array(list(self.log_dict.values())))

    # ---- FMS ---------------------------------------------------------------
    def _load_fms(self, path):
        """(24, 16, 4) -> pads side by side (24, 64) -> bilinear resize -> per-sample min-max to [-1, 1]."""
        fms = np.load(path)
        assert fms.ndim == 3 and fms.shape[2] == 4, 'FMS patch must have shape (depth, samples, 4 pads)'
        unrolled = np.concatenate([fms[:, :, i] for i in range(4)], axis=1)[:, :, np.newaxis]
        t = torch.from_numpy(unrolled).float().permute(2, 0, 1)
        size = (self.opt.fineSize, self.opt.fineSize)
        if t.shape[1:] != size:
            t = F.interpolate(t.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t = (t - t_min) / (t_max - t_min)
        return t * 2.0 - 1.0

    # ---- core image ----------------------------------------------------------
    def _load_core(self, path):
        img = Image.open(path).convert('RGB')
        # random crop offsets are drawn as in pix2pixHD (unused without cropping) to keep the RNG sequence
        w, h = img.size
        random.randint(0, max(0, w - self.opt.fineSize))
        random.randint(0, max(0, h - self.opt.fineSize))
        flip = random.random() > 0.5
        # round the size to a multiple of 2^n_downsampling
        base = float(2 ** self.opt.n_downsample_global)
        nw, nh = int(round(w / base) * base), int(round(h / base) * base)
        if (nw, nh) != (w, h):
            img = img.resize((nw, nh), Image.BICUBIC)
        if self.flip and flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        t = transforms.ToTensor()(img)
        return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(t)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        name = os.path.basename(A_path)
        if name not in self.log_dict:
            raise KeyError('no log values for %s in the CSV' % name)
        log = torch.from_numpy(self.normalize_logs(self.log_dict[name].copy())).float()
        return {'fms': self._load_fms(A_path), 'log': log,
                'image': self._load_core(self.B_paths[index]), 'path': A_path}

    def __len__(self):
        return len(self.A_paths)


def create_dataloader(opt, shuffle):
    dataset = FMSLogDataset(opt)
    loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=shuffle,
                                         num_workers=int(opt.nThreads))
    return dataset, loader
