"""Normalisation of the eight well-log values.

--log_norm phase_minmax (default)
    Min-max scale each feature to [-1, 1] with the minimum and maximum of the CSV being read
    (train CSV when training, test CSV when testing). No special treatment of missing values.

--log_norm train_robust
    1. values <= -999 (the log null value -999.25) are missing and replaced by the training-set median;
    2. IDPH -> log10(IDPH), Inv -> sign(x) * log10(1 + |x|);
    3. clip to the training-set 1st..99th percentile and scale linearly to [-1, 1].
    The statistics are fitted on the training CSV only and saved to checkpoints_dir/name/log_norm.json;
    testing loads that file.
"""
import json
import os

import numpy as np

MISSING_THRESHOLD = -999.0


class PhaseMinMax:
    def __init__(self, values):
        """values: (N, 8) float32 array of the CSV being read."""
        self.mins = np.min(values, axis=0)
        self.maxs = np.max(values, axis=0)

    def __call__(self, x):
        out = np.zeros_like(x)
        for i in range(len(x)):
            if self.maxs[i] > self.mins[i]:
                out[i] = 2.0 * (x[i] - self.mins[i]) / (self.maxs[i] - self.mins[i]) - 1.0
                out[i] = np.clip(out[i], -1.0, 1.0)
            else:
                out[i] = 0.0
        return out


def _transform(col, x):
    if col == 'IDPH':
        return np.log10(np.maximum(x, 1e-3))
    if col == 'Inv':
        return np.sign(x) * np.log10(1.0 + np.abs(x))
    return x


class TrainRobust:
    def __init__(self, stats):
        self.stats = stats

    @classmethod
    def fit(cls, columns, values):
        """values: (N, 8) array of the training CSV."""
        values = np.asarray(values, dtype=np.float64)
        stats = {'columns': list(columns), 'missing_threshold': MISSING_THRESHOLD,
                 'transforms': {'IDPH': 'log10', 'Inv': 'signed_log10_1p'},
                 'n_train': int(values.shape[0]), 'n_missing': {}, 'median': [], 'p1': [], 'p99': []}
        for i, col in enumerate(columns):
            v = values[:, i]
            valid = v > MISSING_THRESHOLD
            t = _transform(col, v[valid])
            med = float(np.median(t))
            t_all = np.full(v.shape, med)
            t_all[valid] = t
            stats['n_missing'][col] = int((~valid).sum())
            stats['median'].append(med)
            stats['p1'].append(float(np.percentile(t_all, 1)))
            stats['p99'].append(float(np.percentile(t_all, 99)))
        return cls(stats)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.stats, f, indent=2)

    @classmethod
    def load(cls, path, columns):
        with open(path) as f:
            stats = json.load(f)
        assert stats['columns'] == list(columns), 'log columns differ from the training CSV'
        return cls(stats)

    def __call__(self, x):
        s = self.stats
        out = np.zeros(len(x), dtype=np.float32)
        for i, col in enumerate(s['columns']):
            v = float(x[i])
            t = s['median'][i] if v <= s['missing_threshold'] else float(_transform(col, np.array(v)))
            lo, hi = s['p1'][i], s['p99'][i]
            out[i] = 0.0 if hi <= lo else 2.0 * (min(max(t, lo), hi) - lo) / (hi - lo) - 1.0
        return out


def build_log_normalizer(opt, columns, values):
    """Returns a callable that maps the raw 8 log values of one sample to [-1, 1]."""
    if opt.log_norm == 'phase_minmax':
        return PhaseMinMax(values)
    if opt.log_norm == 'train_robust':
        path = os.path.join(opt.checkpoints_dir, opt.name, 'log_norm.json')
        if opt.isTrain:
            norm = TrainRobust.fit(columns, values)
            norm.save(path)
            print('log normalisation fitted on the training set, saved to %s' % path)
        else:
            if not os.path.exists(path):
                raise FileNotFoundError('--log_norm train_robust needs the training statistics: %s' % path)
            norm = TrainRobust.load(path, columns)
            print('log normalisation loaded from %s' % path)
        return norm
    raise ValueError('unknown --log_norm %s' % opt.log_norm)
