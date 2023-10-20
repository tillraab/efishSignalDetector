import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from tqdm.auto import tqdm

import itertools

import sys
import os

from IPython import embed


def load_data(folder):
    fill_freqs, fill_times, fill_spec = [], [], []

    if os.path.exists(os.path.join(folder, 'fill_spec.npy')):
        fill_freqs = np.load(os.path.join(folder, 'fill_freqs.npy'))
        fill_times = np.load(os.path.join(folder, 'fill_times.npy'))
        fill_spec_shape = np.load(os.path.join(folder, 'fill_spec_shape.npy'))
        fill_spec = np.memmap(os.path.join(folder, 'fill_spec.npy'), dtype='float', mode='r',
                                   shape=(fill_spec_shape[0], fill_spec_shape[1]), order='F')

    elif os.path.exists(os.path.join(folder, 'fine_spec.npy')):
        fill_freqs = np.load(os.path.join(folder, 'fine_freqs.npy'))
        fill_times = np.load(os.path.join(folder, 'fine_times.npy'))
        fill_spec_shape = np.load(os.path.join(folder, 'fine_spec_shape.npy'))
        fill_spec = np.memmap(os.path.join(folder, 'fine_spec.npy'), dtype='float', mode='r',
                                   shape=(fill_spec_shape[0], fill_spec_shape[1]), order='F')

    base_path = Path(folder)
    EODf_v = np.load(base_path / 'fund_v.npy')
    ident_v = np.load(base_path / 'ident_v.npy')
    idx_v = np.load(base_path / 'idx_v.npy')
    times_v = np.load(base_path / 'times.npy')

    return fill_freqs, fill_times, fill_spec, EODf_v, ident_v, idx_v, times_v


def main(folder):
    freq, times, spec, EODf_v, ident_v, idx_v, times_v = load_data(folder)

    unique_ids = np.unique(ident_v[~np.isnan(ident_v)])

    for t0, f0 in tqdm(list(itertools.product(np.arange(0, times_v[-1], 60*15), np.arange(200, 1500, 200)))):
        t1 = t0 + 60*20
        f1 = f0 + 250

        f_idx0, f_idx1 = np.argmin(np.abs(freq - f0)), np.argmin(np.abs(freq - f1))
        t_idx0, t_idx1 = np.argmin(np.abs(times_v - t0)), np.argmin(np.abs(times_v - t1))

        s = torch.from_numpy(spec[f_idx0:f_idx1, t_idx0:t_idx1].copy()).type(torch.float32)
        log_s = torch.log10(s)
        # s_normed = F.normalize(s.view(-1)).view(s.shape[0], s.shape[1])

        transformed = T.Normalize(mean=torch.mean(log_s), std=torch.std(log_s))
        s_trans = transformed(log_s.unsqueeze(0))

        fig_title = (f'{Path(folder).name}__{t0:.0f}s-{t1:.0f}s__{f0:.0f}-{f1:.0f}Hz').replace(' ', '0')
        fig = plt.figure(figsize=(10, 7), num=fig_title)
        gs = gridspec.GridSpec(1, 2, width_ratios=(8, 1), wspace=0)# , bottom=0, left=0, right=1, top=1
        gs2 = gridspec.GridSpec(1, 1, bottom=0, left=0, right=1, top=1)#
        ax = fig.add_subplot(gs2[0, 0])
        # cax = fig.add_subplot(gs[0, 1])
        im = ax.imshow(s_trans.squeeze(), cmap='gray', aspect='auto', origin='lower', extent=(times_v[t_idx0]/3600, times_v[t_idx1+1]/3600, freq[f_idx0], freq[f_idx1+1]))
        # im = ax.imshow(log_s, cmap='gray', aspect='auto')
        # ax.invert_yaxis()
        # fig.colorbar(im, cax=cax)
        ax.axis(False)

        plt.savefig(fig_title + '.png', dpi=300)
        plt.close()
    # # ax.imshow(spec[f0:f1, t0:t1], cmap='gray')





if __name__ == '__main__':
    main(sys.argv[1])