import numpy as np
import argparse
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

    fish_freq = np.load(base_path / 'analysis' / 'fish_freq.npy')
    rise_idx = np.load(base_path / 'analysis' / 'rise_idx.npy')


    return fill_freqs, fill_times, fill_spec, EODf_v, ident_v, idx_v, times_v, fish_freq, rise_idx

def save_spec_pic(folder, s_trans, times, freq, t_idx0, t_idx1, f_idx0, f_idx1, t_res, f_res):
    fig_title = (f'{Path(folder).name}__{t0:.0f}s-{t1:.0f}s__{f0:4.0f}-{f1:4.0f}Hz').replace(' ', '0')
    fig = plt.figure(figsize=(7, 7), num=fig_title)
    gs = gridspec.GridSpec(1, 2, width_ratios=(8, 1), wspace=0)  # , bottom=0, left=0, right=1, top=1
    gs2 = gridspec.GridSpec(1, 1, bottom=0, left=0, right=1, top=1)  #
    ax = fig.add_subplot(gs2[0, 0])
    im = ax.imshow(s_trans.squeeze(), cmap='gray', aspect='auto', origin='lower',
                   extent=(times[t_idx0] / 3600, times[t_idx1] / 3600 + t_res, freq[f_idx0], freq[f_idx1] + f_res))
    ax.axis(False)

    plt.savefig(os.path.join('train', fig_title + '.png'), dpi=256)
    plt.close()


def main(args):
    min_freq = 200
    max_freq = 1500
    d_freq = 200
    freq_overlap = 50
    d_time = 60*15
    time_overlap = 60*5

    freq, times, spec, EODf_v, ident_v, idx_v, times_v, fish_freq, rise_idx = load_data(args.folder)
    f_res, t_res = freq[1] - freq[0], times[1] - times[0]

    unique_ids = np.unique(ident_v[~np.isnan(ident_v)])

    pic_base = tqdm(itertools.product(
        np.arange(0, times[-1], d_time),
        np.arange(min_freq, max_freq, d_freq)
    ),
        total=int(((max_freq-min_freq)//d_freq) * (times[-1] // d_time))
    )

    for t0, f0 in pic_base:
        t1 = t0 + d_time + time_overlap
        f1 = f0 + d_freq + freq_overlap

        present_freqs = EODf_v[(~np.isnan(ident_v)) &
                               (t0 <= times_v[idx_v]) &
                               (times_v[idx_v] <= t1) &
                               (EODf_v >= f0) &
                               (EODf_v <= f1)]
        if len(present_freqs) == 0:
            continue

        f_idx0, f_idx1 = np.argmin(np.abs(freq - f0)), np.argmin(np.abs(freq - f1))
        t_idx0, t_idx1 = np.argmin(np.abs(times - t0)), np.argmin(np.abs(times - t1))

        s = torch.from_numpy(spec[f_idx0:f_idx1, t_idx0:t_idx1].copy()).type(torch.float32)
        log_s = torch.log10(s)
        transformed = T.Normalize(mean=torch.mean(log_s), std=torch.std(log_s))
        s_trans = transformed(log_s.unsqueeze(0))

        if not args.dev:
            save_spec_pic(args.folder, s_trans, times, freq, t_idx0, t_idx1, f_idx0, f_idx1, t_res, f_res)

        else:
            fig_title = (f'{Path(args.folder).name}__{t0:.0f}s-{t1:.0f}s__{f0:4.0f}-{f1:4.0f}Hz').replace(' ', '0')
            fig = plt.figure(figsize=(10, 7), num=fig_title)
            gs = gridspec.GridSpec(1, 2, width_ratios=(8, 1), wspace=0, left=0.1, bottom=0.1, right=0.9, top=0.95)  # , bottom=0, left=0, right=1, top=1
            ax = fig.add_subplot(gs[0, 0])
            cax = fig.add_subplot(gs[0, 1])
            im = ax.imshow(s_trans.squeeze(), cmap='gray', aspect='auto', origin='lower',
                           extent=(times[t_idx0], times[t_idx1] + t_res, freq[f_idx0], freq[f_idx1] + f_res))
            fig.colorbar(im, cax=cax, orientation='vertical')


            times_v_idx0, times_v_idx1 = np.argmin(np.abs(times_v - t0)), np.argmin(np.abs(times_v - t1))
            for id_idx in range(len(fish_freq)):
                ax.plot(times_v[times_v_idx0:times_v_idx1], fish_freq[id_idx][times_v_idx0:times_v_idx1], marker='.', color='k')
                rise_idx_oi = np.array(rise_idx[id_idx][(rise_idx[id_idx] >= times_v_idx0) & (rise_idx[id_idx] <= times_v_idx1)], dtype=int)
                ax.plot(times_v[rise_idx_oi], fish_freq[id_idx][rise_idx_oi], marker='o', color='tab:red')

            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluated electrode array recordings with multiple fish.')
    parser.add_argument('folder', type=str, help='single recording analysis', default='')
    parser.add_argument('-d', "--dev", action="store_true", help="developer mode; no data saved")
    # parser.add_argument('-x', type=int, nargs=2, default=[1272, 1282], help='x-borders of LED detect area (in pixels)')
    # parser.add_argument('-y', type=int, nargs=2, default=[1500, 1516], help='y-borders of LED area (in pixels)')
    args = parser.parse_args()
    main(args)