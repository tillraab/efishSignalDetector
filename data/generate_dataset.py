import time

import numpy as np
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import pandas as pd

from tqdm.auto import tqdm

import itertools
import sys
import os

from IPython import embed

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


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
    rise_size = np.load(base_path / 'analysis' / 'rise_size.npy')

    fish_baseline_freq = np.load(base_path / 'analysis' / 'baseline_freqs.npy')
    fish_baseline_freq_time = np.load(base_path / 'analysis' / 'baseline_freq_times.npy')

    return fill_freqs, fill_times, fill_spec, EODf_v, ident_v, idx_v, times_v, fish_freq, rise_idx, rise_size, fish_baseline_freq, fish_baseline_freq_time

def save_spec_pic(folder, s_trans, times, freq, t_idx0, t_idx1, f_idx0, f_idx1, t_res, f_res):
    size = (7, 7)
    dpi = 256
    fig_title = (f'{Path(folder).name}__{times[t_idx0]:.0f}s-{times[t_idx1]:.0f}s__{freq[f_idx0]:4.0f}-{freq[f_idx1]:4.0f}Hz.png').replace(' ', '0')
    fig = plt.figure(figsize=(7, 7), num=fig_title)
    gs = gridspec.GridSpec(1, 2, width_ratios=(8, 1), wspace=0)  # , bottom=0, left=0, right=1, top=1
    gs2 = gridspec.GridSpec(1, 1, bottom=0, left=0, right=1, top=1)  #
    ax = fig.add_subplot(gs2[0, 0])
    im = ax.imshow(s_trans.squeeze(), cmap='gray', aspect='auto', origin='lower',
                   extent=(times[t_idx0] / 3600, times[t_idx1] / 3600 + t_res, freq[f_idx0], freq[f_idx1] + f_res))
    ax.axis(False)

    plt.savefig(os.path.join('train', fig_title), dpi=256)
    plt.close()

    return fig_title, (size[0]*dpi, size[1]*dpi)

def bboxes_from_file(times_v, fish_freq, rise_idx, rise_size, fish_baseline_freq_time, fish_baseline_freq, pic_save_str,
                     bbox_df, cols, width, height, t0, t1, f0, f1):

    times_v_idx0, times_v_idx1 = np.argmin(np.abs(times_v - t0)), np.argmin(np.abs(times_v - t1))
    for id_idx in range(len(fish_freq)):
        rise_idx_oi = np.array(rise_idx[id_idx][
                                   (rise_idx[id_idx] >= times_v_idx0) &
                                   (rise_idx[id_idx] <= times_v_idx1) &
                                   (rise_size[id_idx] >= 10)], dtype=int)
        rise_size_oi = rise_size[id_idx][(rise_idx[id_idx] >= times_v_idx0) &
                                         (rise_idx[id_idx] <= times_v_idx1) &
                                         (rise_size[id_idx] >= 10)]
        if len(rise_idx_oi) == 0:
            continue
        closest_baseline_idx = list(map(lambda x: np.argmin(np.abs(fish_baseline_freq_time - x)), times_v[rise_idx_oi]))
        closest_baseline_freq = fish_baseline_freq[id_idx][closest_baseline_idx]

        upper_freq_bound = closest_baseline_freq + rise_size_oi
        lower_freq_bound = closest_baseline_freq

        left_time_bound = times_v[rise_idx_oi]
        right_time_bound = np.zeros_like(left_time_bound)

        for enu, Ct_oi in enumerate(times_v[rise_idx_oi]):
            Crise_size = rise_size_oi[enu]
            Cblf = closest_baseline_freq[enu]

            rise_end_t = times_v[(times_v > Ct_oi) &
                                 (fish_freq[id_idx] < Cblf + Crise_size * 0.37)]
            if len(rise_end_t) == 0:
                right_time_bound[enu] = np.nan
            else:
                right_time_bound[enu] = rise_end_t[0]

        mask = (~np.isnan(right_time_bound) & ((right_time_bound - left_time_bound) > 1.))
        left_time_bound = left_time_bound[mask]
        right_time_bound = right_time_bound[mask]
        lower_freq_bound = lower_freq_bound[mask]
        upper_freq_bound = upper_freq_bound[mask]

        dt_bbox = right_time_bound - left_time_bound
        df_bbox = upper_freq_bound - lower_freq_bound

        # embed()
        # quit()
        # left_time_bound -= dt_bbox + 0.01 * (t1 - t0)
        # right_time_bound += dt_bbox + 0.01 * (t1 - t0)
        # lower_freq_bound -= df_bbox + 0.01 * (f1 - f0)
        # upper_freq_bound += df_bbox + 0.01 * (f1 - f0)

        left_time_bound -= 0.01 * (t1 - t0)
        right_time_bound += 0.05 * (t1 - t0)
        lower_freq_bound -= 0.01 * (f1 - f0)
        upper_freq_bound += 0.05 * (f1 - f0)

        # embed()
        # quit()
        mask2 = ((left_time_bound >= t0) &
                (right_time_bound <= t1) &
                (lower_freq_bound >= f0) &
                (upper_freq_bound <= f1)
        )
        left_time_bound = left_time_bound[mask2]
        right_time_bound = right_time_bound[mask2]
        lower_freq_bound = lower_freq_bound[mask2]
        upper_freq_bound = upper_freq_bound[mask2]

        x0 = np.array((left_time_bound - t0) / (t1 - t0) * width, dtype=int)
        x1 = np.array((right_time_bound - t0) / (t1 - t0) * width, dtype=int)
        y0 = np.array((1 - (upper_freq_bound - f0) / (f1 - f0)) * height, dtype=int)
        y1 = np.array((1 - (lower_freq_bound - f0) / (f1 - f0)) * height, dtype=int)

        bbox = np.array([[pic_save_str for i in range(len(left_time_bound))],
                         left_time_bound,
                         right_time_bound,
                         lower_freq_bound,
                         upper_freq_bound,
                         x0, y0, x1, y1])
        # test_s = ['a', 'a', 'a', 'a']
        tmp_df = pd.DataFrame(
            # index= [pic_save_str for i in range(len(left_time_bound))],
            # index= test_s,
            data=bbox.T,
            columns=cols
        )
        bbox_df = pd.concat([bbox_df, tmp_df], ignore_index=True)
        # bbox_df.append(tmp_df)
    return bbox_df

def main(args):
    def development_fn():
        fig_title = (f'{Path(args.folder).name}__{t0:.0f}s-{t1:.0f}s__{f0:4.0f}-{f1:4.0f}Hz').replace(' ', '0')
        fig = plt.figure(figsize=(7, 7), num=fig_title)
        gs = gridspec.GridSpec(1, 2, width_ratios=(8, 1), wspace=0, left=0.1, bottom=0.1, right=0.9,
                               top=0.95)  # , bottom=0, left=0, right=1, top=1
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        im = ax.imshow(s_trans.squeeze(), cmap='gray', aspect='auto', origin='lower',
                       extent=(times[t_idx0], times[t_idx1] + t_res, freq[f_idx0], freq[f_idx1] + f_res))
        fig.colorbar(im, cax=cax, orientation='vertical')


        cols = ['image', 't0', 't1', 'f0', 'f1', 'x0', 'y0', 'x1', 'y1']
        dev_df = pd.DataFrame(columns=cols)

        dev_df = bboxes_from_file(times_v, fish_freq, rise_idx, rise_size, fish_baseline_freq_time, fish_baseline_freq,
                                  fig_title, dev_df, cols, (7*256), (7*256), t0, t1, f0, f1)

        # embed()
        # quit()
        time_freq_bbox = torch.as_tensor(dev_df.loc[:, ['t0', 'f0', 't1', 'f1']].values.astype(np.float32))

        for bbox in time_freq_bbox:
            Ct0, Cf0, Ct1, Cf1 = bbox
            ax.add_patch(
                Rectangle((Ct0, Cf0), Ct1-Ct0, Cf1-Cf0, fill=False, color="white", linewidth=2, zorder=10)
            )
        # for enu in range(len(left_time_bound)):
        #     if np.isnan(right_time_bound[enu]):
        #         continue
        #     ax.add_patch(
        #         Rectangle((left_time_bound[enu], lower_freq_bound[enu]),
        #                   (right_time_bound[enu] - left_time_bound[enu]),
        #                   (upper_freq_bound[enu] - lower_freq_bound[enu]),
        #                   fill=False, color="white", linewidth=2, zorder=10)
        #     )
        plt.show()

    min_freq = 200
    max_freq = 1500
    d_freq = 200
    freq_overlap = 25
    d_time = 60*10
    time_overlap = 60*1

    if not os.path.exists(os.path.join('train', 'bbox_dataset.csv')):
        cols = ['image', 't0', 't1', 'f0', 'f1', 'x0', 'y0', 'x1', 'y1']
        bbox_df = pd.DataFrame(columns=cols)

    else:
        bbox_df = pd.read_csv(os.path.join('train', 'bbox_dataset.csv'), sep=',', index_col=0)
        cols = list(bbox_df.keys())
        eval_files = []
        # ToDo: make sure not same file twice
        for f in pd.unique(bbox_df['image']):
            eval_files.append(f.split('__')[0])

    folders = list(f.parent for f in Path(args.folder).rglob('fill_times.npy'))

    # embed()
    # quit()

    for enu, folder in enumerate(folders):
        print(f'DataSet generation from {folder} | {enu+1}/{len(folders)}')
        if not (folder/'analysis'/'rise_idx.npy').exists():
            continue

        freq, times, spec, EODf_v, ident_v, idx_v, times_v, fish_freq, rise_idx, rise_size, fish_baseline_freq, fish_baseline_freq_time = (
            load_data(folder))
        f_res, t_res = freq[1] - freq[0], times[1] - times[0]

        pic_base = tqdm(itertools.product(
            np.arange(0, times[-1], d_time),
            np.arange(min_freq, max_freq, d_freq)
        ),
            total=int((((max_freq-min_freq)//d_freq)+1) * ((times[-1] // d_time)+1))
        )

        for t0, f0 in pic_base:

            t1 = t0 + d_time + time_overlap
            f1 = f0 + d_freq + freq_overlap

            present_freqs = EODf_v[(~np.isnan(ident_v))     &
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
                pic_save_str, (width, height) = save_spec_pic(folder, s_trans, times, freq, t_idx0, t_idx1, f_idx0, f_idx1, t_res, f_res)

                bbox_df = bboxes_from_file(times_v, fish_freq, rise_idx, rise_size,
                                           fish_baseline_freq_time, fish_baseline_freq,
                                           pic_save_str, bbox_df, cols, width, height, t0, t1, f0, f1)
            else:
                development_fn()

        if not args.dev:
            print('save')
            bbox_df.to_csv(os.path.join('train', 'bbox_dataset.csv'), columns=cols, sep=',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluated electrode array recordings with multiple fish.')
    parser.add_argument('folder', type=str, help='single recording analysis', default='')
    parser.add_argument('-d', "--dev", action="store_true", help="developer mode; no data saved")
    args = parser.parse_args()
    main(args)