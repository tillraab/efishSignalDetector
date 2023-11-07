import itertools
import sys
import os
import argparse

import torch
import torchvision.transforms as T

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from IPython import embed

from confic import (MIN_FREQ, MAX_FREQ, DELTA_FREQ, FREQ_OVERLAP, DELTA_TIME, TIME_OVERLAP, IMG_SIZE, IMG_DPI, DATA_DIR, LABEL_DIR)


def load_spec_data(folder: str):
    """
    Load spectrogram of a given electrode-grid recording generated with the wavetracker package. The spectrograms may
    be to large to load in total, thats why memmory mapping is used (numpy.memmap).

    Parameters
    ----------
    folder: str
        Folder where fine spec numpy files generated for grid recordings with the wavetracker package can be found.

    Returns
    -------
    fill_freqs: ndarray
        Freuqencies corresponding to 1st dimension of the spectrogram.
    fill_times: ndarray
        Times corresponding to the 2nd dimenstion if the spectrigram.
    fill_spec: ndarray
        Spectrigram of the recording refered to in the input folder.
    """
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

    return fill_freqs, fill_times, fill_spec


def load_tracking_data(folder):
    base_path = Path(folder)
    EODf_v = np.load(base_path / 'fund_v.npy')
    ident_v = np.load(base_path / 'ident_v.npy')
    idx_v = np.load(base_path / 'idx_v.npy')
    times_v = np.load(base_path / 'times.npy')

    return EODf_v, ident_v, idx_v, times_v


def load_trial_data(folder):
    base_path = Path(folder)
    fish_freq = np.load(base_path / 'analysis' / 'fish_freq.npy')
    rise_idx = np.load(base_path / 'analysis' / 'rise_idx.npy')
    rise_size = np.load(base_path / 'analysis' / 'rise_size.npy')

    fish_baseline_freq = np.load(base_path / 'analysis' / 'baseline_freqs.npy')
    fish_baseline_freq_time = np.load(base_path / 'analysis' / 'baseline_freq_times.npy')

    return fish_freq, rise_idx, rise_size, fish_baseline_freq, fish_baseline_freq_time


def save_spec_pic(folder, s_trans, times, freq, t_idx0, t_idx1, f_idx0, f_idx1, dataset_folder):
    f_res, t_res = freq[1] - freq[0], times[1] - times[0]

    fig_title = (f'{Path(folder).name}__{times[t_idx0]:5.0f}s-{times[t_idx1]:5.0f}s__{freq[f_idx0]:4.0f}-{freq[f_idx1]:4.0f}Hz.png').replace(' ', '0')
    fig = plt.figure(figsize=IMG_SIZE, num=fig_title)
    gs = gridspec.GridSpec(1, 1, bottom=0, left=0, right=1, top=1)  #
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(s_trans.squeeze(), cmap='gray', aspect='auto', origin='lower',
              extent=(times[t_idx0] / 3600, (times[t_idx1] + t_res) / 3600, freq[f_idx0], freq[f_idx1] + f_res))
    ax.axis(False)

    # plt.savefig(os.path.join(dataset_folder, fig_title), dpi=IMG_DPI)
    plt.savefig(Path(DATA_DIR)/fig_title, dpi=IMG_DPI)
    plt.close()

    return fig_title, (IMG_SIZE[0]*IMG_DPI, IMG_SIZE[1]*IMG_DPI)


def bboxes_from_file(times_v, fish_freq, rise_idx, rise_size, fish_baseline_freq_time, fish_baseline_freq, pic_save_str,
                     bbox_df, cols, width, height, t0, t1, f0, f1):

    times_v_idx0, times_v_idx1 = np.argmin(np.abs(times_v - t0)), np.argmin(np.abs(times_v - t1))

    all_x_center = []
    all_y_center = []
    all_width = []
    all_height= []
    for id_idx in range(len(fish_freq)):
        rise_idx_oi = np.array(rise_idx[id_idx][
                                   (rise_idx[id_idx] >= times_v_idx0) &
                                   (rise_idx[id_idx] <= times_v_idx1) &
                                   (rise_size[id_idx] >= 10)], dtype=int)
        rise_size_oi = rise_size[id_idx][(rise_idx[id_idx] >= times_v_idx0) &
                                         (rise_idx[id_idx] <= times_v_idx1) &
                                         (rise_size[id_idx] >= 10)]
        if len(rise_idx_oi) == 0:
            # np.savetxt(LABEL_DIR / Path(pic_save_str).with_suffix('.txt'), np.array([]))
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

        left_time_bound -= 0.01 * (t1 - t0)
        right_time_bound += 0.05 * (t1 - t0)
        lower_freq_bound -= 0.01 * (f1 - f0)
        upper_freq_bound += 0.05 * (f1 - f0)

        mask2 = ((left_time_bound >= t0) &
                (right_time_bound <= t1) &
                (lower_freq_bound >= f0) &
                (upper_freq_bound <= f1)
        )
        left_time_bound = left_time_bound[mask2]
        right_time_bound = right_time_bound[mask2]
        lower_freq_bound = lower_freq_bound[mask2]
        upper_freq_bound = upper_freq_bound[mask2]

        if len(left_time_bound) == 0:
            continue

        # x0 = np.array((left_time_bound - t0) / (t1 - t0) * width, dtype=int)
        # x1 = np.array((right_time_bound - t0) / (t1 - t0) * width, dtype=int)
        #
        # y0 = np.array((1 - (upper_freq_bound - f0) / (f1 - f0)) * height, dtype=int)
        # y1 = np.array((1 - (lower_freq_bound - f0) / (f1 - f0)) * height, dtype=int)


        rel_x0 = np.array((left_time_bound - t0) / (t1 - t0), dtype=float)
        rel_x1 = np.array((right_time_bound - t0) / (t1 - t0), dtype=float)

        rel_y0 = np.array(1 - (upper_freq_bound - f0) / (f1 - f0), dtype=float)
        rel_y1 = np.array(1 - (lower_freq_bound - f0) / (f1 - f0), dtype=float)

        rel_x_center = rel_x1 - (rel_x1 - rel_x0) / 2
        rel_y_center = rel_y1 - (rel_y1 - rel_y0) / 2
        rel_width = rel_x1 - rel_x0
        rel_height = rel_y1 - rel_y0

        all_x_center.extend(rel_x_center)
        all_y_center.extend(rel_y_center)
        all_width.extend(rel_width)
        all_height.extend(rel_height)

        # bbox = np.array([[pic_save_str for i in range(len(left_time_bound))],
        #                  left_time_bound,
        #                  right_time_bound,
        #                  lower_freq_bound,
        #                  upper_freq_bound,
        #                  x0, y0, x1, y1])

    bbox_yolo_style = np.array([
        np.ones(len(all_x_center)),
        all_x_center,
        all_y_center,
        all_width,
        all_height
    ]).T

    np.savetxt(LABEL_DIR/ Path(pic_save_str).with_suffix('.txt'), bbox_yolo_style)
    return bbox_yolo_style


def main(args):
    folders = list(f.parent for f in Path(args.folder).rglob('fill_times.npy'))

    if not args.inference:
        print('generate training dataset only for files with detected rises')
        folders = [folder for folder in folders if (folder / 'analysis' / 'rise_idx.npy').exists()]
        cols = ['image', 't0', 't1', 'f0', 'f1', 'x0', 'y0', 'x1', 'y1']
        bbox_df = pd.DataFrame(columns=cols)

    else:
        print('generate inference dataset ... only image output')
        bbox_df = {}

    for enu, folder in enumerate(folders):
        print(f'DataSet generation from {folder} | {enu+1}/{len(folders)}')

        # load different categories of data
        freq, times, spec = (
            load_spec_data(folder))
        EODf_v, ident_v, idx_v, times_v = (
            load_tracking_data(folder))
        if not args.inference:
            fish_freq, rise_idx, rise_size, fish_baseline_freq, fish_baseline_freq_time = (
                load_trial_data(folder))

        # generate iterator for analysis window loop
        pic_base = tqdm(itertools.product(
            np.arange(0, times[-1], DELTA_TIME),
            np.arange(MIN_FREQ, MAX_FREQ, DELTA_FREQ)
        ),
            total=int((((MAX_FREQ-MIN_FREQ)//DELTA_FREQ)+1) * ((times[-1] // DELTA_TIME)+1))
        )

        for t0, f0 in pic_base:

            t1 = t0 + DELTA_TIME + TIME_OVERLAP
            f1 = f0 + DELTA_FREQ + FREQ_OVERLAP

            present_freqs = EODf_v[(~np.isnan(ident_v))     &
                                   (t0 <= times_v[idx_v]) &
                                   (times_v[idx_v] <= t1) &
                                   (EODf_v >= f0) &
                                   (EODf_v <= f1)]
            if len(present_freqs) == 0:
                continue

            # get spec_idx for current spec snippet
            f_idx0, f_idx1 = np.argmin(np.abs(freq - f0)), np.argmin(np.abs(freq - f1))
            t_idx0, t_idx1 = np.argmin(np.abs(times - t0)), np.argmin(np.abs(times - t1))

            # get spec snippet and create torch.tensfor from it
            s = torch.from_numpy(spec[f_idx0:f_idx1, t_idx0:t_idx1].copy()).type(torch.float32)
            log_s = torch.log10(s)
            transformed = T.Normalize(mean=torch.mean(log_s), std=torch.std(log_s))
            s_trans = transformed(log_s.unsqueeze(0))

            pic_save_str, (width, height) = save_spec_pic(folder, s_trans, times, freq, t_idx0, t_idx1, f_idx0, f_idx1, args.dataset_folder)

            if not args.inference:
                bbox_yolo_style = bboxes_from_file(times_v, fish_freq, rise_idx, rise_size,
                                                   fish_baseline_freq_time, fish_baseline_freq,
                                                   pic_save_str, bbox_df, cols, width, height, t0, t1, f0, f1)

            #######################################################################
            if False:
                if bbox_yolo_style.shape[0] >= 1:
                    f_res, t_res = freq[1] - freq[0], times[1] - times[0]

                    fig_title = (
                        f'{Path(folder).name}__{times[t_idx0]:5.0f}s-{times[t_idx1]:5.0f}s__{freq[f_idx0]:4.0f}-{freq[f_idx1]:4.0f}Hz.png').replace(
                        ' ', '0')
                    fig = plt.figure(figsize=IMG_SIZE, num=fig_title)
                    gs = gridspec.GridSpec(1, 1, bottom=0.1, left=0.1, right=0.95, top=0.95)  #
                    ax = fig.add_subplot(gs[0, 0])
                    ax.imshow(s_trans.squeeze(), cmap='gray', aspect='auto', origin='lower',
                              extent=(times[t_idx0] / 3600, (times[t_idx1] + t_res) / 3600, freq[f_idx0], freq[f_idx1] + f_res))
                    # ax.invert_yaxis()
                    # ax.axis(False)

                    for i in range(len(bbox_df)):
                        # Cbbox = np.array(bbox_df.loc[i, ['x0', 'y0', 'x1', 'y1']].values, dtype=np.float32)
                        Cbbox = bbox_df.loc[i, ['t0', 'f0', 't1', 'f1']]
                        ax.add_patch(
                            Rectangle((float(Cbbox['t0']) / 3600, float(Cbbox['f0'])),
                                      float(Cbbox['t1']) / 3600 - float(Cbbox['t0']) / 3600,
                                      float(Cbbox['f1']) - float(Cbbox['f0']),
                                      fill=False, color="white", linestyle='-', linewidth=2, zorder=10)
                        )

                    # print(bbox_yolo_style.T)

                    for bbox in bbox_yolo_style:
                        x0 = bbox[1] - bbox[3]/2 # x_center - width/2
                        y0 = 1 - (bbox[2] + bbox[4]/2) # x_center - width/2
                        w = bbox[3]
                        h = bbox[4]
                        ax.add_patch(
                            Rectangle((x0, y0), w, h,
                                      fill=False, color="k", linestyle='--', linewidth=2, zorder=10,
                                      transform=ax.transAxes)
                        )
                    plt.show()
            #######################################################################

        # if not args.inference:
        #     print('save bboxes')
            # bbox_df.to_csv(os.path.join(args.dataset_folder, 'bbox_dataset.csv'), columns=cols, sep=',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluated electrode array recordings with multiple fish.')
    parser.add_argument('folder', type=str, help='single recording analysis', default='')
    parser.add_argument('-d', "--dataset_folder", type=str, help='designated datasef folder', default=DATA_DIR)
    parser.add_argument('-i', "--inference", action="store_true", help="generate inference dataset. Img only")
    args = parser.parse_args()

    if not Path(args.dataset_folder).exists():
        Path(args.dataset_folder).mkdir(parents=True, exist_ok=True)

    main(args)