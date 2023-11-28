import itertools
import pathlib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from tqdm.auto import tqdm

from IPython import embed

def extract_time_freq_range_from_filename(img_path):
    file_name_str, time_span_str, freq_span_str = str(img_path.with_suffix('').name).split('__')
    time_span_str = time_span_str.replace('s', '')
    freq_span_str = freq_span_str.replace('Hz', '')

    t0, t1 = np.array(time_span_str.split('-'), dtype=float)
    f0, f1 = np.array(freq_span_str.split('-'), dtype=float)
    return file_name_str, t0, t1, f0, f1

def bbox_to_data(img_path, t_min, t_max, f_min, f_max):
    label_path = img_path.parent.parent / 'labels' / img_path.with_suffix('.txt').name

    annotations = np.loadtxt(label_path, delimiter=' ')
    if len(annotations.shape) == 1:
        annotations = np.array([annotations])
    if annotations.shape[1] == 0:
        print('no rises detected in this window')
        return [], []


    boxes = np.array([[x[1] - x[3] / 2, 1 - (x[2] + x[4] / 2), x[1] + x[3] / 2, 1 - (x[2] - x[4] / 2)] for x in annotations]) # x0, y0, x1, y1
    boxes[:, 0] = boxes[:, 0] * (t_max - t_min) + t_min
    boxes[:, 2] = boxes[:, 2] * (t_max - t_min) + t_min

    boxes[:, 1] = boxes[:, 1] * (f_max - f_min) + f_min
    boxes[:, 3] = boxes[:, 3] * (f_max - f_min) + f_min

    scores = annotations[:, -1]

    return boxes, scores

def load_wavetracker_data(raw_path):
    fund_v = np.load(raw_path.parent / 'fund_v.npy')
    ident_v = np.load(raw_path.parent / 'ident_v.npy')
    idx_v = np.load(raw_path.parent / 'idx_v.npy')
    times = np.load(raw_path.parent / 'times.npy')

    return fund_v, ident_v, idx_v, times


def assign_rises_to_ids(raw_path, time_frequency_bboxes, overlapping_boxes, bbox_groups):
    fund_v, ident_v, idx_v, times = load_wavetracker_data(raw_path)

    fig, ax = plt.subplots()
    ax.plot(times[idx_v[~np.isnan(ident_v)]], fund_v[~np.isnan(ident_v)], '.')

    mask = time_frequency_bboxes['file_name'] == raw_path.parent.name
    for index, bbox in time_frequency_bboxes[mask].iterrows():
        name, t0, f0, t1, f1 = (bbox[0], *bbox[1:-1].astype(float))
        if bbox_groups[index] == 0:
            color = 'tab:green'
        else:
            color = 'k'
        # if overlapping_boxes[index] == 0:
        #     color='tab:green'
        # elif overlapping_boxes[index] == 1:
        #     color = 'tab:olive'
        # elif overlapping_boxes[index] == 2:
        #     color = 'tab:orange'
        # elif overlapping_boxes[index] == 3:
        #     color = 'tab:red'
        # color = 'tab:green' if overlapping_boxes[index] == 0 else 'tab:orange'
        ax.add_patch(
            Rectangle((t0, f0),
                      (t1 - t0),
                      (f1 - f0),
                      fill=False, color=color, linestyle='--', linewidth=2, zorder=10)
        )
    plt.show()
    # if np.any(overlapping_boxes[mask] >= 2):
    #     print('yay')
    #     embed()
    #     quit()
    # ToDo: eliminate double rises -- overlap
    # ToDo: double detections -- non overlap --> the one with higher probability ?!
    # ToDo: assign rises to traces --> who is at lower right corner

def find_overlapping_bboxes(df_collect):
    file_names = np.array(df_collect)[:, 0]
    bboxes_overlapping_mask = np.zeros(len(df_collect))
    bboxes = np.array(df_collect)[:, 1:].astype(float)

    overlap_bbox_idxs = []

    for file_name in tqdm(np.unique(file_names)):
        file_bbox_idxs = np.arange(len(file_names))[file_names == file_name]
        for ind0, ind1 in itertools.combinations(file_bbox_idxs, r=2):
            bb0 = bboxes[ind0]
            bb1 = bboxes[ind1]
            t0_0, f0_0, t0_1, f0_1 = bb0[:-1]
            t1_0, f1_0, t1_1, f1_1 = bb1[:-1]

            bb_times = np.array([t0_0, t0_1, t1_0, t1_1])
            bb_time_associate = np.array([0, 0, 1, 1])
            time_helper = bb_time_associate[np.argsort(bb_times)]

            if time_helper[0] == time_helper[1]:
                # no temporal overlap
                continue

            # check freq overlap
            bb_freqs = np.array([f0_0, f0_1, f1_0, f1_1])
            bb_freq_associate = np.array([0, 0, 1, 1])

            freq_helper = bb_freq_associate[np.argsort(bb_freqs)]

            if freq_helper[0] == freq_helper[1]:
                continue

            bboxes_overlapping_mask[ind0] +=1
            bboxes_overlapping_mask[ind1] +=1

            overlap_bbox_idxs.append((ind0, ind1))

    return bboxes_overlapping_mask, np.asarray(overlap_bbox_idxs)


def main(args):
    img_paths = sorted(list(pathlib.Path(args.annotations).absolute().rglob('*.png')))
    df_collect = []

    for img_path in img_paths:
        # convert to time_frequency
        file_name_str, t_min, t_max, f_min, f_max = extract_time_freq_range_from_filename(img_path)

        boxes, scores = bbox_to_data(img_path, t_min, t_max, f_min, f_max ) # t0, t1, f0, f1
        # store values in df
        if not len(boxes) == 0:
            for (t0, f0, t1, f1), s in zip(boxes, scores):
                df_collect.append([file_name_str, t0, f0, t1, f1, s])

    df_collect = np.array(df_collect)

    bbox_overlapping_mask, overlap_bbox_idxs = find_overlapping_bboxes(df_collect)

    bbox_groups = delete_double_boxes(bbox_overlapping_mask, overlap_bbox_idxs, df_collect)
    # embed()
    # quit()
    time_frequency_bboxes = pd.DataFrame(data= np.array(df_collect), columns=['file_name', 't0', 'f0', 't1', 'f1', 'score'])

    if args.tracking_data_path:
        file_paths = sorted(list(pathlib.Path(args.tracking_data_path).absolute().rglob('*.raw')))
        for raw_path in file_paths:
            if not raw_path.parent.name in time_frequency_bboxes['file_name'].to_list():
                continue

            assign_rises_to_ids(raw_path, time_frequency_bboxes, bbox_overlapping_mask, bbox_groups)

    pass

def delete_double_boxes(bbox_overlapping_mask, overlap_bbox_idxs, df_collect):
    def get_connected(non_regarded_bbox_idx, overlap_bbox_idxs):
        mask = np.array((np.array(overlap_bbox_idxs) == non_regarded_bbox_idx).sum(1), dtype=bool)
        affected_bbox_idxs = np.unique(overlap_bbox_idxs[mask])
        return affected_bbox_idxs

    handled_bbox_idxs = []
    bbox_groups = np.zeros(len(df_collect))
    for Coverlapping_bbox_idx in tqdm(np.unique(overlap_bbox_idxs)):
        if Coverlapping_bbox_idx in handled_bbox_idxs:
            continue
        # if bbox_overlapping_mask[Coverlapping_bbox_idx] >= 3:
        #     pass
        # else:
        #     continue

        regarded_bbox_idxs = [Coverlapping_bbox_idx]
        mask = np.array((np.array(overlap_bbox_idxs) == Coverlapping_bbox_idx).sum(1), dtype=bool)
        affected_bbox_idxs = np.unique(overlap_bbox_idxs[mask])

        non_regarded_bbox_idxs = list(set(affected_bbox_idxs) - set(regarded_bbox_idxs))
        # non_regarded_bbox_idxs = list(set(non_regarded_bbox_idxs) - set(handled_bbox_idxs))
        while len(non_regarded_bbox_idxs) > 0:
            non_regarded_bbox_idxs_cp = np.copy(non_regarded_bbox_idxs)
            for non_regarded_bbox_idx in non_regarded_bbox_idxs_cp:
                Caffected_bbox_idxs = get_connected(non_regarded_bbox_idx, overlap_bbox_idxs)
                affected_bbox_idxs = np.unique(np.append(affected_bbox_idxs, Caffected_bbox_idxs))

                regarded_bbox_idxs.append(non_regarded_bbox_idx)
                non_regarded_bbox_idxs = list(set(affected_bbox_idxs) - set(regarded_bbox_idxs))

        bbox_idx_group = regarded_bbox_idxs
        bbox_groups[bbox_idx_group] = np.max(bbox_groups) + 1
        # bbox_scores = df_collect[bbox_idx_group][:, -1]
        # overlap_pct = np.full((len(bbox_idx_group), len(bbox_idx_group)), np.nan)
        #
        # for i, j in itertools.product(range(len(bbox_idx_group)), repeat=2):
        #     if i == j:
        #         continue
        #     bb0_idx = bbox_idx_group[i]
        #     bb1_idx = bbox_idx_group[j]
        #
        #     bb0_t0, bb0_t1 = df_collect[bb0_idx][1].astype(float), df_collect[bb0_idx][3].astype(float)
        #     bb1_t0, bb1_t1 = df_collect[bb1_idx][1].astype(float), df_collect[bb1_idx][3].astype(float)
        #
        #     helper = np.array([0, 0, 1, 1])
        #     bb_times = np.array([bb0_t0, bb0_t1, bb1_t0, bb1_t1])
        #
        #     sorted_helper = helper[bb_times.argsort()]
        #
        #     if sorted_helper[0] == sorted_helper[1]:
        #         continue
        #
        #     elif sorted_helper[1] == sorted_helper[2] == 0:
        #         overlap_pct[i, j] = 1
        #
        #     elif sorted_helper[1] == sorted_helper[2] == 1:
        #         overlap_pct[i, j] =  (bb1_t1 - bb1_t0) / (bb0_t1 - bb0_t0)
        #
        #     else:
        #         overlap_pct[i, j] = np.diff(sorted(bb_times)[1:3])[0] / ((bb0_t1 - bb0_t0))
        # embed()
        # quit()

        handled_bbox_idxs.extend(bbox_idx_group)

    return bbox_groups

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract time, frequency and identity association of bboxes')
    parser.add_argument('annotations', nargs='?', type=str, help='path to annotations')
    parser.add_argument('-t', '--tracking_data_path', type=str, help='path to tracking dataa')
    args = parser.parse_args()
    main(args)