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

    scores = annotations[:, 5]

    return boxes, scores

def load_wavetracker_data(raw_path):
    fund_v = np.load(raw_path.parent / 'fund_v.npy')
    ident_v = np.load(raw_path.parent / 'ident_v.npy')
    idx_v = np.load(raw_path.parent / 'idx_v.npy')
    times = np.load(raw_path.parent / 'times.npy')

    return fund_v, ident_v, idx_v, times


def assign_rises_to_ids(raw_path, time_frequency_bboxes, bbox_groups):
    def identify_most_likely_rise_id(possible_ids, t0, t1, f0, f1, fund_v, ident_v, times, idx_v):

        mean_id_box_f_rel_to_bbox = []
        for id in possible_ids:
            id_box_f = fund_v[(ident_v == id) & (times[idx_v] >= t0) & (times[idx_v] <= t1)]
            id_box_f_rel_to_bbox = (id_box_f - f0) / (f1 - f0)
            mean_id_box_f_rel_to_bbox.append(np.mean(id_box_f_rel_to_bbox))
            # print(id, np.mean(id_box_f), f0, f1, np.mean(id_box_f_rel_to_bbox))

        most_likely_id = possible_ids[np.argsort(mean_id_box_f_rel_to_bbox)[0]]
        return most_likely_id

    fund_v, ident_v, idx_v, times = load_wavetracker_data(raw_path)

    # rise_id = np.full(len(time_frequency_bboxes), np.nan)
    # embed()
    # quit()

    fig, ax = plt.subplots()
    ax.plot(times[idx_v[~np.isnan(ident_v)]], fund_v[~np.isnan(ident_v)], '.')

    mask = time_frequency_bboxes['file_name'] == raw_path.parent.name
    for index, bbox in time_frequency_bboxes[mask].iterrows():
        name, t0, f0, t1, f1, score = (bbox[0], *bbox[1:-1].astype(float))
        if bbox_groups[index] == 0:
            color = 'tab:green'
        elif bbox_groups[index] > 0:
            color = 'tab:red'
        else:
            color = 'k'

        ax.add_patch(
            Rectangle((t0, f0),
                      (t1 - t0),
                      (f1 - f0),
                      fill=False, color=color, linestyle='--', linewidth=2, zorder=10)

        )
        ax.text(t1, f1, f'{score:.1%}', ha='right', va='bottom')

        possible_ids = np.unique(
            ident_v[~np.isnan(ident_v) &
                    (t0 <= times[idx_v]) &
                    (t1 >= times[idx_v]) &
                    (f0 <= fund_v) &
                    (f1 >= fund_v)]
        )
        if len(possible_ids) == 1:
            time_frequency_bboxes.at[index, 'id'] = possible_ids[0]
            # rise_id[index] = possible_ids[0]
        elif len(possible_ids) > 1:
            time_frequency_bboxes.at[index, 'id']= identify_most_likely_rise_id(possible_ids, t0, t1, f0, f1, fund_v, ident_v, times, idx_v)
            # rise_id[index] = identify_most_likely_rise_id(possible_ids, t0, t1, f0, f1, fund_v, ident_v, times, idx_v)

    # time_frequency_bboxes['id'] = rise_id
    # embed()
    plt.close()
    return time_frequency_bboxes


def find_overlapping_bboxes(df_collect):
    file_names = np.array(df_collect)[:, 0]
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

            overlap_bbox_idxs.append((ind0, ind1))

    return np.asarray(overlap_bbox_idxs)


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

    overlap_bbox_idxs = find_overlapping_bboxes(df_collect)

    bbox_groups = delete_double_boxes(overlap_bbox_idxs, df_collect)

    time_frequency_bboxes = pd.DataFrame(data= np.array(df_collect), columns=['file_name', 't0', 'f0', 't1', 'f1', 'score'])
    time_frequency_bboxes['id'] = np.full(len(time_frequency_bboxes), np.nan)

    ###########################################
    # for file_name in time_frequency_bboxes['file_name'].unique():
    #     fig, ax = plt.subplots()
    #
    #     mask = time_frequency_bboxes['file_name'] == file_name
    #     for index, bbox in time_frequency_bboxes[mask].iterrows():
    #         name, t0, f0, t1, f1 = (bbox[0], *bbox[1:-1].astype(float))
    #         if bbox_groups[index] == 0:
    #             color = 'tab:green'
    #         elif bbox_groups[index] > 0:
    #             color = 'tab:red'
    #         else:
    #             color = 'k'
    #
    #         ax.add_patch(
    #             Rectangle((t0, f0),
    #                       (t1 - t0),
    #                       (f1 - f0),
    #                       fill=False, color=color, linestyle='--', linewidth=2, zorder=10)
    #         )
    #     # ax.set_xlim(float(time_frequency_bboxes[mask]['t0'].min()), float(time_frequency_bboxes[mask]['t1'].max()))
    #     ax.set_xlim(0, float(time_frequency_bboxes[mask]['t1'].max()))
    #     # ax.set_ylim(float(time_frequency_bboxes[mask]['f0'].min()), float(time_frequency_bboxes[mask]['f1'].max()))
    #     ax.set_ylim(400, 1200)
    #     plt.show()
    # exit()
    ###########################################

    if args.tracking_data_path:
        file_paths = sorted(list(pathlib.Path(args.tracking_data_path).absolute().rglob('*.raw')))
        for raw_path in file_paths:
            if not raw_path.parent.name in time_frequency_bboxes['file_name'].to_list():
                continue
            time_frequency_bboxes = assign_rises_to_ids(raw_path, time_frequency_bboxes, bbox_groups)
        for raw_path in file_paths:
            # mask = (time_frequency_bboxes['file_name'] == raw_path.parent.name)
            mask = ((time_frequency_bboxes['file_name'] == raw_path.parent.name) & (~np.isnan(time_frequency_bboxes['id'])))
            save_df = pd.DataFrame(time_frequency_bboxes[mask][['t0', 't1', 'f0', 'f1', 'score', 'id']].values, columns=['t0', 't1', 'f0', 'f1', 'score', 'id'])
            save_df['label'] = np.ones(len(save_df), dtype=int)
            save_df.to_csv(raw_path.parent / 'risedetector_bboxes.csv', sep = ',', index = False)
    quit()

def delete_double_boxes(overlap_bbox_idxs, df_collect, overlap_th = 0.2):
    def get_connected(non_regarded_bbox_idx, overlap_bbox_idxs):
        mask = np.array((np.array(overlap_bbox_idxs) == non_regarded_bbox_idx).sum(1), dtype=bool)
        affected_bbox_idxs = np.unique(overlap_bbox_idxs[mask])
        return affected_bbox_idxs

    handled_bbox_idxs = []
    bbox_groups = np.zeros(len(df_collect))
    # detele_bbox_idxs = []

    for Coverlapping_bbox_idx in tqdm(np.unique(overlap_bbox_idxs)):
        if Coverlapping_bbox_idx in handled_bbox_idxs:
            continue

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

        bbox_idx_group = np.array(regarded_bbox_idxs)
        bbox_scores = df_collect[bbox_idx_group][:, -1].astype(float)

        bbox_groups[bbox_idx_group] = np.max(bbox_groups) + 1

        remove_idx_combinations = [()]
        remove_idx_combinations_scores = [0]
        for r in range(1, len(bbox_idx_group)):
            remove_idx_combinations.extend(list(itertools.combinations(bbox_idx_group, r=r)))
            remove_idx_combinations_scores.extend(list(itertools.combinations(bbox_scores, r=r)))
        for enu, combi_score in enumerate(remove_idx_combinations_scores):
            remove_idx_combinations_scores[enu] = np.sum(combi_score)
        if len(bbox_idx_group) > 1:
            remove_idx_combinations = [remove_idx_combinations[ind] for ind in np.argsort(remove_idx_combinations_scores)]
            remove_idx_combinations_scores = [remove_idx_combinations_scores[ind] for ind in np.argsort(remove_idx_combinations_scores)]


        for remove_idx in remove_idx_combinations:
            select_bbox_idx_group = list(set(bbox_idx_group) - set(remove_idx))
            time_overlap_pct, freq_overlap_pct = (
                compute_time_frequency_overlap_for_bbox_group(select_bbox_idx_group,df_collect))

            if np.all(np.min([time_overlap_pct, freq_overlap_pct], axis=0) < overlap_th):
                break

        if len(remove_idx) > 0:
            bbox_groups[np.array(remove_idx)] *= -1

        handled_bbox_idxs.extend(bbox_idx_group)

    return bbox_groups

def compute_time_frequency_overlap_for_bbox_group(bbox_idx_group, df_collect):
    time_overlap_pct = np.zeros((len(bbox_idx_group), len(bbox_idx_group)))
    freq_overlap_pct = np.zeros((len(bbox_idx_group), len(bbox_idx_group)))


    for i, j in itertools.product(range(len(bbox_idx_group)), repeat=2):
        if i == j:
            continue
        bb0_idx = bbox_idx_group[i]
        bb1_idx = bbox_idx_group[j]

        bb0_t0, bb0_t1 = df_collect[bb0_idx][1].astype(float), df_collect[bb0_idx][3].astype(float)
        bb1_t0, bb1_t1 = df_collect[bb1_idx][1].astype(float), df_collect[bb1_idx][3].astype(float)

        bb0_f0, bb0_f1 = df_collect[bb0_idx][2].astype(float), df_collect[bb0_idx][4].astype(float)
        bb1_f0, bb1_f1 = df_collect[bb1_idx][2].astype(float), df_collect[bb1_idx][4].astype(float)

        bb_times_idx = np.array([0, 0, 1, 1])
        bb_times = np.array([bb0_t0, bb0_t1, bb1_t0, bb1_t1])
        sorted_bb_times_idx = bb_times_idx[bb_times.argsort()]

        if sorted_bb_times_idx[0] == sorted_bb_times_idx[1]:
            time_overlap_pct[i, j] = 0
        elif sorted_bb_times_idx[1] == sorted_bb_times_idx[2] == 0:
            time_overlap_pct[i, j] = 1
        elif sorted_bb_times_idx[1] == sorted_bb_times_idx[2] == 1:
            time_overlap_pct[i, j] =  (bb1_t1 - bb1_t0) / (bb0_t1 - bb0_t0)
        else:
            time_overlap_pct[i, j] = np.diff(sorted(bb_times)[1:3])[0] / ((bb0_t1 - bb0_t0))

        bb_freqs_idx = np.array([0, 0, 1, 1])
        bb_freqs = np.array([bb0_f0, bb0_f1, bb1_f0, bb1_f1])
        sorted_bb_freqs_idx = bb_freqs_idx[bb_freqs.argsort()]

        if sorted_bb_freqs_idx[0] == sorted_bb_freqs_idx[1]:
            freq_overlap_pct[i, j] = 0
        elif sorted_bb_freqs_idx[1] == sorted_bb_freqs_idx[2] == 0:
            freq_overlap_pct[i, j] = 1
        elif sorted_bb_freqs_idx[1] == sorted_bb_freqs_idx[2] == 1:
            freq_overlap_pct[i, j] =  (bb1_f1 - bb1_f0) / (bb0_f1 - bb0_f0)
        else:
            freq_overlap_pct[i, j] = np.diff(sorted(bb_freqs)[1:3])[0] / ((bb0_f1 - bb0_f0))

    return time_overlap_pct, freq_overlap_pct


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract time, frequency and identity association of bboxes')
    parser.add_argument('annotations', nargs='?', type=str, help='path to annotations')
    parser.add_argument('-t', '--tracking_data_path', type=str, help='path to tracking dataa')
    args = parser.parse_args()
    main(args)