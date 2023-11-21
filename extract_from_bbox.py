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
        return []


    boxes = np.array([[x[1] - x[3] / 2, 1 - (x[2] + x[4] / 2), x[1] + x[3] / 2, 1 - (x[2] - x[4] / 2)] for x in annotations]) # x0, y0, x1, y1
    boxes[:, 0] = boxes[:, 0] * (t_max - t_min) + t_min
    boxes[:, 2] = boxes[:, 2] * (t_max - t_min) + t_min

    boxes[:, 1] = boxes[:, 1] * (f_max - f_min) + f_min
    boxes[:, 3] = boxes[:, 3] * (f_max - f_min) + f_min

    return boxes

def load_wavetracker_data(raw_path):
    fund_v = np.load(raw_path.parent / 'fund_v.npy')
    ident_v = np.load(raw_path.parent / 'ident_v.npy')
    idx_v = np.load(raw_path.parent / 'idx_v.npy')
    times = np.load(raw_path.parent / 'times.npy')

    return fund_v, ident_v, idx_v, times


def assign_rises_to_ids(raw_path, time_frequency_bboxes):
    fund_v, ident_v, idx_v, times = load_wavetracker_data(raw_path)

    fig, ax = plt.subplots()
    ax.plot(times[idx_v[~np.isnan(ident_v)]], fund_v[~np.isnan(ident_v)], '.')

    mask = time_frequency_bboxes['file_name'] == raw_path.parent.name
    for index, bbox in time_frequency_bboxes[mask].iterrows():
        name, t0, f0, t1, f1 = (bbox[0], *bbox[1:].astype(float))
        ax.add_patch(
            Rectangle((t0, f0),
                      (t1 - t0),
                      (f1 - f0),
                      fill=False, color="tab:green", linestyle='--', linewidth=2, zorder=10)
        )
    plt.show()

    # ToDo: eliminate double rises -- overlap
    # ToDo: double detections -- non overlap --> the one with higher probability ?!
    # ToDo: assign rises to traces --> who is at lower right corner

def main(args):
    img_paths = sorted(list(pathlib.Path(args.annotations).absolute().rglob('*.png')))
    df_collect = []

    for img_path in img_paths:
        # convert to time_frequency
        file_name_str, t_min, t_max, f_min, f_max = extract_time_freq_range_from_filename(img_path)

        boxes = bbox_to_data(img_path, t_min, t_max, f_min, f_max ) # t0, t1, f0, f1

        # store values in df
        if not len(boxes) == 0:
            for x0, y0, x1, y1 in boxes:
                df_collect.append([file_name_str, x0, y0, x1, y1])

    time_frequency_bboxes = pd.DataFrame(data= np.array(df_collect), columns=['file_name', 't0', 'f0', 't1', 'f1'])

    if args.tracking_data_path:
        file_paths = sorted(list(pathlib.Path(args.tracking_data_path).absolute().rglob('*.raw')))
        for raw_path in file_paths:
            if not raw_path.parent.name in time_frequency_bboxes['file_name'].to_list():
                continue

            assign_rises_to_ids(raw_path, time_frequency_bboxes)

    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract time, frequency and identity association of bboxes')
    parser.add_argument('annotations', nargs='?', type=str, help='path to annotations')
    parser.add_argument('-t', '--tracking_data_path', type=str, help='path to tracking dataa')
    args = parser.parse_args()
    main(args)