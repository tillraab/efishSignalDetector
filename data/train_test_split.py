import pandas as pd
from pathlib import Path
import numpy as np
import os

from IPython import embed

def define_train_test_img_names(bbox, test_size = 0.2):
    np.random.seed(42)
    unique_imgs = np.asarray(pd.unique(bbox['image']))
    np.random.shuffle(unique_imgs)

    test_img = sorted(unique_imgs[:int(len(unique_imgs) * test_size)])
    train_img = sorted(unique_imgs[int(len(unique_imgs) * test_size):])

    return test_img, train_img

def split_data_df_in_test_train_df(bbox, test_img, train_img):
    cols = list(bbox.keys())

    test_bbox = pd.DataFrame(columns=cols)
    train_bbox = pd.DataFrame(columns=cols)

    for img_name in test_img:
        tmp_df = bbox[bbox['image'] == img_name]
        test_bbox = pd.concat([test_bbox, tmp_df], ignore_index=True)

    for img_name in train_img:
        tmp_df = bbox[bbox['image'] == img_name]
        train_bbox = pd.concat([train_bbox, tmp_df], ignore_index=True)

    return train_bbox, test_bbox, cols
def main(path):
    bbox = pd.read_csv(path/'bbox_dataset.csv', sep=',', index_col=0)

    test_img, train_img = define_train_test_img_names(bbox)

    train_bbox, test_bbox, cols = split_data_df_in_test_train_df(bbox, test_img, train_img)

    train_bbox.to_csv(path/'bbox_train.csv', columns=cols, sep=',')
    test_bbox.to_csv(path/'bbox_test.csv', columns=cols, sep=',')

if __name__ == '__main__':
    main(Path('./dataset'))