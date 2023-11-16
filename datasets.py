import os
import glob

import torch
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image

from confic import (CLASSES, RESIZE_TO, DATA_DIR, LABEL_DIR, BATCH_SIZE, IMG_SIZE, IMG_DPI)
from custom_utils import collate_fn

from IPython import embed

from sklearn.model_selection import train_test_split


class InferenceDataset(Dataset):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.all_images = sorted(list(Path(self.dir_path).rglob(f'*.png')))
        self.file_names = np.array([Path(x).with_suffix('') for x in self.all_images])
        # self.images = np.array(sorted(os.listdir(DATA_DIR)))
    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        image_path = self.all_images[idx]
        image_name = image_path.name
        img = Image.open(image_path)
        img_tensor = F.to_tensor(img.convert('RGB'))

        return img_tensor, image_name


class CustomDataset(Dataset):
    def __init__(self, limited_idxs=None):
        self.images = np.array(sorted(os.listdir(DATA_DIR)))
        self.labels = np.array(sorted(os.listdir(LABEL_DIR)))

        if hasattr(limited_idxs, '__len__'):
            self.images = np.array(sorted(os.listdir(DATA_DIR)))[limited_idxs]
            self.labels = np.array(sorted(os.listdir(LABEL_DIR)))[limited_idxs]

        self.file_names = np.array([Path(x).with_suffix('') for x in self.images])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(Path(DATA_DIR) / self.images[idx])
        img_tensor = F.to_tensor(img.convert('RGB'))

        annotations = np.loadtxt(Path(LABEL_DIR) / Path(self.images[idx]).with_suffix('.txt'), delimiter=' ')

        boxes, labels, area, iscrowd = self.extract_bboxes(annotations)

        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        target["image_name"] = self.images[idx]

        return img_tensor, target

    def extract_bboxes(self, annotations):
        if len(annotations.shape) == 1:
            annotations = np.array([annotations])

        if annotations.shape[1] == 0:
            boxes = area = torch.tensor([], dtype=torch.float32)
            labels = iscrowd = torch.tensor([], dtype=torch.int64)
            return boxes, labels, area, iscrowd

        boxes = np.array([[x[1] - x[3] / 2, x[2] - x[4] / 2, x[1] + x[3] / 2, x[2] + x[4] / 2] for x in annotations])
        boxes[:, 0] *= IMG_SIZE[0] * IMG_DPI
        boxes[:, 2] *= IMG_SIZE[0] * IMG_DPI
        boxes[:, 1] *= IMG_SIZE[1] * IMG_DPI
        boxes[:, 3] *= IMG_SIZE[1] * IMG_DPI
        boxes = torch.from_numpy(boxes).type(torch.float32)

        labels = torch.from_numpy(annotations[:, 0]).type(torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        return boxes, labels, area, iscrowd

def custom_train_test_split():
    file_list = sorted(list(Path(LABEL_DIR).rglob('*.txt')))
    data_idxs = np.arange(len(file_list))
    empty_mask = np.array([os.stat(x).st_size == 0 for x in file_list], dtype=bool)
    data_idxs = data_idxs[~empty_mask]

    # ToDo: do this witch labels and remove empty shit !!!
    np.random.shuffle(data_idxs)

    train_idxs = np.sort(data_idxs[int(0.2 * len(data_idxs)):])
    test_idxs = np.sort(data_idxs[:int(0.2 * len(data_idxs))])

    train_data = CustomDataset(limited_idxs=train_idxs)
    test_data = CustomDataset(limited_idxs=test_idxs)

    return train_data, test_data


def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader


# ToDo the next two functions are redundant!
def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader

if __name__ == '__main__':
    train_data, test_data = custom_train_test_split()

    train_loader = create_train_loader(train_data)
    test_loader = create_valid_loader(test_data)

    for samples, targets in train_loader:
        for s, t in zip(samples, targets):

            fig, ax = plt.subplots()
            ax.imshow(s.permute(1, 2, 0), aspect='auto')
            for (x0, y0, x1, y1), l in zip(t['boxes'], t['labels']):
                print(x0, y0, x1, y1, l)
                ax.add_patch(
                    Rectangle((x0, y0),
                              (x1 - x0),
                              (y1 - y0),
                              fill=False, color="white", linewidth=2, zorder=10)
                )
            ax.set_title(t['image_name'])
            plt.show()