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

from confic import (CLASSES, RESIZE_TO, TRAIN_DIR, BATCH_SIZE)
from custom_utils import collate_fn

from IPython import embed

class CustomDataset(Dataset):
    def __init__(self, dir_path, use_idxs = None):
        self.dir_path = dir_path
        self.image_paths = glob.glob(f'{self.dir_path}/*.png')
        self.all_images = [img_path.split(os.path.sep)[-1] for img_path in self.image_paths]
        self.all_images = np.array(sorted(self.all_images), dtype=str)
        if hasattr(use_idxs, '__len__'):
            self.all_images = self.all_images[use_idxs]
        self.bbox_df = pd.read_csv(os.path.join(dir_path, 'bbox_dataset.csv'), sep=',', index_col=0)

    def __getitem__(self, idx):
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)

        img = Image.open(image_path)
        img_tensor = F.to_tensor(img.convert('RGB'))

        Cbbox = self.bbox_df[self.bbox_df['image'] == image_name]

        labels = np.ones(len(Cbbox), dtype=int)
        boxes = torch.as_tensor(Cbbox.loc[:, ['x0', 'y0', 'x1', 'y1']].values, dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        return img_tensor, target

    def __len__(self):
        return len(self.all_images)

def create_train_test_dataset(path, test_size=0.2):
    files = glob.glob(os.path.join(path, '*.png'))
    train_test_idx = np.arange(len(files), dtype=int)
    np.random.shuffle(train_test_idx)

    train_idx = train_test_idx[int(test_size*len(train_test_idx)):]
    test_idx = train_test_idx[:int(test_size*len(train_test_idx))]

    train_data = CustomDataset(path, use_idxs=train_idx)
    test_data = CustomDataset(path, use_idxs=test_idx)

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

    train_data, test_data = create_train_test_dataset(TRAIN_DIR)

    train_loader = create_train_loader(train_data)
    test_loader = create_valid_loader(test_data)

    for samples, targets in train_loader:
        for s, t in zip(samples, targets):
            fig, ax = plt.subplots()
            ax.imshow(s.permute(1, 2, 0), aspect='auto')
            for (x0, x1, y0, y1), l in zip(t['boxes'], t['labels']):
                print(x0, x1, y0, y1, l)
                ax.add_patch(
                    Rectangle((x0, y0),
                              (x1 - x0),
                              (y1 - y0),
                              fill=False, color="white", linewidth=2, zorder=10)
                )
            plt.show()