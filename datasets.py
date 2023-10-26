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

from confic import (CLASSES, RESIZE_TO, DATA_DIR, BATCH_SIZE)
from custom_utils import collate_fn

from IPython import embed

class CustomDataset(Dataset):
    def __init__(self, dir_path, bbox_df):
        self.dir_path = dir_path
        self.bbox_df = bbox_df

        self.all_images = np.array(sorted(pd.unique(self.bbox_df['image'])), dtype=str)
        self.image_paths = list(map(lambda x: Path(self.dir_path)/x, self.all_images))

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
        target["image_name"] = image_name #ToDo: implement this as 3rd return...

        return img_tensor, target

    def __len__(self):
        return len(self.all_images)

def create_train_or_test_dataset(path, train=True):
    if train == True:
        pfx='train'
        print('Generate train dataset !')
    else:
        print('Generate test dataset !')
        pfx='test'

    csv_candidates = list(Path(path).rglob(f'*{pfx}*.csv'))
    if len(csv_candidates) == 0:
        print(f'no .csv files for *{pfx}* found in {Path(path)}')
        quit()
    else:
        bboxes = pd.read_csv(csv_candidates[0], sep=',', index_col=0)
    return CustomDataset(path, bboxes)

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

    # train_data, test_data = create_train_test_dataset(TRAIN_DIR)
    train_data = create_train_or_test_dataset(DATA_DIR)
    test_data = create_train_or_test_dataset(DATA_DIR, train=False)

    train_loader = create_train_loader(train_data)
    test_loader = create_valid_loader(test_data)

    for samples, targets in test_loader:
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
            plt.show()