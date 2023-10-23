from confic import (DEVICE, NUM_CLASSES, NUM_EPOCHS, OUTDIR, NUM_WORKERS, TRAIN_DIR)
from model import create_model

from tqdm.auto import tqdm
from datasets import create_train_test_dataset, create_train_loader, create_valid_loader

import torch
import matplotlib.pyplot as plt
import time

from IPython import embed

if __name__ == '__main__':
    train_data, test_data = create_train_test_dataset(TRAIN_DIR)
    train_loader = create_train_loader(train_data)
    test_loader = create_train_loader(test_data)

    model = create_model(num_classes=1)
    model = model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    for epoch in range(NUM_EPOCHS):
        prog_bar = tqdm(train_loader, total=len(train_loader))
        for samples, targets in prog_bar:
            images = list(image.to(DEVICE) for image in samples)

            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            try:
                loss_dict = model(images, targets)
            except:
                embed()
                quit()

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")