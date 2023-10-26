from confic import (DEVICE, NUM_CLASSES, NUM_EPOCHS, OUTDIR, NUM_WORKERS, DATA_DIR)
from model import create_model

from tqdm.auto import tqdm

from datasets import create_train_loader, create_valid_loader, create_train_or_test_dataset
from custom_utils import Averager, SaveBestModel, save_model, save_loss_plot

import torch
import matplotlib.pyplot as plt
import time

from IPython import embed

def train(train_loader, model, optimizer, train_loss):
    print('Training')

    prog_bar = tqdm(train_loader, total=len(train_loader))
    for samples, targets in prog_bar:
        images = list(image.to(DEVICE) for image in samples)

        # targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        targets = [{k: v.to(DEVICE) for k, v in t.items() if k != 'image_name'} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_hist.send(loss_value) # this is a global instance !!!
        train_loss.append(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return train_loss

def validate(test_loader, model, val_loss):
    print('Validation')

    prog_bar = tqdm(test_loader, total=len(test_loader))
    for samples, targets in prog_bar:
        images = list(image.to(DEVICE) for image in samples)

        # targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        targets = [{k: v.to(DEVICE) for k, v in t.items() if k != 'image_name'} for t in targets]


        with torch.inference_mode():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_hist.send(loss_value) # this is a global instance !!!
        val_loss.append(loss_value)

        # optimizer.zero_grad()
        # losses.backward()
        # optimizer.step()

        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return val_loss

if __name__ == '__main__':
    train_data = create_train_or_test_dataset(DATA_DIR)
    test_data = create_train_or_test_dataset(DATA_DIR, train=False)

    train_loader = create_train_loader(train_data)
    test_loader = create_valid_loader(test_data)

    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_loss = []
    val_loss = []

    save_best_model = SaveBestModel()

    for epoch in range(NUM_EPOCHS):
        print(f'\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---')

        train_loss_hist.reset()
        val_loss_hist.reset()

        train_loss = train(train_loader, model, optimizer, train_loss)
        val_loss = validate(test_loader, model, val_loss)


        save_best_model(
            val_loss_hist.value, epoch, model, optimizer
        )

        save_model(epoch, model, optimizer)

        save_loss_plot(OUTDIR, train_loss, val_loss)

