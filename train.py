from confic import (DEVICE, NUM_CLASSES, NUM_EPOCHS, OUTDIR, NUM_WORKERS, DATA_DIR, IMG_SIZE, IMG_DPI, INFERENCE_OUTDIR)
from model import create_model
from datasets import create_train_loader, create_valid_loader, custom_train_test_split
from custom_utils import Averager, SaveBestModel, save_model, save_loss_plot

from tqdm.auto import tqdm

import os
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import time

import pathlib
from pathlib import Path

from IPython import embed

def train(train_loader, model, optimizer, train_loss):
    print('Training')

    prog_bar = tqdm(train_loader, total=len(train_loader))
    for samples, targets in prog_bar:
        images = list(image.to(DEVICE) for image in samples)
        # img_names = [t['image_name'] for t in targets]
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
        targets = [{k: v.to(DEVICE) for k, v in t.items() if k != 'image_name'} for t in targets]

        with torch.inference_mode():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_hist.send(loss_value) # this is a global instance !!!
        val_loss.append(loss_value)
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return val_loss

def best_model_validation_with_plots(test_loader):
    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load(f'{OUTDIR}/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    if not pathlib.Path(Path(INFERENCE_OUTDIR)/Path(DATA_DIR).name).exists():
        pathlib.Path(Path(INFERENCE_OUTDIR)/Path(DATA_DIR).name).mkdir(parents=True, exist_ok=True)
    validate_with_plots(test_loader, model)


def validate_with_plots(test_loader, model, detection_th=0.8):
    print('Final validation with image putput')

    prog_bar = tqdm(test_loader, total=len(test_loader))
    for samples, targets in prog_bar:
        images = list(image.to(DEVICE) for image in samples)

        img_names = [t['image_name'] for t in targets]
        targets = [{k: v for k, v in t.items() if k != 'image_name'} for t in targets]

        with torch.inference_mode():
            outputs = model(images)

        for image, img_name, output, target in zip(images, img_names, outputs, targets):
            plot_validation(image, img_name, output, target, detection_th)


def plot_validation(img_tensor, img_name, output, target, detection_threshold):

    fig = plt.figure(figsize=IMG_SIZE, num=img_name)
    gs = gridspec.GridSpec(1, 1, bottom=0, left=0, right=1, top=1)  #
    ax = fig.add_subplot(gs[0, 0])

    ax.imshow(img_tensor.cpu().squeeze().permute(1, 2, 0),  aspect='auto')
    for (x0, y0, x1, y1), l, score in zip(output['boxes'].cpu(), output['labels'].cpu(), output['scores'].cpu()):
        if score < detection_threshold:
            continue
    #     print(x0, y0, x1, y1, l)
        ax.text(x0, y0, f'{score:.2f}', ha='left', va='bottom', fontsize=12, color='white')
        ax.add_patch(
            Rectangle((x0, y0),
                      (x1 - x0),
                      (y1 - y0),
                      fill=False, color="tab:green", linestyle='--', linewidth=2, zorder=10)
        )
    for (x0, y0, x1, y1), l in zip(target['boxes'], target['labels']):
        ax.add_patch(
            Rectangle((x0, y0),
                      (x1 - x0),
                      (y1 - y0),
                      fill=False, color="white", linewidth=2, zorder=9)
        )

    ax.set_axis_off()

    plt.savefig(Path(INFERENCE_OUTDIR)/Path(DATA_DIR).name/(os.path.splitext(img_name)[0] +'_predicted.png'), dpi=IMG_DPI)
    plt.close()
    # plt.show()

if __name__ == '__main__':

    train_data, test_data = custom_train_test_split()

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

    # load best model and perform inference with plot output
    best_model_validation_with_plots(test_loader)