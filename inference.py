import numpy as np
import torch
import torchvision.transforms.functional as F
import glob
import os
from PIL import Image

from model import create_model
from confic import NUM_CLASSES, DEVICE, CLASSES, OUTDIR, TRAIN_DIR, INFERENCE_OUTDIR, IMG_DPI, IMG_SIZE
from datasets import create_train_or_test_dataset, create_valid_loader

from IPython import embed
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle


def plot_inference(img_tensor, img_name, output, target, detection_threshold):

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
    plt.savefig(Path(INFERENCE_OUTDIR)/(os.path.splitext(img_name)[0] +'_inferred.png'), IMG_DPI)
    plt.close()
    # plt.show()

def infere_model(test_loader, model, detection_th=0.8):

    print('Validation')

    prog_bar = tqdm(test_loader, total=len(test_loader))
    for samples, targets in prog_bar:
        images = list(image.to(DEVICE) for image in samples)

        img_names = [t['image_name'] for t in targets]
        targets = [{k: v for k, v in t.items() if k != 'image_name'} for t in targets]

        with torch.inference_mode():
            outputs = model(images)

        for image, img_name, output, target in zip(images, img_names, outputs, targets):
            plot_inference(image, img_name, output, target, detection_th)


if __name__ == '__main__':
    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load(f'{OUTDIR}/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    test_data = create_train_or_test_dataset(TRAIN_DIR, train=False)
    test_loader = create_valid_loader(test_data)

    infere_model(test_loader, model)

    # detection_threshold = 0.8
    # frame_count = 0
    # total_fps = 0
    # test_images = glob.glob(f"{TRAIN_DIR}/*.png")

    # for i in tqdm(np.arange(len(test_images))):
    #     image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
    #
    #     img = Image.open(test_images[i])
    #     img_tensor = F.to_tensor(img.convert('RGB')).unsqueeze(dim=0)
    #
    #     with torch.inference_mode():
    #         outputs = model(img_tensor.to(DEVICE))
    #
    #     print(len(outputs[0]['boxes']))

        # show_sample(img_tensor, outputs, detection_threshold)