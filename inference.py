import numpy as np
import torch
import torchvision.transforms.functional as F
import glob
import os
from PIL import Image
import argparse

from model import create_model
from confic import NUM_CLASSES, DEVICE, CLASSES, OUTDIR, DATA_DIR, INFERENCE_OUTDIR, IMG_DPI, IMG_SIZE
from datasets import InferenceDataset, create_inference_loader

from IPython import embed
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle


def plot_inference(img_tensor, img_name, output, detection_threshold):

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

    ax.set_axis_off()
    plt.savefig(Path(INFERENCE_OUTDIR)/(os.path.splitext(img_name)[0] +'_inferred.png'), dpi=IMG_DPI)
    plt.close()
    # plt.show()

def infere_model(inference_loader, model, detection_th=0.8):

    print('Inference')

    prog_bar = tqdm(inference_loader, total=len(inference_loader))
    for samples, targets in prog_bar:
        images = list(image.to(DEVICE) for image in samples)

        img_names = [t['image_name'] for t in targets]
        targets = [{k: v for k, v in t.items() if k != 'image_name'} for t in targets]

        with torch.inference_mode():
            outputs = model(images)

        for image, img_name, output, target in zip(images, img_names, outputs, targets):
            plot_inference(image, img_name, output, detection_th)


def main(args):
    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load(f'{OUTDIR}/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    inference_data = InferenceDataset(args.folder)
    inference_loader = create_inference_loader(inference_data)

    embed()
    quit()
    infere_model(inference_loader, model)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluated electrode array recordings with multiple fish.')
    parser.add_argument('folder', type=str, help='folder to infer picutes', default='')
    args = parser.parse_args()

    main(args)
