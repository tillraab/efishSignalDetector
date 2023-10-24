import numpy as np
import torch
import torchvision.transforms.functional as F
import glob
import os
from PIL import Image

from model import create_model
from confic import NUM_CLASSES, DEVICE, CLASSES, OUTDIR

from IPython import embed
from tqdm.auto import tqdm

if __name__ == '__main__':
    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load(f'{OUTDIR}/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    DIR_TEST = 'data/train'
    test_images = glob.glob(f"{DIR_TEST}/*.png")

    detection_threshold = 0.8

    frame_count = 0
    total_fps = 0

    for i in tqdm(np.arange(len(test_images))):
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]

        img = Image.open(test_images[i])
        img_tensor = F.to_tensor(img.convert('RGB')).unsqueeze(dim=0)

        with torch.inference_mode():
            outputs = model(img_tensor.to(DEVICE))

        print(len(outputs[0]['boxes']))