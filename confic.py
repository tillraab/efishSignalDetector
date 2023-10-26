import torch
import pathlib

BATCH_SIZE = 8
RESIZE_TO = 416
NUM_EPOCHS = 10
NUM_WORKERS = 4

IMG_SIZE = (7, 7) # inches
IMG_DPI = 256

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CLASSES = ['__backgroud__', '1']

NUM_CLASSES = len(CLASSES)

TRAIN_DIR = 'data/train'
OUTDIR = 'model_outputs'
INFERENCE_OUTDIR = 'inference_outputs'

for required_folders in [TRAIN_DIR, OUTDIR, INFERENCE_OUTDIR]:
    if not pathlib.Path(required_folders).exists():
        pathlib.Path(required_folders).mkdir(parents=True, exist_ok=True)
