import torch
import pathlib

BATCH_SIZE = 8
RESIZE_TO = 416
NUM_EPOCHS = 10
NUM_WORKERS = 4

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_DIR = 'data/train'

CLASSES = ['__backgroud__', '1']

NUM_CLASSES = len(CLASSES)

OUTDIR = 'model_outputs'


if not pathlib.Path(OUTDIR).exists():
    pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)