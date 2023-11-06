import torch
import pathlib

# training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 10
NUM_WORKERS = 4
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# input parameters
IMG_SIZE = (7, 7) # inches
IMG_DPI = 256
RESIZE_TO = 416 # ToDo: alter this parameter to (7, 7) * 256 [IMG_SIZE * IMG_DPI]

# dataset parameters
CLASSES = ['__backgroud__', '1']
NUM_CLASSES = len(CLASSES)

# data snippet paramters
MIN_FREQ = 200
MAX_FREQ = 1500
DELTA_FREQ = 200
FREQ_OVERLAP = 25

DELTA_TIME = 60*10
TIME_OVERLAP = 60*1

# output parameters
DATA_DIR = 'data/dataset'
OUTDIR = 'model_outputs'
INFERENCE_OUTDIR = 'inference_outputs'
for required_folders in [DATA_DIR, OUTDIR, INFERENCE_OUTDIR]:
    if not pathlib.Path(required_folders).exists():
        pathlib.Path(required_folders).mkdir(parents=True, exist_ok=True)
