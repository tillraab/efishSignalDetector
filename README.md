# efishSignalDetector

Welcome to the efishSignalDetector, a **neural network** framework adapted 
for detecting electrocommunication signals in wavetype electric fish based on 
spectrogram images. The model itself is a pretrained **FasterRCNN** Model with a 
**ResNet50** Backbone. Only the final predictor is replaced to not predict the 91 classes 
present in the coco-dataset the model is trained to but the (currently) 1 category it should detect.

## Long-Term and major ToDos:
* implement gui to correct bounding boxes
* implement reinforced learning

## Data preparation
### Data structure
The algorithm learns patterns based on **.png-images** and corresponding bounding boxes 
which are stored in **.csv-files**.
* image name includes the file it is derived form as well as time and frequency bounds
* image size defined in *config.py* with size (IMG_SIZE=(7, 7)) and dpi (IMG_DPI=256).
* .cvs file where each row represents one assigned signal:
  * image: image name
  * x0, x1, y0, y1: image coordinates of bounding box
  * t0, t1, f0, f1: time and frequency boinding box
* .png images and .csv file is stored in ./data/dataset

### Test-train-split

Use the script **./data/train_test_split.py** to split the original .csv file into one for
training and one for testing (both also stored in ./data/dataset).

### ToDos:
* on a long scale: only save raw file bounding boxes in frequency and time (t0, t1, f0, f1) and the hyperparameters of the corresponding spectrogram. USE THESE PARAMETERS IN DATASET_FN.
* rescale image input to (7, 7) * 256 --> width/height in inch * dpi
* when dataset input it spectrogram use resize transform.
* replace datastructure with yolo structure ... per pic 1 .csv saved as .txt 

## model.py

Contains the detection neural network model with its adjustments. The model itself is a pretrained **FasterRCNN** Model with a 
**ResNet50** Backbone. Only the final predictor is replaced to not predict the 91 classes 
present in the coco-dataset the model is trained to but the (currently) 1 category it should detect.

### ToDos:
* replace backbone entry to not take RGB input, but grayscale images or even spectrograms.
~~~ py
# Hint:
model.backbone.body.conv1 = torch.nn.Conv2d(1, 64,
                            kernel_size=(7, 7), stride=(2, 2),
                            padding=(3, 3), bias=False).requires_grad_(True)
~~~
~~~ py
# Hint:
from PIL import Image, ImageOps   

im1 = Image.open(img_path) 
im2 = ImageOps.grayscale(im1) 
~~~

* check other pretrained models from torchvision.models.detection, e.g. fasterrcnn_resnet50_fpn_v2

## datasets.py
Contains custom datasets and dataloader functions and classes.  

### ToDos:
* load/compute spectrogram directly and perform signal detection. E.g. spectrogram calculation as part of __getitem__


## config.py
Containes Hyperparameters used by the scripts.

### ToDos:
* Do we need the RESIZE_TO parameter ?

## custom_utils.py
Classes and functions to save models and store loss values for later illustration.
Also includes helper functions...

### ToDos:

## train.py
Code training the model using the stored images in ./data/dataset and the .csv files
containing the bounding boxes meant for training. For each epoch test-loss (without 
gradient tracking) is computed and used to infer whether the model is better than the one
of the previous epochs. If the new model is the best model, the model.state_dict is saved in 
./model_outputs as best_model.pth.

After training a final validation is performed where images showing both, the predicted and the
true bounding boxes in an image. Images are stored in ./inference_output/dataset.

### ToDos:

## inference.py
Script is used to infere unknown .png images. Results are stored in ./inference_output/<Dataset-Name>

### ToDo:


