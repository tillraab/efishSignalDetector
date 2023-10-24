import torch.nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes: int) -> torch.nn.Module:
    """
    Create a pretrained Faster RCNN Model and replaces the final predictor in order to fit
    to a specific detection task.

    Parameters
    ----------
    num_classes: int
        Number of classes (+1) that shall be detected with the model.
        One more class is required because of background.

    Returns
    -------
    model: torch.nn.Module
        Adapted FasterRCNN Model
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model