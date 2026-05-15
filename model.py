import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES, USE_PRETRAINED


def build_model(num_classes=NUM_CLASSES):
    weights = models.ResNet18_Weights.DEFAULT if USE_PRETRAINED else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
