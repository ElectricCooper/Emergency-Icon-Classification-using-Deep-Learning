"""Module for setting up different models from torchvision"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, ResNet18_Weights


def get_model(model_name, num_classes, device):
    """
    Gets the wanted model from torch and returns it
    """

    if model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_name == "vit_b_16":
        weights = ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)

        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, num_classes)

    else:
        print("Error. Wrong model name")
        return None

    return model.to(device)
