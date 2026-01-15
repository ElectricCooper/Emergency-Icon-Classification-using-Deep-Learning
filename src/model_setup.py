"""Module for setting up different models from torchvision"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, ResNet18_Weights

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512), 
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


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

    elif model_name == "simple_cnn":
        model = SimpleCNN(num_classes)

    else:
        print("Error. Wrong model name")
        return None

    return model.to(device)
