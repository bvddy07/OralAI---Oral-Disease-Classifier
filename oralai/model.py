import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    resnet50 = models.resnet50(pretrained=True)
    resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)
    return resnet50
