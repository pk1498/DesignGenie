import os
import torch
import torchvision
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torchvision.models import ResNet152_Weights, EfficientNet_B0_Weights, Inception_V3_Weights, ViT_B_16_Weights

random_seed = 123
torch.manual_seed(random_seed)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    acc = torch.sum(preds == labels).item() / len(preds)
    return acc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvClassifier(nn.Module):
    def __init__(self, model_name='resnet18', dataset=None):
        super().__init__()
        self.model_name = model_name
        num_classes = len(dataset.classes)

        if self.model_name == 'resnet18':
            self.network = torchvision.models.resnet18(pretrained=True)
            in_features = self.network.fc.in_features
            self.network.fc = nn.Identity()  # Remove the existing classification head

        elif self.model_name == 'resnet152':
            self.network = torchvision.models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
            in_features = self.network.fc.in_features
            self.network.fc = nn.Identity()

        elif self.model_name == 'efficientnet_b0':
            self.network = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = self.network.classifier[1].in_features
            self.network.classifier = nn.Identity()

        elif self.model_name == 'vit_b_16':
            self.network = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            in_features = self.network.heads.head.in_features
            self.network.heads = nn.Identity()

        elif self.model_name == 'inception_v3':
            self.network = torchvision.models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            in_features = self.network.fc.in_features
            self.network.fc = nn.Identity()

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features // 2, in_features // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features // 4, num_classes)
        )

    def forward(self, xb):
        features = self.network(xb)
        return self.classifier(features)

    def training_step(self, batch):
        images, labels = batch[0].to(device), batch[1].to(device)
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    @torch.no_grad()
    def valid_step(self, batch):
        images, labels = batch[0].to(device), batch[1].to(device)
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.item(), 'val_acc': acc}