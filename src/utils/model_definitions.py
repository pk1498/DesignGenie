import os
import torch
import torchvision
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F


random_seed = 213
torch.manual_seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    acc = torch.sum(preds == labels).item() / len(preds)
    return acc

class ConvClassifier(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.network = torchvision.models.resnet18(pretrained=True)
        self.network.fc = nn.Linear(self.network.fc.in_features, len(dataset.classes))
    
    def forward(self, xb):
        return self.network(xb)
    
    def training_step(self, batch):
        images, labels = batch[0].to(device), batch[1].to(device)
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # cross entropy 内部已包含 softmax， 所以外部就不激活了
        return loss
    
    @torch.no_grad()
    def valid_step(self, batch):
        images, labels = batch[0].to(device), batch[1].to(device) 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.item(), 'val_acc': acc}
