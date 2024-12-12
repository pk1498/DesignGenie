import torch.nn as nn
import torchvision
from torchvision.models import ViT_B_16_Weights


class ConvClassifier(nn.Module):
    def __init__(self):
        super(ConvClassifier, self).__init__()
        self.network = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.network.heads = nn.Linear(self.network.heads.head.in_features, 12)

    def forward(self, xb):
        return self.network(xb)
