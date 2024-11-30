import torch.nn as nn
from torchvision.models import resnet18


class SceneClassifier(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the SceneClassifier model using ResNet18.
        """
        super(SceneClassifier, self).__init__()
        self.model = resnet18(pretrained=True)  # Load pre-trained ResNet18
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Replace last layer

    def forward(self, x):
        return self.model(x)
