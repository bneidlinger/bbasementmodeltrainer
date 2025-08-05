import torch
import torch.nn as nn
import torchvision.models as models
from . import register_model

@register_model("ResNet18")
class ResNet18(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()
        # Load pretrained ResNet18 or create from scratch
        self.model = models.resnet18(pretrained=pretrained)
        
        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    @staticmethod
    def get_config_options():
        """Return configurable options for the GUI."""
        return {
            'num_classes': {'type': 'int', 'default': 10, 'min': 2, 'max': 1000},
            'pretrained': {'type': 'bool', 'default': False}
        }

@register_model("ResNet34")
class ResNet34(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()
        self.model = models.resnet34(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    @staticmethod
    def get_config_options():
        """Return configurable options for the GUI."""
        return {
            'num_classes': {'type': 'int', 'default': 10, 'min': 2, 'max': 1000},
            'pretrained': {'type': 'bool', 'default': False}
        }