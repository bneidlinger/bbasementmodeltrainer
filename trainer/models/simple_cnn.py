import torch
import torch.nn as nn
from . import register_model

def conv_block(in_channels, out_channels, kernel_size=3):
    """Returns a block of Conv -> BatchNorm -> ReLU -> MaxPool."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )

@register_model("Basic CNN")
class BasicCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        # Convolutional layers
        self.features = nn.Sequential(
            conv_block(in_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128)
        )
        
        # Calculate the size after 3 pooling layers
        # For 32x32 input: 32 -> 16 -> 8 -> 4
        # For 28x28 input: 28 -> 14 -> 7 -> 3
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
    
    @staticmethod
    def get_config_options():
        """Return configurable options for the GUI."""
        return {
            'in_channels': {'type': 'int', 'default': 3, 'min': 1, 'max': 10},
            'num_classes': {'type': 'int', 'default': 10, 'min': 2, 'max': 1000}
        }

@register_model("LeNet-5")
class LeNet5(nn.Module):
    """Classic LeNet-5 architecture for MNIST."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            nn.Conv2d(6, 16, kernel_size=5),  # 14x14 -> 10x10
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 10x10 -> 5x5
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    @staticmethod
    def get_config_options():
        """Return configurable options for the GUI."""
        return {
            'num_classes': {'type': 'int', 'default': 10, 'min': 2, 'max': 1000}
        }