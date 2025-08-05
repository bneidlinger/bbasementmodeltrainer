import torch
import torch.nn as nn
from . import register_model

@register_model("Simple MLP")
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        # Flatten image tensors
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.layers(x)
    
    @staticmethod
    def get_config_options():
        """Return configurable options for the GUI."""
        return {
            'input_size': {'type': 'int', 'default': 784, 'min': 1, 'max': 10000},
            'hidden_size': {'type': 'int', 'default': 128, 'min': 16, 'max': 1024},
            'output_size': {'type': 'int', 'default': 10, 'min': 2, 'max': 1000}
        }