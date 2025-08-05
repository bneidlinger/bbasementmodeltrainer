import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional, Dict, Any
import os

from .core import BaseDataset, register_dataset, TensorDataset

@register_dataset("CIFAR-10")
class CIFAR10Dataset(BaseDataset):
    """CIFAR-10 image classification dataset."""
    
    def load(self) -> Tuple[Dataset, Optional[Dataset]]:
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load training dataset
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=True,
            download=self.download,
            transform=transform
        )
        
        # Load test dataset as validation
        val_dataset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=False,
            download=self.download,
            transform=transform
        )
        
        return train_dataset, val_dataset
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'name': 'CIFAR-10',
            'source': 'torchvision',
            'input_shape': (3, 32, 32),
            'num_classes': 10,
            'classes': ['plane', 'car', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck'],
            'train_samples': 50000,
            'test_samples': 10000,
            'license': 'MIT'
        }

@register_dataset("MNIST")
class MNISTDataset(BaseDataset):
    """MNIST handwritten digits dataset."""
    
    def load(self) -> Tuple[Dataset, Optional[Dataset]]:
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load training dataset
        train_dataset = torchvision.datasets.MNIST(
            root=self.root,
            train=True,
            download=self.download,
            transform=transform
        )
        
        # Load test dataset as validation
        val_dataset = torchvision.datasets.MNIST(
            root=self.root,
            train=False,
            download=self.download,
            transform=transform
        )
        
        return train_dataset, val_dataset
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'name': 'MNIST',
            'source': 'torchvision',
            'input_shape': (1, 28, 28),
            'num_classes': 10,
            'classes': [str(i) for i in range(10)],
            'train_samples': 60000,
            'test_samples': 10000,
            'license': 'CC BY-SA 3.0'
        }

@register_dataset("FashionMNIST")
class FashionMNISTDataset(BaseDataset):
    """Fashion-MNIST clothing classification dataset."""
    
    def load(self) -> Tuple[Dataset, Optional[Dataset]]:
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Load training dataset
        train_dataset = torchvision.datasets.FashionMNIST(
            root=self.root,
            train=True,
            download=self.download,
            transform=transform
        )
        
        # Load test dataset as validation
        val_dataset = torchvision.datasets.FashionMNIST(
            root=self.root,
            train=False,
            download=self.download,
            transform=transform
        )
        
        return train_dataset, val_dataset
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'name': 'Fashion-MNIST',
            'source': 'torchvision',
            'input_shape': (1, 28, 28),
            'num_classes': 10,
            'classes': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
            'train_samples': 60000,
            'test_samples': 10000,
            'license': 'MIT'
        }

@register_dataset("Random Dummy")
class RandomDummyDataset(BaseDataset):
    """Random dummy dataset for testing."""
    
    def __init__(self, root: str = "./data/cache", download: bool = True,
                 num_samples: int = 1000, input_size: int = 784, 
                 num_classes: int = 10):
        super().__init__(root, download)
        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes = num_classes
    
    def load(self) -> Tuple[Dataset, Optional[Dataset]]:
        # Generate random data
        np.random.seed(42)
        
        # Training data
        train_data = torch.randn(self.num_samples, self.input_size)
        train_targets = torch.randint(0, self.num_classes, (self.num_samples,))
        train_dataset = TensorDataset(train_data, train_targets)
        
        # Validation data (20% of training size)
        val_samples = self.num_samples // 5
        val_data = torch.randn(val_samples, self.input_size)
        val_targets = torch.randint(0, self.num_classes, (val_samples,))
        val_dataset = TensorDataset(val_data, val_targets)
        
        return train_dataset, val_dataset
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'name': 'Random Dummy',
            'source': 'generated',
            'input_shape': (self.input_size,),
            'num_classes': self.num_classes,
            'classes': [f'Class {i}' for i in range(self.num_classes)],
            'train_samples': self.num_samples,
            'test_samples': self.num_samples // 5,
            'license': 'None'
        }

@register_dataset("Random Images")
class RandomImageDataset(BaseDataset):
    """Random image dataset for testing CNNs."""
    
    def __init__(self, root: str = "./data/cache", download: bool = True,
                 num_samples: int = 1000, channels: int = 3, 
                 height: int = 32, width: int = 32, num_classes: int = 10):
        super().__init__(root, download)
        self.num_samples = num_samples
        self.channels = channels
        self.height = height
        self.width = width
        self.num_classes = num_classes
    
    def load(self) -> Tuple[Dataset, Optional[Dataset]]:
        # Generate random image data
        np.random.seed(42)
        
        # Training data
        train_data = torch.randn(self.num_samples, self.channels, self.height, self.width)
        train_targets = torch.randint(0, self.num_classes, (self.num_samples,))
        train_dataset = TensorDataset(train_data, train_targets)
        
        # Validation data (20% of training size)
        val_samples = self.num_samples // 5
        val_data = torch.randn(val_samples, self.channels, self.height, self.width)
        val_targets = torch.randint(0, self.num_classes, (val_samples,))
        val_dataset = TensorDataset(val_data, val_targets)
        
        return train_dataset, val_dataset
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'name': 'Random Images',
            'source': 'generated',
            'input_shape': (self.channels, self.height, self.width),
            'num_classes': self.num_classes,
            'classes': [f'Class {i}' for i in range(self.num_classes)],
            'train_samples': self.num_samples,
            'test_samples': self.num_samples // 5,
            'license': 'None'
        }