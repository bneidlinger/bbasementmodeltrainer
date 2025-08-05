import os
import hashlib
import pickle
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, random_split

# Registry for dataset loaders
DATASET_REGISTRY = {}

def register_dataset(name: str):
    """Decorator to register dataset loaders."""
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator

class BaseDataset(ABC):
    """Base class for all dataset loaders."""
    
    def __init__(self, root: str = "./data/cache", download: bool = True):
        self.root = root
        self.download = download
        os.makedirs(root, exist_ok=True)
    
    @abstractmethod
    def load(self) -> Tuple[Dataset, Optional[Dataset]]:
        """Load and return train and validation datasets."""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return dataset information."""
        pass
    
    def _get_cache_path(self, filename: str) -> str:
        """Get the full path for a cached file."""
        return os.path.join(self.root, filename)
    
    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate SHA1 hash of a file."""
        sha1 = hashlib.sha1()
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                sha1.update(chunk)
        return sha1.hexdigest()
    
    def _save_to_cache(self, data: Any, filename: str):
        """Save data to cache using pickle."""
        filepath = self._get_cache_path(filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def _load_from_cache(self, filename: str) -> Optional[Any]:
        """Load data from cache if it exists."""
        filepath = self._get_cache_path(filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None
    
    def split_dataset(self, dataset: Dataset, val_split: float = 0.2) -> Tuple[Dataset, Dataset]:
        """Split a dataset into train and validation sets."""
        total_size = len(dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        return train_dataset, val_dataset


class TensorDataset(Dataset):
    """Simple tensor dataset for custom data."""
    
    def __init__(self, data: torch.Tensor, targets: torch.Tensor):
        assert len(data) == len(targets), "Data and targets must have the same length"
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def discover_datasets():
    """Discover and import all dataset loaders."""
    # Import built-in datasets
    from . import builtin
    
    # Import plugins
    import pkgutil
    import importlib
    
    plugins_path = os.path.join(os.path.dirname(__file__), 'plugins')
    if os.path.exists(plugins_path):
        for (_, name, _) in pkgutil.iter_modules([plugins_path]):
            importlib.import_module(f".plugins.{name}", __package__)


# Run discovery when module is imported
discover_datasets()