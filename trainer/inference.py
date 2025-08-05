import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from typing import Dict, Any, List, Tuple, Optional
import json

class ModelInference:
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize inference engine with a saved model.
        
        Args:
            model_path: Path to saved model file
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.metadata = None
        self.transform = None
        
        self.load_model()
        self.setup_transforms()
    
    def load_model(self):
        """Load the model and metadata from checkpoint."""
        # Try loading full model first
        full_model_path = self.model_path.replace('.pth', '_full.pth')
        
        try:
            if os.path.exists(full_model_path):
                # Load full model
                self.model = torch.load(full_model_path, map_location=self.device)
                self.model.eval()
                
                # Load metadata from checkpoint
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
            else:
                # Load from state dict (requires model class)
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
                
                # Try to recreate model from metadata
                model_class_name = self.metadata.get('model_class')
                model_params = self.metadata.get('model_params', {})
                
                # Import models to get registry
                from models import MODEL_REGISTRY
                
                # Find model class
                model_class = None
                for name, cls in MODEL_REGISTRY.items():
                    if cls.__name__ == model_class_name:
                        model_class = cls
                        break
                
                if model_class:
                    self.model = model_class(**model_params)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model.to(self.device)
                    self.model.eval()
                else:
                    raise ValueError(f"Could not find model class: {model_class_name}")
                    
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def setup_transforms(self):
        """Set up image transforms based on dataset."""
        dataset_name = self.metadata.get('dataset_name', 'unknown')
        input_shape = self.metadata.get('input_shape', (3, 32, 32))
        
        # Get number of channels and image size
        if len(input_shape) == 3:
            channels, height, width = input_shape
        elif len(input_shape) == 1:
            # Flatten input (like MNIST MLP)
            channels = 1
            height = width = int(np.sqrt(input_shape[0]))
        else:
            channels, height, width = 3, 32, 32
        
        # Create transforms based on dataset
        transform_list = [
            transforms.Resize((height, width)),
            transforms.ToTensor()
        ]
        
        # Add normalization based on dataset
        if dataset_name == 'CIFAR-10':
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        elif dataset_name == 'MNIST':
            transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        elif dataset_name == 'FashionMNIST':
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        else:
            # Generic normalization
            if channels == 1:
                transform_list.append(transforms.Normalize((0.5,), (0.5,)))
            else:
                transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        
        self.transform = transforms.Compose(transform_list)
        self.expected_channels = channels
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess an image for inference."""
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB or L based on expected channels
        if self.expected_channels == 1:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        return tensor.unsqueeze(0).to(self.device)
    
    def predict(self, image_path: str) -> Tuple[int, List[float]]:
        """
        Predict class for a single image.
        
        Returns:
            Tuple of (predicted_class, probabilities)
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image_path)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        return predicted_class, probabilities[0].cpu().numpy().tolist()
    
    def predict_batch(self, image_paths: List[str]) -> List[Tuple[int, List[float]]]:
        """Predict classes for multiple images."""
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append((-1, []))
        return results
    
    def get_class_names(self) -> List[str]:
        """Get class names based on dataset."""
        dataset_name = self.metadata.get('dataset_name', 'unknown')
        
        class_names = {
            'CIFAR-10': ['plane', 'car', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck'],
            'MNIST': [str(i) for i in range(10)],
            'FashionMNIST': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        }
        
        if dataset_name in class_names:
            return class_names[dataset_name]
        else:
            # Generic class names
            num_classes = self.metadata.get('num_classes', 10)
            return [f'Class {i}' for i in range(num_classes)]
    
    def export_onnx(self, output_path: str, example_input: Optional[torch.Tensor] = None):
        """Export model to ONNX format."""
        if example_input is None:
            # Create dummy input based on input shape
            input_shape = self.metadata.get('input_shape', (3, 32, 32))
            if len(input_shape) == 1:
                example_input = torch.randn(1, input_shape[0])
            else:
                example_input = torch.randn(1, *input_shape)
            example_input = example_input.to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            example_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'output': {0: 'batch_size'}}
        )
    
    def export_torchscript(self, output_path: str):
        """Export model to TorchScript format."""
        scripted_model = torch.jit.script(self.model)
        scripted_model.save(output_path)


def test_on_dataset(model_path: str, dataset_loader, num_samples: int = 100) -> Dict[str, float]:
    """Test model on a dataset and return metrics."""
    inference = ModelInference(model_path)
    
    # Get test data
    _, test_dataset = dataset_loader.load()
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    # Test metrics
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    # Run inference
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if total >= num_samples:
                break
                
            inputs = inputs.to(inference.device)
            outputs = inference.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted.cpu() == targets).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # Calculate metrics
    accuracy = 100 * correct / total
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_targets[:total], all_predictions[:total])
    
    return {
        'accuracy': accuracy,
        'total_samples': total,
        'correct': correct,
        'confusion_matrix': cm.tolist()
    }


if __name__ == "__main__":
    # Test the inference module
    print("Testing inference module...")
    
    # You would need to provide a path to a saved model
    # Example: inference = ModelInference("saved_models/run_1_model.pth")
    
    print("Inference module ready for use!")