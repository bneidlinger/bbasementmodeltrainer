"""
Demo script showing how to use a trained model for inference.
"""

import os
import sys
import torch
from PIL import Image
import numpy as np

# Add trainer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trainer'))

from inference import ModelInference

def demo_inference():
    """Demonstrate model inference capabilities."""
    print("ModelBuilder - Model Testing Demo")
    print("=" * 50)
    
    # Check for saved models
    models_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    
    if not os.path.exists(models_dir):
        print("\nNo saved models found!")
        print("Please train a model first using the GUI (run.py)")
        return
    
    # Find model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth') and not f.endswith('_full.pth')]
    
    if not model_files:
        print("\nNo model files found!")
        return
    
    print(f"\nFound {len(model_files)} saved model(s):")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    # Use the first model for demo
    model_path = os.path.join(models_dir, model_files[0])
    print(f"\nUsing model: {model_files[0]}")
    
    # Load model
    print("\nLoading model...")
    inference = ModelInference(model_path)
    
    # Display model info
    print("\nModel Information:")
    print(f"- Model Class: {inference.metadata.get('model_class')}")
    print(f"- Dataset: {inference.metadata.get('dataset_name')}")
    print(f"- Input Shape: {inference.metadata.get('input_shape')}")
    print(f"- Number of Classes: {inference.metadata.get('num_classes')}")
    print(f"- Best Validation Accuracy: {inference.metadata.get('best_val_acc', 0):.2%}")
    
    # Create a dummy image for testing
    print("\nCreating test image...")
    input_shape = inference.metadata.get('input_shape', (3, 32, 32))
    
    if len(input_shape) == 3:
        channels, height, width = input_shape
    else:
        channels = 1
        height = width = 28
    
    # Create random test image
    if channels == 1:
        test_image = Image.fromarray(np.random.randint(0, 255, (height, width), dtype=np.uint8), mode='L')
    else:
        test_image = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8), mode='RGB')
    
    # Save test image
    test_image_path = "test_image.png"
    test_image.save(test_image_path)
    print(f"Test image saved to: {test_image_path}")
    
    # Run prediction
    print("\nRunning prediction...")
    predicted_class, probabilities = inference.predict(test_image_path)
    
    # Get class names
    class_names = inference.get_class_names()
    
    # Display results
    print(f"\nPredicted Class: {class_names[predicted_class]}")
    print(f"Confidence: {probabilities[predicted_class]:.2%}")
    
    print("\nTop 5 predictions:")
    probs_with_idx = [(i, p) for i, p in enumerate(probabilities)]
    probs_with_idx.sort(key=lambda x: x[1], reverse=True)
    
    for i, (idx, prob) in enumerate(probs_with_idx[:5]):
        print(f"{i+1}. {class_names[idx]}: {prob:.2%}")
    
    # Export options
    print("\nExport Options:")
    print("1. ONNX format - for deployment with ONNX Runtime")
    print("2. TorchScript - for deployment with PyTorch")
    
    # Clean up
    os.remove(test_image_path)
    
    print("\n" + "=" * 50)
    print("Demo complete! You can now:")
    print("1. Use the GUI Testing tab to upload and test real images")
    print("2. Export your model to ONNX or TorchScript format")
    print("3. Test on the full dataset to get accuracy metrics")


if __name__ == "__main__":
    demo_inference()