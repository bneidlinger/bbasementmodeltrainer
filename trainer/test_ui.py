import dearpygui.dearpygui as dpg
import os
import glob
from tkinter import filedialog
import tkinter as tk
from PIL import Image
import numpy as np
from typing import Optional, List
import torch

from inference import ModelInference
from data.core import DATASET_REGISTRY
from retro_theme import COLORS

class TestingUI:
    def __init__(self, parent_app):
        self.app = parent_app
        self.current_model: Optional[ModelInference] = None
        self.current_model_path: Optional[str] = None
        
        # UI tags
        self.tags = {
            'model_combo': 'test_model_combo',
            'load_button': 'test_load_button',
            'model_info': 'test_model_info',
            'image_preview': 'test_image_preview',
            'upload_button': 'test_upload_button',
            'predict_button': 'test_predict_button',
            'results_text': 'test_results_text',
            'probability_bars': 'test_probability_bars',
            'test_dataset_button': 'test_dataset_button',
            'export_onnx_button': 'export_onnx_button',
            'export_torchscript_button': 'export_torchscript_button'
        }
        
        self.test_image_path: Optional[str] = None
    
    def create_ui(self, parent):
        """Create the testing interface UI."""
        with dpg.group(parent=parent):
            dpg.add_text("[ MODEL TESTING & INFERENCE ]", color=COLORS['orange_bright'])
            dpg.add_separator()
            
            with dpg.group(horizontal=True):
                # Left panel - Model selection and info
                with dpg.child_window(width=350, height=-1):
                    dpg.add_text("[ SELECT MODEL ]", color=COLORS['orange_bright'])
                    dpg.add_separator()
                    
                    # Model selection
                    dpg.add_text("Saved Models:", color=COLORS['green_normal'])
                    dpg.add_combo(
                        [],
                        tag=self.tags['model_combo'],
                        width=-1,
                        callback=self.on_model_selected
                    )
                    
                    dpg.add_button(
                        label="(R) REFRESH LIST",
                        callback=self.refresh_model_list,
                        width=-1
                    )
                    
                    dpg.add_separator()
                    
                    # Model info
                    dpg.add_text("Model Information:", color=COLORS['green_normal'])
                    dpg.add_text("No model loaded", 
                               tag=self.tags['model_info'],
                               wrap=330,
                               color=(150, 150, 150))
                    
                    dpg.add_separator()
                    
                    # Export options
                    dpg.add_text("[ EXPORT MODEL ]", color=COLORS['orange_bright'])
                    
                    dpg.add_button(
                        label="[>] EXPORT ONNX",
                        tag=self.tags['export_onnx_button'],
                        callback=self.export_onnx,
                        width=-1,
                        enabled=False
                    )
                    
                    dpg.add_button(
                        label="[>] EXPORT TORCHSCRIPT",
                        tag=self.tags['export_torchscript_button'],
                        callback=self.export_torchscript,
                        width=-1,
                        enabled=False
                    )
                    
                    dpg.add_separator()
                    
                    # Test on dataset
                    dpg.add_text("[ BATCH TESTING ]", color=COLORS['orange_bright'])
                    
                    dpg.add_button(
                        label="[*] TEST ON DATASET",
                        tag=self.tags['test_dataset_button'],
                        callback=self.test_on_dataset,
                        width=-1,
                        enabled=False
                    )
                
                # Middle panel - Image upload and preview
                with dpg.child_window(width=300, height=-1):
                    dpg.add_text("[ IMAGE TESTING ]", color=COLORS['orange_bright'])
                    dpg.add_separator()
                    
                    # Upload button
                    dpg.add_button(
                        label="[^] UPLOAD IMAGE",
                        tag=self.tags['upload_button'],
                        callback=self.upload_image,
                        width=-1,
                        height=30
                    )
                    
                    # Image preview placeholder
                    dpg.add_text("No image loaded")
                    dpg.add_text("Image will appear here", 
                               tag=self.tags['image_preview'],
                               color=(150, 150, 150))
                    
                    dpg.add_separator()
                    
                    # Predict button
                    dpg.add_button(
                        label=">> RUN PREDICTION",
                        tag=self.tags['predict_button'],
                        callback=self.predict_image,
                        width=-1,
                        height=30,
                        enabled=False
                    )
                
                # Right panel - Results
                with dpg.child_window(width=-1, height=-1):
                    dpg.add_text("[ PREDICTION RESULTS ]", color=COLORS['orange_bright'])
                    dpg.add_separator()
                    
                    # Results text
                    dpg.add_text("No predictions yet", 
                               tag=self.tags['results_text'],
                               wrap=400)
                    
                    dpg.add_separator()
                    
                    # Probability visualization
                    dpg.add_text("Class Probabilities:")
                    with dpg.group(tag=self.tags['probability_bars']):
                        dpg.add_text("Run a prediction to see probabilities",
                                   color=(150, 150, 150))
        
        # Refresh model list on creation
        self.refresh_model_list()
    
    def refresh_model_list(self):
        """Refresh the list of saved models."""
        models_dir = os.path.join(os.path.dirname(self.app.db.db_path), 'saved_models')
        
        if not os.path.exists(models_dir):
            dpg.configure_item(self.tags['model_combo'], items=["No saved models found"])
            return
        
        # Find all model files
        model_files = glob.glob(os.path.join(models_dir, "*.pth"))
        # Filter out _full.pth files
        model_files = [f for f in model_files if not f.endswith('_full.pth')]
        
        if model_files:
            # Extract model names
            model_names = [os.path.basename(f) for f in model_files]
            dpg.configure_item(self.tags['model_combo'], items=model_names)
            
            # Auto-select first model
            if model_names:
                dpg.set_value(self.tags['model_combo'], model_names[0])
                self.on_model_selected(None, model_names[0])
        else:
            dpg.configure_item(self.tags['model_combo'], items=["No saved models found"])
    
    def on_model_selected(self, sender, app_data):
        """Handle model selection."""
        model_name = app_data if app_data else dpg.get_value(self.tags['model_combo'])
        
        if model_name and model_name != "No saved models found":
            models_dir = os.path.join(os.path.dirname(self.app.db.db_path), 'saved_models')
            model_path = os.path.join(models_dir, model_name)
            
            try:
                # Load model
                self.current_model = ModelInference(model_path)
                self.current_model_path = model_path
                
                # Update model info
                metadata = self.current_model.metadata
                info_text = f"Model Class: {metadata.get('model_class', 'Unknown')}\n"
                info_text += f"Dataset: {metadata.get('dataset_name', 'Unknown')}\n"
                info_text += f"Input Shape: {metadata.get('input_shape', 'Unknown')}\n"
                info_text += f"Classes: {metadata.get('num_classes', 'Unknown')}\n"
                info_text += f"Best Val Acc: {metadata.get('best_val_acc', 0):.2%}\n"
                info_text += f"Epochs Trained: {metadata.get('epochs_trained', 'Unknown')}\n"
                info_text += f"Run ID: {metadata.get('run_id', 'Unknown')}"
                
                dpg.set_value(self.tags['model_info'], info_text)
                
                # Enable buttons
                dpg.configure_item(self.tags['export_onnx_button'], enabled=True)
                dpg.configure_item(self.tags['export_torchscript_button'], enabled=True)
                dpg.configure_item(self.tags['test_dataset_button'], enabled=True)
                if self.test_image_path:
                    dpg.configure_item(self.tags['predict_button'], enabled=True)
                
            except Exception as e:
                dpg.set_value(self.tags['model_info'], f"Error loading model: {str(e)}")
                self.current_model = None
                dpg.configure_item(self.tags['export_onnx_button'], enabled=False)
                dpg.configure_item(self.tags['export_torchscript_button'], enabled=False)
                dpg.configure_item(self.tags['test_dataset_button'], enabled=False)
                dpg.configure_item(self.tags['predict_button'], enabled=False)
    
    def upload_image(self):
        """Handle image upload."""
        # Create a hidden tkinter window
        root = tk.Tk()
        root.withdraw()
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        root.destroy()
        
        if file_path:
            self.test_image_path = file_path
            
            # Update preview text (we can't display actual images in DearPyGui easily)
            filename = os.path.basename(file_path)
            dpg.set_value(self.tags['image_preview'], 
                         f"Loaded: {filename}\n\nPath: {file_path}")
            
            # Enable predict button if model is loaded
            if self.current_model:
                dpg.configure_item(self.tags['predict_button'], enabled=True)
    
    def predict_image(self):
        """Run prediction on uploaded image."""
        if not self.current_model or not self.test_image_path:
            return
        
        try:
            # Run prediction
            predicted_class, probabilities = self.current_model.predict(self.test_image_path)
            
            # Get class names
            class_names = self.current_model.get_class_names()
            
            # Update results text
            result_text = f"Predicted Class: {class_names[predicted_class]}\n"
            result_text += f"Confidence: {probabilities[predicted_class]:.2%}\n\n"
            result_text += "Top 5 Predictions:\n"
            
            # Get top 5 predictions
            probs_with_idx = [(i, p) for i, p in enumerate(probabilities)]
            probs_with_idx.sort(key=lambda x: x[1], reverse=True)
            
            for i, (idx, prob) in enumerate(probs_with_idx[:5]):
                result_text += f"{i+1}. {class_names[idx]}: {prob:.2%}\n"
            
            dpg.set_value(self.tags['results_text'], result_text)
            
            # Update probability bars
            self.update_probability_bars(probabilities, class_names)
            
        except Exception as e:
            dpg.set_value(self.tags['results_text'], f"Error during prediction: {str(e)}")
    
    def update_probability_bars(self, probabilities: List[float], class_names: List[str]):
        """Update probability visualization."""
        # Clear existing bars
        dpg.delete_item(self.tags['probability_bars'], children_only=True)
        
        # Sort probabilities
        probs_with_idx = [(i, p) for i, p in enumerate(probabilities)]
        probs_with_idx.sort(key=lambda x: x[1], reverse=True)
        
        # Show top 10 classes
        for idx, prob in probs_with_idx[:10]:
            with dpg.group(horizontal=True, parent=self.tags['probability_bars']):
                # Class name
                dpg.add_text(f"{class_names[idx]}:", width=100)
                
                # Progress bar for probability
                dpg.add_progress_bar(default_value=prob, width=200)
                
                # Percentage text
                dpg.add_text(f"{prob:.1%}")
    
    def test_on_dataset(self):
        """Test model on its training dataset."""
        if not self.current_model:
            return
        
        dataset_name = self.current_model.metadata.get('dataset_name', 'unknown')
        
        if dataset_name in DATASET_REGISTRY:
            dpg.set_value(self.tags['results_text'], "Testing on dataset... This may take a moment.")
            
            try:
                from inference import test_on_dataset
                
                # Get dataset loader
                dataset_class = DATASET_REGISTRY[dataset_name]
                dataset_loader = dataset_class()
                
                # Test model
                results = test_on_dataset(self.current_model_path, dataset_loader, num_samples=1000)
                
                # Display results
                result_text = f"Test Results on {dataset_name}:\n\n"
                result_text += f"Accuracy: {results['accuracy']:.2f}%\n"
                result_text += f"Tested on: {results['total_samples']} samples\n"
                result_text += f"Correct: {results['correct']}\n\n"
                
                # Add confusion matrix summary
                cm = np.array(results['confusion_matrix'])
                result_text += "Per-class accuracy:\n"
                class_names = self.current_model.get_class_names()
                
                for i in range(min(len(class_names), cm.shape[0])):
                    if cm[i].sum() > 0:
                        class_acc = cm[i, i] / cm[i].sum() * 100
                        result_text += f"{class_names[i]}: {class_acc:.1f}%\n"
                
                dpg.set_value(self.tags['results_text'], result_text)
                
            except Exception as e:
                dpg.set_value(self.tags['results_text'], f"Error testing model: {str(e)}")
        else:
            dpg.set_value(self.tags['results_text'], 
                         f"Dataset '{dataset_name}' not found in registry.")
    
    def export_onnx(self):
        """Export model to ONNX format."""
        if not self.current_model:
            return
        
        try:
            output_path = self.current_model_path.replace('.pth', '.onnx')
            self.current_model.export_onnx(output_path)
            dpg.set_value(self.tags['results_text'], 
                         f"Model exported to ONNX:\n{output_path}")
        except Exception as e:
            dpg.set_value(self.tags['results_text'], 
                         f"Error exporting to ONNX: {str(e)}")
    
    def export_torchscript(self):
        """Export model to TorchScript format."""
        if not self.current_model:
            return
        
        try:
            output_path = self.current_model_path.replace('.pth', '.pt')
            self.current_model.export_torchscript(output_path)
            dpg.set_value(self.tags['results_text'], 
                         f"Model exported to TorchScript:\n{output_path}")
        except Exception as e:
            dpg.set_value(self.tags['results_text'], 
                         f"Error exporting to TorchScript: {str(e)}")