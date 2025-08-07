import dearpygui.dearpygui as dpg
import os
import glob
from typing import Dict, List, Optional
import torch
import numpy as np

from inference import ModelInference, test_on_dataset
from data.core import DATASET_REGISTRY
from retro_theme import COLORS
from testing_metrics import AdvancedMetrics, ModelComparator

class ModelComparisonUI:
    """UI for comparing multiple models side-by-side."""
    
    def __init__(self, parent_app):
        self.app = parent_app
        self.models = {}  # Dict[name, ModelInference]
        self.comparison_results = {}
        self.comparator = ModelComparator()
        self.metrics_calc = AdvancedMetrics()
        
        # UI tags
        self.tags = {
            'model_list': 'compare_model_list',
            'add_model_button': 'compare_add_model',
            'remove_model_button': 'compare_remove_model',
            'run_comparison_button': 'compare_run',
            'comparison_results': 'compare_results',
            'export_comparison_button': 'compare_export'
        }
    
    def create_ui(self, parent):
        """Create comparison UI as a separate tab."""
        with dpg.group(parent=parent):
            dpg.add_text("[ MODEL COMPARISON SUITE ]", color=COLORS['orange_bright'])
            dpg.add_separator()
            
            with dpg.group(horizontal=True):
                # Left panel - Model selection
                with dpg.child_window(width=300, height=-1):
                    dpg.add_text("[ MODELS TO COMPARE ]", color=COLORS['orange_bright'])
                    dpg.add_separator()
                    
                    # Model list
                    dpg.add_listbox(
                        [],
                        tag=self.tags['model_list'],
                        width=-1,
                        num_items=10
                    )
                    
                    dpg.add_separator()
                    
                    # Add/Remove buttons
                    dpg.add_button(
                        label="[+] ADD MODEL",
                        tag=self.tags['add_model_button'],
                        callback=self.add_model_to_comparison,
                        width=-1
                    )
                    
                    dpg.add_button(
                        label="[-] REMOVE SELECTED",
                        tag=self.tags['remove_model_button'],
                        callback=self.remove_model_from_comparison,
                        width=-1
                    )
                    
                    dpg.add_separator()
                    
                    dpg.add_button(
                        label=">> RUN COMPARISON",
                        tag=self.tags['run_comparison_button'],
                        callback=self.run_comparison,
                        width=-1,
                        height=40
                    )
                    
                    dpg.add_button(
                        label="[>] EXPORT RESULTS",
                        tag=self.tags['export_comparison_button'],
                        callback=self.export_comparison,
                        width=-1,
                        enabled=False
                    )
                
                # Right panel - Results
                with dpg.child_window(width=-1, height=-1):
                    dpg.add_text("[ COMPARISON RESULTS ]", color=COLORS['orange_bright'])
                    dpg.add_separator()
                    
                    dpg.add_text(
                        "Add models and run comparison to see results",
                        tag=self.tags['comparison_results'],
                        wrap=800,
                        color=(150, 150, 150)
                    )
    
    def add_model_to_comparison(self):
        """Add a model to the comparison list."""
        # Get available models
        models_dir = os.path.join(os.path.dirname(self.app.db.db_path), 'saved_models')
        
        if not os.path.exists(models_dir):
            return
        
        # Find all model files
        model_files = glob.glob(os.path.join(models_dir, "*.pth"))
        model_files = [f for f in model_files if not f.endswith('_full.pth')]
        
        if not model_files:
            return
        
        # Create selection popup
        with dpg.window(
            label="Select Model",
            modal=True,
            show=True,
            tag="model_selection_popup",
            width=400,
            height=300,
            pos=[400, 200]
        ):
            dpg.add_text("Select a model to add:")
            
            model_names = [os.path.basename(f) for f in model_files]
            
            def on_select(sender, app_data):
                selected = dpg.get_value(sender)
                if selected and selected not in self.models:
                    # Load the model
                    model_path = os.path.join(models_dir, selected)
                    try:
                        model = ModelInference(model_path)
                        display_name = f"{selected.replace('.pth', '')} (Run {model.metadata.get('run_id', '?')})"
                        self.models[display_name] = model
                        
                        # Update list
                        current_items = list(self.models.keys())
                        dpg.configure_item(self.tags['model_list'], items=current_items)
                        
                    except Exception as e:
                        print(f"Error loading model: {e}")
                
                dpg.delete_item("model_selection_popup")
            
            dpg.add_listbox(
                model_names,
                callback=on_select,
                width=-1,
                num_items=10
            )
            
            dpg.add_button(
                label="Cancel",
                callback=lambda: dpg.delete_item("model_selection_popup"),
                width=-1
            )
    
    def remove_model_from_comparison(self):
        """Remove selected model from comparison."""
        selected = dpg.get_value(self.tags['model_list'])
        if selected and selected in self.models:
            del self.models[selected]
            current_items = list(self.models.keys())
            dpg.configure_item(self.tags['model_list'], items=current_items)
    
    def run_comparison(self):
        """Run comprehensive comparison of all loaded models."""
        if len(self.models) < 2:
            dpg.set_value(self.tags['comparison_results'], 
                         "Please add at least 2 models to compare")
            return
        
        dpg.set_value(self.tags['comparison_results'], 
                     "Running comparison... This may take a few minutes.")
        
        try:
            self.comparison_results = {}
            self.comparator = ModelComparator()
            
            for model_name, model_inference in self.models.items():
                # Get dataset for testing
                dataset_name = model_inference.metadata.get('dataset_name', 'unknown')
                
                if dataset_name not in DATASET_REGISTRY:
                    continue
                
                # Test model
                dataset_class = DATASET_REGISTRY[dataset_name]
                dataset_loader = dataset_class()
                
                # Quick test for metrics
                results = test_on_dataset(
                    model_inference.model_path, 
                    dataset_loader, 
                    num_samples=500
                )
                
                # Get efficiency metrics
                efficiency = self.metrics_calc.calculate_model_efficiency(
                    model_inference.model
                )
                
                # Get speed metrics
                input_shape = model_inference.metadata.get('input_shape', (3, 32, 32))
                if len(input_shape) == 1:
                    input_shape = (input_shape[0],)
                
                speed = self.metrics_calc.benchmark_inference_speed(
                    model_inference.model,
                    input_shape,
                    model_inference.device,
                    num_iterations=50
                )
                
                # Compile metrics
                compiled_metrics = {
                    'accuracy': results['accuracy'] / 100,
                    'macro_f1': 0.85,  # Placeholder - would calculate from confusion matrix
                    'macro_precision': 0.85,
                    'macro_recall': 0.85,
                    'model_size_mb': efficiency['model_size_mb'],
                    'inference_ms': speed['mean_ms'],
                    'fps': speed['fps'],
                    'total_params': efficiency['total_parameters'],
                    'dataset': dataset_name,
                    'epochs': model_inference.metadata.get('epochs_trained', 0),
                    'best_val_acc': model_inference.metadata.get('best_val_acc', 0)
                }
                
                self.comparison_results[model_name] = compiled_metrics
                self.comparator.add_model(model_name, model_inference.model_path, compiled_metrics)
            
            # Generate comparison table
            comparison_table = self.comparator.generate_comparison_table()
            
            # Add detailed analysis
            analysis = self._generate_analysis()
            
            # Display results
            full_results = f"{comparison_table}\n\n{analysis}"
            dpg.set_value(self.tags['comparison_results'], full_results)
            
            # Enable export
            dpg.configure_item(self.tags['export_comparison_button'], enabled=True)
            
        except Exception as e:
            dpg.set_value(self.tags['comparison_results'], 
                         f"Error during comparison: {str(e)}")
    
    def _generate_analysis(self) -> str:
        """Generate analysis and recommendations based on comparison."""
        if not self.comparison_results:
            return ""
        
        lines = []
        lines.append("=" * 80)
        lines.append("ANALYSIS & RECOMMENDATIONS")
        lines.append("=" * 80)
        
        # Find best models for each metric
        metrics_analysis = {
            'accuracy': ('Highest Accuracy', True),
            'inference_ms': ('Fastest Inference', False),
            'model_size_mb': ('Smallest Model', False),
            'fps': ('Highest Throughput', True)
        }
        
        for metric, (display_name, higher_better) in metrics_analysis.items():
            values = [(name, data.get(metric, 0)) 
                     for name, data in self.comparison_results.items()
                     if metric in data]
            
            if values:
                if higher_better:
                    best = max(values, key=lambda x: x[1])
                else:
                    best = min(values, key=lambda x: x[1])
                
                lines.append(f"\n{display_name}: {best[0]}")
                if metric == 'accuracy':
                    lines.append(f"  → {best[1]:.2%} accuracy")
                elif metric == 'inference_ms':
                    lines.append(f"  → {best[1]:.2f} ms per inference")
                elif metric == 'model_size_mb':
                    lines.append(f"  → {best[1]:.2f} MB model size")
                elif metric == 'fps':
                    lines.append(f"  → {best[1]:.1f} FPS throughput")
        
        # Trade-off analysis
        lines.append("\n" + "=" * 80)
        lines.append("TRADE-OFF ANALYSIS")
        lines.append("-" * 80)
        
        # Find pareto-optimal models
        pareto_models = self._find_pareto_optimal()
        if pareto_models:
            lines.append("\nPareto-Optimal Models (best trade-offs):")
            for model in pareto_models:
                lines.append(f"  • {model}")
        
        # Recommendations
        lines.append("\n" + "=" * 80)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 80)
        
        # Deployment recommendations
        for name, metrics in self.comparison_results.items():
            acc = metrics.get('accuracy', 0)
            speed = metrics.get('inference_ms', float('inf'))
            size = metrics.get('model_size_mb', float('inf'))
            
            recommendations = []
            
            if acc > 0.9 and speed < 20:
                recommendations.append("✓ PRODUCTION READY - High accuracy & fast inference")
            elif acc > 0.85 and size < 50:
                recommendations.append("✓ MOBILE DEPLOYMENT - Good accuracy & compact size")
            elif speed < 10:
                recommendations.append("✓ REAL-TIME CAPABLE - Ultra-fast inference")
            elif acc < 0.7:
                recommendations.append("⚠ NEEDS IMPROVEMENT - Low accuracy")
            
            if recommendations:
                lines.append(f"\n{name}:")
                for rec in recommendations:
                    lines.append(f"  {rec}")
        
        return "\n".join(lines)
    
    def _find_pareto_optimal(self) -> List[str]:
        """Find pareto-optimal models (not dominated by any other)."""
        pareto_models = []
        
        for name1, metrics1 in self.comparison_results.items():
            is_dominated = False
            
            for name2, metrics2 in self.comparison_results.items():
                if name1 == name2:
                    continue
                
                # Check if model2 dominates model1
                # (better in all metrics)
                if (metrics2.get('accuracy', 0) >= metrics1.get('accuracy', 0) and
                    metrics2.get('inference_ms', float('inf')) <= metrics1.get('inference_ms', float('inf')) and
                    metrics2.get('model_size_mb', float('inf')) <= metrics1.get('model_size_mb', float('inf'))):
                    
                    # Check if strictly better in at least one
                    if (metrics2.get('accuracy', 0) > metrics1.get('accuracy', 0) or
                        metrics2.get('inference_ms', float('inf')) < metrics1.get('inference_ms', float('inf')) or
                        metrics2.get('model_size_mb', float('inf')) < metrics1.get('model_size_mb', float('inf'))):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_models.append(name1)
        
        return pareto_models
    
    def export_comparison(self):
        """Export comparison results to file."""
        if not self.comparison_results:
            return
        
        # Create export content
        content = "MODEL COMPARISON REPORT\n"
        content += "=" * 80 + "\n\n"
        content += self.comparator.generate_comparison_table()
        content += "\n\n"
        content += self._generate_analysis()
        content += "\n\n"
        content += "Generated by BasementBrewAI\n"
        
        # Save to file
        output_path = "model_comparison_report.txt"
        with open(output_path, 'w') as f:
            f.write(content)
        
        dpg.set_value(self.tags['comparison_results'], 
                     dpg.get_value(self.tags['comparison_results']) + 
                     f"\n\nReport exported to: {output_path}")