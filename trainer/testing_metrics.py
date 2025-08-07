import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_recall_fscore_support, classification_report
import torch
from collections import defaultdict
import time

class AdvancedMetrics:
    def __init__(self):
        self.results = {}
        
    def calculate_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  class_names: List[str]) -> Dict:
        """Calculate precision, recall, F1-score for each class."""
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics = {
            'per_class': {},
            'macro_avg': {},
            'weighted_avg': {}
        }
        
        # Per-class metrics
        for i, class_name in enumerate(class_names):
            if i < len(precision):
                metrics['per_class'][class_name] = {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i]) if i < len(support) else 0
                }
        
        # Calculate averages
        macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        metrics['macro_avg'] = {
            'precision': float(macro_prec),
            'recall': float(macro_rec),
            'f1_score': float(macro_f1)
        }
        
        metrics['weighted_avg'] = {
            'precision': float(weighted_prec),
            'recall': float(weighted_rec),
            'f1_score': float(weighted_f1)
        }
        
        # Overall accuracy
        metrics['accuracy'] = float(np.mean(y_true == y_pred))
        
        return metrics
    
    def create_confusion_matrix_ascii(self, cm: np.ndarray, 
                                     class_names: List[str],
                                     normalize: bool = False) -> str:
        """Create ASCII art confusion matrix visualization."""
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)
        
        # Truncate class names for display
        max_name_len = 8
        display_names = [name[:max_name_len].center(max_name_len) 
                        for name in class_names]
        
        # Build ASCII art
        lines = []
        lines.append("=" * (len(display_names) * 10 + 15))
        lines.append("     CONFUSION MATRIX     ")
        lines.append("=" * (len(display_names) * 10 + 15))
        
        # Header
        header = "TRUE\\PRED |"
        for name in display_names:
            header += f" {name} |"
        lines.append(header)
        lines.append("-" * len(header))
        
        # Matrix rows
        for i, true_name in enumerate(display_names):
            row = f"{true_name} |"
            for j in range(len(class_names)):
                if i < cm.shape[0] and j < cm.shape[1]:
                    if normalize:
                        value = f"{cm[i, j]:.2%}".center(9)
                    else:
                        value = f"{int(cm[i, j])}".center(9)
                    
                    # Highlight diagonal (correct predictions)
                    if i == j:
                        value = f"[{value.strip()}]"
                    row += f" {value}|"
                else:
                    row += "    -    |"
            lines.append(row)
        
        lines.append("=" * (len(display_names) * 10 + 15))
        
        return "\n".join(lines)
    
    def benchmark_inference_speed(self, model, input_shape: Tuple, 
                                 device: str, num_iterations: int = 100) -> Dict:
        """Benchmark model inference speed."""
        model.eval()
        
        # Warmup
        dummy_input = torch.randn(1, *input_shape).to(device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Actual benchmarking
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        return {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'p50_ms': float(np.percentile(times, 50)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'fps': float(1000 / np.mean(times))  # Frames per second
        }
    
    def calculate_model_efficiency(self, model) -> Dict:
        """Calculate model efficiency metrics."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate model size in MB
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        # Calculate FLOPs (simplified estimate)
        # This is a rough estimate, actual FLOPs depend on input size
        flops_estimate = total_params * 2  # Rough estimate
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'estimated_flops': flops_estimate,
            'parameter_efficiency': trainable_params / total_params if total_params > 0 else 0
        }
    
    def generate_model_report_card(self, metrics: Dict, 
                                   efficiency: Dict,
                                   speed: Dict) -> str:
        """Generate a comprehensive model report card."""
        lines = []
        lines.append("╔" + "═" * 58 + "╗")
        lines.append("║" + "MODEL EVALUATION REPORT CARD".center(58) + "║")
        lines.append("╠" + "═" * 58 + "╣")
        
        # Performance Grade
        accuracy = metrics.get('accuracy', 0)
        grade = self._calculate_grade(accuracy)
        lines.append(f"║ OVERALL GRADE: {grade}".ljust(59) + "║")
        lines.append(f"║ Accuracy: {accuracy:.2%}".ljust(59) + "║")
        lines.append("╠" + "═" * 58 + "╣")
        
        # Detailed Metrics
        lines.append("║ PERFORMANCE METRICS:".ljust(59) + "║")
        lines.append(f"║   Macro F1-Score: {metrics['macro_avg']['f1_score']:.3f}".ljust(59) + "║")
        lines.append(f"║   Weighted F1-Score: {metrics['weighted_avg']['f1_score']:.3f}".ljust(59) + "║")
        lines.append(f"║   Macro Precision: {metrics['macro_avg']['precision']:.3f}".ljust(59) + "║")
        lines.append(f"║   Macro Recall: {metrics['macro_avg']['recall']:.3f}".ljust(59) + "║")
        lines.append("╠" + "═" * 58 + "╣")
        
        # Efficiency Metrics
        lines.append("║ EFFICIENCY METRICS:".ljust(59) + "║")
        lines.append(f"║   Total Parameters: {efficiency['total_parameters']:,}".ljust(59) + "║")
        lines.append(f"║   Model Size: {efficiency['model_size_mb']:.2f} MB".ljust(59) + "║")
        lines.append(f"║   Parameter Efficiency: {efficiency['parameter_efficiency']:.2%}".ljust(59) + "║")
        lines.append("╠" + "═" * 58 + "╣")
        
        # Speed Metrics
        lines.append("║ INFERENCE SPEED:".ljust(59) + "║")
        lines.append(f"║   Mean Latency: {speed['mean_ms']:.2f} ms".ljust(59) + "║")
        lines.append(f"║   95th Percentile: {speed['p95_ms']:.2f} ms".ljust(59) + "║")
        lines.append(f"║   Throughput: {speed['fps']:.1f} FPS".ljust(59) + "║")
        lines.append("╠" + "═" * 58 + "╣")
        
        # Recommendations
        lines.append("║ RECOMMENDATIONS:".ljust(59) + "║")
        recommendations = self._generate_recommendations(accuracy, efficiency, speed)
        for rec in recommendations:
            wrapped = self._wrap_text(rec, 56)
            for line in wrapped:
                lines.append(f"║   {line}".ljust(59) + "║")
        
        lines.append("╚" + "═" * 58 + "╝")
        
        return "\n".join(lines)
    
    def _calculate_grade(self, accuracy: float) -> str:
        """Calculate letter grade based on accuracy."""
        if accuracy >= 0.95:
            return "A+ ⭐ EXCELLENT"
        elif accuracy >= 0.90:
            return "A  ✓ VERY GOOD"
        elif accuracy >= 0.85:
            return "B+ ✓ GOOD"
        elif accuracy >= 0.80:
            return "B  ✓ SATISFACTORY"
        elif accuracy >= 0.75:
            return "C+ → ACCEPTABLE"
        elif accuracy >= 0.70:
            return "C  → NEEDS IMPROVEMENT"
        elif accuracy >= 0.60:
            return "D  ⚠ POOR"
        else:
            return "F  ✗ FAILING"
    
    def _generate_recommendations(self, accuracy: float, 
                                 efficiency: Dict, 
                                 speed: Dict) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        if accuracy < 0.80:
            recommendations.append("Consider training for more epochs or adjusting hyperparameters")
        
        if efficiency['model_size_mb'] > 100:
            recommendations.append("Model is large; consider pruning or quantization for deployment")
        
        if speed['mean_ms'] > 50:
            recommendations.append("Inference is slow; consider model optimization or hardware acceleration")
        
        if accuracy >= 0.95:
            recommendations.append("Excellent performance! Consider testing on more challenging datasets")
        
        if efficiency['parameter_efficiency'] < 0.9:
            recommendations.append("Many frozen parameters; ensure this is intentional")
        
        if len(recommendations) == 0:
            recommendations.append("Model performs well across all metrics!")
        
        return recommendations
    
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to specified width."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + len(current_line) <= width:
                current_line.append(word)
                current_length += len(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return lines if lines else [text[:width]]


class ModelComparator:
    """Compare multiple models side-by-side."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_model(self, name: str, model_path: str, metrics: Dict):
        """Add a model to comparison."""
        self.models[name] = model_path
        self.results[name] = metrics
    
    def generate_comparison_table(self) -> str:
        """Generate ASCII comparison table."""
        if not self.models:
            return "No models to compare"
        
        lines = []
        model_names = list(self.models.keys())
        
        # Header
        lines.append("╔" + "═" * 80 + "╗")
        lines.append("║" + "MODEL COMPARISON".center(80) + "║")
        lines.append("╠" + "═" * 20 + "╦" + ("═" * 19 + "╦") * (len(model_names) - 1) + "═" * (80 - 20 - 20 * len(model_names)) + "╣")
        
        # Model names header
        header = "║ METRIC".ljust(21) + "║"
        for name in model_names:
            header += f" {name[:18].center(18)}║"
        lines.append(header)
        lines.append("╠" + "═" * 20 + "╬" + ("═" * 19 + "╬") * (len(model_names) - 1) + "═" * (80 - 20 - 20 * len(model_names)) + "╣")
        
        # Metrics rows
        metrics_to_compare = [
            ('Accuracy', 'accuracy'),
            ('F1-Score (Macro)', 'macro_f1'),
            ('Precision (Macro)', 'macro_precision'),
            ('Recall (Macro)', 'macro_recall'),
            ('Model Size (MB)', 'model_size_mb'),
            ('Inference (ms)', 'inference_ms'),
            ('FPS', 'fps')
        ]
        
        for display_name, key in metrics_to_compare:
            row = f"║ {display_name[:19].ljust(19)}║"
            best_value = None
            values = []
            
            # Collect values for comparison
            for name in model_names:
                if name in self.results and key in self.results[name]:
                    value = self.results[name][key]
                    values.append(value)
                else:
                    values.append(None)
            
            # Find best value
            if values and any(v is not None for v in values):
                valid_values = [v for v in values if v is not None]
                if key in ['inference_ms', 'model_size_mb']:  # Lower is better
                    best_value = min(valid_values)
                else:  # Higher is better
                    best_value = max(valid_values)
            
            # Format values
            for value in values:
                if value is None:
                    row += " N/A".center(19) + "║"
                else:
                    if key in ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall']:
                        formatted = f"{value:.3f}"
                    elif key == 'model_size_mb':
                        formatted = f"{value:.2f} MB"
                    elif key == 'inference_ms':
                        formatted = f"{value:.2f} ms"
                    elif key == 'fps':
                        formatted = f"{value:.1f}"
                    else:
                        formatted = str(value)
                    
                    # Highlight best value
                    if value == best_value:
                        formatted = f"[{formatted}]"
                    
                    row += f" {formatted.center(17)} ║"
            
            lines.append(row)
        
        lines.append("╚" + "═" * 20 + "╩" + ("═" * 19 + "╩") * (len(model_names) - 1) + "═" * (80 - 20 - 20 * len(model_names)) + "╝")
        
        return "\n".join(lines)