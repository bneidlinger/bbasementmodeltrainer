import os
import json
import csv
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from pathlib import Path

class BatchTester:
    """Batch testing functionality with export capabilities."""
    
    def __init__(self, model_inference):
        self.model = model_inference
        self.results = []
        
    def test_folder(self, folder_path: str, recursive: bool = False) -> Dict:
        """Test all images in a folder."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        results = {
            'folder': folder_path,
            'timestamp': datetime.now().isoformat(),
            'model': self.model.model_path,
            'predictions': [],
            'summary': {}
        }
        
        # Find all images
        if recursive:
            image_files = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        image_files.append(os.path.join(root, file))
        else:
            image_files = [
                os.path.join(folder_path, f) 
                for f in os.listdir(folder_path)
                if Path(f).suffix.lower() in image_extensions
            ]
        
        # Test each image
        class_names = self.model.get_class_names()
        class_counts = {name: 0 for name in class_names}
        
        for image_path in image_files:
            try:
                pred_class, probs = self.model.predict(image_path)
                
                # Get top 3 predictions
                probs_with_idx = [(i, p) for i, p in enumerate(probs)]
                probs_with_idx.sort(key=lambda x: x[1], reverse=True)
                top_3 = [
                    {
                        'class': class_names[idx],
                        'confidence': float(prob)
                    }
                    for idx, prob in probs_with_idx[:3]
                ]
                
                results['predictions'].append({
                    'file': os.path.basename(image_path),
                    'path': image_path,
                    'predicted_class': class_names[pred_class],
                    'confidence': float(probs[pred_class]),
                    'top_3': top_3
                })
                
                class_counts[class_names[pred_class]] += 1
                
            except Exception as e:
                results['predictions'].append({
                    'file': os.path.basename(image_path),
                    'path': image_path,
                    'error': str(e)
                })
        
        # Add summary statistics
        results['summary'] = {
            'total_images': len(image_files),
            'successful': len([p for p in results['predictions'] if 'error' not in p]),
            'failed': len([p for p in results['predictions'] if 'error' in p]),
            'class_distribution': class_counts,
            'average_confidence': np.mean([
                p['confidence'] 
                for p in results['predictions'] 
                if 'confidence' in p
            ]) if results['predictions'] else 0
        }
        
        self.results.append(results)
        return results
    
    def export_to_csv(self, output_path: str):
        """Export results to CSV file."""
        if not self.results:
            return
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'timestamp', 'folder', 'file', 'path',
                'predicted_class', 'confidence',
                'top_1', 'top_1_conf',
                'top_2', 'top_2_conf',
                'top_3', 'top_3_conf',
                'error'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for batch_result in self.results:
                for pred in batch_result['predictions']:
                    row = {
                        'timestamp': batch_result['timestamp'],
                        'folder': batch_result['folder'],
                        'file': pred['file'],
                        'path': pred['path']
                    }
                    
                    if 'error' in pred:
                        row['error'] = pred['error']
                    else:
                        row['predicted_class'] = pred['predicted_class']
                        row['confidence'] = f"{pred['confidence']:.4f}"
                        
                        for i, top in enumerate(pred['top_3'], 1):
                            row[f'top_{i}'] = top['class']
                            row[f'top_{i}_conf'] = f"{top['confidence']:.4f}"
                    
                    writer.writerow(row)
    
    def export_to_json(self, output_path: str):
        """Export results to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def export_to_html(self, output_path: str):
        """Export results to HTML report."""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Batch Testing Results</title>
    <style>
        body {
            font-family: 'Courier New', monospace;
            background-color: #1a1a1a;
            color: #FF8C00;
            padding: 20px;
        }
        h1, h2 {
            color: #32CD32;
            border-bottom: 2px solid #32CD32;
            padding-bottom: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th {
            background-color: #2a2a2a;
            color: #FF8C00;
            padding: 10px;
            text-align: left;
            border: 1px solid #FF8C00;
        }
        td {
            padding: 8px;
            border: 1px solid #444;
        }
        tr:nth-child(even) {
            background-color: #2a2a2a;
        }
        .confidence-high { color: #32CD32; }
        .confidence-medium { color: #FFA500; }
        .confidence-low { color: #FF4444; }
        .summary {
            background-color: #2a2a2a;
            padding: 15px;
            border: 2px solid #FF8C00;
            margin: 20px 0;
        }
        .error { color: #FF4444; }
    </style>
</head>
<body>
    <h1>🤖 BasementBrewAI - Batch Testing Results</h1>
"""
        
        for batch_result in self.results:
            html_content += f"""
    <div class="summary">
        <h2>Test Session: {batch_result['timestamp']}</h2>
        <p><strong>Folder:</strong> {batch_result['folder']}</p>
        <p><strong>Model:</strong> {batch_result['model']}</p>
        <p><strong>Total Images:</strong> {batch_result['summary']['total_images']}</p>
        <p><strong>Successful:</strong> {batch_result['summary']['successful']}</p>
        <p><strong>Failed:</strong> {batch_result['summary']['failed']}</p>
        <p><strong>Average Confidence:</strong> {batch_result['summary']['average_confidence']:.2%}</p>
    </div>
    
    <table>
        <thead>
            <tr>
                <th>File</th>
                <th>Predicted Class</th>
                <th>Confidence</th>
                <th>Top 3 Predictions</th>
            </tr>
        </thead>
        <tbody>
"""
            
            for pred in batch_result['predictions']:
                if 'error' in pred:
                    html_content += f"""
            <tr>
                <td>{pred['file']}</td>
                <td colspan="3" class="error">Error: {pred['error']}</td>
            </tr>
"""
                else:
                    conf = pred['confidence']
                    conf_class = 'confidence-high' if conf > 0.8 else 'confidence-medium' if conf > 0.5 else 'confidence-low'
                    
                    top_3_str = ', '.join([
                        f"{t['class']} ({t['confidence']:.1%})"
                        for t in pred['top_3']
                    ])
                    
                    html_content += f"""
            <tr>
                <td>{pred['file']}</td>
                <td>{pred['predicted_class']}</td>
                <td class="{conf_class}">{conf:.2%}</td>
                <td>{top_3_str}</td>
            </tr>
"""
            
            html_content += """
        </tbody>
    </table>
"""
        
        # Add class distribution chart (ASCII style)
        if self.results:
            latest_result = self.results[-1]
            if 'class_distribution' in latest_result['summary']:
                html_content += """
    <div class="summary">
        <h2>Class Distribution</h2>
        <pre>
"""
                distribution = latest_result['summary']['class_distribution']
                max_count = max(distribution.values()) if distribution else 1
                
                for class_name, count in distribution.items():
                    bar_length = int((count / max_count) * 50) if max_count > 0 else 0
                    bar = '█' * bar_length
                    html_content += f"{class_name:15} [{bar:50}] {count:3d}\n"
                
                html_content += """
        </pre>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def generate_summary_report(self) -> str:
        """Generate ASCII summary report."""
        if not self.results:
            return "No test results available"
        
        lines = []
        lines.append("=" * 60)
        lines.append("BATCH TESTING SUMMARY REPORT")
        lines.append("=" * 60)
        
        for i, batch_result in enumerate(self.results, 1):
            lines.append(f"\nBatch #{i}")
            lines.append("-" * 40)
            lines.append(f"Timestamp: {batch_result['timestamp']}")
            lines.append(f"Folder: {batch_result['folder']}")
            lines.append(f"Total Images: {batch_result['summary']['total_images']}")
            lines.append(f"Successful: {batch_result['summary']['successful']}")
            lines.append(f"Failed: {batch_result['summary']['failed']}")
            lines.append(f"Avg Confidence: {batch_result['summary']['average_confidence']:.2%}")
            
            # Class distribution
            lines.append("\nClass Distribution:")
            distribution = batch_result['summary']['class_distribution']
            max_count = max(distribution.values()) if distribution else 1
            
            for class_name, count in sorted(distribution.items(), 
                                           key=lambda x: x[1], reverse=True):
                if count > 0:
                    bar_length = int((count / max_count) * 30)
                    bar = '█' * bar_length
                    lines.append(f"  {class_name:15} [{bar:30}] {count:3d}")
        
        lines.append("=" * 60)
        return "\n".join(lines)