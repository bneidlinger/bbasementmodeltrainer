#!/usr/bin/env python
"""
ModelBuilder - PyTorch Training GUI
Run this script to start the application.
"""

import sys
import os

# Add trainer directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trainer'))

# Import and run the application
from trainer.app import ModelBuilderApp
import multiprocessing as mp

# Optional torch import for environments without CUDA/torch installed
try:
    import torch
except Exception:  # torch may not be installed in minimal environments
    torch = None

if __name__ == "__main__":
    # Enable multiprocessing support on Windows
    mp.freeze_support()
    
    print("Starting ModelBuilder...")
    if torch is not None:
        gpu_available = "Yes" if torch.cuda.is_available() else "No"
    else:
        gpu_available = "No (torch not installed)"
    print("GPU Available:", gpu_available)
    print()
    
    # Create and run the application
    app = ModelBuilderApp()
    app.run()
