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

if __name__ == "__main__":
    # Enable multiprocessing support on Windows
    mp.freeze_support()
    
    print("Starting ModelBuilder...")
    print("GPU Available:", "Yes" if __import__('torch').cuda.is_available() else "No")
    print()
    
    # Create and run the application
    app = ModelBuilderApp()
    app.run()