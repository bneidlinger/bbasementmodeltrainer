# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

B's BasementBrewAI (formerly ModelBuilder) is a retro-themed desktop GUI application for machine learning training using PyTorch and Dear PyGui. The project provides an industrial terminal-style interface for beginners to train ML models with a distinctive 80's aesthetic.

## Current State

The project is fully implemented with:
- Complete training and testing UI
- Model registry system with built-in models (MLP, CNN, ResNet)
- Dataset loading system with plugin architecture
- Real-time training visualization
- Model inference and export capabilities
- Retro industrial theme with orange/green color scheme

## Architecture

### Multi-Process Design
- **Main Process**: Dear PyGui GUI (`app.py`) - handles UI and user interaction
- **Worker Process**: PyTorch training (`train_worker.py`) - runs training in isolation
- **Communication**: multiprocessing.Queue for real-time updates
- **Storage**: SQLite database (`experiments.db`) for experiment tracking

### Key Components

1. **`trainer/app.py`** - Main application entry point
   - Creates tabbed interface (Training, Testing)
   - Manages training lifecycle and monitoring
   - Implements retro theme with ASCII art headers

2. **`trainer/train_worker.py`** - Training process
   - Runs in separate process to prevent GUI freezing
   - Sends progress updates via Queue
   - Saves models to `saved_models/` directory
   - Handles graceful shutdown and error reporting

3. **`trainer/db.py`** - Database operations
   - Tables: `runs`, `datasets`, `metrics`
   - Tracks experiment history and metrics
   - Provides context manager for safe operations

4. **Model System** (`trainer/models/`)
   - Registry pattern with `@register_model` decorator
   - Auto-discovery of models on startup
   - Built-in: MLP, BasicCNN, LeNet-5, ResNet18/34

5. **Data System** (`trainer/data/`)
   - Registry pattern for datasets
   - Built-in loaders: CIFAR-10, MNIST, FashionMNIST
   - Plugin architecture in `data/plugins/`

6. **Testing/Inference** (`trainer/test_ui.py`, `trainer/inference.py`)
   - Model loading and inference
   - Single image prediction
   - Batch testing on datasets
   - Export to ONNX/TorchScript

7. **Theme System** (`trainer/retro_theme.py`, `trainer/ascii_blocks.py`)
   - Industrial orange (#FF8C00) and IBM green (#32CD32)
   - Custom button themes
   - ASCII art headers with block characters

## Commands

### Development
- **Run with retro branding**: `run_basement.bat`
- **Run standard**: `python run.py` or `venv\Scripts\python trainer\app.py`
- **Activate venv**: `venv\Scripts\activate`

### Dependencies
- **Install all**: `venv\Scripts\pip install -r requirements.txt`
- **Install PyTorch GPU**: `venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`

### Testing Components
- **Test database**: `venv\Scripts\python trainer\db.py`
- **Test training worker**: `venv\Scripts\python trainer\train_worker.py`
- **Test inference**: `python test_model_demo.py`

### Maintenance
- **Clean failed runs**: `python cleanup_failed_runs.py` (interactive)
- **Auto cleanup**: `python cleanup_auto.py` (marks stuck runs as failed)

### Packaging
- **Build executable**: `venv\Scripts\pyinstaller --onefile --windowed --icon=trainer\assets\icon.ico --name "BasementBrewAI" trainer\app.py`

## Process Management

The app implements robust process handling:
- 5-minute timeout for unresponsive training
- Force termination after 3 seconds on stop
- Automatic cleanup on app exit
- Database status tracking (running/completed/failed)

## Error Handling Patterns

- Training errors are caught and logged to database
- Failed runs are marked with error messages in notes
- Queue timeouts trigger automatic cleanup
- GPU availability is checked before training

## Theme Customization

The retro theme uses:
- `COLORS` dict in `retro_theme.py` for all color values
- ASCII art in `ascii_blocks.py` using only standard characters
- Status prefixes: `[LOADING]`, `[ACTIVE]`, `[ERROR]`, `[TERMINATED]`
- Icon prefixes for buttons: `>`, `[]`, `>>`, `(R)`

## Model Save Format

Models are saved to `saved_models/` with:
- State dict: `run_{id}_model.pth`
- Full model: `run_{id}_model_full.pth`
- Metadata includes: dataset, accuracy, parameters, training config

## Important Notes

### Virtual Environment
- Always use the virtual environment: `venv\Scripts\activate`
- PyTorch installation requires special handling - see requirements.txt for GPU/CPU instructions

### File Creation Policy
- DO NOT create new files unless explicitly necessary
- NEVER create documentation files (*.md) unless explicitly requested
- Always prefer editing existing files over creating new ones

### Testing
- No formal test suite exists - use component testing commands above
- The `test_model_demo.py` script demonstrates inference capabilities

### Database Cleanup
- Failed/stuck runs need periodic cleanup using the cleanup scripts
- Training processes have a 5-minute timeout for safety