# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BasementBrewAI (formerly ModelBuilder) is a retro-themed desktop GUI application for machine learning training using PyTorch and Dear PyGui. The project provides an industrial terminal-style interface with a distinctive 80's aesthetic (orange/green color scheme, ASCII art headers).

## Architecture

### Multi-Process Design
The application uses a two-process architecture to prevent GUI freezing:
- **Main Process**: Dear PyGui GUI (`trainer/app.py`) - handles UI and user interaction
- **Worker Process**: PyTorch training (`trainer/train_worker.py`) - runs training in isolation
- **Communication**: `multiprocessing.Queue` for real-time updates between processes
- **Storage**: SQLite database (`experiments.db`) for experiment tracking

### Registry Pattern
Both models and datasets use auto-registration:
- Models: `@register_model()` decorator in `trainer/models/__init__.py`
- Datasets: `@register_dataset()` decorator in `trainer/data/core.py`
- Auto-discovery on startup using `pkgutil.iter_modules()`

### Key Components

1. **`trainer/app.py`** - Main GUI application
   - Tabbed interface: Training, Testing, Comparison, LLM Lab
   - Process lifecycle management with 5-minute timeout
   - Real-time training visualization

2. **`trainer/train_worker.py`** - Isolated training process
   - Sends progress updates every 10 batches via Queue
   - Saves models to `saved_models/` directory
   - Handles graceful shutdown and error reporting

3. **`trainer/db.py`** - Database operations
   - Tables: `runs`, `datasets`, `metrics`
   - Context manager for safe transactions
   - Run status tracking: running/completed/failed

4. **Model System** (`trainer/models/`)
   - Built-in: MLP, BasicCNN, LeNet-5, ResNet18/34
   - Models provide `get_config_options()` for GUI integration

5. **Data System** (`trainer/data/`)
   - Built-in loaders: CIFAR-10, MNIST, FashionMNIST
   - Plugin architecture in `data/plugins/`

## Commands

### Development
```bash
# Run application
python run.py                    # Standard launch
run_basement.bat                # Retro-themed launch (Windows)
venv\Scripts\python trainer\app.py  # Direct launch

# Virtual environment
venv\Scripts\activate           # Always activate before development
```

### Dependencies
```bash
# Install all dependencies
venv\Scripts\pip install -r requirements.txt

# PyTorch GPU (if needed)
venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Testing Components
```bash
venv\Scripts\python trainer\db.py          # Test database operations
venv\Scripts\python trainer\train_worker.py  # Test training worker
python test_model_demo.py                   # Test inference capabilities
```

### Maintenance
```bash
python cleanup_auto.py          # Auto-cleanup stuck runs (non-interactive)
python cleanup_failed_runs.py   # Interactive cleanup with prompts
python force_clean_runs.py      # Force cleanup of all stuck runs
python kill_stuck_runs.py       # Kill stuck training processes
```

### Packaging
```bash
venv\Scripts\pyinstaller --onefile --windowed --icon=trainer\assets\icon.ico --name "BasementBrewAI" trainer\app.py
```

## Process Management

The app implements robust process handling:
- **5-minute timeout** for unresponsive training processes
- **3-second grace period** before force termination on stop
- **Automatic cleanup** on app exit
- **Database status tracking** prevents stuck processes

### Queue Message Types
- `'status'`: Status updates for GUI
- `'batch_update'`: Progress during training
- `'epoch_complete'`: End-of-epoch metrics
- `'complete'`: Training finished successfully
- `'error'`: Training failed with error details
- `'stop'`: Signal from GUI to halt training

## Theme System

Located in `trainer/retro_theme.py` and `trainer/ascii_blocks.py`:
- **Colors**: Industrial orange (#FF8C00) + IBM green (#32CD32)
- **Status Prefixes**: `[LOADING]`, `[ACTIVE]`, `[ERROR]`, `[TERMINATED]`
- **Button Icons**: `>`, `[]`, `>>`, `(R)` for different actions
- **ASCII Art**: Professional block characters only

## Model Save Format

Models saved to `saved_models/` with:
- State dict: `run_{id}_model.pth`
- Full model: `run_{id}_model_full.pth`
- Metadata: dataset, accuracy, parameters, training config

## LLM Integration

Recent additions in `trainer/llm_ui.py` and `trainer/llm/`:
- QLoRA fine-tuning support
- Data scraping (ArXiv, Reddit, web)
- Danger mode controller for experimental features
- Model comparison interface

## Important Patterns

### Error Handling
- Training errors caught and logged to database
- Failed runs marked with error messages in notes
- Queue timeouts trigger automatic cleanup
- GPU availability checked before training

### Virtual Environment
- **Critical**: Always use `venv\Scripts\activate`
- PyTorch installation requires special handling for GPU support
- See requirements.txt for detailed instructions

### Testing
- No formal test suite - use component testing commands
- `test_model_demo.py` demonstrates inference capabilities
- Manual testing through UI interaction

## File Creation Policy
- DO NOT create new files unless explicitly necessary
- NEVER create documentation files (*.md) unless explicitly requested
- Always prefer editing existing files over creating new ones