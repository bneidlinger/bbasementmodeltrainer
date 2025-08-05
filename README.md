# B's BasementBrewAI 🟧 Industrial ML Terminal

```
================================================================================
                                                                                
  ####    B A S E M E N T   B R E W   A I                                     
  #  #    +-----------------------------+                                     
  ####    | Industrial ML Terminal v1.0 |                                     
  #  #    +-----------------------------+                                     
  ####                                                                         
                                                                                
================================================================================
```

> *"Where neural networks meet the underground"* 

[![Python](https://img.shields.io/badge/Python-3.8+-orange.svg?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-green.svg?style=flat-square)](https://pytorch.org)
[![Dear PyGui](https://img.shields.io/badge/Dear%20PyGui-1.0+-yellow.svg?style=flat-square)](https://github.com/hoffstadt/DearPyGui)
[![License](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](LICENSE)

## 🟧 System Overview

Welcome to the basement, operator. This is **BasementBrewAI** - a retro-themed neural network training terminal built for those who appreciate the aesthetics of 80's industrial computing while harnessing modern deep learning power.

### ⚡ Features

- **Industrial Terminal Interface** - Orange & green phosphor display aesthetics
- **Real-time Training Visualization** - Watch your losses drop in glorious ASCII
- **Multi-Model Support** - MLP, CNN, ResNet architectures ready to deploy
- **Dataset Plugin System** - MNIST, CIFAR-10, FashionMNIST + extensible framework
- **GPU Acceleration** - CUDA support for serious number crunching
- **Model Export** - ONNX & TorchScript for production deployment

## 🟧 Quick Start

### Prerequisites

```bash
[SYSTEM] Checking requirements...
> Python 3.8+ [REQUIRED]
> NVIDIA GPU with CUDA [OPTIONAL]
> 4GB+ RAM [RECOMMENDED]
```

### Installation

```bash
# Clone the basement
git clone https://github.com/yourusername/BasementBrewAI.git
cd BasementBrewAI

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Launch Terminal

```bash
# Windows - Full retro experience
run_basement.bat

# Cross-platform
python run.py
```

## 🟧 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   MAIN PROCESS (GUI)                    │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Dear PyGui │  │ Retro Theme  │  │ ASCII Header  │  │
│  │   Display   │  │  Controller  │  │   Generator   │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│         │                                      │         │
│         └──────────────┬───────────────────────┘         │
│                        │                                 │
│                   ┌────▼────┐                            │
│                   │ Queue   │                            │
│                   │ Manager │                            │
│                   └────┬────┘                            │
└────────────────────────┼────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                 WORKER PROCESS (Training)               │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   PyTorch   │  │   Dataset    │  │     Model     │  │
│  │   Engine    │  │   Loaders    │  │   Registry    │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 🟧 Usage Guide

### Training a Model

1. **Select Dataset** - Choose from available data modules
2. **Pick Architecture** - MLP for basics, ResNet for power
3. **Configure Hyperparameters** - Epochs, batch size, learning rate
4. **Enable GPU** - If available (check that sweet VRAM meter)
5. **START TRAINING** - Watch the magic happen

### Testing & Inference

Switch to the `TESTING` tab to:
- Load trained models
- Upload custom images for prediction
- Batch test on entire datasets
- Export to ONNX/TorchScript

## 🟧 Model Zoo

| Model | Parameters | Best For |
|-------|------------|----------|
| **Simple MLP** | ~800K | Quick experiments, MNIST |
| **BasicCNN** | ~1.2M | Image classification basics |
| **LeNet-5** | ~60K | Classic CNN architecture |
| **ResNet-18** | ~11M | Serious image classification |
| **ResNet-34** | ~21M | When you need more depth |

## 🟧 Extending the System

### Adding Custom Models

```python
# trainer/models/your_model.py
from models import register_model

@register_model('YourModel')
class YourModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Your architecture here
```

### Creating Dataset Plugins

```python
# trainer/data/plugins/your_dataset.py
from data.core import BaseDataset, register_dataset

@register_dataset('YourDataset')
class YourDataset(BaseDataset):
    def load(self):
        # Return train, val datasets
```

## 🟧 Command Reference

```bash
# Development
venv\Scripts\python trainer\app.py    # Run application
venv\Scripts\python trainer\db.py     # Test database
venv\Scripts\python test_model_demo.py # Demo inference

# Maintenance  
python cleanup_failed_runs.py          # Clean failed runs
python cleanup_auto.py                 # Auto cleanup stuck runs

# Build executable
venv\Scripts\pyinstaller --onefile --windowed --name "BasementBrewAI" trainer\app.py
```

## 🟧 System Requirements

- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8 - 3.11
- **RAM**: 4GB minimum, 8GB recommended  
- **GPU**: NVIDIA with CUDA support (optional)
- **Storage**: 2GB for application + dataset cache

## 🟧 Troubleshooting

### Common Issues

**Q: GUI shows question marks instead of graphics**  
A: Fixed in v1.0 - all Unicode replaced with ASCII

**Q: Training gets stuck**  
A: 5-minute timeout protection built-in, run cleanup scripts

**Q: Can't see saved models**  
A: Check `saved_models/` directory or use Testing tab

## 🟧 Contributing

Pull requests welcome! Please maintain the retro aesthetic:
- Use ASCII art only (no Unicode)
- Keep the orange/green color scheme
- Follow the industrial terminal theme

## 🟧 License

MIT License - See [LICENSE](LICENSE) file

## 🟧 Acknowledgments

- Built with [Dear PyGui](https://github.com/hoffstadt/DearPyGui)
- Powered by [PyTorch](https://pytorch.org)
- Inspired by 80's industrial control systems
- ASCII art hand-crafted in the basement

---

```
[SYSTEM] README.md loaded successfully
[STATUS] Ready for neural network operations
[MODE]   Industrial ML Terminal v1.0
>>> _
```

*Remember: Real hackers train in the basement.* 🟧