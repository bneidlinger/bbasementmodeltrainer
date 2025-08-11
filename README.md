# BasementBrewAI 🟧 Industrial ML Terminal

```
================================================================================
                                                                                
  ####    B A S E M E N T   B R E W   A I                                     
  #  #    +-----------------------------+                                     
  ####    | Industrial ML Terminal v2.0 |                                     
  #  #    +-----------------------------+                                     
  ####    NOW WITH LLM CAPABILITIES                                          
                                                                                
================================================================================
```

> *"Where neural networks meet the underground - now brewing language models"* 

[![Python](https://img.shields.io/badge/Python-3.8+-orange.svg?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-green.svg?style=flat-square)](https://pytorch.org)
[![Dear PyGui](https://img.shields.io/badge/Dear%20PyGui-1.0+-yellow.svg?style=flat-square)](https://github.com/hoffstadt/DearPyGui)
[![License](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](LICENSE)

## 🟧 System Overview

Welcome to the basement, operator. **BasementBrewAI** is a retro-themed ML/AI training terminal that combines 80's industrial computing aesthetics with modern deep learning capabilities. Now featuring **LLM fine-tuning** with GPT-OSS support!

### ⚡ Core Features

#### Traditional ML
- **Industrial Terminal Interface** - Orange & green phosphor display aesthetics
- **Real-time Training Visualization** - Watch losses drop in glorious ASCII
- **Multi-Model Support** - MLP, CNN, LeNet-5, ResNet-18/34
- **Dataset Plugin System** - MNIST, CIFAR-10, FashionMNIST + custom loaders
- **GPU Acceleration** - CUDA support with real-time VRAM monitoring
- **Model Export** - ONNX & TorchScript for production

#### LLM Capabilities (NEW!)
- **GPT-OSS-20B Support** - OpenAI's open-source model integration
- **QLoRA Fine-tuning** - 4-bit quantization for 12GB GPUs
- **Data Pipeline** - Web scraping, cleaning, tokenization
- **Safety Controls** - Danger mode with explicit warnings
- **Inference Engine** - Chat interface for testing models
- **Multi-source Scraping** - ArXiv, Reddit, web content

## 🟧 Quick Start

### Prerequisites

```bash
[SYSTEM] Checking requirements...
> Python 3.8-3.11 [REQUIRED]
> NVIDIA GPU with CUDA [RECOMMENDED]
> 8GB+ RAM [MINIMUM]
> 50GB+ Storage [FOR LLMs]
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

# Direct launch
python trainer/app.py
```

## 🟧 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   MAIN PROCESS (GUI)                    │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Dear PyGui │  │ Retro Theme  │  │  LLM UI Tab   │  │
│  │   Display   │  │  Controller  │  │   Interface   │  │
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
│  │   PyTorch   │  │  LLM/QLoRA   │  │  Data Pipeline│  │
│  │   Engine    │  │   Training   │  │   Processing  │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 🟧 Feature Modules

### Traditional ML Training
- **Models**: MLP, CNN, LeNet-5, ResNet-18/34
- **Datasets**: MNIST, CIFAR-10, FashionMNIST, custom plugins
- **Training**: Real-time monitoring, auto-save, experiment tracking
- **Testing**: Batch inference, single image prediction
- **Export**: ONNX, TorchScript formats

### LLM Fine-tuning (NEW!)
- **Models**: GPT-OSS-20B (fits 12GB VRAM with QLoRA)
- **Data Sources**: 
  - ArXiv papers
  - Reddit posts
  - Web content
- **Processing Pipeline**:
  - Tokenization (GPT-2, tiktoken)
  - Cleaning (PII removal, deduplication)
  - Formatting (Alpaca, Vicuna, ChatML)
- **Safety**: Danger mode toggle with comprehensive warnings

## 🟧 Usage Guide

### Training Traditional Models

1. **Select Dataset** - Choose from available data modules
2. **Pick Architecture** - MLP for basics, ResNet for power
3. **Configure Hyperparameters** - Epochs, batch size, learning rate
4. **Enable GPU** - Check VRAM meter
5. **START TRAINING** - Monitor in real-time

### Fine-tuning LLMs

1. **Navigate to LLM Lab** tab
2. **Configure Data Sources** - Select scrapers and queries
3. **Set QLoRA Parameters** - Rank, alpha, quantization
4. **Review Safety Settings** - Keep guardrails enabled
5. **Start Fine-tuning** - Monitor perplexity and loss

## 🟧 Model Zoo

### Computer Vision Models
| Model | Parameters | Best For |
|-------|------------|----------|
| **Simple MLP** | ~800K | Quick experiments, MNIST |
| **BasicCNN** | ~1.2M | Image classification basics |
| **LeNet-5** | ~60K | Classic CNN architecture |
| **ResNet-18** | ~11M | Serious image classification |
| **ResNet-34** | ~21M | When you need more depth |

### Language Models
| Model | Parameters | VRAM Required | Status |
|-------|------------|---------------|---------|
| **GPT-OSS-20B** | 21B (3.6B active) | 12-16GB | Ready |
| **GPT-OSS-120B** | 117B (5.1B active) | 60GB+ | Planned |

## 🟧 Data Pipeline

```python
from trainer.llm.data import DatasetManager, DatasetConfig

# Configure data collection
config = DatasetConfig(
    name="my_dataset",
    sources=["arxiv", "reddit", "web"],
    queries=["machine learning", "AI safety"],
    max_items_per_source=1000
)

# Run pipeline
manager = DatasetManager(config)
result = manager.create_dataset(dataset_type="instruction")
```

## 🟧 Command Reference

```bash
# Application
python run.py                          # Launch with header
python trainer/app.py                  # Direct launch

# Testing
python test_model_demo.py              # Test inference
python test_llm_pipeline.py            # Test LLM data pipeline

# Maintenance
python cleanup_failed_runs.py          # Interactive cleanup
python cleanup_auto.py                 # Auto cleanup
python force_clean_runs.py             # Force clean all
python kill_stuck_runs.py              # Kill stuck processes

# Database
python trainer/db.py                   # Test database operations

# Build
pyinstaller --onefile --windowed --icon=trainer\assets\icon.ico --name "BasementBrewAI" trainer\app.py
```

## 🟧 System Requirements

### Minimum
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8 - 3.11
- **RAM**: 8GB
- **Storage**: 10GB
- **GPU**: Optional (CPU training supported)

### Recommended (for LLMs)
- **RAM**: 32GB+
- **GPU**: NVIDIA RTX 3070 Ti or better (12GB+ VRAM)
- **Storage**: 100GB+ (for models and datasets)
- **CUDA**: 11.8 or 12.4

## 🟧 Project Structure

```
BasementBrewAI/
├── trainer/
│   ├── app.py                 # Main application
│   ├── llm/                   # LLM module
│   │   ├── models/            # Model implementations
│   │   ├── training/          # Training pipelines
│   │   ├── data/              # Data processing
│   │   │   ├── scrapers/      # Web scrapers
│   │   │   └── processors/    # Tokenizers, cleaners
│   │   ├── inference/         # Generation engine
│   │   └── safety/            # Danger mode controls
│   ├── models/                # Traditional ML models
│   ├── data/                  # Dataset loaders
│   └── assets/                # Icons and resources
├── saved_models/              # Trained model storage
├── datasets/                  # Processed datasets
├── experiments.db             # Training history
└── requirements.txt           # Dependencies
```

## 🟧 Troubleshooting

### Common Issues

**Q: Stuck training runs won't terminate**  
A: Fixed! Auto-cleanup on startup + enhanced kill functions

**Q: GUI shows question marks**  
A: All Unicode replaced with ASCII blocks

**Q: LLM training OOM errors**  
A: Enable QLoRA, reduce batch size, use gradient accumulation

**Q: Data pipeline fails**  
A: Check internet connection, install all dependencies from requirements.txt

## 🟧 Recent Updates

### v2.0 - LLM Integration
- ✅ GPT-OSS-20B support with QLoRA
- ✅ Complete data pipeline (scraping → cleaning → formatting)
- ✅ Safety controls and danger mode
- ✅ Enhanced process management
- ✅ Startup auto-cleanup for stuck runs

### v1.0 - Initial Release
- ✅ Multi-process architecture
- ✅ Retro terminal interface
- ✅ Traditional ML models
- ✅ Real-time training visualization

## 🟧 Contributing

Pull requests welcome! Please maintain:
- ASCII art only (no Unicode except where necessary)
- Orange/green color scheme
- Industrial terminal theme
- Comprehensive error handling
- Safety-first approach for LLM features

## 🟧 License

MIT License - See [LICENSE](LICENSE) file

## 🟧 Acknowledgments

- Built with [Dear PyGui](https://github.com/hoffstadt/DearPyGui)
- Powered by [PyTorch](https://pytorch.org)
- LLM support via [Transformers](https://huggingface.co/transformers)
- Inspired by 80's industrial control systems
- ASCII art hand-crafted in the basement

---

```
[SYSTEM] BasementBrewAI v2.0 initialized
[STATUS] Ready for neural network operations
[MODE]   Industrial ML Terminal + LLM Laboratory
[SAFETY] Guardrails ENABLED
>>> _
```

*Remember: Real hackers train in the basement. Now with language models.* 🟧