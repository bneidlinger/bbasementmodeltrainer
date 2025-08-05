# Dear PyGui PyTorch Trainer - Requirements for Windows

# -----------------------------------------------------------------------------
# Core Application & GUI
# -----------------------------------------------------------------------------
# For the main application window and widgets
dearpygui==1.*

# For packaging the application into a single executable
pyinstaller==6.*

# -----------------------------------------------------------------------------
# Data Handling
# -----------------------------------------------------------------------------
# Used for handling tabular datasets (e.g., from Kaggle, UCI)
# and for convenient data manipulation.
pandas==2.*

# Used by some dataset loaders (e.g., OpenML)
scikit-learn==1.5.*

# -----------------------------------------------------------------------------
# PyTorch (Machine Learning)
# -----------------------------------------------------------------------------
# IMPORTANT: The installation of PyTorch with GPU (CUDA) support is best
# handled by a specific command. Do NOT just run "pip install torch".
#
# 1. First, uninstall any existing versions of PyTorch:
#    pip uninstall torch torchvision torchaudio
#
# 2. Visit the PyTorch official website to get the correct command:
#    https://pytorch.org/get-started/locally/
#
# 3. Select the appropriate options (e.g., Stable, Windows, Pip, Python, CUDA version).
#    As of late 2025, the command for CUDA 12.1 is typically:
#
#    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#
# Because the line above needs to be run manually, `torch` is commented out below.
# If you are running on a machine without an NVIDIA GPU (CPU only), you can
# uncomment the following lines.
#
# torch==2.*
# torchvision==0.17.*
# torchaudio==2.*