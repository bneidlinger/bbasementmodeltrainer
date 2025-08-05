
Dear PyGui PyTorch Trainer – Custom Model Prototyping Guide
===========================================================

*   Companion document to **Dear PyGui PyTorch Trainer**. Focus: enabling users to rapidly prototype, integrate, and experiment with their own PyTorch model architectures.

1\. Key Objectives
------------------

*   **Rapid Prototyping**: Allow users to define a new model architecture in a single Python file and use it in the trainer immediately.
*   **Simple Integration**: Automatically discover and load any custom model placed in a designated `models/` directory.
*   **Flexible Architecture**: Support a variety of common architectures, from simple MLPs to more complex CNNs with custom blocks.
*   **Clear Separation**: Keep user-defined model code separate from the core application logic.
*   **Seamless UI Integration**: Custom models should populate a dropdown menu in the GUI, just like built-in models.

2\. Proposed File Structure
---------------------------

    trainer/
    ├── models/
    │   ├── __init__.py
    │   ├── mlp.py
    │   ├── simple_cnn.py
    │   └── custom_resnet_block.py
    ├── app.py
    ├── train_worker.py
    └── ... (other files)

3\. Model Registration Pattern
------------------------------

Similar to the data loader, a registry pattern can be used to discover models.

**`models/__init__.py`**:

    import importlib
    import pkgutil

    # Dictionary to hold model names and their classes
    MODEL_REGISTRY = {}

    def register_model(name):
        """A decorator to add model classes to the registry."""
        def decorator(cls):
            MODEL_REGISTRY[name] = cls
            return cls
        return decorator

    def discover_models():
        """Finds and imports all models in the 'models' package."""
        for (_, name, _) in pkgutil.iter_modules(__path__):
            importlib.import_module(f".{name}", __package__)

    # Run discovery when the package is imported
    discover_models()

4\. Creating a Custom Model
---------------------------

Users can create a new `.py` file in the `models/` directory and use the `@register_model` decorator. The base class must be `torch.nn.Module`.

### Example 1: Simple Multi-Layer Perceptron (MLP)

**`models/mlp.py`**:

    import torch
    import torch.nn as nn
    from . import register_model

    @register_model("Simple MLP")
    class MLP(nn.Module):
        def __init__(self, input_size=784, hidden_size=128, output_size=10):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )

        def forward(self, x):
            # Flatten image tensors
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            return self.layers(x)

### Example 2: Basic CNN Block

This demonstrates creating reusable modules for more complex models.

**`models/simple_cnn.py`**:

    import torch.nn as nn
    from . import register_model

    def conv_block(in_channels, out_channels, kernel_size=3):
        """Returns a block of Conv -> BatchNorm -> ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    @register_model("Basic CNN")
    class BasicCNN(nn.Module):
        def __init__(self, in_channels=1, num_classes=10):
            super().__init__()
            self.block1 = conv_block(in_channels, 16)
            self.block2 = conv_block(16, 32)
            self.classifier = nn.Linear(32 * 28 * 28, num_classes) # Assuming 28x28 input

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = x.view(x.size(0), -1) # Flatten for classifier
            return self.classifier(x)

5\. Integration with the Trainer App
------------------------------------

1.  **Model Discovery**: The main application will call `models.discover_models()` at startup.
2.  **GUI Population**: The keys from `models.MODEL_REGISTRY` will be used to populate the model selection dropdown in the UI.
3.  **Instantiation**: When the user clicks "Start Training", the application will retrieve the selected model class from the registry and instantiate it: `model_class = MODEL_REGISTRY[selected_model_name]`
4.  **Training**: The instantiated model is passed to the `Training Worker` process, which then proceeds with the training loop.

6\. Data Flow for Model Selection
---------------------------------

    ┌───────────────────────────┐
    │     Trainer Starts Up     │
    └─────────────┬─────────────┘
                  │ Calls models.discover_models()
    ┌─────────────▼─────────────┐
    │  Populates GUI Dropdown   │
    │  with MODEL_REGISTRY keys │
    └─────────────┬─────────────┘
                  │ User selects "Simple MLP"
                  │ and clicks "Start"
    ┌─────────────▼─────────────┐
    │      app.py Gets Class    │
    │ `MLP = REGISTRY["Simple MLP"]` │
    │   `model = MLP(params)`   │
    └─────────────┬─────────────┘
                  │  Passes model instance
                  │    to Training Worker
    ┌─────────────▼─────────────┐
    │     Training Process      │
    │  Receives PyTorch object  │
    └───────────────────────────┘

7\. Next Steps
--------------

*   Develop a parameter manager that allows the GUI to ask a model for its required parameters (e.g., `input_size`, `num_classes`) and dynamically generate UI widgets for them.
*   Implement a "Create New Model" button in the UI that generates a template `.py` file in the `models/` directory.
*   Add error handling for model loading and instantiation to provide clear feedback to the user.
*   Extend the `runs` table in the SQLite database to log model architecture details, perhaps by saving the model's string representation.