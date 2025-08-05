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