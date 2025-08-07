"""
LLM Model Registry System
Manages registration and loading of language models
"""

from typing import Dict, Any, Optional, Type
import logging

logger = logging.getLogger(__name__)

# Global registry for LLM models
LLM_REGISTRY: Dict[str, Type] = {}

def register_llm(name: str):
    """
    Decorator to register an LLM model class.
    
    Usage:
        @register_llm("gpt-oss-20b")
        class GPTOSS20B(BaseLLM):
            ...
    """
    def decorator(cls):
        if name in LLM_REGISTRY:
            logger.warning(f"Model {name} already registered, overwriting...")
        
        LLM_REGISTRY[name] = cls
        logger.info(f"Registered LLM model: {name}")
        return cls
    
    return decorator

def get_llm(name: str, **kwargs):
    """
    Get an LLM model instance by name.
    
    Args:
        name: Registered model name
        **kwargs: Arguments to pass to model constructor
    
    Returns:
        Instantiated LLM model
    """
    if name not in LLM_REGISTRY:
        available = ", ".join(LLM_REGISTRY.keys())
        raise ValueError(f"Model {name} not found. Available: {available}")
    
    model_class = LLM_REGISTRY[name]
    return model_class(**kwargs)

def list_available_llms() -> list:
    """List all registered LLM models."""
    return list(LLM_REGISTRY.keys())

# Auto-import all model definitions to register them
def auto_import_models():
    """Auto-import model definitions to populate registry."""
    import os
    import importlib
    import sys
    
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(models_dir):
        return
    
    for filename in os.listdir(models_dir):
        if filename.endswith('.py') and not filename.startswith('_'):
            module_name = filename[:-3]
            try:
                # Try relative import first
                try:
                    importlib.import_module(f'.models.{module_name}', package='llm')
                except:
                    # Fallback to absolute import
                    importlib.import_module(f'llm.models.{module_name}')
                logger.debug(f"Imported LLM module: {module_name}")
            except Exception as e:
                logger.warning(f"Failed to import {module_name}: {e}")

# Run auto-import when module is loaded
auto_import_models()