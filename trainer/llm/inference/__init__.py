"""
LLM Inference Module
Handles model loading and text generation for chat interface
"""

from .generation import LLMInference, InferenceConfig

__all__ = ['LLMInference', 'InferenceConfig']