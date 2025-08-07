"""
QLoRA (Quantized Low-Rank Adaptation) implementation
Efficient fine-tuning for large language models
"""

import torch
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class QLoRAConfig:
    """Configuration for QLoRA fine-tuning."""
    # LoRA parameters
    r: int = 64  # Rank
    lora_alpha: int = 128  # Alpha scaling parameter
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    
    # Quantization parameters
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # Training parameters
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3
    
    def __post_init__(self):
        """Set default target modules if not specified."""
        if self.target_modules is None:
            # Common attention/MLP modules for transformers
            self.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj",      # MLP
                "lm_head"  # Output layer
            ]

def prepare_model_for_qlora(model, config: QLoRAConfig):
    """
    Prepare a model for QLoRA training.
    
    Args:
        model: The base model to prepare
        config: QLoRA configuration
    
    Returns:
        Prepared model with LoRA adapters
    """
    try:
        from peft import (
            prepare_model_for_kbit_training,
            LoraConfig,
            get_peft_model,
            TaskType
        )
    except ImportError:
        raise ImportError("Please install peft: pip install peft")
    
    logger.info("Preparing model for QLoRA training...")
    
    # Prepare model for k-bit training (handles quantization)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.gradient_checkpointing
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params
    
    logger.info(f"QLoRA Configuration:")
    logger.info(f"  Rank (r): {config.r}")
    logger.info(f"  Alpha: {config.lora_alpha}")
    logger.info(f"  Dropout: {config.lora_dropout}")
    logger.info(f"  Target modules: {config.target_modules}")
    logger.info(f"Model Statistics:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable: {trainable_percent:.2f}%")
    
    return model

def get_quantization_config(config: QLoRAConfig):
    """
    Get BitsAndBytes quantization configuration.
    
    Args:
        config: QLoRA configuration
    
    Returns:
        BitsAndBytesConfig for model loading
    """
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")
    
    if not config.load_in_4bit:
        return None
    
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )

def calculate_memory_requirements(model_params: int, config: QLoRAConfig) -> Dict[str, float]:
    """
    Estimate memory requirements for QLoRA training.
    
    Args:
        model_params: Number of model parameters
        config: QLoRA configuration
    
    Returns:
        Dictionary with memory estimates in GB
    """
    # Base model memory (4-bit quantization)
    if config.load_in_4bit:
        model_memory = model_params * 0.5 / 1e9  # 0.5 bytes per param
    else:
        model_memory = model_params * 2 / 1e9  # 2 bytes per param (fp16)
    
    # LoRA adapter memory (rough estimate)
    # Assuming ~0.1% of model params for typical LoRA config
    lora_memory = model_params * 0.001 * 4 / 1e9  # 4 bytes per param (fp32)
    
    # Gradient and optimizer memory
    # For AdamW: ~8 bytes per trainable param (momentum + variance)
    trainable_params = model_params * 0.001  # Rough estimate
    optimizer_memory = trainable_params * 8 / 1e9
    
    # Activation memory (very rough estimate, depends on batch size)
    activation_memory = 2.0  # GB, typical for batch_size=1-4
    
    return {
        'model': model_memory,
        'lora_adapters': lora_memory,
        'optimizer': optimizer_memory,
        'activations': activation_memory,
        'total_estimated': model_memory + lora_memory + optimizer_memory + activation_memory
    }