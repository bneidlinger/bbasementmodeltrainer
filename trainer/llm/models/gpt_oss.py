"""
GPT-OSS Model Implementation
Support for OpenAI's open-source GPT-OSS-20B and GPT-OSS-120B models
"""

import torch
from typing import Dict, Any, Optional
import logging
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)

from .base import BaseLLM, LLMConfig, GenerationConfig
from ..registry import register_llm

logger = logging.getLogger(__name__)

@register_llm("gpt-oss-20b")
class GPTOSS20B(BaseLLM):
    """
    GPT-OSS 20B model implementation.
    21B parameters total, 3.6B active per token.
    Optimized for 16GB VRAM.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            config = LLMConfig(
                model_name="openai/gpt-oss-20b",
                load_in_4bit=True,  # Default to 4-bit for efficiency
                max_length=128000,  # Native 128k context
                use_flash_attention=True
            )
        super().__init__(config)
        
        self.model_info.update({
            'total_parameters': '21B',
            'active_parameters': '3.6B',
            'architecture': 'MoE Transformer',
            'experts': '12 active out of 64',
            'vram_required': 16,  # GB
            'optimal_batch_size': 4
        })
    
    def load_model(self):
        """Load GPT-OSS-20B model with optimizations."""
        logger.info(f"Loading {self.config.model_name}...")
        
        # Configure quantization if needed
        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            logger.info("Using 4-bit quantization (NF4)")
        elif self.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Using 8-bit quantization")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
                use_fast=True
            )
            
            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map=self.config.device_map,
                torch_dtype=torch.bfloat16 if self.config.torch_dtype == "auto" else getattr(torch, self.config.torch_dtype),
                trust_remote_code=self.config.trust_remote_code,
                use_flash_attention_2=self.config.use_flash_attention
            )
            
            # Enable gradient checkpointing if specified
            if self.config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            
            # Set device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                logger.warning("CUDA not available, using CPU (will be slow)")
            
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Memory footprint: {self.get_memory_footprint()}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def prepare_for_training(self, training_config: Dict[str, Any]):
        """Prepare model for QLoRA fine-tuning."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.info("Preparing model for QLoRA training...")
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=training_config.get('lora_r', 64),
            lora_alpha=training_config.get('lora_alpha', 128),
            target_modules=training_config.get('target_modules', [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_dropout=training_config.get('lora_dropout', 0.1),
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        logger.info(f"Total params: {total_params:,}")
        
        return self.model
    
    def generate(self, 
                prompt: str, 
                generation_config: Optional[GenerationConfig] = None) -> str:
        """Generate text from prompt."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if generation_config is None:
            generation_config = GenerationConfig()
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config.max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=generation_config.max_new_tokens,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                repetition_penalty=generation_config.repetition_penalty,
                do_sample=generation_config.do_sample,
                num_beams=generation_config.num_beams,
                early_stopping=generation_config.early_stopping,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize input text."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")
        
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True
        )


@register_llm("gpt-oss-120b")
class GPTOSS120B(GPTOSS20B):
    """
    GPT-OSS 120B model implementation.
    117B parameters total, 5.1B active per token.
    Requires 60GB+ VRAM (or 80GB with MXFP4 quantization).
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            config = LLMConfig(
                model_name="openai/gpt-oss-120b",
                load_in_4bit=True,  # Essential for fitting in memory
                max_length=128000,
                use_flash_attention=True,
                gradient_checkpointing=True  # Essential for training
            )
        
        # Initialize with parent class but update model name
        super().__init__(config)
        
        self.model_info.update({
            'total_parameters': '117B',
            'active_parameters': '5.1B',
            'vram_required': 80,  # GB with quantization
            'optimal_batch_size': 1,  # Very limited batch size
            'warning': 'Requires high-end GPU (A100/H100) for efficient training'
        })


@register_llm("gpt-oss-20b-uncensored")
class GPTOSS20BUncensored(GPTOSS20B):
    """
    GPT-OSS 20B with safety features disabled.
    ⚠️ DANGER MODE - Use with extreme caution!
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self.safety_disabled = True
        self.model_info['warning'] = 'UNCENSORED MODEL - No safety filters active!'
    
    def load_model(self):
        """Load model with safety features disabled."""
        super().load_model()
        
        # Disable any safety layers if they exist
        if hasattr(self.model, 'safety_checker'):
            self.model.safety_checker = None
            logger.warning("⚠️ Safety checker DISABLED - Uncensored mode active")
        
        # Remove any content filtering in generation
        if hasattr(self.model.config, 'use_safety_checker'):
            self.model.config.use_safety_checker = False
        
        logger.warning("=" * 60)
        logger.warning("DANGER MODE ACTIVATED - SAFETY FEATURES DISABLED")
        logger.warning("You are responsible for all generated content!")
        logger.warning("=" * 60)
    
    def generate(self, prompt: str, generation_config: Optional[GenerationConfig] = None) -> str:
        """Generate without any safety filtering."""
        # Log uncensored generation
        logger.warning(f"Uncensored generation requested for prompt: {prompt[:50]}...")
        
        # Generate without safety checks
        result = super().generate(prompt, generation_config)
        
        # No post-processing or filtering
        return result