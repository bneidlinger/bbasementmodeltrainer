"""
Text Generation and Inference for LLM Models
Handles loading trained models and generating text
"""

import os
import torch
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig as HFGenerationConfig
)
from peft import PeftModel

logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """Configuration for inference."""
    model_path: str  # Path to saved model/checkpoint
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    load_in_4bit: bool = True
    max_memory: Optional[Dict] = None
    torch_dtype: str = "auto"
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1
    
    # Safety
    apply_safety_filter: bool = True

class LLMInference:
    """Manages LLM model loading and text generation."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.model_type = None
        self.base_model_name = None
        
    def load_model(self, checkpoint_path: str = None):
        """
        Load a trained model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint. If None, uses config.model_path
        """
        model_path = checkpoint_path or self.config.model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        
        try:
            # Check if this is a LoRA checkpoint or full model
            is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
            
            if is_lora:
                self._load_lora_model(model_path)
            else:
                self._load_full_model(model_path)
            
            self.is_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_lora_model(self, checkpoint_path: str):
        """Load a LoRA fine-tuned model."""
        import json
        
        # Read adapter config to get base model
        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        base_model_name = adapter_config.get("base_model_name_or_path")
        if not base_model_name:
            # Try to infer from training config if saved
            training_config_path = os.path.join(checkpoint_path, "training_config.json")
            if os.path.exists(training_config_path):
                with open(training_config_path, 'r') as f:
                    training_config = json.load(f)
                base_model_name = training_config.get("model_name", "gpt2")  # Default to gpt2
            else:
                base_model_name = "gpt2"  # Fallback default
        
        self.base_model_name = base_model_name
        logger.info(f"Loading base model: {base_model_name}")
        
        # Configure quantization if needed
        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.config.device == "cuda" else None,
            torch_dtype=torch.bfloat16 if self.config.torch_dtype == "auto" else getattr(torch, self.config.torch_dtype),
            trust_remote_code=True
        )
        
        # Load LoRA weights
        logger.info(f"Loading LoRA weights from {checkpoint_path}")
        self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
        
        # Move to device if not using device_map
        if self.config.device == "cpu":
            self.model = self.model.to(self.config.device)
    
    def _load_full_model(self, model_path: str):
        """Load a full model checkpoint."""
        logger.info(f"Loading full model from {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization if needed
        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto" if self.config.device == "cuda" else None,
            torch_dtype=torch.bfloat16 if self.config.torch_dtype == "auto" else getattr(torch, self.config.torch_dtype),
            trust_remote_code=True
        )
        
        if self.config.device == "cpu":
            self.model = self.model.to(self.config.device)
    
    def generate(self, 
                prompt: str,
                max_new_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                top_k: Optional[int] = None,
                stream: bool = False) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Override config max tokens
            temperature: Override config temperature
            top_p: Override config top_p
            top_k: Override config top_k
            stream: Whether to stream tokens (not implemented yet)
            
        Returns:
            Generated text
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        # Move to device
        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Set generation parameters
        gen_config = HFGenerationConfig(
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            temperature=temperature or self.config.temperature,
            top_p=top_p or self.config.top_p,
            top_k=top_k or self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            do_sample=self.config.do_sample,
            num_beams=self.config.num_beams,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Apply safety filter if enabled
        if self.config.apply_safety_filter:
            generated_text = self._apply_safety_filter(generated_text)
        
        return generated_text
    
    def _apply_safety_filter(self, text: str) -> str:
        """Apply basic safety filtering to generated text."""
        # This is a placeholder for more sophisticated filtering
        # In production, you'd want to use a proper content filter
        
        # Basic profanity filter (very simple example)
        blocked_words = []  # Add blocked words if needed
        
        for word in blocked_words:
            if word.lower() in text.lower():
                return "[Content filtered for safety]"
        
        return text
    
    def chat(self, 
             message: str,
             conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Chat with the model, maintaining conversation context.
        
        Args:
            message: User message
            conversation_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            
        Returns:
            Model response
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Build conversation prompt
        prompt = self._build_chat_prompt(message, conversation_history)
        
        # Generate response
        response = self.generate(prompt)
        
        return response
    
    def _build_chat_prompt(self, 
                          message: str,
                          history: Optional[List[Dict[str, str]]] = None) -> str:
        """Build a chat prompt with conversation history."""
        # Simple format, can be customized based on model
        prompt_parts = []
        
        if history:
            for msg in history[-10:]:  # Keep last 10 messages for context
                role = msg['role']
                content = msg['content']
                if role == 'user':
                    prompt_parts.append(f"User: {content}")
                else:
                    prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append(f"User: {message}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"status": "No model loaded"}
        
        info = {
            "status": "Model loaded",
            "base_model": self.base_model_name or "Unknown",
            "device": str(self.config.device),
            "quantization": "4-bit" if self.config.load_in_4bit else "Full precision"
        }
        
        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            info["total_parameters"] = f"{total_params:,}"
            info["trainable_parameters"] = f"{trainable_params:,}"
        
        return info
    
    def unload_model(self):
        """Unload the model to free memory."""
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        self.is_loaded = False
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded")