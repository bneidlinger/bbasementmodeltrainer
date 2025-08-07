"""
Base LLM class for all language models
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import torch
from dataclasses import dataclass

@dataclass
class LLMConfig:
    """Configuration for LLM models."""
    model_name: str
    model_path: Optional[str] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    device_map: str = "auto"
    max_length: int = 2048
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False
    rope_scaling: Optional[Dict] = None
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    use_cache: bool = True

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = False

class BaseLLM(ABC):
    """Abstract base class for LLM models."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_info = {
            'name': config.model_name,
            'parameters': 0,
            'vram_required': 0,
            'supports_qlora': True,
            'context_length': config.max_length
        }
    
    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def prepare_for_training(self, training_config: Dict[str, Any]):
        """Prepare model for fine-tuning."""
        pass
    
    @abstractmethod
    def generate(self, 
                prompt: str, 
                generation_config: Optional[GenerationConfig] = None) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize input text."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and requirements."""
        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            self.model_info['parameters'] = total_params
            
            # Estimate VRAM (rough calculation)
            bytes_per_param = 2 if self.config.load_in_8bit else 4
            if self.config.load_in_4bit:
                bytes_per_param = 0.5
            
            vram_gb = (total_params * bytes_per_param) / (1024**3)
            self.model_info['vram_required'] = vram_gb
        
        return self.model_info
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded")
        
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        self.model.save_pretrained(path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(path)
    
    def get_memory_footprint(self) -> Dict[str, float]:
        """Get current memory usage."""
        if not torch.cuda.is_available():
            return {'ram_gb': 0, 'vram_gb': 0}
        
        # Get VRAM usage
        vram_bytes = torch.cuda.memory_allocated()
        vram_gb = vram_bytes / (1024**3)
        
        # Get RAM usage (approximate)
        import psutil
        process = psutil.Process()
        ram_bytes = process.memory_info().rss
        ram_gb = ram_bytes / (1024**3)
        
        return {
            'ram_gb': ram_gb,
            'vram_gb': vram_gb,
            'vram_reserved_gb': torch.cuda.memory_reserved() / (1024**3)
        }