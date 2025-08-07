"""
Main LLM Training Pipeline
Handles the complete training workflow for language models
"""

import os
import json
import torch
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import multiprocessing as mp
from queue import Empty
import traceback

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)

from .qlora import QLoRAConfig, prepare_model_for_qlora
from ..models.base import BaseLLM
from ..safety.danger_mode import DangerModeController

logger = logging.getLogger(__name__)

@dataclass
class LLMTrainingConfig:
    """Complete configuration for LLM training."""
    # Model settings
    model_name: str
    model_type: str = "gpt-oss-20b"  # Registry key
    
    # Dataset settings  
    dataset_path: str = None
    max_seq_length: int = 2048
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    
    # QLoRA settings
    use_qlora: bool = True
    qlora_config: Optional[QLoRAConfig] = None
    
    # Training options
    fp16: bool = True
    gradient_checkpointing: bool = True
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3
    
    # Output settings
    output_dir: str = "llm_outputs"
    run_name: Optional[str] = None
    
    # Safety settings
    enable_safety: bool = True
    danger_mode: bool = False
    
    def __post_init__(self):
        """Initialize default values."""
        if self.qlora_config is None and self.use_qlora:
            self.qlora_config = QLoRAConfig()
        
        if self.run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"{self.model_type}_{timestamp}"

class LLMTrainer:
    """Main trainer for language models."""
    
    def __init__(self, config: LLMTrainingConfig, message_queue: Optional[mp.Queue] = None):
        """
        Initialize LLM trainer.
        
        Args:
            config: Training configuration
            message_queue: Optional queue for progress updates
        """
        self.config = config
        self.message_queue = message_queue
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.danger_controller = None
        
        # Setup safety
        if config.danger_mode:
            self.danger_controller = DangerModeController()
            if not self.danger_controller.activate_danger_mode(
                "I ACCEPT FULL RESPONSIBILITY", 
                password=None
            ):
                raise ValueError("Failed to activate danger mode")
    
    def send_message(self, msg_type: str, data: Any):
        """Send message to UI if queue is available."""
        if self.message_queue:
            try:
                self.message_queue.put({
                    'type': msg_type,
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                })
            except:
                pass  # Ignore queue errors
    
    def load_model(self):
        """Load the model and tokenizer."""
        self.send_message('status', 'Loading model...')
        
        try:
            # Import registry
            from .. import get_llm
            
            # Get model from registry
            if self.config.danger_mode and self.config.model_type.endswith('-uncensored'):
                # Use uncensored variant if in danger mode
                model_type = self.config.model_type
            else:
                model_type = self.config.model_type
            
            # Load model
            self.model = get_llm(model_type)
            self.model.load_model()
            
            # Get tokenizer
            self.tokenizer = self.model.tokenizer
            
            # Apply QLoRA if configured
            if self.config.use_qlora:
                self.model.model = prepare_model_for_qlora(
                    self.model.model,
                    self.config.qlora_config
                )
            
            self.send_message('status', f'Model loaded: {self.config.model_name}')
            
        except Exception as e:
            self.send_message('error', f'Failed to load model: {str(e)}')
            raise
    
    def load_dataset(self):
        """Load and prepare the dataset."""
        self.send_message('status', 'Loading dataset...')
        
        try:
            from datasets import load_dataset
            
            if self.config.dataset_path:
                # Load from local path
                if os.path.isfile(self.config.dataset_path):
                    # Single file
                    dataset = load_dataset(
                        'text',
                        data_files=self.config.dataset_path,
                        split='train'
                    )
                else:
                    # Directory of files
                    dataset = load_dataset(
                        'text',
                        data_dir=self.config.dataset_path,
                        split='train'
                    )
            else:
                # Use a default demo dataset
                dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:1000]')
            
            # Tokenize dataset
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'],
                    truncation=True,
                    padding='max_length',
                    max_length=self.config.max_seq_length
                )
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            self.send_message('status', f'Dataset loaded: {len(tokenized_dataset)} samples')
            return tokenized_dataset
            
        except Exception as e:
            self.send_message('error', f'Failed to load dataset: {str(e)}')
            raise
    
    def setup_trainer(self, train_dataset, eval_dataset=None):
        """Setup the Hugging Face Trainer."""
        self.send_message('status', 'Setting up trainer...')
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.output_dir, self.config.run_name),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True if eval_dataset else False,
            report_to="none",  # Disable wandb/tensorboard
            gradient_checkpointing=self.config.gradient_checkpointing,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM
        )
        
        # Custom callback for progress updates
        class ProgressCallback:
            def __init__(self, trainer_instance):
                self.trainer = trainer_instance
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    self.trainer.send_message('metrics', {
                        'step': state.global_step,
                        'epoch': state.epoch,
                        'loss': logs.get('loss', 0),
                        'learning_rate': logs.get('learning_rate', 0),
                    })
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[ProgressCallback(self)]
        )
    
    def train(self):
        """Run the training loop."""
        self.send_message('status', 'Starting training...')
        
        try:
            # Start training
            train_result = self.trainer.train()
            
            # Save final model
            self.trainer.save_model()
            
            # Save training metrics
            metrics = train_result.metrics
            self.trainer.save_metrics("train", metrics)
            
            self.send_message('complete', {
                'status': 'success',
                'metrics': metrics,
                'model_path': self.trainer.args.output_dir
            })
            
            return train_result
            
        except Exception as e:
            self.send_message('error', {
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            raise
    
    def run(self):
        """Complete training pipeline."""
        try:
            # Load model
            self.load_model()
            
            # Load dataset
            dataset = self.load_dataset()
            
            # Split dataset if needed
            if len(dataset) > 100:
                split = dataset.train_test_split(test_size=0.1, seed=42)
                train_dataset = split['train']
                eval_dataset = split['test']
            else:
                train_dataset = dataset
                eval_dataset = None
            
            # Setup trainer
            self.setup_trainer(train_dataset, eval_dataset)
            
            # Run training
            return self.train()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.send_message('error', str(e))
            raise

def train_llm_worker(config_dict: Dict, message_queue: mp.Queue):
    """
    Worker function for multiprocess training.
    
    Args:
        config_dict: Configuration dictionary
        message_queue: Queue for progress updates
    """
    try:
        # Reconstruct config
        config = LLMTrainingConfig(**config_dict)
        
        # Create trainer
        trainer = LLMTrainer(config, message_queue)
        
        # Run training
        trainer.run()
        
    except Exception as e:
        message_queue.put({
            'type': 'error',
            'data': {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        })