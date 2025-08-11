"""
Dataset Formatting for LLM Training
Converts cleaned data into training-ready formats
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

@dataclass
class FormatterConfig:
    """Configuration for dataset formatting."""
    output_format: str = "jsonl"  # jsonl, parquet, arrow, pt
    max_seq_length: int = 2048
    
    # Instruction formatting
    add_eos_token: bool = True
    add_bos_token: bool = True
    instruction_template: str = "alpaca"  # alpaca, vicuna, chatml, plain
    
    # Data splitting
    train_split: float = 0.9
    val_split: float = 0.05
    test_split: float = 0.05
    shuffle: bool = True
    seed: int = 42
    
    # Batching
    batch_size: int = 1000  # For writing to disk
    
    # Paths
    output_dir: str = "formatted_datasets"

class DataFormatter:
    """
    Format cleaned data for LLM training.
    Supports various output formats and instruction templates.
    """
    
    # Instruction templates
    TEMPLATES = {
        'alpaca': {
            'prompt': "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}",
            'prompt_input': "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
        },
        'vicuna': {
            'prompt': "USER: {instruction}\nASSISTANT: {output}",
            'prompt_input': "USER: {instruction}\n{input}\nASSISTANT: {output}"
        },
        'chatml': {
            'prompt': "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>",
            'prompt_input': "<|im_start|>user\n{instruction}\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        },
        'plain': {
            'prompt': "{instruction}\n{output}",
            'prompt_input': "{instruction}\n{input}\n{output}"
        }
    }
    
    def __init__(self, config: Optional[FormatterConfig] = None):
        """
        Initialize formatter with configuration.
        
        Args:
            config: Formatter configuration
        """
        self.config = config or FormatterConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        random.seed(self.config.seed)
        
    def format_instruction(self, 
                          instruction: str, 
                          output: str, 
                          input_text: str = "") -> str:
        """
        Format instruction-following data.
        
        Args:
            instruction: The instruction/prompt
            output: The expected output
            input_text: Optional input context
            
        Returns:
            Formatted text
        """
        template = self.TEMPLATES.get(self.config.instruction_template, self.TEMPLATES['alpaca'])
        
        if input_text:
            text = template['prompt_input'].format(
                instruction=instruction,
                input=input_text,
                output=output
            )
        else:
            text = template['prompt'].format(
                instruction=instruction,
                output=output
            )
        
        # Add special tokens if configured
        if self.config.add_bos_token:
            text = "<s>" + text
        if self.config.add_eos_token:
            text = text + "</s>"
        
        return text
    
    def format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """
        Format multi-turn conversation data.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Formatted conversation
        """
        if self.config.instruction_template == 'chatml':
            formatted = []
            for msg in messages:
                role = msg['role']
                content = msg['content']
                formatted.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            return "\n".join(formatted)
        
        elif self.config.instruction_template == 'vicuna':
            formatted = []
            for msg in messages:
                role = "USER" if msg['role'] == 'user' else "ASSISTANT"
                formatted.append(f"{role}: {msg['content']}")
            return "\n".join(formatted)
        
        else:
            # Default format
            formatted = []
            for msg in messages:
                formatted.append(f"{msg['role'].title()}: {msg['content']}")
            return "\n\n".join(formatted)
    
    def format_dataset(self, 
                      data: List[Dict[str, Any]], 
                      dataset_type: str = "instruction") -> List[Dict[str, str]]:
        """
        Format a dataset based on type.
        
        Args:
            data: Raw data entries
            dataset_type: Type of dataset (instruction, conversation, completion)
            
        Returns:
            List of formatted examples
        """
        formatted_data = []
        
        for entry in data:
            if dataset_type == "instruction":
                text = self.format_instruction(
                    instruction=entry.get('instruction', ''),
                    output=entry.get('output', ''),
                    input_text=entry.get('input', '')
                )
                formatted_data.append({
                    'text': text,
                    'metadata': entry.get('metadata', {})
                })
            
            elif dataset_type == "conversation":
                text = self.format_conversation(entry.get('messages', []))
                formatted_data.append({
                    'text': text,
                    'metadata': entry.get('metadata', {})
                })
            
            elif dataset_type == "completion":
                # Simple text completion
                text = entry.get('text', '')
                if self.config.add_bos_token:
                    text = "<s>" + text
                if self.config.add_eos_token:
                    text = text + "</s>"
                formatted_data.append({
                    'text': text,
                    'metadata': entry.get('metadata', {})
                })
        
        return formatted_data
    
    def split_dataset(self, 
                     data: List[Dict[str, Any]]) -> Tuple[List, List, List]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            data: Full dataset
            
        Returns:
            Tuple of (train, val, test) datasets
        """
        if self.config.shuffle:
            random.shuffle(data)
        
        total_size = len(data)
        train_size = int(total_size * self.config.train_split)
        val_size = int(total_size * self.config.val_split)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        logger.info(f"Split dataset: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def save_jsonl(self, data: List[Dict], filepath: str):
        """Save data in JSONL format."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(data)} entries to {filepath}")
    
    def save_parquet(self, data: List[Dict], filepath: str):
        """Save data in Parquet format."""
        df = pd.DataFrame(data)
        df.to_parquet(filepath, engine='pyarrow', compression='snappy')
        logger.info(f"Saved {len(data)} entries to {filepath}")
    
    def save_arrow(self, data: List[Dict], filepath: str):
        """Save data in Arrow format."""
        # Convert to Arrow table
        texts = [d['text'] for d in data]
        metadata = [json.dumps(d.get('metadata', {})) for d in data]
        
        table = pa.table({
            'text': texts,
            'metadata': metadata
        })
        
        # Save to Arrow file
        with pa.OSFile(filepath, 'wb') as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        
        logger.info(f"Saved {len(data)} entries to {filepath}")
    
    def save_pytorch(self, data: List[Dict], filepath: str):
        """Save data in PyTorch format."""
        torch.save(data, filepath)
        logger.info(f"Saved {len(data)} entries to {filepath}")
    
    def save_dataset(self, 
                    data: List[Dict], 
                    name: str = "dataset",
                    split_data: bool = True):
        """
        Save formatted dataset to disk.
        
        Args:
            data: Formatted data
            name: Dataset name
            split_data: Whether to split into train/val/test
        """
        output_dir = Path(self.config.output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split dataset if requested
        if split_data:
            train_data, val_data, test_data = self.split_dataset(data)
            datasets = {
                'train': train_data,
                'val': val_data,
                'test': test_data
            }
        else:
            datasets = {'full': data}
        
        # Save each split
        for split_name, split_data in datasets.items():
            if not split_data:
                continue
            
            # Determine file extension based on format
            ext_map = {
                'jsonl': '.jsonl',
                'parquet': '.parquet',
                'arrow': '.arrow',
                'pt': '.pt'
            }
            ext = ext_map.get(self.config.output_format, '.jsonl')
            filepath = output_dir / f"{split_name}{ext}"
            
            # Save based on format
            if self.config.output_format == 'jsonl':
                self.save_jsonl(split_data, str(filepath))
            elif self.config.output_format == 'parquet':
                self.save_parquet(split_data, str(filepath))
            elif self.config.output_format == 'arrow':
                self.save_arrow(split_data, str(filepath))
            elif self.config.output_format == 'pt':
                self.save_pytorch(split_data, str(filepath))
        
        # Save configuration
        config_path = output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        logger.info(f"Dataset saved to {output_dir}")
    
    def load_dataset(self, path: str) -> List[Dict]:
        """
        Load a formatted dataset from disk.
        
        Args:
            path: Path to dataset file
            
        Returns:
            List of data entries
        """
        path = Path(path)
        
        if path.suffix == '.jsonl':
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        
        elif path.suffix == '.parquet':
            df = pd.read_parquet(path)
            return df.to_dict('records')
        
        elif path.suffix == '.arrow':
            with pa.memory_map(str(path), 'r') as source:
                table = pa.ipc.RecordBatchFileReader(source).read_all()
            return [
                {'text': text, 'metadata': json.loads(meta)}
                for text, meta in zip(table['text'].to_pylist(), 
                                     table['metadata'].to_pylist())
            ]
        
        elif path.suffix == '.pt':
            return torch.load(path)
        
        else:
            raise ValueError(f"Unknown format: {path.suffix}")
    
    def create_torch_dataset(self, data: List[Dict]) -> 'FormattedDataset':
        """
        Create a PyTorch Dataset from formatted data.
        
        Args:
            data: Formatted data entries
            
        Returns:
            PyTorch Dataset
        """
        return FormattedDataset(data, self.config.max_seq_length)


class FormattedDataset(Dataset):
    """PyTorch Dataset for formatted LLM training data."""
    
    def __init__(self, data: List[Dict], max_length: int = 2048):
        """
        Initialize dataset.
        
        Args:
            data: List of formatted data entries
            max_length: Maximum sequence length
        """
        self.data = data
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset."""
        item = self.data[idx]
        return {
            'text': item['text'],
            'metadata': item.get('metadata', {})
        }
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Custom collate function for batching."""
        texts = [item['text'] for item in batch]
        metadata = [item.get('metadata', {}) for item in batch]
        
        return {
            'texts': texts,
            'metadata': metadata
        }