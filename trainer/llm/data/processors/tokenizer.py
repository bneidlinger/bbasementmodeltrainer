"""
Text Tokenization for LLM Training
Handles tokenization with various strategies and models
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
import tiktoken

logger = logging.getLogger(__name__)

@dataclass
class TokenizerConfig:
    """Configuration for tokenization."""
    tokenizer_name: str = "gpt2"  # Default tokenizer
    max_length: int = 2048
    padding: str = "max_length"
    truncation: bool = True
    add_special_tokens: bool = True
    return_tensors: str = "pt"
    
    # Chunking settings
    chunk_overlap: int = 128
    stride: int = 0
    
    # Special tokens
    bos_token: Optional[str] = None
    eos_token: Optional[str] = None
    pad_token: Optional[str] = None
    
class DataTokenizer:
    """
    Universal tokenizer for various LLM models.
    Supports HuggingFace tokenizers and tiktoken.
    """
    
    def __init__(self, config: Optional[TokenizerConfig] = None):
        """
        Initialize tokenizer with configuration.
        
        Args:
            config: Tokenizer configuration
        """
        self.config = config or TokenizerConfig()
        self.tokenizer = None
        self.is_tiktoken = False
        self._load_tokenizer()
        
    def _load_tokenizer(self):
        """Load the appropriate tokenizer based on config."""
        try:
            # Check if it's a tiktoken model
            if self.config.tokenizer_name in ['gpt-4', 'gpt-3.5-turbo', 'cl100k_base']:
                self.is_tiktoken = True
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.info(f"Loaded tiktoken tokenizer: cl100k_base")
            else:
                # Load HuggingFace tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.tokenizer_name,
                    trust_remote_code=True
                )
                
                # Set special tokens if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
                
                if self.config.bos_token:
                    self.tokenizer.bos_token = self.config.bos_token
                if self.config.eos_token:
                    self.tokenizer.eos_token = self.config.eos_token
                if self.config.pad_token:
                    self.tokenizer.pad_token = self.config.pad_token
                    
                logger.info(f"Loaded HuggingFace tokenizer: {self.config.tokenizer_name}")
                
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            # Fallback to GPT-2 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.warning("Falling back to GPT-2 tokenizer")
    
    def tokenize(self, 
                 text: Union[str, List[str]], 
                 return_offsets: bool = False) -> Dict[str, Any]:
        """
        Tokenize text or list of texts.
        
        Args:
            text: Input text(s) to tokenize
            return_offsets: Whether to return character offsets
            
        Returns:
            Dictionary containing token IDs and optionally attention masks
        """
        if self.is_tiktoken:
            return self._tokenize_tiktoken(text)
        else:
            return self._tokenize_huggingface(text, return_offsets)
    
    def _tokenize_tiktoken(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """Tokenize using tiktoken."""
        if isinstance(text, str):
            text = [text]
        
        encoded_texts = []
        for t in text:
            tokens = self.tokenizer.encode(t)
            # Truncate if needed
            if len(tokens) > self.config.max_length:
                tokens = tokens[:self.config.max_length]
            # Pad if needed
            elif len(tokens) < self.config.max_length:
                tokens = tokens + [0] * (self.config.max_length - len(tokens))
            encoded_texts.append(tokens)
        
        return {
            'input_ids': torch.tensor(encoded_texts),
            'attention_mask': torch.tensor([[1 if t != 0 else 0 for t in tokens] 
                                           for tokens in encoded_texts])
        }
    
    def _tokenize_huggingface(self, 
                              text: Union[str, List[str]], 
                              return_offsets: bool = False) -> Dict[str, Any]:
        """Tokenize using HuggingFace tokenizer."""
        return self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            add_special_tokens=self.config.add_special_tokens,
            return_tensors=self.config.return_tensors,
            return_offsets_mapping=return_offsets
        )
    
    def tokenize_for_training(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts specifically for training.
        Handles concatenation and chunking for efficient training.
        
        Args:
            texts: List of texts to tokenize
            
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        all_input_ids = []
        all_attention_masks = []
        
        for text in texts:
            # Tokenize individual text
            tokens = self.tokenize(text)
            
            # Handle long sequences with sliding window
            if tokens['input_ids'].shape[1] > self.config.max_length:
                # Use sliding window approach
                for i in range(0, tokens['input_ids'].shape[1] - self.config.chunk_overlap, 
                             self.config.max_length - self.config.chunk_overlap):
                    chunk_ids = tokens['input_ids'][:, i:i + self.config.max_length]
                    chunk_mask = tokens['attention_mask'][:, i:i + self.config.max_length]
                    
                    # Pad if necessary
                    if chunk_ids.shape[1] < self.config.max_length:
                        pad_length = self.config.max_length - chunk_ids.shape[1]
                        chunk_ids = torch.nn.functional.pad(chunk_ids, (0, pad_length), value=self.tokenizer.pad_token_id)
                        chunk_mask = torch.nn.functional.pad(chunk_mask, (0, pad_length), value=0)
                    
                    all_input_ids.append(chunk_ids)
                    all_attention_masks.append(chunk_mask)
            else:
                all_input_ids.append(tokens['input_ids'])
                all_attention_masks.append(tokens['attention_mask'])
        
        return {
            'input_ids': torch.cat(all_input_ids, dim=0),
            'attention_mask': torch.cat(all_attention_masks, dim=0)
        }
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if self.is_tiktoken:
            return self.tokenizer.decode(token_ids)
        else:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer."""
        if self.is_tiktoken:
            return self.tokenizer.n_vocab
        else:
            return len(self.tokenizer)
    
    def prepare_instruction_format(self, 
                                  instruction: str, 
                                  input_text: str = "", 
                                  output: str = "") -> str:
        """
        Format text for instruction fine-tuning.
        
        Args:
            instruction: The instruction/prompt
            input_text: Optional input context
            output: Expected output (for training)
            
        Returns:
            Formatted text ready for tokenization
        """
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        return prompt
    
    def prepare_chat_format(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages for chat-based training.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted chat text
        """
        formatted = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                formatted.append(f"System: {content}")
            elif role == 'user':
                formatted.append(f"User: {content}")
            elif role == 'assistant':
                formatted.append(f"Assistant: {content}")
        
        return "\n\n".join(formatted)
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if self.is_tiktoken:
            return len(self.tokenizer.encode(text))
        else:
            return len(self.tokenizer.encode(text, add_special_tokens=True))
    
    def batch_encode(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Encode texts in batches for memory efficiency.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            List of encoded batches
        """
        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenize(batch)
            batches.append(encoded)
        
        return batches