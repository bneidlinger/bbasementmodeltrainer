"""
Data Cleaning and Deduplication for LLM Training
Removes noise, duplicates, and filters low-quality content
"""

import re
import hashlib
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np
from collections import Counter
import unicodedata
import langdetect
from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)

@dataclass
class CleanerConfig:
    """Configuration for data cleaning."""
    # Text cleaning
    min_length: int = 50  # Minimum text length in characters
    max_length: int = 100000  # Maximum text length
    remove_html: bool = True
    remove_urls: bool = False  # Keep URLs for some training
    normalize_whitespace: bool = True
    remove_emoji: bool = False  # Keep emoji for modern LLMs
    
    # Language filtering
    target_languages: List[str] = None  # None means all languages
    language_threshold: float = 0.8  # Confidence threshold
    
    # Quality filtering
    min_word_count: int = 10
    max_word_repetition: float = 0.3  # Max ratio of repeated words
    min_avg_word_length: float = 3.0
    max_avg_word_length: float = 20.0
    
    # Deduplication
    dedup_threshold: float = 0.85  # Similarity threshold for dedup
    minhash_num_perm: int = 128  # Number of permutations for MinHash
    
    # PII removal
    remove_emails: bool = True
    remove_phone_numbers: bool = True
    remove_ssn: bool = True
    remove_credit_cards: bool = True
    anonymize_names: bool = False  # More complex, requires NER

class DataCleaner:
    """
    Comprehensive data cleaning for LLM training.
    Handles text normalization, quality filtering, and deduplication.
    """
    
    def __init__(self, config: Optional[CleanerConfig] = None):
        """
        Initialize cleaner with configuration.
        
        Args:
            config: Cleaner configuration
        """
        self.config = config or CleanerConfig()
        self.lsh = MinHashLSH(threshold=self.config.dedup_threshold, 
                             num_perm=self.config.minhash_num_perm)
        self.seen_hashes: Set[str] = set()
        
        # Compile regex patterns
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.patterns = {
            'html': re.compile(r'<[^>]+>'),
            'url': re.compile(r'https?://\S+|www\.\S+'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            'multiple_spaces': re.compile(r'\s+'),
            'multiple_newlines': re.compile(r'\n{3,}'),
        }
    
    def clean_text(self, text: str) -> Optional[str]:
        """
        Clean a single text document.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text or None if filtered out
        """
        if not text or len(text.strip()) < self.config.min_length:
            return None
        
        # Remove HTML if configured
        if self.config.remove_html:
            text = self.patterns['html'].sub('', text)
        
        # Remove or mask PII
        text = self._remove_pii(text)
        
        # Remove URLs if configured
        if self.config.remove_urls:
            text = self.patterns['url'].sub('[URL]', text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = self.patterns['multiple_spaces'].sub(' ', text)
            text = self.patterns['multiple_newlines'].sub('\n\n', text)
            text = text.strip()
        
        # Check length constraints
        if len(text) < self.config.min_length or len(text) > self.config.max_length:
            return None
        
        # Quality checks
        if not self._passes_quality_checks(text):
            return None
        
        # Language detection
        if self.config.target_languages and not self._check_language(text):
            return None
        
        return text
    
    def _remove_pii(self, text: str) -> str:
        """Remove or mask personally identifiable information."""
        if self.config.remove_emails:
            text = self.patterns['email'].sub('[EMAIL]', text)
        
        if self.config.remove_phone_numbers:
            text = self.patterns['phone'].sub('[PHONE]', text)
        
        if self.config.remove_ssn:
            text = self.patterns['ssn'].sub('[SSN]', text)
        
        if self.config.remove_credit_cards:
            text = self.patterns['credit_card'].sub('[CREDIT_CARD]', text)
        
        return text
    
    def _passes_quality_checks(self, text: str) -> bool:
        """Check if text passes quality thresholds."""
        words = text.split()
        
        # Word count check
        if len(words) < self.config.min_word_count:
            return False
        
        # Word repetition check
        word_counts = Counter(words)
        if words:
            repetition_ratio = max(word_counts.values()) / len(words)
            if repetition_ratio > self.config.max_word_repetition:
                return False
        
        # Average word length check
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            if (avg_word_length < self.config.min_avg_word_length or 
                avg_word_length > self.config.max_avg_word_length):
                return False
        
        # Check for excessive special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.3:  # More than 30% special characters
            return False
        
        return True
    
    def _check_language(self, text: str) -> bool:
        """Check if text is in target language(s)."""
        try:
            detected_lang = langdetect.detect(text)
            confidence = langdetect.detect_langs(text)[0].prob
            
            if confidence < self.config.language_threshold:
                return False
            
            return detected_lang in self.config.target_languages
        except:
            # If language detection fails, keep the text
            return True
    
    def deduplicate(self, texts: List[str]) -> List[str]:
        """
        Remove duplicate or near-duplicate texts using MinHash LSH.
        
        Args:
            texts: List of texts to deduplicate
            
        Returns:
            List of unique texts
        """
        unique_texts = []
        
        for text in texts:
            # Create MinHash for text
            minhash = self._create_minhash(text)
            
            # Check if similar document exists
            similar = self.lsh.query(minhash)
            
            if not similar:
                # Add to LSH index
                doc_id = f"doc_{len(unique_texts)}"
                self.lsh.insert(doc_id, minhash)
                unique_texts.append(text)
        
        return unique_texts
    
    def _create_minhash(self, text: str) -> MinHash:
        """Create MinHash signature for text."""
        minhash = MinHash(num_perm=self.config.minhash_num_perm)
        
        # Create shingles (n-grams of words)
        words = text.lower().split()
        for i in range(len(words) - 2):
            shingle = ' '.join(words[i:i+3])
            minhash.update(shingle.encode('utf-8'))
        
        return minhash
    
    def exact_dedup(self, texts: List[str]) -> List[str]:
        """
        Remove exact duplicates using hashing.
        
        Args:
            texts: List of texts
            
        Returns:
            List of unique texts
        """
        unique_texts = []
        
        for text in texts:
            # Create hash of normalized text
            normalized = text.lower().strip()
            text_hash = hashlib.sha256(normalized.encode()).hexdigest()
            
            if text_hash not in self.seen_hashes:
                self.seen_hashes.add(text_hash)
                unique_texts.append(text)
        
        return unique_texts
    
    def clean_batch(self, texts: List[str], deduplicate: bool = True) -> List[str]:
        """
        Clean a batch of texts.
        
        Args:
            texts: List of raw texts
            deduplicate: Whether to remove duplicates
            
        Returns:
            List of cleaned texts
        """
        # Clean individual texts
        cleaned = []
        for text in texts:
            clean_text = self.clean_text(text)
            if clean_text:
                cleaned.append(clean_text)
        
        logger.info(f"Cleaned {len(cleaned)}/{len(texts)} texts")
        
        # Remove duplicates if requested
        if deduplicate and cleaned:
            # First exact dedup
            cleaned = self.exact_dedup(cleaned)
            logger.info(f"After exact dedup: {len(cleaned)} texts")
            
            # Then fuzzy dedup
            cleaned = self.deduplicate(cleaned)
            logger.info(f"After fuzzy dedup: {len(cleaned)} texts")
        
        return cleaned
    
    def calculate_perplexity_filter(self, texts: List[str], threshold: float = 1000.0) -> List[str]:
        """
        Filter texts based on perplexity scores.
        High perplexity often indicates low-quality or nonsensical text.
        
        Args:
            texts: List of texts
            threshold: Maximum perplexity threshold
            
        Returns:
            Filtered list of texts
        """
        # This would require a language model to calculate actual perplexity
        # For now, using a simplified heuristic based on character distribution
        filtered = []
        
        for text in texts:
            # Calculate character entropy as proxy for perplexity
            char_counts = Counter(text)
            total_chars = len(text)
            entropy = 0
            
            for count in char_counts.values():
                if count > 0:
                    prob = count / total_chars
                    entropy -= prob * np.log2(prob)
            
            # Lower entropy suggests repetitive/low-quality text
            if 2.0 < entropy < 6.0:  # Reasonable entropy range for natural text
                filtered.append(text)
        
        return filtered
    
    def get_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get statistics about the texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Dictionary of statistics
        """
        if not texts:
            return {}
        
        lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        return {
            'num_documents': len(texts),
            'avg_length': np.mean(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_word_count': np.mean(word_counts),
            'total_characters': sum(lengths),
            'unique_hashes': len(self.seen_hashes)
        }
    
    def reset_dedup_cache(self):
        """Reset deduplication cache for new dataset."""
        self.lsh = MinHashLSH(threshold=self.config.dedup_threshold,
                             num_perm=self.config.minhash_num_perm)
        self.seen_hashes.clear()
        logger.info("Reset deduplication cache")