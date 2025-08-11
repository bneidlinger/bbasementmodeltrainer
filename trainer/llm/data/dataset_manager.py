"""
Unified Dataset Manager for LLM Training
Orchestrates the complete data pipeline from scraping to training-ready datasets
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .scrapers import ArxivScraper, RedditScraper, WebScraper
from .processors import DataTokenizer, DataCleaner, DataFormatter
from .processors.tokenizer import TokenizerConfig
from .processors.cleaner import CleanerConfig
from .processors.formatter import FormatterConfig

logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """Complete configuration for dataset creation."""
    # Dataset info
    name: str = "custom_dataset"
    description: str = ""
    
    # Data sources
    sources: List[str] = None  # ['arxiv', 'reddit', 'web']
    queries: List[str] = None  # Search queries for each source
    max_items_per_source: int = 1000
    
    # Processing configs
    tokenizer_config: TokenizerConfig = None
    cleaner_config: CleanerConfig = None
    formatter_config: FormatterConfig = None
    
    # Output
    output_dir: str = "datasets"
    save_intermediate: bool = True  # Save raw and cleaned data
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = ['web']
        if self.queries is None:
            self.queries = ['machine learning']
        if self.tokenizer_config is None:
            self.tokenizer_config = TokenizerConfig()
        if self.cleaner_config is None:
            self.cleaner_config = CleanerConfig()
        if self.formatter_config is None:
            self.formatter_config = FormatterConfig()

class DatasetManager:
    """
    Manages the complete data pipeline for LLM training.
    Coordinates scraping, cleaning, and formatting.
    """
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        """
        Initialize dataset manager.
        
        Args:
            config: Dataset configuration
        """
        self.config = config or DatasetConfig()
        
        # Initialize components
        self.scrapers = self._init_scrapers()
        self.tokenizer = DataTokenizer(self.config.tokenizer_config)
        self.cleaner = DataCleaner(self.config.cleaner_config)
        self.formatter = DataFormatter(self.config.formatter_config)
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir) / self.config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'raw_documents': 0,
            'cleaned_documents': 0,
            'formatted_documents': 0,
            'total_tokens': 0,
            'processing_time': 0
        }
    
    def _init_scrapers(self) -> Dict[str, Any]:
        """Initialize available scrapers."""
        scrapers = {}
        
        if 'arxiv' in self.config.sources:
            scrapers['arxiv'] = ArxivScraper(
                cache_dir=str(self.output_dir / 'arxiv_cache')
            )
        
        if 'reddit' in self.config.sources:
            scrapers['reddit'] = RedditScraper(
                cache_dir=str(self.output_dir / 'reddit_cache')
            )
        
        if 'web' in self.config.sources:
            scrapers['web'] = WebScraper(
                cache_dir=str(self.output_dir / 'web_cache')
            )
        
        return scrapers
    
    def scrape_data(self, parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Scrape data from configured sources.
        
        Args:
            parallel: Whether to scrape sources in parallel
            
        Returns:
            List of scraped documents
        """
        logger.info(f"Starting data scraping from {len(self.config.sources)} sources")
        all_data = []
        
        if parallel and len(self.config.sources) > 1:
            # Parallel scraping
            with ThreadPoolExecutor(max_workers=len(self.config.sources)) as executor:
                futures = {}
                
                for source in self.config.sources:
                    if source in self.scrapers:
                        for query in self.config.queries:
                            future = executor.submit(
                                self._scrape_single_source,
                                source,
                                query
                            )
                            futures[future] = (source, query)
                
                for future in as_completed(futures):
                    source, query = futures[future]
                    try:
                        data = future.result()
                        all_data.extend(data)
                        logger.info(f"Scraped {len(data)} items from {source} for query '{query}'")
                    except Exception as e:
                        logger.error(f"Error scraping {source} for '{query}': {e}")
        else:
            # Sequential scraping
            for source in self.config.sources:
                if source in self.scrapers:
                    for query in self.config.queries:
                        try:
                            data = self._scrape_single_source(source, query)
                            all_data.extend(data)
                            logger.info(f"Scraped {len(data)} items from {source} for query '{query}'")
                        except Exception as e:
                            logger.error(f"Error scraping {source} for '{query}': {e}")
        
        self.stats['raw_documents'] = len(all_data)
        logger.info(f"Total scraped documents: {len(all_data)}")
        
        # Save raw data if configured
        if self.config.save_intermediate:
            self._save_intermediate(all_data, 'raw_data.jsonl')
        
        return all_data
    
    def _scrape_single_source(self, source: str, query: str) -> List[Dict]:
        """Scrape from a single source."""
        scraper = self.scrapers.get(source)
        if not scraper:
            return []
        
        if source == 'arxiv':
            return scraper.search_papers(
                query=query,
                max_results=self.config.max_items_per_source
            )
        elif source == 'reddit':
            return scraper.scrape_subreddit(
                subreddit='MachineLearning',  # Default subreddit
                query=query,
                limit=self.config.max_items_per_source
            )
        elif source == 'web':
            return scraper.scrape_query(
                query=query,
                max_pages=self.config.max_items_per_source // 10  # Assuming ~10 items per page
            )
        
        return []
    
    def clean_data(self, data: List[Dict]) -> List[Dict]:
        """
        Clean and deduplicate scraped data.
        
        Args:
            data: Raw scraped data
            
        Returns:
            Cleaned data
        """
        logger.info("Starting data cleaning")
        
        # Extract text content from documents
        texts = []
        metadata = []
        
        for doc in data:
            # Handle different document structures
            text = doc.get('text') or doc.get('content') or doc.get('abstract', '')
            if doc.get('title'):
                text = doc['title'] + '\n\n' + text
            
            texts.append(text)
            metadata.append({
                'source': doc.get('source', 'unknown'),
                'url': doc.get('url', ''),
                'date': doc.get('date', ''),
                'author': doc.get('author', '')
            })
        
        # Clean texts
        cleaned_texts = self.cleaner.clean_batch(texts, deduplicate=True)
        
        # Match cleaned texts with metadata
        cleaned_data = []
        for i, text in enumerate(cleaned_texts):
            if i < len(metadata):
                cleaned_data.append({
                    'text': text,
                    'metadata': metadata[i]
                })
        
        self.stats['cleaned_documents'] = len(cleaned_data)
        logger.info(f"Cleaned documents: {len(cleaned_data)}/{len(data)}")
        
        # Get cleaning statistics
        clean_stats = self.cleaner.get_statistics(cleaned_texts)
        self.stats.update(clean_stats)
        
        # Save cleaned data if configured
        if self.config.save_intermediate:
            self._save_intermediate(cleaned_data, 'cleaned_data.jsonl')
        
        return cleaned_data
    
    def format_data(self, 
                   data: List[Dict], 
                   dataset_type: str = "completion") -> List[Dict]:
        """
        Format cleaned data for training.
        
        Args:
            data: Cleaned data
            dataset_type: Type of dataset formatting
            
        Returns:
            Formatted data
        """
        logger.info(f"Formatting data as {dataset_type}")
        
        # Format based on type
        formatted_data = self.formatter.format_dataset(data, dataset_type)
        
        self.stats['formatted_documents'] = len(formatted_data)
        
        # Calculate token statistics
        total_tokens = 0
        for entry in formatted_data[:100]:  # Sample for speed
            total_tokens += self.tokenizer.count_tokens(entry['text'])
        avg_tokens = total_tokens / min(100, len(formatted_data))
        self.stats['total_tokens'] = int(avg_tokens * len(formatted_data))
        
        logger.info(f"Formatted {len(formatted_data)} documents")
        logger.info(f"Estimated total tokens: {self.stats['total_tokens']:,}")
        
        return formatted_data
    
    def create_dataset(self, 
                      dataset_type: str = "completion",
                      parallel_scraping: bool = True) -> Dict[str, Any]:
        """
        Run the complete pipeline to create a dataset.
        
        Args:
            dataset_type: Type of dataset to create
            parallel_scraping: Whether to scrape in parallel
            
        Returns:
            Dictionary with dataset info and statistics
        """
        start_time = time.time()
        logger.info(f"Creating dataset: {self.config.name}")
        
        try:
            # Step 1: Scrape data
            logger.info("Step 1/3: Scraping data...")
            raw_data = self.scrape_data(parallel=parallel_scraping)
            
            if not raw_data:
                logger.error("No data scraped!")
                return {'error': 'No data scraped', 'stats': self.stats}
            
            # Step 2: Clean data
            logger.info("Step 2/3: Cleaning data...")
            cleaned_data = self.clean_data(raw_data)
            
            if not cleaned_data:
                logger.error("No data survived cleaning!")
                return {'error': 'No data survived cleaning', 'stats': self.stats}
            
            # Step 3: Format data
            logger.info("Step 3/3: Formatting data...")
            formatted_data = self.format_data(cleaned_data, dataset_type)
            
            # Save final dataset
            self.formatter.save_dataset(
                formatted_data,
                name=self.config.name,
                split_data=True
            )
            
            # Calculate processing time
            self.stats['processing_time'] = time.time() - start_time
            
            # Save configuration and statistics
            self._save_config_and_stats()
            
            logger.info(f"Dataset creation complete in {self.stats['processing_time']:.2f} seconds")
            
            return {
                'success': True,
                'dataset_path': str(self.output_dir),
                'stats': self.stats
            }
            
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'stats': self.stats
            }
    
    def _save_intermediate(self, data: List[Dict], filename: str):
        """Save intermediate data."""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logger.info(f"Saved intermediate data to {filepath}")
    
    def _save_config_and_stats(self):
        """Save configuration and statistics."""
        # Save config
        config_path = self.output_dir / 'dataset_config.json'
        config_dict = {
            'name': self.config.name,
            'description': self.config.description,
            'sources': self.config.sources,
            'queries': self.config.queries,
            'max_items_per_source': self.config.max_items_per_source,
            'tokenizer': asdict(self.config.tokenizer_config),
            'cleaner': asdict(self.config.cleaner_config),
            'formatter': asdict(self.config.formatter_config)
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save statistics
        stats_path = self.output_dir / 'dataset_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Saved config and stats to {self.output_dir}")
    
    def load_existing_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Load an existing dataset.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Dataset information
        """
        dataset_path = Path(dataset_path)
        
        # Load configuration
        config_path = dataset_path / 'dataset_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Load statistics
        stats_path = dataset_path / 'dataset_stats.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
        else:
            stats = {}
        
        # Check for data files
        data_files = {
            'train': dataset_path / 'train.jsonl',
            'val': dataset_path / 'val.jsonl',
            'test': dataset_path / 'test.jsonl'
        }
        
        available_splits = {
            split: str(path) for split, path in data_files.items() 
            if path.exists()
        }
        
        return {
            'config': config,
            'stats': stats,
            'splits': available_splits,
            'path': str(dataset_path)
        }
    
    def estimate_training_time(self, batch_size: int = 4, gpu_memory: int = 12) -> Dict[str, float]:
        """
        Estimate training time based on dataset size.
        
        Args:
            batch_size: Training batch size
            gpu_memory: Available GPU memory in GB
            
        Returns:
            Time estimates in hours
        """
        total_tokens = self.stats.get('total_tokens', 0)
        num_documents = self.stats.get('formatted_documents', 0)
        
        if not total_tokens or not num_documents:
            return {'error': 'No dataset statistics available'}
        
        # Rough estimates based on hardware
        if gpu_memory >= 24:  # High-end GPU
            tokens_per_second = 5000
        elif gpu_memory >= 16:  # Mid-range GPU
            tokens_per_second = 3000
        else:  # Your 12GB GPU
            tokens_per_second = 1500
        
        # Calculate estimates
        seconds_per_epoch = total_tokens / tokens_per_second
        hours_per_epoch = seconds_per_epoch / 3600
        
        return {
            'per_epoch': hours_per_epoch,
            'three_epochs': hours_per_epoch * 3,
            'tokens_per_second': tokens_per_second,
            'total_tokens': total_tokens
        }