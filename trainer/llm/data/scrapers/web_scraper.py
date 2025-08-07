"""
Universal Web Scraping System
Collects data from multiple sources for LLM training
"""

import os
import json
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re

logger = logging.getLogger(__name__)

class WebScraper:
    """Basic web scraper with rate limiting and caching."""
    
    def __init__(self, 
                 cache_dir: str = "scraped_data",
                 rate_limit: float = 1.0,
                 respect_robots: bool = True):
        """
        Initialize web scraper.
        
        Args:
            cache_dir: Directory to cache scraped data
            rate_limit: Seconds between requests
            respect_robots: Whether to respect robots.txt
        """
        self.cache_dir = cache_dir
        self.rate_limit = rate_limit
        self.respect_robots = respect_robots
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BasementBrewAI/1.0 (Educational Research Bot)'
        })
        
        os.makedirs(cache_dir, exist_ok=True)
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _get_cache_path(self, url: str) -> str:
        """Get cache file path for URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.json")
    
    def _is_cached(self, url: str, max_age: int = 86400) -> bool:
        """Check if URL is cached and fresh."""
        cache_path = self._get_cache_path(url)
        if os.path.exists(cache_path):
            age = time.time() - os.path.getmtime(cache_path)
            return age < max_age
        return False
    
    def scrape_url(self, url: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Scrape a single URL.
        
        Args:
            url: URL to scrape
            use_cache: Whether to use cached data
        
        Returns:
            Dictionary with scraped data
        """
        # Check cache
        if use_cache and self._is_cached(url):
            cache_path = self._get_cache_path(url)
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Rate limit
        self._rate_limit()
        
        try:
            # Fetch page
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract metadata
            title = soup.find('title')
            title = title.string if title else ''
            
            meta_description = soup.find('meta', attrs={'name': 'description'})
            description = meta_description.get('content', '') if meta_description else ''
            
            # Build result
            result = {
                'url': url,
                'title': title,
                'description': description,
                'text': text,
                'word_count': len(text.split()),
                'scraped_at': datetime.now().isoformat(),
                'success': True
            }
            
            # Cache result
            if use_cache:
                cache_path = self._get_cache_path(url)
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            
            return result
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return {
                'url': url,
                'error': str(e),
                'success': False,
                'scraped_at': datetime.now().isoformat()
            }
    
    def scrape_urls(self, urls: List[str], max_workers: int = 5) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs in parallel.
        
        Args:
            urls: List of URLs to scrape
            max_workers: Number of parallel workers
        
        Returns:
            List of scraped data dictionaries
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self.scrape_url, url): url 
                for url in urls
            }
            
            for future in as_completed(future_to_url):
                result = future.result()
                results.append(result)
                
                if result['success']:
                    logger.info(f"Scraped: {result['url']} ({result['word_count']} words)")
                else:
                    logger.warning(f"Failed: {result['url']}")
        
        return results
    
    def extract_links(self, url: str, same_domain: bool = True) -> List[str]:
        """Extract all links from a page."""
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            links = []
            base_domain = urlparse(url).netloc
            
            for link in soup.find_all('a', href=True):
                href = urljoin(url, link['href'])
                parsed = urlparse(href)
                
                # Filter links
                if same_domain and parsed.netloc != base_domain:
                    continue
                
                if parsed.scheme not in ['http', 'https']:
                    continue
                
                links.append(href)
            
            return list(set(links))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting links from {url}: {e}")
            return []


class UniversalScraper:
    """Universal scraper that combines multiple data sources."""
    
    def __init__(self, cache_dir: str = "scraped_data"):
        """Initialize universal scraper with all sub-scrapers."""
        self.scrapers = {
            'web': WebScraper(cache_dir=cache_dir),
        }
        
        # Try to import optional scrapers
        try:
            from .reddit_scraper import RedditScraper
            self.scrapers['reddit'] = RedditScraper()
        except ImportError:
            logger.info("Reddit scraper not available")
        
        try:
            from .arxiv_scraper import ArxivScraper
            self.scrapers['arxiv'] = ArxivScraper()
        except ImportError:
            logger.info("ArXiv scraper not available")
    
    def scrape_multi_source(self, 
                           sources: List[str], 
                           query: str,
                           max_items: int = 100) -> Dict[str, List[Dict]]:
        """
        Scrape from multiple sources.
        
        Args:
            sources: List of source names ('web', 'reddit', 'arxiv', etc.)
            query: Search query or topic
            max_items: Maximum items per source
        
        Returns:
            Dictionary mapping source to list of scraped items
        """
        results = {}
        
        for source in sources:
            if source not in self.scrapers:
                logger.warning(f"Unknown source: {source}")
                continue
            
            try:
                scraper = self.scrapers[source]
                
                if source == 'web':
                    # For web, use Google search or similar
                    # This is a placeholder - implement actual search
                    results[source] = []
                elif hasattr(scraper, 'search'):
                    results[source] = scraper.search(query, max_items)
                else:
                    logger.warning(f"Scraper {source} doesn't support search")
                    
            except Exception as e:
                logger.error(f"Error scraping {source}: {e}")
                results[source] = []
        
        return results
    
    def clean_and_dedupe(self, data: List[Dict]) -> List[Dict]:
        """
        Clean and deduplicate scraped data.
        
        Args:
            data: List of scraped items
        
        Returns:
            Cleaned and deduplicated list
        """
        seen_hashes = set()
        cleaned = []
        
        for item in data:
            # Get text content
            text = item.get('text', '') or item.get('content', '')
            
            if not text or len(text) < 100:  # Skip very short texts
                continue
            
            # Clean text
            text = self._clean_text(text)
            
            # Check for duplicates using hash
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in seen_hashes:
                continue
            
            seen_hashes.add(text_hash)
            item['cleaned_text'] = text
            cleaned.append(item)
        
        return cleaned
    
    def _clean_text(self, text: str) -> str:
        """Clean text for training."""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = text.strip()
        
        return text
    
    def save_dataset(self, data: List[Dict], output_path: str):
        """Save scraped data as training dataset."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to simple text format
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                text = item.get('cleaned_text', '') or item.get('text', '')
                if text:
                    f.write(text + '\n\n')
        
        logger.info(f"Saved {len(data)} items to {output_path}")