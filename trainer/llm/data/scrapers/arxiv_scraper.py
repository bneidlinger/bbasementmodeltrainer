"""
ArXiv Paper Scraper
Collects research papers and abstracts for training data
"""

import os
import re
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class ArxivScraper:
    """Scraper for ArXiv research papers."""
    
    def __init__(self, cache_dir: str = "arxiv_cache"):
        """
        Initialize ArXiv scraper.
        
        Args:
            cache_dir: Directory to cache papers
        """
        self.base_url = "http://export.arxiv.org/api/query"
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def search(self, query: str, max_results: int = 100, 
              categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search ArXiv for papers.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            categories: List of ArXiv categories (e.g., ['cs.AI', 'cs.LG'])
        
        Returns:
            List of paper metadata and abstracts
        """
        # Build query
        search_query = query
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            search_query = f"({query}) AND ({cat_query})"
        
        # API parameters
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        try:
            # Make API request
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.text)
            
            # Extract papers
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                paper = self._parse_entry(entry)
                if paper:
                    papers.append(paper)
            
            logger.info(f"Found {len(papers)} papers for query: {query}")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
            return []
    
    def get_paper_by_id(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific paper by ArXiv ID.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., '2301.00234')
        
        Returns:
            Paper metadata and content
        """
        params = {
            'id_list': arxiv_id
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.text)
            entry = root.find('{http://www.w3.org/2005/Atom}entry')
            
            if entry is not None:
                return self._parse_entry(entry)
            
        except Exception as e:
            logger.error(f"Error fetching paper {arxiv_id}: {e}")
        
        return None
    
    def _parse_entry(self, entry) -> Dict[str, Any]:
        """Parse an ArXiv entry from XML."""
        try:
            # Extract basic metadata
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
            
            # Extract authors
            authors = []
            for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                name = author.find('{http://www.w3.org/2005/Atom}name').text
                authors.append(name)
            
            # Extract links
            pdf_link = None
            abs_link = None
            for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
                if link.get('type') == 'application/pdf':
                    pdf_link = link.get('href')
                elif link.get('type') == 'text/html':
                    abs_link = link.get('href')
            
            # Extract ID
            id_text = entry.find('{http://www.w3.org/2005/Atom}id').text
            arxiv_id = id_text.split('/')[-1]
            
            # Extract categories
            categories = []
            for category in entry.findall('{http://arxiv.org/schemas/atom}category'):
                categories.append(category.get('term'))
            
            # Published date
            published = entry.find('{http://www.w3.org/2005/Atom}published').text
            
            return {
                'arxiv_id': arxiv_id,
                'title': title,
                'abstract': summary,
                'authors': authors,
                'categories': categories,
                'pdf_url': pdf_link,
                'abs_url': abs_link,
                'published': published,
                'combined_text': f"Title: {title}\n\nAuthors: {', '.join(authors)}\n\nAbstract:\n{summary}"
            }
            
        except Exception as e:
            logger.error(f"Error parsing ArXiv entry: {e}")
            return None
    
    def download_pdf(self, paper: Dict[str, Any], output_dir: Optional[str] = None) -> Optional[str]:
        """
        Download PDF for a paper.
        
        Args:
            paper: Paper metadata dict
            output_dir: Directory to save PDF
        
        Returns:
            Path to downloaded PDF or None
        """
        if not paper.get('pdf_url'):
            return None
        
        output_dir = output_dir or self.cache_dir
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{paper['arxiv_id'].replace('/', '_')}.pdf"
        filepath = os.path.join(output_dir, filename)
        
        # Check if already cached
        if os.path.exists(filepath):
            logger.info(f"PDF already cached: {filepath}")
            return filepath
        
        try:
            # Download PDF
            response = requests.get(paper['pdf_url'], timeout=60)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded PDF: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            return None
    
    def extract_text_from_papers(self, papers: List[Dict]) -> List[str]:
        """
        Extract text content from papers (abstracts for now).
        
        Args:
            papers: List of paper metadata
        
        Returns:
            List of text content
        """
        texts = []
        
        for paper in papers:
            # For now, just use title and abstract
            # Full PDF extraction would require PyPDF2 or similar
            text = paper.get('combined_text', '')
            if text:
                texts.append(text)
        
        return texts
    
    def save_dataset(self, papers: List[Dict], output_path: str):
        """Save papers as training dataset."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for paper in papers:
                text = paper.get('combined_text', '')
                if text:
                    f.write(text + '\n\n' + '='*50 + '\n\n')
        
        logger.info(f"Saved {len(papers)} papers to {output_path}")


def search_papers_for_training(query: str, max_papers: int = 100) -> str:
    """
    Convenience function to search and prepare papers for training.
    
    Args:
        query: Search query
        max_papers: Maximum number of papers
    
    Returns:
        Path to generated dataset file
    """
    scraper = ArxivScraper()
    
    # Search for papers
    papers = scraper.search(query, max_results=max_papers)
    
    # Save as dataset
    output_path = f"datasets/arxiv_{query.replace(' ', '_')}_{len(papers)}.txt"
    scraper.save_dataset(papers, output_path)
    
    return output_path