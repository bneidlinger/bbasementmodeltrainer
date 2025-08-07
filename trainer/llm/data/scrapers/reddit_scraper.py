"""
Reddit Data Scraper
Collects posts and comments from Reddit for training data
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class RedditScraper:
    """Scraper for Reddit content using PRAW or API."""
    
    def __init__(self, client_id: Optional[str] = None, 
                 client_secret: Optional[str] = None,
                 user_agent: str = "BasementBrewAI/1.0"):
        """
        Initialize Reddit scraper.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent for requests
        """
        self.reddit = None
        self.use_api = False
        
        # Try to setup PRAW if credentials provided
        if client_id and client_secret:
            try:
                import praw
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent
                )
                self.use_api = True
                logger.info("Reddit API initialized with PRAW")
            except ImportError:
                logger.warning("PRAW not installed. Install with: pip install praw")
            except Exception as e:
                logger.error(f"Failed to initialize Reddit API: {e}")
    
    def search(self, query: str, max_items: int = 100, 
               subreddits: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search Reddit for content.
        
        Args:
            query: Search query
            max_items: Maximum number of items to return
            subreddits: List of subreddits to search (None for all)
        
        Returns:
            List of Reddit posts/comments
        """
        if not self.use_api:
            return self._scrape_without_api(query, max_items)
        
        results = []
        
        try:
            # Search across Reddit or specific subreddits
            if subreddits:
                for subreddit_name in subreddits:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    for submission in subreddit.search(query, limit=max_items//len(subreddits)):
                        results.append(self._extract_submission_data(submission))
            else:
                # Search all of Reddit
                for submission in self.reddit.subreddit("all").search(query, limit=max_items):
                    results.append(self._extract_submission_data(submission))
            
        except Exception as e:
            logger.error(f"Error searching Reddit: {e}")
        
        return results
    
    def scrape_subreddit(self, subreddit_name: str, 
                        sort: str = "hot", 
                        max_items: int = 100) -> List[Dict[str, Any]]:
        """
        Scrape posts from a specific subreddit.
        
        Args:
            subreddit_name: Name of subreddit
            sort: Sort method (hot, new, top, rising)
            max_items: Maximum posts to scrape
        
        Returns:
            List of posts
        """
        if not self.use_api:
            return []
        
        results = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get posts based on sort method
            if sort == "hot":
                posts = subreddit.hot(limit=max_items)
            elif sort == "new":
                posts = subreddit.new(limit=max_items)
            elif sort == "top":
                posts = subreddit.top(limit=max_items)
            elif sort == "rising":
                posts = subreddit.rising(limit=max_items)
            else:
                posts = subreddit.hot(limit=max_items)
            
            for submission in posts:
                results.append(self._extract_submission_data(submission))
            
        except Exception as e:
            logger.error(f"Error scraping r/{subreddit_name}: {e}")
        
        return results
    
    def _extract_submission_data(self, submission) -> Dict[str, Any]:
        """Extract data from a Reddit submission."""
        try:
            # Get top comments
            submission.comments.replace_more(limit=0)
            comments = []
            
            for comment in submission.comments.list()[:10]:  # Top 10 comments
                if hasattr(comment, 'body'):
                    comments.append({
                        'text': comment.body,
                        'score': comment.score,
                        'author': str(comment.author) if comment.author else '[deleted]'
                    })
            
            return {
                'title': submission.title,
                'text': submission.selftext,
                'subreddit': str(submission.subreddit),
                'score': submission.score,
                'num_comments': submission.num_comments,
                'url': f"https://reddit.com{submission.permalink}",
                'created_utc': submission.created_utc,
                'author': str(submission.author) if submission.author else '[deleted]',
                'comments': comments,
                'combined_text': f"{submission.title}\n\n{submission.selftext}\n\n" + 
                               "\n".join([c['text'] for c in comments])
            }
        except Exception as e:
            logger.error(f"Error extracting submission data: {e}")
            return {}
    
    def _scrape_without_api(self, query: str, max_items: int) -> List[Dict[str, Any]]:
        """
        Fallback scraping without API (limited functionality).
        Note: This is for educational purposes and respects Reddit's ToS.
        """
        logger.warning("Reddit API not configured. Limited scraping available.")
        
        # This would need to use requests and BeautifulSoup
        # but Reddit actively blocks scraping, so we return empty
        return []
    
    def save_to_file(self, data: List[Dict], output_path: str):
        """Save Reddit data to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                # Write combined text for training
                text = item.get('combined_text', '')
                if text:
                    f.write(text + '\n\n' + '='*50 + '\n\n')
        
        logger.info(f"Saved {len(data)} Reddit posts to {output_path}")