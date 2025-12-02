"""
Social Media Content Scraper
Extracts text content from social media posts for fake news verification
"""

import re
import json
import logging
from typing import Dict, List, Optional
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

class SocialMediaScraper:
    """
    Social media content extractor for fake news verification
    Supports Instagram, Facebook, and Twitter/X URLs
    """
    
    def __init__(self):
        """Initialize social media scraper"""
        self.supported_platforms = {
            'instagram': ['instagram.com', 'www.instagram.com', 'instagr.am'],
            'facebook': ['facebook.com', 'www.facebook.com', 'fb.com'],
            'twitter': ['twitter.com', 'www.twitter.com', 'x.com', 'www.x.com', 't.co']
        }
    
    def extract_content_from_url(self, url: str) -> Dict:
        """
        Extract text content from social media URL
        
        Args:
            url: Social media post URL
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        try:
            platform = self._detect_platform(url)
            
            if platform == 'instagram':
                return self._extract_instagram_content(url)
            elif platform == 'facebook':
                return self._extract_facebook_content(url)
            elif platform == 'twitter':
                return self._extract_twitter_content(url)
            else:
                return {
                    'success': False,
                    'error': 'Unsupported platform',
                    'platform': 'unknown'
                }
                
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'platform': 'error'
            }
    
    def _detect_platform(self, url: str) -> str:
        """Detect social media platform from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            for platform, domains in self.supported_platforms.items():
                if any(d in domain for d in domains):
                    return platform
            
            return 'unknown'
        except:
            return 'unknown'
    
    def _extract_instagram_content(self, url: str) -> Dict:
        """Extract content from Instagram URL"""
        # For demo purposes, simulate content extraction
        # In production, this would use Instagram API or web scraping
        return {
            'success': True,
            'platform': 'instagram',
            'content': self._extract_text_from_url(url),
            'metadata': {
                'url': url,
                'type': 'post',
                'extracted_method': 'url_analysis'
            }
        }
    
    def _extract_facebook_content(self, url: str) -> Dict:
        """Extract content from Facebook URL"""
        # For demo purposes, simulate content extraction
        # In production, this would use Facebook Graph API or web scraping
        return {
            'success': True,
            'platform': 'facebook',
            'content': self._extract_text_from_url(url),
            'metadata': {
                'url': url,
                'type': 'post',
                'extracted_method': 'url_analysis'
            }
        }
    
    def _extract_twitter_content(self, url: str) -> Dict:
        """Extract content from Twitter/X URL"""
        # For demo purposes, simulate content extraction
        # In production, this would use Twitter API or web scraping
        return {
            'success': True,
            'platform': 'twitter',
            'content': self._extract_text_from_url(url),
            'metadata': {
                'url': url,
                'type': 'tweet',
                'extracted_method': 'url_analysis'
            }
        }
    
    def _extract_text_from_url(self, url: str) -> str:
        """
        Extract text content from URL path and parameters
        This is a simplified extraction for demo purposes
        """
        try:
            # Extract text from URL path and parameters
            parsed = urlparse(url)
            text_parts = []
            
            # Add path segments
            if parsed.path:
                path_parts = [part for part in parsed.path.split('/') if part and not part.isdigit()]
                text_parts.extend(path_parts)
            
            # Add query parameters
            if parsed.query:
                query_params = parse_qs(parsed.query)
                for key, values in query_params.items():
                    if key.lower() in ['text', 'status', 'caption', 'message']:
                        text_parts.extend(values)
            
            # Clean and combine text
            if text_parts:
                # Replace URL encoding and separators
                combined_text = ' '.join(text_parts)
                combined_text = re.sub(r'[-_]', ' ', combined_text)
                combined_text = re.sub(r'[^\w\s\u0600-\u06FF]', '', combined_text)
                return combined_text.strip()
            
            return "Social media post content - please verify manually"
            
        except Exception as e:
            logger.error(f"Error extracting text from URL: {str(e)}")
            return "Social media post content - please verify manually"
    
    def get_supported_platforms(self) -> List[str]:
        """Get list of supported social media platforms"""
        return list(self.supported_platforms.keys())
    
    def validate_url(self, url: str) -> Dict:
        """
        Validate if URL is supported social media link
        
        Args:
            url: URL to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            if not url or not isinstance(url, str):
                return {
                    'valid': False,
                    'error': 'Invalid URL format'
                }
            
            platform = self._detect_platform(url)
            
            if platform == 'unknown':
                return {
                    'valid': False,
                    'error': 'Unsupported social media platform',
                    'supported_platforms': self.get_supported_platforms()
                }
            
            return {
                'valid': True,
                'platform': platform,
                'url': url
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }