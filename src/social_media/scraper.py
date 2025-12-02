"""
Social Media Content Scraper
Extracts text content from social media posts for fake news verification
"""

import re
import json
import logging
from typing import Dict, List, Optional
from urllib.parse import urlparse, parse_qs
import requests
from bs4 import BeautifulSoup
from ..preprocessing.text_cleaner import TextCleaner

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
        self.text_cleaner = TextCleaner()
        
        # Request headers to mimic browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Alternative methods for content extraction
        self.alternative_extractors = {
            'nitter': 'https://nitter.net',
            'twitpic': 'https://r.jina.ai/http://twitter.com',
            'textise': 'https://r.jina.ai/http://'
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
        """Extract content from Instagram URL using web scraping"""
        try:
            # Try to fetch the Instagram page
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract meta tags that often contain the caption
            caption = ""
            
            # Try different meta tags that Instagram uses
            meta_selectors = [
                'meta[property="og:description"]',
                'meta[name="description"]',
                'meta[property="og:title"]'
            ]
            
            for selector in meta_selectors:
                meta_tag = soup.select_one(selector)
                if meta_tag and meta_tag.get('content'):
                    caption = meta_tag['content']
                    break
            
            # Also try to find script tags with embedded data
            scripts = soup.find_all('script', type='application/ld+json')
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict) and 'articleBody' in data:
                        caption = data['articleBody']
                        break
                except:
                    continue
            
            # If still no caption, try to extract from page title
            if not caption:
                title_tag = soup.find('title')
                if title_tag:
                    caption = title_tag.get_text().strip()
            
            if not caption:
                caption = "Instagram post content - could not extract automatically"
            
            return {
                'success': True,
                'platform': 'instagram',
                'content': caption,
                'metadata': {
                    'url': url,
                    'type': 'post',
                    'extracted_method': 'web_scraping'
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting Instagram content: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to extract Instagram content: {str(e)}',
                'platform': 'instagram',
                'content': self._extract_text_from_url(url),
                'metadata': {
                    'url': url,
                    'type': 'post',
                    'extracted_method': 'fallback_url_analysis'
                }
            }
    
    def _extract_facebook_content(self, url: str) -> Dict:
        """Extract content from Facebook URL using web scraping"""
        try:
            # Try to fetch the Facebook page
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract content from Facebook page
            content = ""
            
            # Try meta tags first
            meta_selectors = [
                'meta[property="og:description"]',
                'meta[name="description"]',
                'meta[property="og:title"]'
            ]
            
            for selector in meta_selectors:
                meta_tag = soup.select_one(selector)
                if meta_tag and meta_tag.get('content'):
                    content = meta_tag['content']
                    break
            
            # Try to find post content in divs
            if not content:
                content_selectors = [
                    '[data-testid="post_message"]',
                    '.userContent',
                    '.mtm',
                    'div[data-ft]'
                ]
                
                for selector in content_selectors:
                    element = soup.select_one(selector)
                    if element:
                        content = element.get_text().strip()
                        if len(content) > 10:  # Ensure we got meaningful content
                            break
            
            # Fallback to title
            if not content:
                title_tag = soup.find('title')
                if title_tag:
                    content = title_tag.get_text().strip()
            
            if not content:
                content = "Facebook post content - could not extract automatically"
            
            return {
                'success': True,
                'platform': 'facebook',
                'content': content,
                'metadata': {
                    'url': url,
                    'type': 'post',
                    'extracted_method': 'web_scraping'
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting Facebook content: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to extract Facebook content: {str(e)}',
                'platform': 'facebook',
                'content': self._extract_text_from_url(url),
                'metadata': {
                    'url': url,
                    'type': 'post',
                    'extracted_method': 'fallback_url_analysis'
                }
            }
    
    def _extract_twitter_content(self, url: str) -> Dict:
        """Extract content from Twitter/X URL using multiple methods"""
        
        # Method 1: Try direct scraping first
        direct_result = self._try_direct_twitter_scraping(url)
        if direct_result['success'] and not self._is_javascript_blocked(direct_result['content']):
            return direct_result
        
        # Method 2: Try alternative services (nitter, jina.ai)
        alternative_result = self._try_alternative_twitter_extraction(url)
        if alternative_result['success']:
            return alternative_result
        
        # Method 3: Fallback to URL analysis
        logger.warning(f"All extraction methods failed for {url}, using fallback")
        return {
            'success': True,
            'platform': 'twitter',
            'content': self._extract_text_from_url(url),
            'metadata': {
                'url': url,
                'type': 'tweet',
                'extracted_method': 'fallback_url_analysis'
            }
        }
    
    def _try_direct_twitter_scraping(self, url: str) -> Dict:
        """Try to extract content directly from Twitter/X"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Check if JavaScript is blocked
            if self._is_javascript_blocked(response.text):
                return {
                    'success': False,
                    'error': 'JavaScript blocked - cannot extract content directly'
                }
            
            # Extract tweet content
            tweet_text = ""
            
            # Try multiple selectors for tweet content
            tweet_selectors = [
                '[data-testid="tweetText"]',
                '.tweet-text',
                '.js-tweet-text-container',
                '[data-aria-label]',
                'p'
            ]
            
            for selector in tweet_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text().strip()
                    if len(text) > 10 and not text.startswith('Twitter') and not text.startswith('We'):
                        tweet_text = text
                        break
                if tweet_text:
                    break
            
            # Try meta tags as fallback
            if not tweet_text:
                meta_selectors = [
                    'meta[property="og:description"]',
                    'meta[name="description"]',
                    'meta[property="og:title"]'
                ]
                
                for selector in meta_selectors:
                    meta_tag = soup.select_one(selector)
                    if meta_tag and meta_tag.get('content'):
                        tweet_text = meta_tag['content']
                        break
            
            # Try title as last resort
            if not tweet_text:
                title_tag = soup.find('title')
                if title_tag:
                    title_text = title_tag.get_text().strip()
                    # Remove "Twitter" suffix if present
                    if title_text.endswith(' / X'):
                        tweet_text = title_text[:-4].strip()
                    elif title_text.endswith(' / Twitter'):
                        tweet_text = title_text[:-9].strip()
                    else:
                        tweet_text = title_text
            
            if not tweet_text:
                tweet_text = "Twitter/X post content - could not extract automatically"
            
            return {
                'success': True,
                'platform': 'twitter',
                'content': tweet_text,
                'metadata': {
                    'url': url,
                    'type': 'tweet',
                    'extracted_method': 'direct_scraping'
                }
            }
            
        except Exception as e:
            logger.error(f"Direct Twitter scraping failed: {str(e)}")
            return {
                'success': False,
                'error': f'Direct scraping failed: {str(e)}'
            }
    
    def _try_alternative_twitter_extraction(self, url: str) -> Dict:
        """Try alternative services to extract Twitter content"""
        
        # Method 1: Try Nitter (privacy-focused Twitter frontend)
        nitter_url = url.replace('x.com/', 'nitter.net/').replace('twitter.com/', 'nitter.net/')
        
        try:
            response = requests.get(nitter_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Nitter-specific selectors
            tweet_selectors = [
                '.tweet-content',
                '.main-tweet',
                'p'
            ]
            
            for selector in tweet_selectors:
                element = soup.select_one(selector)
                if element:
                    text = element.get_text().strip()
                    if len(text) > 10 and not text.startswith('Nitter'):
                        return {
                            'success': True,
                            'platform': 'twitter',
                            'content': text,
                            'metadata': {
                                'url': url,
                                'type': 'tweet',
                                'extracted_method': 'nitter_alternative'
                            }
                        }
        except Exception as e:
            logger.debug(f"Nitter extraction failed: {str(e)}")
        
        # Method 2: Try Jina.ai summarizer
        try:
            jina_url = f"https://r.jina.ai/http://{url.replace('https://', '').replace('http://', '')}"
            response = requests.get(jina_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            if response.status_code == 200:
                content = response.text.strip()
                if len(content) > 20 and not self._is_javascript_blocked(content):
                    return {
                        'success': True,
                        'platform': 'twitter',
                        'content': content,
                        'metadata': {
                            'url': url,
                            'type': 'tweet',
                            'extracted_method': 'jina_ai_alternative'
                        }
                    }
        except Exception as e:
            logger.debug(f"Jina.ai extraction failed: {str(e)}")
        
        return {
            'success': False,
            'error': 'All alternative extraction methods failed'
        }
    
    def _is_javascript_blocked(self, content: str) -> bool:
        """Check if JavaScript is blocking content extraction"""
        blocked_indicators = [
            'JavaScript is disabled',
            'Please enable JavaScript',
            'switch to a supported browser',
            'list of supported browsers',
            'Help Center'
        ]
        
        content_lower = content.lower()
        return any(indicator.lower() in content_lower for indicator in blocked_indicators)
    
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
    
    def extract_and_clean_content(self, url: str, raw_text: str = None) -> Dict:
        """
        Extract content from social media URL and clean it for analysis
        
        Args:
            url: Social media URL
            raw_text: Optional raw text content (if already available)
            
        Returns:
            Dictionary containing cleaned content and metadata
        """
        try:
            platform = self._detect_platform(url)
            
            # If raw text is provided, use it directly
            if raw_text:
                content = raw_text
                extraction_method = 'provided_text'
            else:
                # Extract content from URL
                extraction_result = self.extract_content_from_url(url)
                if not extraction_result['success']:
                    return extraction_result
                
                content = extraction_result['content']
                extraction_method = extraction_result['metadata']['extracted_method']
            
            # Clean the content using text cleaner
            if platform:
                clean_result = self.text_cleaner.clean_social_media_text(content, platform)
            else:
                # Use general text cleaning if platform not detected
                clean_result = self.text_cleaner.clean_text(content, level='standard')
            
            # Extract text only (for ML processing)
            text_only = self.text_cleaner.extract_text_only(content)
            
            return {
                'success': True,
                'platform': platform,
                'original_content': content,
                'cleaned_content': clean_result['cleaned_text'],
                'text_only': text_only,
                'metadata': {
                    'url': url,
                    'platform': platform,
                    'extraction_method': extraction_method,
                    'changes_made': clean_result['changes_made'],
                    'statistics': clean_result['statistics']
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting and cleaning content from {url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'platform': 'error'
            }
    
    def process_social_media_post(self, post_data: Dict) -> Dict:
        """
        Process social media post data (URL + text content)
        
        Args:
            post_data: Dictionary containing 'url' and optionally 'text'
            
        Returns:
            Dictionary containing processed and cleaned content
        """
        try:
            url = post_data.get('url', '')
            text = post_data.get('text', '')
            
            if not url:
                return {
                    'success': False,
                    'error': 'URL is required'
                }
            
            # Extract and clean content
            result = self.extract_and_clean_content(url, text)
            
            if result['success']:
                # Add language detection
                if result['text_only']:
                    lang_detection = self.text_cleaner.language_detector.detect_language(result['text_only'])
                    result['language_analysis'] = lang_detection
                else:
                    result['language_analysis'] = {'primary_language': 'unknown'}
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing social media post: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
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