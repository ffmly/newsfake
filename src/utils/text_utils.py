"""
Text utilities for Arabic/Darija text processing
"""

import re
from typing import List, Dict, Optional

class TextUtils:
    """Utility functions for text processing"""
    
    @staticmethod
    def clean_text_for_ml(text: str) -> str:
        """
        Clean text for machine learning processing
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation but keep basic separators
        text = re.sub(r'[^\w\s]', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def tokenize_arabic(text: str) -> List[str]:
        """
        Simple tokenization for Arabic text
        
        Args:
            text: Arabic text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Arabic word boundaries (simplified)
        tokens = re.findall(r'[\u0600-\u06FF]+', text)
        return [token for token in tokens if token.strip()]
    
    @staticmethod
    def is_arabic_text(text: str) -> bool:
        """
        Check if text contains Arabic characters
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains Arabic characters
        """
        return bool(re.search(r'[\u0600-\u06FF]', text))
    
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """
        Extract URLs from text
        
        Args:
            text: Text to extract URLs from
            
        Returns:
            List of URLs found in text
        """
        url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        return url_pattern.findall(text)
    
    @staticmethod
    def extract_mentions(text: str) -> List[str]:
        """
        Extract mentions from text
        
        Args:
            text: Text to extract mentions from
            
        Returns:
            List of mentions found in text
        """
        mention_pattern = re.compile(r'@\w+')
        return mention_pattern.findall(text)
    
    @staticmethod
    def extract_hashtags(text: str) -> List[str]:
        """
        Extract hashtags from text
        
        Args:
            text: Text to extract hashtags from
            
        Returns:
            List of hashtags found in text
        """
        hashtag_pattern = re.compile(r'#\w+')
        return hashtag_pattern.findall(text)
    
    @staticmethod
    def count_arabic_words(text: str) -> int:
        """
        Count Arabic words in text
        
        Args:
            text: Text to count Arabic words in
            
        Returns:
            Number of Arabic words
        """
        if not text:
            return 0
        
        arabic_words = re.findall(r'[\u0600-\u06FF]+', text)
        return len(arabic_words)
    
    @staticmethod
    def calculate_readability_score(text: str) -> float:
        """
        Calculate a simple readability score for text
        
        Args:
            text: Text to analyze
            
        Returns:
            Readability score (0-1, higher is more readable)
        """
        if not text:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        # Simple metrics
        avg_word_length = sum(len(word) for word in words) / len(words)
        total_chars = len(text)
        
        # Simple readability formula (inverse of complexity)
        readability = min(1.0, (avg_word_length / 10) * (100 / total_chars))
        
        return readability