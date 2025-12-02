"""
Text Feature Extraction Module
Extracts basic text features for fake news detection
"""

import re
import string
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np

class TextFeatures:
    """
    Extract basic text features including:
    - Text length statistics
    - Punctuation ratios
    - Case ratios
    - Character distributions
    - Word-level statistics
    """
    
    def __init__(self):
        """Initialize text feature extractor"""
        # Punctuation categories
        self.arabic_punctuation = '،؛؟!.؛،«»""''()[]{}'
        self.latin_punctuation = string.punctuation
        self.all_punctuation = self.arabic_punctuation + self.latin_punctuation
        
        # Character sets
        self.arabic_chars = set('ابثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئىة')
        self.latin_chars = set(string.ascii_letters)
        self.digits = set('0123456789٠١٢٣٤٥٦٧٨٩')
        
        # Question and exclamation indicators
        self.question_marks = '؟?؟'
        self.exclamation_marks = '!！!؟؟'
        
    def extract_features(self, text: str) -> Dict:
        """
        Extract comprehensive text features
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing extracted features
        """
        if not text or not text.strip():
            return self._empty_features()
        
        # Basic text statistics
        length_features = self._extract_length_features(text)
        
        # Character-level features
        char_features = self._extract_character_features(text)
        
        # Word-level features
        word_features = self._extract_word_features(text)
        
        # Punctuation features
        punct_features = self._extract_punctuation_features(text)
        
        # Case features (for Latin text)
        case_features = self._extract_case_features(text)
        
        # Special patterns
        pattern_features = self._extract_pattern_features(text)
        
        # Combine all features
        all_features = {}
        all_features.update(length_features)
        all_features.update(char_features)
        all_features.update(word_features)
        all_features.update(punct_features)
        all_features.update(case_features)
        all_features.update(pattern_features)
        
        return all_features
    
    def _empty_features(self) -> Dict:
        """Return empty feature dictionary"""
        return {
            'text_length': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'char_count': 0,
            'arabic_char_ratio': 0,
            'latin_char_ratio': 0,
            'digit_ratio': 0,
            'punctuation_ratio': 0,
            'uppercase_ratio': 0,
            'lowercase_ratio': 0,
            'question_ratio': 0,
            'exclamation_ratio': 0,
            'url_count': 0,
            'email_count': 0,
            'mention_count': 0,
            'hashtag_count': 0,
            'number_ratio': 0,
            'whitespace_ratio': 0,
            'unique_word_ratio': 0,
            'repeated_char_ratio': 0
        }
    
    def _extract_length_features(self, text: str) -> Dict:
        """Extract text length and count features"""
        words = text.split()
        sentences = re.split(r'[.!?؟]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            'text_length': len(text),
            'char_count': len(text.replace(' ', '')),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': np.mean([len(sent.split()) for sent in sentences]) if sentences else 0,
            'max_word_length': max([len(word) for word in words]) if words else 0,
            'min_word_length': min([len(word) for word in words]) if words else 0
        }
    
    def _extract_character_features(self, text: str) -> Dict:
        """Extract character-level features"""
        total_chars = len(text)
        if total_chars == 0:
            return {
                'arabic_char_ratio': 0,
                'latin_char_ratio': 0,
                'digit_ratio': 0,
                'whitespace_ratio': 0,
                'arabic_char_count': 0,
                'latin_char_count': 0,
                'digit_count': 0,
                'whitespace_count': 0
            }
        
        arabic_count = sum(1 for char in text if char in self.arabic_chars)
        latin_count = sum(1 for char in text if char in self.latin_chars)
        digit_count = sum(1 for char in text if char in self.digits)
        whitespace_count = sum(1 for char in text if char.isspace())
        
        return {
            'arabic_char_ratio': arabic_count / total_chars,
            'latin_char_ratio': latin_count / total_chars,
            'digit_ratio': digit_count / total_chars,
            'whitespace_ratio': whitespace_count / total_chars,
            'arabic_char_count': arabic_count,
            'latin_char_count': latin_count,
            'digit_count': digit_count,
            'whitespace_count': whitespace_count
        }
    
    def _extract_word_features(self, text: str) -> Dict:
        """Extract word-level features"""
        words = text.split()
        if not words:
            return {
                'unique_word_ratio': 0,
                'repeated_word_ratio': 0,
                'long_word_ratio': 0,
                'short_word_ratio': 0,
                'avg_word_frequency': 0,
                'lexical_diversity': 0
            }
        
        # Word frequency analysis
        word_freq = Counter(words)
        unique_words = len(word_freq)
        total_words = len(words)
        
        # Calculate ratios
        unique_ratio = unique_words / total_words
        repeated_words = sum(1 for count in word_freq.values() if count > 1)
        repeated_ratio = repeated_words / total_words
        
        # Long and short words (thresholds can be adjusted)
        long_words = [w for w in words if len(w) > 8]
        short_words = [w for w in words if len(w) < 4]
        
        long_ratio = len(long_words) / total_words
        short_ratio = len(short_words) / total_words
        
        # Average word frequency
        avg_frequency = np.mean(list(word_freq.values()))
        
        # Lexical diversity (Type-Token Ratio)
        lexical_diversity = unique_ratio
        
        return {
            'unique_word_ratio': unique_ratio,
            'repeated_word_ratio': repeated_ratio,
            'long_word_ratio': long_ratio,
            'short_word_ratio': short_ratio,
            'avg_word_frequency': avg_frequency,
            'lexical_diversity': lexical_diversity,
            'unique_word_count': unique_words,
            'repeated_word_count': repeated_words,
            'long_word_count': len(long_words),
            'short_word_count': len(short_words)
        }
    
    def _extract_punctuation_features(self, text: str) -> Dict:
        """Extract punctuation-related features"""
        total_chars = len(text)
        if total_chars == 0:
            return {
                'punctuation_ratio': 0,
                'question_ratio': 0,
                'exclamation_ratio': 0,
                'comma_ratio': 0,
                'period_ratio': 0,
                'punctuation_count': 0
            }
        
        # Count different types of punctuation
        punct_count = sum(1 for char in text if char in self.all_punctuation)
        question_count = sum(1 for char in text if char in self.question_marks)
        exclamation_count = sum(1 for char in text if char in self.exclamation_marks)
        comma_count = sum(1 for char in text if char in ',،')
        period_count = sum(1 for char in text if char in '.؟')
        
        return {
            'punctuation_ratio': punct_count / total_chars,
            'question_ratio': question_count / total_chars,
            'exclamation_ratio': exclamation_count / total_chars,
            'comma_ratio': comma_count / total_chars,
            'period_ratio': period_count / total_chars,
            'punctuation_count': punct_count,
            'question_count': question_count,
            'exclamation_count': exclamation_count,
            'comma_count': comma_count,
            'period_count': period_count
        }
    
    def _extract_case_features(self, text: str) -> Dict:
        """Extract case-related features (mainly for Latin text)"""
        latin_chars = [char for char in text if char in self.latin_chars]
        if not latin_chars:
            return {
                'uppercase_ratio': 0,
                'lowercase_ratio': 0,
                'titlecase_ratio': 0,
                'uppercase_count': 0,
                'lowercase_count': 0,
                'titlecase_count': 0
            }
        
        uppercase_count = sum(1 for char in latin_chars if char.isupper())
        lowercase_count = sum(1 for char in latin_chars if char.islower())
        
        # Title case (first letter of word capitalized)
        words = text.split()
        titlecase_count = sum(1 for word in words if word and word[0].isupper() and word[0] in self.latin_chars)
        
        total_latin = len(latin_chars)
        
        return {
            'uppercase_ratio': uppercase_count / total_latin,
            'lowercase_ratio': lowercase_count / total_latin,
            'titlecase_ratio': titlecase_count / len(words) if words else 0,
            'uppercase_count': uppercase_count,
            'lowercase_count': lowercase_count,
            'titlecase_count': titlecase_count
        }
    
    def _extract_pattern_features(self, text: str) -> Dict:
        """Extract special pattern features"""
        # URL pattern
        url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        urls = url_pattern.findall(text)
        
        # Email pattern
        email_pattern = re.compile(r'\S+@\S+\.\S+')
        emails = email_pattern.findall(text)
        
        # Mention pattern (@username)
        mention_pattern = re.compile(r'@\w+')
        mentions = mention_pattern.findall(text)
        
        # Hashtag pattern
        hashtag_pattern = re.compile(r'#\w+')
        hashtags = hashtag_pattern.findall(text)
        
        # Number patterns
        number_pattern = re.compile(r'\d+')
        numbers = number_pattern.findall(text)
        
        # Repeated characters (e.g., "!!!!", "aaaa")
        repeated_char_pattern = re.compile(r'(.)\1{2,}')
        repeated_chars = repeated_char_pattern.findall(text)
        
        # Capitalized words (all caps)
        caps_pattern = re.compile(r'\b[A-Z]{2,}\b')
        caps_words = caps_pattern.findall(text)
        
        # Calculate ratios
        total_words = len(text.split())
        total_chars = len(text)
        
        return {
            'url_count': len(urls),
            'email_count': len(emails),
            'mention_count': len(mentions),
            'hashtag_count': len(hashtags),
            'number_count': len(numbers),
            'repeated_char_count': len(repeated_chars),
            'caps_word_count': len(caps_words),
            'url_ratio': len(urls) / total_words if total_words > 0 else 0,
            'email_ratio': len(emails) / total_words if total_words > 0 else 0,
            'mention_ratio': len(mentions) / total_words if total_words > 0 else 0,
            'hashtag_ratio': len(hashtags) / total_words if total_words > 0 else 0,
            'number_ratio': len(numbers) / total_words if total_words > 0 else 0,
            'repeated_char_ratio': len(repeated_chars) / total_chars if total_chars > 0 else 0,
            'caps_word_ratio': len(caps_words) / total_words if total_words > 0 else 0
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return list(self._empty_features().keys())
    
    def extract_feature_vector(self, text: str) -> List[float]:
        """
        Extract features as a numeric vector
        
        Args:
            text: Input text
            
        Returns:
            List of feature values
        """
        features = self.extract_features(text)
        feature_names = self.get_feature_names()
        return [features.get(name, 0.0) for name in feature_names]
    
    def extract_batch_features(self, texts: List[str]) -> List[Dict]:
        """
        Extract features from multiple texts
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of feature dictionaries
        """
        return [self.extract_features(text) for text in texts]
    
    def extract_batch_vectors(self, texts: List[str]) -> List[List[float]]:
        """
        Extract feature vectors from multiple texts
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of feature vectors
        """
        return [self.extract_feature_vector(text) for text in texts]