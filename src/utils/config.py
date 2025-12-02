"""
Configuration settings for the fake news detection system
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for the fake news detection system"""
    
    # API Configuration
    HAQIQA_API_URL = os.getenv('HAQIQA_API_URL', 'https://haqiqa-api.example.com/predict')
    HAQIQA_API_KEY = os.getenv('HAQIQA_API_KEY', '')
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))
    
    # Feature Extraction Settings
    MAX_FEATURES = int(os.getenv('MAX_FEATURES', '10000'))
    NGRAM_RANGE = (1, 2)  # unigrams and bigrams
    MIN_DF = int(os.getenv('MIN_DF', '2'))
    MAX_DF = float(os.getenv('MAX_DF', '0.95'))
    
    # Language Detection Settings
    SUPPORTED_LANGUAGES = ['ar', 'fr', 'en']
    DEFAULT_LANGUAGE = 'ar'
    
    # Text Processing Settings
    MIN_TEXT_LENGTH = int(os.getenv('MIN_TEXT_LENGTH', '10'))
    MAX_TEXT_LENGTH = int(os.getenv('MAX_TEXT_LENGTH', '10000'))
    
    # Risk Scoring Weights
    HAQIQA_WEIGHT = float(os.getenv('HAQIQA_WEIGHT', '0.6'))
    FEATURE_WEIGHT = float(os.getenv('FEATURE_WEIGHT', '0.4'))
    
    # Threshold Settings
    FAKE_NEWS_THRESHOLD = float(os.getenv('FAKE_NEWS_THRESHOLD', '0.5'))
    
    # File Paths
    DATA_DIR = os.getenv('DATA_DIR', 'data')
    MODELS_DIR = os.getenv('MODELS_DIR', 'models')
    LEXICONS_DIR = os.getenv('LEXICONS_DIR', 'lexicons')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/fake_news_detector.log')
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        errors = []
        
        if not cls.HAQIQA_API_URL:
            errors.append("HAQIQA_API_URL is required")
        
        if cls.HAQIQA_WEIGHT + cls.FEATURE_WEIGHT != 1.0:
            errors.append("HAQIQA_WEIGHT and FEATURE_WEIGHT must sum to 1.0")
        
        if cls.FAKE_NEWS_THRESHOLD < 0 or cls.FAKE_NEWS_THRESHOLD > 1:
            errors.append("FAKE_NEWS_THRESHOLD must be between 0 and 1")
        
        return errors