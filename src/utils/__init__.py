"""
Utility functions for fake news detection system
"""

from .config import Config
from .text_utils import TextUtils

# Import individual functions from language_utils
from .language_utils import (
    is_arabic_text,
    is_latin_arabic,
    detect_script_type,
    normalize_arabic_text,
    extract_arabic_features,
    clean_darija_text,
    get_language_confidence
)

__all__ = [
    'Config',
    'TextUtils',
    'is_arabic_text',
    'is_latin_arabic',
    'detect_script_type',
    'normalize_arabic_text',
    'extract_arabic_features',
    'clean_darija_text',
    'get_language_confidence'
]