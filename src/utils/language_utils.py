"""
Language utilities for Arabic/Darija text processing
"""

import re
import sys
import os
from typing import Dict, List, Tuple

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

def is_arabic_text(text: str) -> bool:
    """Check if text contains Arabic characters"""
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    return bool(arabic_pattern.search(text))

def is_latin_arabic(text: str) -> bool:
    """Check if text contains Latin script with Arabic words (Darija)"""
    darija_words = [
        'salam', 'marhaba', 'shukran', 'mafi', 'mushkil', 'bezaf', 
        'khoud', 'jey', 'mzyan', 'la', 'kan', 'kayn', 'machi',
        'bghit', 'n9der', 'm3ana', '3la', 'f', 'b', 'men'
    ]
    text_lower = text.lower()
    return any(word in text_lower for word in darija_words)

def detect_script_type(text: str) -> str:
    """Detect the script type of the text"""
    if is_arabic_text(text):
        return 'arabic'
    elif is_latin_arabic(text):
        return 'latin_arabic'
    else:
        return 'latin'

def normalize_arabic_text(text: str) -> str:
    """Normalize Arabic text by removing diacritics and standardizing characters"""
    # Remove diacritics
    text = re.sub(r'[\u064B-\u065F\u0670\u06D6-\u06ED\u08E3-\u08FF\uFB8B\uFC5E-\uFC62]', '', text)
    
    # Standardize Arabic letters
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ة', 'ه').replace('ى', 'ي')
    text = text.replace('ؤ', 'و').replace('ئ', 'ي')
    
    return text

def extract_arabic_features(text: str) -> Dict[str, int]:
    """Extract Arabic-specific features from text"""
    features = {
        'arabic_chars_count': len(re.findall(r'[\u0600-\u06FF]', text)),
        'diacritics_count': len(re.findall(r'[\u064B-\u065F\u0670]', text)),
        'arabic_words_count': len(re.findall(r'[\u0600-\u06FF]+', text)),
        'question_marks_count': text.count('؟'),
        'exclamation_marks_count': text.count('！'),
        'has_arabic_punctuation': int(any(p in text for p in '،؛؟!.')),
    }
    
    return features

def clean_darija_text(text: str) -> str:
    """Clean and normalize Darija text written in Latin script"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize common Darija patterns
    replacements = {
        r'3': 'ع',
        r'7': 'ح',
        r'5': 'خ',
        r'9': 'ق',
        r'2': 'ق',
        r'4': 'ش',
        r'6': 'ط',
        r'8': 'غ',
        r'gh': 'غ',
        r'kh': 'خ',
        r'sh': 'ش',
        r'ch': 'ش',
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    return text.strip()

def get_language_confidence(text: str) -> Tuple[str, float]:
    """Get language detection with confidence score"""
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    total_chars = len(re.findall(r'[a-zA-Z\u0600-\u06FF]', text))
    
    if total_chars == 0:
        return 'unknown', 0.0
    
    arabic_ratio = arabic_chars / total_chars
    latin_ratio = latin_chars / total_chars
    
    if arabic_ratio > 0.7:
        return 'arabic', arabic_ratio
    elif latin_ratio > 0.7:
        return 'latin', latin_ratio
    elif is_latin_arabic(text):
        return 'darija_latin', 0.5
    else:
        return 'mixed', max(arabic_ratio, latin_ratio)