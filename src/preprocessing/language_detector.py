"""
Language Detection Module for Arabic/Darija/French/English
Supports code-switching detection and language identification
"""

import re
from typing import Dict, List, Tuple
from langdetect import detect, DetectorFactory
from ..utils.config import Config

# Set seed for consistent language detection
DetectorFactory.seed = 0

class LanguageDetector:
    """
    Detects language and code-switching in multilingual text
    Supports Arabic, Darija, French, and English
    """
    
    def __init__(self):
        """Initialize language detector with character patterns"""
        # Arabic character ranges
        self.arabic_chars = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
        
        # Darija-specific patterns (Latin script with Arabic numbers/symbols)
        self.darija_latin_patterns = [
            r'\b[3-7]\w*',  # Arabic numbers in Latin script (3=ain, 7=ha, etc.)
            r'\b[a-zA-Z]+[3-7][a-zA-Z]*\b',  # Mixed Latin-Arabic
            r'\b[gh][a-zA-Z]*\b',  # Common Darija patterns
        ]
        
        # French indicators
        self.french_indicators = [
            'le', 'la', 'les', 'de', 'du', 'des', 'et', 'est', 'dans', 'pour',
            'avec', 'par', 'sur', 'que', 'qui', 'quoi', 'où', 'quand', 'comment',
            'pourquoi', 'mais', 'ou', 'donc', 'or', 'ni', 'car', 'ne', 'pas',
            'très', 'plus', 'moins', 'bien', 'mal', 'bon', 'mauvais', 'grand',
            'petit', 'nouveau', 'vieux', 'jeune', 'beau', 'joli'
        ]
        
        # English indicators
        self.english_indicators = [
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that',
            'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could'
        ]
        
        # Darija-specific words (in Arabic script)
        self.darija_words = [
            'زوين', 'بحال', 'كاين', 'ماكاينش', 'دار', 'ديال', 'فاش', 'فاشكاين',
            'معاك', 'معاكum', 'شنو', 'اشنو', 'كيفاش', 'فين', 'واش', 'لا', 'به',
            'بلا', 'حتى', 'ولا', 'أو', 'ملي', 'لي', 'اللي', 'هاد', 'هادي',
            'هناك', 'هنا', 'تم', 'تما', 'دابا', 'داباا', 'باه', 'باهي'
        ]
    
    def detect_language(self, text: str) -> Dict:
        """
        Detect the primary language and code-switching patterns
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing language detection results
        """
        if not text or not text.strip():
            return {
                'primary_language': 'unknown',
                'confidence': 0.0,
                'is_code_switched': False,
                'language_distribution': {},
                'segments': []
            }
        
        # Clean text for detection
        clean_text = self._clean_for_detection(text)
        
        # Detect primary language using langdetect
        try:
            detected_lang = detect(clean_text)
            confidence = 0.8  # Default confidence for langdetect
        except:
            detected_lang = 'unknown'
            confidence = 0.0
        
        # Analyze character distribution
        char_analysis = self._analyze_characters(text)
        
        # Detect code-switching
        code_switching = self._detect_code_switching(text)
        
        # Determine primary language based on multiple factors
        primary_lang = self._determine_primary_language(
            detected_lang, char_analysis, code_switching
        )
        
        # Calculate language distribution
        lang_distribution = self._calculate_language_distribution(text)
        
        return {
            'primary_language': primary_lang,
            'confidence': confidence,
            'is_code_switched': code_switching['is_switched'],
            'language_distribution': lang_distribution,
            'segments': code_switching['segments'],
            'character_analysis': char_analysis,
            'langdetect_result': detected_lang
        }
    
    def _clean_for_detection(self, text: str) -> str:
        """Clean text for language detection"""
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _analyze_characters(self, text: str) -> Dict:
        """Analyze character distribution in text"""
        total_chars = len(text)
        if total_chars == 0:
            return {'arabic': 0.0, 'latin': 0.0, 'digits': 0.0, 'other': 0.0}
        
        arabic_count = len(self.arabic_chars.findall(text))
        latin_count = len(re.findall(r'[a-zA-Z]', text))
        digit_count = len(re.findall(r'\d', text))
        other_count = total_chars - arabic_count - latin_count - digit_count
        
        return {
            'arabic': arabic_count / total_chars,
            'latin': latin_count / total_chars,
            'digits': digit_count / total_chars,
            'other': other_count / total_chars,
            'counts': {
                'arabic': arabic_count,
                'latin': latin_count,
                'digits': digit_count,
                'other': other_count,
                'total': total_chars
            }
        }
    
    def _detect_code_switching(self, text: str) -> Dict:
        """Detect code-switching patterns in text"""
        segments = []
        current_segment = {'text': '', 'language': 'unknown', 'start': 0}
        
        words = text.split()
        position = 0
        
        for i, word in enumerate(words):
            word_lang = self._detect_word_language(word)
            
            if word_lang != current_segment['language'] and i > 0:
                # Save current segment
                if current_segment['text'].strip():
                    current_segment['end'] = position - 1
                    segments.append(current_segment.copy())
                
                # Start new segment
                current_segment = {
                    'text': word + ' ',
                    'language': word_lang,
                    'start': position
                }
            else:
                current_segment['text'] += word + ' '
                if current_segment['language'] == 'unknown':
                    current_segment['language'] = word_lang
            
            position += len(word) + 1  # +1 for space
        
        # Add final segment
        if current_segment['text'].strip():
            current_segment['end'] = len(text) - 1
            segments.append(current_segment)
        
        # Determine if text is code-switched
        unique_languages = set(seg['language'] for seg in segments if seg['language'] != 'unknown')
        is_switched = len(unique_languages) > 1
        
        return {
            'is_switched': is_switched,
            'segments': segments,
            'language_count': len(unique_languages),
            'unique_languages': list(unique_languages)
        }
    
    def _detect_word_language(self, word: str) -> str:
        """Detect language of individual word"""
        # Check for Arabic characters
        if self.arabic_chars.search(word):
            # Check if it's Darija
            if self._is_darija_word(word):
                return 'darija'
            return 'arabic'
        
        # Check for Darija in Latin script
        for pattern in self.darija_latin_patterns:
            if re.search(pattern, word, re.IGNORECASE):
                return 'darija'
        
        # Check French indicators
        word_lower = word.lower()
        if word_lower in self.french_indicators:
            return 'french'
        
        # Check English indicators
        if word_lower in self.english_indicators:
            return 'english'
        
        # Check for French characteristics
        if re.search(r'[àâäéèêëïîôöùûüÿç]', word_lower):
            return 'french'
        
        # Default to Latin-based detection
        if re.search(r'[a-zA-Z]', word):
            # Simple heuristic based on common patterns
            if len(word) > 3 and any(word_lower.endswith(suffix) for suffix in ['tion', 'ment', 'ance', 'ence']):
                return 'french'
            return 'english'
        
        return 'unknown'
    
    def _is_darija_word(self, word: str) -> bool:
        """Check if word is specifically Darija"""
        word_clean = re.sub(r'[^\u0600-\u06FF]', '', word.lower())
        
        # Check against Darija word list
        if word_clean in self.darija_words:
            return True
        
        # Check for Darija patterns
        darija_patterns = [
            r'^[أإا][لن][يى]',  # Common prefixes
            r'[يى]ة$',        # Common suffix
            r'و[شن]',         # Common patterns
            r'ك[ا][ن]',       # Conjugation patterns
        ]
        
        for pattern in darija_patterns:
            if re.search(pattern, word_clean):
                return True
        
        return False
    
    def _determine_primary_language(self, langdetect_result: str, 
                                   char_analysis: Dict, code_switching: Dict) -> str:
        """Determine primary language using multiple factors"""
        
        # If mostly Arabic characters, prioritize Arabic variants
        if char_analysis['arabic'] > 0.5:
            # Check if Darija
            darija_segments = [s for s in code_switching['segments'] if s['language'] == 'darija']
            if len(darija_segments) > len(code_switching['segments']) * 0.3:
                return 'darija'
            return 'arabic'
        
        # If mostly Latin characters, use langdetect result
        if char_analysis['latin'] > 0.5:
            if langdetect_result in ['fr', 'en']:
                return langdetect_result
            return 'unknown'
        
        # Use langdetect as fallback
        lang_mapping = {
            'ar': 'arabic',
            'fr': 'french', 
            'en': 'english'
        }
        
        return lang_mapping.get(langdetect_result, 'unknown')
    
    def _calculate_language_distribution(self, text: str) -> Dict:
        """Calculate percentage distribution of languages in text"""
        segments = self._detect_code_switching(text)['segments']
        
        if not segments:
            return {'unknown': 1.0}
        
        total_length = sum(len(seg['text']) for seg in segments)
        if total_length == 0:
            return {'unknown': 1.0}
        
        distribution = {}
        for segment in segments:
            lang = segment['language']
            length = len(segment['text'])
            distribution[lang] = distribution.get(lang, 0) + length
        
        # Convert to percentages
        for lang in distribution:
            distribution[lang] = distribution[lang] / total_length
        
        return distribution
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return ['arabic', 'darija', 'french', 'english', 'unknown']
    
    def is_arabic_text(self, text: str) -> bool:
        """Check if text is primarily Arabic"""
        char_analysis = self._analyze_characters(text)
        return char_analysis['arabic'] > 0.3
    
    def is_darija_text(self, text: str) -> bool:
        """Check if text contains Darija"""
        detection = self.detect_language(text)
        return detection['primary_language'] == 'darija' or 'darija' in detection['language_distribution']