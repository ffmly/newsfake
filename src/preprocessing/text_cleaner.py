"""
Text Cleaning Module for Arabic/Darija/French/English
Handles normalization, tokenization, and preprocessing
"""

import re
import string
from typing import Dict, List, Tuple
from .language_detector import LanguageDetector

class TextCleaner:
    """
    Comprehensive text cleaning and normalization for multilingual text
    Supports Arabic, Darija, French, and English
    """
    
    def __init__(self):
        """Initialize text cleaner with normalization patterns"""
        self.language_detector = LanguageDetector()
        
        # Arabic normalization mappings
        self.arabic_normalization = {
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا',  # Alif variants
            'ة': 'ه', 'ت': 'ت',           # Teh marbuta
            'ى': 'ي', 'ي': 'ي',           # Alif maqsura
            'ؤ': 'و', 'ئ': 'ي',           # Hamza variants
            'ك': 'ك', 'ق': 'ق',           # Kaf and Qaf (keep separate)
        }
        
        # Darija Latin script normalization
        self.darija_latin_map = {
            '3': 'ع', '7': 'ح', '5': 'خ', '9': 'ق',
            '2': 'ء', '4': 'ش', '6': 'ط', '8': 'غ',
            'gh': 'غ', 'kh': 'خ', 'sh': 'ش', 'th': 'ث',
            'dh': 'ذ', 'zh': 'ژ', 'aa': 'آ', 'ee': 'ي',
            'oo': 'و', 'ou': 'و'
        }
        
        # Punctuation and special characters to handle
        self.punctuation = string.punctuation + '،؛؟!.؛،«»""''()[]{}'
        
        # Common Arabic diacritics to remove
        self.diacritics = re.compile(r'[\u064B-\u065F\u0670\u0640]')
        
        # URLs and mentions patterns
        self.url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        
        # Enhanced emoji patterns
        self.emoji_pattern = re.compile(
            '['
            '\U0001F600-\U0001F64F'  # emoticons
            '\U0001F300-\U0001F5FF'  # symbols & pictographs
            '\U0001F680-\U0001F6FF'  # transport & map symbols
            '\U0001F1E0-\U0001F1FF'  # flags (iOS)
            '\U00002702-\U000027B0'  # dingbats
            '\U000024C2-\U0001F251'  # enclosed characters
            '\U0001F900-\U0001F9FF'  # supplemental symbols
            '\U0001FA70-\U0001FAFF'  # symbols and pictographs extended
            ']+', flags=re.UNICODE
        )
        
        # Social media specific patterns
        self.social_media_patterns = {
            'instagram_tags': re.compile(r'#[\w]+'),
            'twitter_handles': re.compile(r'@[\w]+'),
            'tiktok_sounds': re.compile(r'♬ [\w\s]+'),
            'facebook_reactions': re.compile(r'👍|👎|❤️|😂|😮|😢|😡'),
            'instagram_stories': re.compile(r'👤|📍|🎵|📷|🎥|🎬|🎮|🎯|🎪|🎭|🎨|🎸|🎺|🎻|🥁|🎹|🎧|🎤|🎙️|🎚️|🎛️|📻|📱|📲|💻|🖥️|⌨️|🖱️|🖨️|📠|📞|☎️|📟|📱|📲|📞|☎️|📟|📠|📱|📲|📞|☎️|📟|📠')
        }
        
        # Numbers and special patterns
        self.number_pattern = re.compile(r'\d+')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        
        # Multiple whitespace
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Arabic-specific patterns
        self.arabic_punctuation = re.compile(r'[،؛؟!.؛،«»""''()[]{}]')
        self.latin_numbers_in_arabic = re.compile(r'[0-9]')
        
    def clean_text(self, text: str, language: str = None, 
                   level: str = 'standard') -> Dict:
        """
        Clean and normalize text based on language and cleaning level
        
        Args:
            text: Input text to clean
            language: Target language (auto-detected if None)
            level: Cleaning level ('minimal', 'standard', 'aggressive')
            
        Returns:
            Dictionary containing cleaned text and metadata
        """
        if not text or not text.strip():
            return {
                'original_text': text,
                'cleaned_text': '',
                'language': 'unknown',
                'level': level,
                'changes_made': [],
                'statistics': {}
            }
        
        original_text = text
        
        # Detect language if not provided
        if language is None:
            lang_detection = self.language_detector.detect_language(text)
            language = lang_detection['primary_language']
        
        changes_made = []
        
        # Apply cleaning based on level
        if level == 'minimal':
            cleaned_text = self._minimal_cleaning(text, language, changes_made)
        elif level == 'standard':
            cleaned_text = self._standard_cleaning(text, language, changes_made)
        elif level == 'aggressive':
            cleaned_text = self._aggressive_cleaning(text, language, changes_made)
        else:
            cleaned_text = self._standard_cleaning(text, language, changes_made)
        
        # Calculate statistics
        stats = self._calculate_text_statistics(original_text, cleaned_text)
        
        return {
            'original_text': original_text,
            'cleaned_text': cleaned_text,
            'language': language,
            'level': level,
            'changes_made': changes_made,
            'statistics': stats
        }
    
    def _minimal_cleaning(self, text: str, language: str, changes: List) -> str:
        """Minimal cleaning - preserve most original content"""
        cleaned = text
        
        # Remove only URLs and emails
        if self.url_pattern.search(cleaned):
            cleaned = self.url_pattern.sub(' ', cleaned)
            changes.append('removed_urls')
        
        if self.email_pattern.search(cleaned):
            cleaned = self.email_pattern.sub(' ', cleaned)
            changes.append('removed_emails')
        
        # Normalize whitespace
        cleaned = self.whitespace_pattern.sub(' ', cleaned).strip()
        
        return cleaned
    
    def _standard_cleaning(self, text: str, language: str, changes: List) -> str:
        """Standard cleaning - balanced approach"""
        cleaned = text
        
        # Remove URLs, emails, mentions, hashtags
        if self.url_pattern.search(cleaned):
            cleaned = self.url_pattern.sub(' ', cleaned)
            changes.append('removed_urls')
        
        if self.email_pattern.search(cleaned):
            cleaned = self.email_pattern.sub(' ', cleaned)
            changes.append('removed_emails')
        
        if self.mention_pattern.search(cleaned):
            cleaned = self.mention_pattern.sub(' ', cleaned)
            changes.append('removed_mentions')
        
        if self.hashtag_pattern.search(cleaned):
            cleaned = self.hashtag_pattern.sub(' ', cleaned)
            changes.append('removed_hashtags')
        
        # Remove emojis
        if self.emoji_pattern.search(cleaned):
            cleaned = self.emoji_pattern.sub(' ', cleaned)
            changes.append('removed_emojis')
        
        # Language-specific normalization
        if language in ['arabic', 'darija']:
            cleaned = self._normalize_arabic(cleaned, changes)
        elif language == 'french':
            cleaned = self._normalize_french(cleaned, changes)
        elif language == 'english':
            cleaned = self._normalize_english(cleaned, changes)
        
        # Handle Darija Latin script
        cleaned = self._normalize_darija_latin(cleaned, changes)
        
        # Normalize punctuation
        cleaned = self._normalize_punctuation(cleaned, changes)
        
        # Normalize whitespace
        cleaned = self.whitespace_pattern.sub(' ', cleaned).strip()
        
        return cleaned
    
    def _aggressive_cleaning(self, text: str, language: str, changes: List) -> str:
        """Aggressive cleaning - maximum normalization"""
        cleaned = text
        
        # Remove all URLs, emails, mentions, hashtags
        cleaned = self.url_pattern.sub(' ', cleaned)
        cleaned = self.email_pattern.sub(' ', cleaned)
        cleaned = self.mention_pattern.sub(' ', cleaned)
        cleaned = self.hashtag_pattern.sub(' ', cleaned)
        changes.extend(['removed_urls', 'removed_emails', 'removed_mentions', 'removed_hashtags', 'removed_emojis'])
        
        # Remove emojis aggressively
        cleaned = self.emoji_pattern.sub(' ', cleaned)
        
        # Remove all numbers
        if self.number_pattern.search(cleaned):
            cleaned = self.number_pattern.sub(' ', cleaned)
            changes.append('removed_numbers')
        
        # Language-specific aggressive normalization
        if language in ['arabic', 'darija']:
            cleaned = self._aggressive_normalize_arabic(cleaned, changes)
        
        # Handle Darija Latin script
        cleaned = self._normalize_darija_latin(cleaned, changes)
        
        # Remove most punctuation
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        changes.append('removed_punctuation')
        
        # Convert to lowercase for Latin scripts
        cleaned = cleaned.lower()
        changes.append('lowercased')
        
        # Normalize whitespace
        cleaned = self.whitespace_pattern.sub(' ', cleaned).strip()
        
        return cleaned
    
    def _normalize_arabic(self, text: str, changes: List) -> str:
        """Normalize Arabic text"""
        normalized = text
        
        # Remove diacritics
        if self.diacritics.search(normalized):
            normalized = self.diacritics.sub('', normalized)
            changes.append('removed_diacritics')
        
        # Normalize Arabic letters
        for old, new in self.arabic_normalization.items():
            if old in normalized:
                normalized = normalized.replace(old, new)
        
        if any(old in text for old in self.arabic_normalization.keys()):
            changes.append('normalized_arabic_letters')
        
        # Normalize Arabic punctuation
        normalized = self.arabic_punctuation.sub(' ', normalized)
        
        # Replace Latin numbers with Arabic numbers if in Arabic context
        if self.latin_numbers_in_arabic.search(normalized):
            latin_to_arabic = {
                '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
                '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩'
            }
            for latin, arabic in latin_to_arabic.items():
                normalized = normalized.replace(latin, arabic)
            changes.append('normalized_numbers')
        
        return normalized
    
    def _aggressive_normalize_arabic(self, text: str, changes: List) -> str:
        """Aggressive Arabic normalization"""
        normalized = self._normalize_arabic(text, changes)
        
        # Remove all non-Arabic characters (except spaces)
        normalized = re.sub(r'[^\u0600-\u06FF\s]', ' ', normalized)
        changes.append('kept_arabic_only')
        
        return normalized
    
    def _normalize_french(self, text: str, changes: List) -> str:
        """Normalize French text"""
        normalized = text
        
        # Normalize French accents (optional - keep for now)
        # Could add accent removal if needed
        
        # Handle French-specific punctuation
        normalized = normalized.replace('«', '"').replace('»', '"')
        
        return normalized
    
    def _normalize_english(self, text: str, changes: List) -> str:
        """Normalize English text"""
        normalized = text
        # Basic English normalization
        return normalized
    
    def _normalize_darija_latin(self, text: str, changes: List) -> str:
        """Normalize Darija written in Latin script"""
        normalized = text.lower()
        
        # Replace Darija Latin numbers with Arabic letters
        for latin_num, arabic_letter in self.darija_latin_map.items():
            if latin_num in normalized:
                normalized = normalized.replace(latin_num, arabic_letter)
        
        if any(latin_num in text.lower() for latin_num in self.darija_latin_map.keys()):
            changes.append('normalized_darija_latin')
        
        return normalized
    
    def _normalize_punctuation(self, text: str, changes: List) -> str:
        """Normalize punctuation across languages"""
        normalized = text
        
        # Replace multiple punctuation with single
        normalized = re.sub(r'[!?]{2,}', '!', normalized)
        normalized = re.sub(r'[.]{2,}', '.', normalized)
        
        # Normalize quotes
        normalized = normalized.replace('"', '"').replace('"', '"')
        normalized = normalized.replace(''', "'").replace(''', "'")
        
        # Add spaces around punctuation for better tokenization
        normalized = re.sub(r'([,.!?;:])', r' \1 ', normalized)
        
        return normalized
    
    def _calculate_text_statistics(self, original: str, cleaned: str) -> Dict:
        """Calculate text statistics before and after cleaning"""
        return {
            'original_length': len(original),
            'cleaned_length': len(cleaned),
            'original_words': len(original.split()),
            'cleaned_words': len(cleaned.split()),
            'reduction_ratio': 1 - (len(cleaned) / len(original)) if len(original) > 0 else 0,
            'compression_ratio': len(cleaned) / len(original) if len(original) > 0 else 0
        }
    
    def tokenize_text(self, text: str, language: str = None) -> List[str]:
        """
        Tokenize text based on language
        
        Args:
            text: Text to tokenize
            language: Language for tokenization rules
            
        Returns:
            List of tokens
        """
        if not text or not text.strip():
            return []
        
        # Detect language if not provided
        if language is None:
            lang_detection = self.language_detector.detect_language(text)
            language = lang_detection['primary_language']
        
        # Basic tokenization - split on whitespace
        tokens = text.split()
        
        # Language-specific tokenization improvements
        if language in ['arabic', 'darija']:
            tokens = self._tokenize_arabic(tokens)
        elif language in ['french', 'english']:
            tokens = self._tokenize_latin(tokens)
        
        # Filter out empty tokens and very short tokens
        tokens = [token.strip() for token in tokens if len(token.strip()) > 1]
        
        return tokens
    
    def _tokenize_arabic(self, tokens: List[str]) -> List[str]:
        """Arabic-specific tokenization improvements"""
        result = []
        
        for token in tokens:
            # Handle attached punctuation
            token = re.sub(r'^([^\w\s])(\w+)', r'\1 \2', token)
            token = re.sub(r'(\w+)([^\w\s])$', r'\1 \2', token)
            
            # Split further if needed
            sub_tokens = token.split()
            result.extend(sub_tokens)
        
        return result
    
    def _tokenize_latin(self, tokens: List[str]) -> List[str]:
        """Latin script tokenization improvements"""
        result = []
        
        for token in tokens:
            # Handle contractions and special cases
            if token.endswith("'s"):
                result.extend([token[:-2], "'s"])
            elif token.endswith("n't"):
                result.extend([token[:-3], "n't"])
            else:
                result.append(token)
        
        return result
    
    def get_cleaning_levels(self) -> List[str]:
        """Get available cleaning levels"""
        return ['minimal', 'standard', 'aggressive']
    
    def preprocess_for_ml(self, text: str, language: str = None) -> str:
        """
        Preprocess text specifically for machine learning models
        
        Args:
            text: Input text
            language: Text language
            
        Returns:
            Preprocessed text ready for ML models
        """
        # Use standard cleaning for ML
        result = self.clean_text(text, language, level='standard')
        cleaned_text = result['cleaned_text']
        
        # Additional ML-specific preprocessing
        # Convert to lowercase for consistency
        cleaned_text = cleaned_text.lower()
        
        # Remove extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
    
    def clean_social_media_text(self, text: str, platform: str = None) -> Dict:
        """
        Clean social media text specifically for fake news detection
        
        Args:
            text: Social media text to clean
            platform: Platform type (instagram, facebook, twitter, tiktok, etc.)
            
        Returns:
            Dictionary containing cleaned social media text and metadata
        """
        if not text or not text.strip():
            return {
                'original_text': text,
                'cleaned_text': '',
                'platform': platform,
                'changes_made': [],
                'statistics': {}
            }
        
        original_text = text
        changes_made = []
        cleaned = text
        
        # Remove URLs aggressively
        if self.url_pattern.search(cleaned):
            cleaned = self.url_pattern.sub(' ', cleaned)
            changes_made.append('removed_urls')
        
        # Remove emojis and special characters
        if self.emoji_pattern.search(cleaned):
            cleaned = self.emoji_pattern.sub(' ', cleaned)
            changes_made.append('removed_emojis')
        
        # Platform-specific cleaning
        if platform == 'instagram':
            # Remove Instagram-specific elements
            cleaned = self.social_media_patterns['instagram_tags'].sub(' ', cleaned)
            cleaned = self.social_media_patterns['instagram_stories'].sub(' ', cleaned)
            changes_made.append('cleaned_instagram_elements')
            
        elif platform == 'twitter':
            # Remove Twitter-specific elements
            cleaned = self.social_media_patterns['twitter_handles'].sub(' ', cleaned)
            changes_made.append('cleaned_twitter_handles')
            
        elif platform == 'facebook':
            # Remove Facebook reactions
            cleaned = self.social_media_patterns['facebook_reactions'].sub(' ', cleaned)
            changes_made.append('cleaned_facebook_reactions')
            
        elif platform == 'tiktok':
            # Remove TikTok-specific elements
            cleaned = self.social_media_patterns['tiktok_sounds'].sub(' ', cleaned)
            changes_made.append('cleaned_tiktok_sounds')
        
        # Remove mentions and hashtags
        if self.mention_pattern.search(cleaned):
            cleaned = self.mention_pattern.sub(' ', cleaned)
            changes_made.append('removed_mentions')
        
        if self.hashtag_pattern.search(cleaned):
            cleaned = self.hashtag_pattern.sub(' ', cleaned)
            changes_made.append('removed_hashtags')
        
        # Remove emails
        if self.email_pattern.search(cleaned):
            cleaned = self.email_pattern.sub(' ', cleaned)
            changes_made.append('removed_emails')
        
        # Remove extra whitespace
        cleaned = self.whitespace_pattern.sub(' ', cleaned).strip()
        
        # Calculate statistics
        stats = self._calculate_text_statistics(original_text, cleaned)
        
        return {
            'original_text': original_text,
            'cleaned_text': cleaned,
            'platform': platform,
            'changes_made': changes_made,
            'statistics': stats
        }
    
    def extract_text_only(self, text: str) -> str:
        """
        Extract only meaningful text content from social media posts
        Removes URLs, emojis, mentions, hashtags, and other social media artifacts
        
        Args:
            text: Social media text
            
        Returns:
            Clean text content only
        """
        if not text or not text.strip():
            return ""
        
        # Remove URLs
        cleaned = self.url_pattern.sub(' ', text)
        
        # Remove emojis
        cleaned = self.emoji_pattern.sub(' ', cleaned)
        
        # Remove mentions and hashtags
        cleaned = self.mention_pattern.sub(' ', cleaned)
        cleaned = self.hashtag_pattern.sub(' ', cleaned)
        
        # Remove emails
        cleaned = self.email_pattern.sub(' ', cleaned)
        
        # Remove social media specific patterns
        for pattern_name, pattern in self.social_media_patterns.items():
            cleaned = pattern.sub(' ', cleaned)
        
        # Remove extra whitespace and normalize
        cleaned = self.whitespace_pattern.sub(' ', cleaned).strip()
        
        return cleaned