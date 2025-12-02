"""
Arabic Text Processor
Specialized processing for Modern Standard Arabic
"""

import re
from typing import Dict, List, Tuple
from .text_cleaner import TextCleaner
from .language_detector import LanguageDetector

class ArabicProcessor:
    """
    Specialized processor for Arabic text including:
    - Arabic-specific normalization
    - Stemming and lemmatization
    - Morphological analysis
    - Named entity recognition patterns
    """
    
    def __init__(self):
        """Initialize Arabic processor"""
        self.text_cleaner = TextCleaner()
        self.language_detector = LanguageDetector()
        
        # Arabic prefixes (common)
        self.prefixes = [
            'ال', 'و', 'ف', 'ب', 'ك', 'ل', 'س', 'ي', 'ت', 'ن', 'م'
        ]
        
        # Arabic suffixes (common)
        self.suffixes = [
            'ة', 'ه', 'ها', 'هم', 'هن', 'ك', 'كم', 'نا', 'ون', 'ين', 'ات', 'ان'
        ]
        
        # Arabic stop words (comprehensive list)
        self.stop_words = {
            'من', 'إلى', 'عن', 'على', 'في', 'مع', 'خلال', 'بعد', 'قبل', 'حتى', 'منذ',
            'أو', 'و', 'ثم', 'لكن', 'بل', 'حتى', 'أم', 'إما', 'لا', 'لم', 'لن',
            'هذا', 'هذه', 'هذاك', 'هذهك', 'ذلك', 'تلك', 'هؤلاء', 'أولئك',
            'أنا', 'أنت', 'أنتِ', 'أنتما', 'أنتم', 'أنتن', 'هو', 'هي', 'هما', 'هم', 'هن',
            'ما', 'ماذا', 'متى', 'أين', 'كيف', 'لماذا', 'كم', 'أي', 'أين', 'متى',
            'كان', 'كانت', 'كانوا', 'كن', 'كنن', 'أصبح', 'أصبحت', 'أمسى', 'أمسى',
            'ليس', 'ليست', 'ليسوا', 'لست', 'لستن',
            'الذي', 'التي', 'الذين', 'اللاتي', 'اللائي', 'ما', 'من', 'ذات', 'ذو',
            'كل', 'بعض', 'جميع', 'كلا', 'كلتا', 'غير', 'سوى', 'فقط', 'أيضا',
            'جدا', 'أكثر', 'أقل', 'أفضل', 'أسوأ', 'كثير', 'قليل',
            'هنا', 'هناك', 'حيث', 'حيثما', 'أينما', 'مهما', 'كيفما',
            'حين', 'عند', 'عندما', 'إذ', 'إذا', 'لو', 'لولا'
        }
        
        # Named entity patterns
        self.person_patterns = [
            r'\b[أآإ][ب-ت][م-د][ةه]\s+[أ-ي]+\b',  # Common name patterns
            r'\b[أ-ي]+\s+بن\s+[أ-ي]+\b',         # "bin" pattern
            r'\b[أ-ي]+\s+أبو\s+[أ-ي]+\b',        # "abu" pattern
        ]
        
        self.location_patterns = [
            r'\bمدينة\s+[أ-ي]+\b',
            r'\b[أ-ي]+\s+مدينة\b',
            r'\b[أ-ي]+\s+محافظة\b',
            r'\b[أ-ي]+\s+دولة\b',
            r'\b[أ-ي]+\s+منطقة\b',
        ]
        
        self.organization_patterns = [
            r'\b[أ-ي]+\s+شركة\b',
            r'\b[أ-ي]+\s+منظمة\b',
            r'\b[أ-ي]+\s+جمعية\b',
            r'\b[أ-ي]+\s+وزارة\b',
            r'\b[أ-ي]+\s+مؤسسة\b',
        ]
        
        # Date patterns
        self.date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',
            r'\b\d{4}/\d{1,2}/\d{1,2}\b',
            r'\b[أ-ي]+\s+\d{1,4}\s+\d{4}\b',  # Arabic month
        ]
        
        # Number patterns
        self.number_patterns = [
            r'\b\d+\b',                           # Western numbers
            r'[٠-٩]+',                            # Arabic-Indic numbers
            r'\b[أ-ي]+\s+مليون\b',               # "million"
            r'\b[أ-ي]+\s+مليار\b',               # "billion"
            r'\b[أ-ي]+\s+ألف\b',                 # "thousand"
        ]
    
    def process_arabic_text(self, text: str) -> Dict:
        """
        Comprehensive Arabic text processing
        
        Args:
            text: Arabic text to process
            
        Returns:
            Dictionary containing processed text and analysis
        """
        # Clean text first
        clean_result = self.text_cleaner.clean_text(text, language='arabic', level='standard')
        cleaned_text = clean_result['cleaned_text']
        
        # Tokenize
        tokens = self.text_cleaner.tokenize_text(cleaned_text, language='arabic')
        
        # Remove stop words
        filtered_tokens = self._remove_stop_words(tokens)
        
        # Extract features
        features = {
            'named_entities': self._extract_named_entities(cleaned_text),
            'numbers': self._extract_numbers(cleaned_text),
            'dates': self._extract_dates(cleaned_text),
            'morphological_features': self._extract_morphological_features(tokens),
            'syntactic_features': self._extract_syntactic_features(tokens),
            'semantic_features': self._extract_semantic_features(tokens)
        }
        
        # Stemming (simple light stemming)
        stemmed_tokens = self._light_stem(filtered_tokens)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'filtered_tokens': filtered_tokens,
            'stemmed_tokens': stemmed_tokens,
            'features': features,
            'statistics': self._calculate_statistics(tokens, filtered_tokens, stemmed_tokens)
        }
    
    def _remove_stop_words(self, tokens: List[str]) -> List[str]:
        """Remove Arabic stop words"""
        return [token for token in tokens if token not in self.stop_words]
    
    def _extract_named_entities(self, text: str) -> Dict:
        """Extract named entities from Arabic text"""
        entities = {
            'persons': [],
            'locations': [],
            'organizations': []
        }
        
        # Extract persons
        for pattern in self.person_patterns:
            matches = re.findall(pattern, text)
            entities['persons'].extend(matches)
        
        # Extract locations
        for pattern in self.location_patterns:
            matches = re.findall(pattern, text)
            entities['locations'].extend(matches)
        
        # Extract organizations
        for pattern in self.organization_patterns:
            matches = re.findall(pattern, text)
            entities['organizations'].extend(matches)
        
        # Remove duplicates
        for entity_type in entities:
            entities[entity_type] = list(set(entities[entity_type]))
        
        return entities
    
    def _extract_numbers(self, text: str) -> List[str]:
        """Extract numbers from Arabic text"""
        numbers = []
        
        for pattern in self.number_patterns:
            matches = re.findall(pattern, text)
            numbers.extend(matches)
        
        return numbers
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from Arabic text"""
        dates = []
        
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        
        return dates
    
    def _extract_morphological_features(self, tokens: List[str]) -> Dict:
        """Extract morphological features from tokens"""
        features = {
            'prefix_count': 0,
            'suffix_count': 0,
            'average_token_length': 0,
            'long_tokens': [],
            'short_tokens': [],
            'has_diacritics': False,
            'has_hamza': False,
            'has_shadda': False
        }
        
        if not tokens:
            return features
        
        total_length = 0
        long_threshold = 8
        short_threshold = 3
        
        for token in tokens:
            total_length += len(token)
            
            if len(token) > long_threshold:
                features['long_tokens'].append(token)
            elif len(token) < short_threshold:
                features['short_tokens'].append(token)
            
            # Check for morphological markers
            for prefix in self.prefixes:
                if token.startswith(prefix):
                    features['prefix_count'] += 1
                    break
            
            for suffix in self.suffixes:
                if token.endswith(suffix):
                    features['suffix_count'] += 1
                    break
            
            # Check for diacritics
            if re.search(r'[\u064B-\u065F]', token):
                features['has_diacritics'] = True
            
            # Check for hamza
            if re.search(r'[ؤئئأإآ]', token):
                features['has_hamza'] = True
            
            # Check for shadda
            if re.search(r'[\u0651]', token):
                features['has_shadda'] = True
        
        features['average_token_length'] = total_length / len(tokens)
        
        return features
    
    def _extract_syntactic_features(self, tokens: List[str]) -> Dict:
        """Extract syntactic features"""
        features = {
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens)),
            'lexical_diversity': 0,
            'question_indicators': 0,
            'exclamation_indicators': 0,
            'conjunctions': 0,
            'prepositions': 0
        }
        
        if not tokens:
            return features
        
        features['lexical_diversity'] = features['unique_tokens'] / features['token_count']
        
        # Question indicators
        question_words = ['ماذا', 'متى', 'أين', 'كيف', 'لماذا', 'كم', 'هل']
        for token in tokens:
            if token in question_words:
                features['question_indicators'] += 1
        
        # Conjunctions
        conjunctions = ['و', 'ف', 'ثم', 'لكن', 'بل', 'أو', 'أم']
        for token in tokens:
            if token in conjunctions:
                features['conjunctions'] += 1
        
        # Prepositions
        prepositions = ['من', 'إلى', 'عن', 'على', 'في', 'مع', 'خلال', 'بعد', 'قبل', 'حتى']
        for token in tokens:
            if token in prepositions:
                features['prepositions'] += 1
        
        return features
    
    def _extract_semantic_features(self, tokens: List[str]) -> Dict:
        """Extract semantic features"""
        features = {
            'positive_words': 0,
            'negative_words': 0,
            'neutral_words': 0,
            'emotional_intensity': 0
        }
        
        # Simple sentiment word lists (can be expanded)
        positive_words = [
            'جيد', 'ممتاز', 'رائع', 'جميل', 'سعيد', 'نجاح', 'تفوق', 'حب', 'سلام',
            'أمل', 'فرح', 'سعادة', 'تقدم', 'ازدهار', 'خير', 'بركة'
        ]
        
        negative_words = [
            'سيء', 'فشل', 'حزن', 'غضب', 'خوف', 'كره', 'حرب', 'موت', 'مرض',
            'فقر', 'بطالة', 'فساد', 'ظلم', 'عنف', 'إرهاب', 'كارثة'
        ]
        
        for token in tokens:
            if token in positive_words:
                features['positive_words'] += 1
            elif token in negative_words:
                features['negative_words'] += 1
            else:
                features['neutral_words'] += 1
        
        # Calculate emotional intensity
        total_sentiment_words = features['positive_words'] + features['negative_words']
        if total_sentiment_words > 0:
            features['emotional_intensity'] = total_sentiment_words / len(tokens)
        
        return features
    
    def _light_stem(self, tokens: List[str]) -> List[str]:
        """
        Apply light stemming to Arabic tokens
        Removes common prefixes and suffixes
        """
        stemmed_tokens = []
        
        for token in tokens:
            stemmed = token
            
            # Remove prefixes
            for prefix in sorted(self.prefixes, key=len, reverse=True):
                if stemmed.startswith(prefix) and len(stemmed) > len(prefix) + 2:
                    stemmed = stemmed[len(prefix):]
                    break
            
            # Remove suffixes
            for suffix in sorted(self.suffixes, key=len, reverse=True):
                if stemmed.endswith(suffix) and len(stemmed) > len(suffix) + 2:
                    stemmed = stemmed[:-len(suffix)]
                    break
            
            stemmed_tokens.append(stemmed)
        
        return stemmed_tokens
    
    def _calculate_statistics(self, original_tokens: List[str], 
                             filtered_tokens: List[str], 
                             stemmed_tokens: List[str]) -> Dict:
        """Calculate processing statistics"""
        return {
            'original_token_count': len(original_tokens),
            'filtered_token_count': len(filtered_tokens),
            'stemmed_token_count': len(stemmed_tokens),
            'stop_words_removed': len(original_tokens) - len(filtered_tokens),
            'reduction_ratio': 1 - (len(filtered_tokens) / len(original_tokens)) if original_tokens else 0,
            'unique_original_tokens': len(set(original_tokens)),
            'unique_filtered_tokens': len(set(filtered_tokens)),
            'unique_stemmed_tokens': len(set(stemmed_tokens))
        }
    
    def is_arabic_text(self, text: str) -> bool:
        """Check if text is primarily Arabic"""
        return self.language_detector.is_arabic_text(text)
    
    def get_arabic_stop_words(self) -> set:
        """Get Arabic stop words set"""
        return self.stop_words.copy()
    
    def add_stop_words(self, words: List[str]):
        """Add custom stop words"""
        self.stop_words.update(words)
    
    def remove_stop_words(self, words: List[str]):
        """Remove words from stop words list"""
        for word in words:
            self.stop_words.discard(word)