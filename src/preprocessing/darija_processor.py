"""
Darija Text Processor
Specialized processing for Moroccan Darija (Arabic dialect)
"""

import re
from typing import Dict, List, Tuple
from .text_cleaner import TextCleaner
from .language_detector import LanguageDetector

class DarijaProcessor:
    """
    Specialized processor for Moroccan Darija including:
    - Darija-specific normalization
    - Code-switching handling (Arabic script + Latin script)
    - Dialect-specific vocabulary and patterns
    - Morphological analysis for colloquial Arabic
    """
    
    def __init__(self):
        """Initialize Darija processor"""
        self.text_cleaner = TextCleaner()
        self.language_detector = LanguageDetector()
        
        # Darija-specific vocabulary
        self.darija_words = {
            # Common expressions
            'زوين': 'good', 'بحال': 'like', 'كاين': 'there is', 'ماكاينش': 'there is no',
            'دار': 'did/made', 'ديال': 'of', 'فاش': 'how', 'فاشكاين': 'how is',
            'معاك': 'with you', 'معاكم': 'with you (plural)', 'معاكum': 'with you (fem)',
            'شنو': 'what', 'اشنو': 'what', 'كيفاش': 'how', 'فين': 'where',
            'واش': 'is', 'لا': 'no', 'به': 'yes', 'باه': 'yes', 'باهي': 'yes',
            'بلا': 'without', 'حتى': 'until', 'ولا': 'or', 'أو': 'or',
            'ملي': 'when', 'لي': 'that', 'اللي': 'that', 'هاد': 'this', 'هادي': 'this',
            'هناك': 'there', 'هنا': 'here', 'تم': 'there', 'تما': 'there',
            'دابا': 'now', 'داباا': 'now', 'بغيت': 'I want', 'بغيتي': 'I want',
            'علاش': 'why', 'علاه': 'why', 'اش': 'what', 'كاش': 'is there',
            
            # Time expressions
            'بارح': 'yesterday', 'لبارح': 'yesterday', 'غدوة': 'tomorrow', 'لغدوة': 'tomorrow',
            'النهارده': 'today', 'هاد النهار': 'today', 'دلوقت': 'now', 'هادا': 'this',
            
            # People and family
            'بزاف': 'a lot', 'كتر': 'more', 'شحال': 'how much', 'قل': 'less',
            'واحد': 'one', 'جوج': 'two', 'تلاتة': 'three', 'ربعة': 'four',
            'خمسة': 'five', 'ستة': 'six', 'سبعة': 'seven', 'تمنية': 'eight',
            'تسعة': 'nine', 'عشرة': 'ten',
            
            # Verbs
            'شفت': 'I saw', 'شفتي': 'you saw', 'شاف': 'he saw', 'شافت': 'she saw',
            'كليت': 'I ate', 'كليتي': 'you ate', 'كلا': 'he ate', 'كلات': 'she ate',
            'مشيت': 'I went', 'مشيتي': 'you went', 'مشا': 'he went', 'مشات': 'she went',
            'جيت': 'I came', 'جيتي': 'you came', 'جا': 'he came', 'جات': 'she came',
            
            # Adjectives
            'كبير': 'big', 'صغير': 'small', 'طويل': 'tall', 'قصير': 'short',
            'جميل': 'beautiful', 'حلو': 'sweet/nice', 'مر': 'bitter', 'خطر': 'dangerous',
            'سهل': 'easy', 'صعب': 'difficult', 'غالي': 'expensive', 'رخيص': 'cheap'
        }
        
        # Darija Latin script mappings
        self.latin_to_arabic = {
            '3': 'ع', '7': 'ح', '5': 'خ', '9': 'ق', '2': 'ء',
            '4': 'ش', '6': 'ط', '8': 'غ', 'gh': 'غ', 'kh': 'خ',
            'sh': 'ش', 'ch': 'ش', 'th': 'ث', 'dh': 'ذ', 'zh': 'ژ',
            'aa': 'آ', 'ee': 'يي', 'oo': 'وو', 'ou': 'و'
        }
        
        # Darija prefixes
        self.prefixes = [
            'ت', 'ن', 'م', 'ي', 'كا', 'ك', 'ال', 'و', 'ف'
        ]
        
        # Darija suffixes
        self.suffixes = [
            'ة', 'ه', 'ها', 'هم', 'هن', 'ك', 'كم', 'نا', 'ين', 'ات', 'ي', 'ت'
        ]
        
        # Darija stop words
        self.stop_words = {
            'من', 'في', 'على', 'مع', 'عن', 'بعد', 'قبل', 'حتى', 'ملي', 'لي', 'اللي',
            'هاد', 'هادي', 'هناك', 'هنا', 'تم', 'تما', 'دابا', 'واش', 'لا', 'به', 'باه',
            'بلا', 'حتى', 'ولا', 'أو', 'و', 'ف', 'ب', 'ك', 'ل', 'س', 'ي', 'ت', 'ن', 'م',
            'كن', 'كنا', 'كان', 'كانت', 'كانوا', 'بقى', 'بقات', 'بقوا',
            'شنو', 'اشنو', 'كيفاش', 'فين', 'علاش', 'شحال', 'كاش', 'واشكاين'
        }
        
        # Code-switching patterns
        self.code_switch_patterns = [
            r'\b\w*[3-7]\w*\b',           # Words with Arabic numbers
            r'\b[a-zA-Z]+[3-7][a-zA-Z]*\b',  # Mixed Latin-Arabic
            r'\b[ghkhsh][a-zA-Z]+\b',     # Darija digraphs
            r'\b\w+(ing|ed|tion|ment)\b',  # English suffixes in Darija
        ]
        
        # Darija-specific named entity patterns
        self.darija_person_patterns = [
            r'\b[أ-ي]+\s+بن\s+[أ-ي]+\b',
            r'\b[أ-ي]+\s+أبو\s+[أ-ي]+\b',
            r'\b[أ-ي]+\s+أم\s+[أ-ي]+\b',
        ]
        
        self.darija_location_patterns = [
            r'\bدار\s+[أ-ي]+\b',
            r'\b[أ-ي]+\s+دار\b',
            r'\b[أ-ي]+\s+ديال\b',
            r'\b[أ-ي]+\s+بلاد\b',
        ]
    
    def process_darija_text(self, text: str) -> Dict:
        """
        Comprehensive Darija text processing
        
        Args:
            text: Darija text to process (Arabic or Latin script)
            
        Returns:
            Dictionary containing processed text and analysis
        """
        # Detect if text is code-switched
        is_code_switched = self._detect_code_switching(text)
        
        # Normalize script (convert Latin Darija to Arabic if needed)
        normalized_text = self._normalize_script(text)
        
        # Clean text
        clean_result = self.text_cleaner.clean_text(normalized_text, language='darija', level='standard')
        cleaned_text = clean_result['cleaned_text']
        
        # Tokenize
        tokens = self.text_cleaner.tokenize_text(cleaned_text, language='darija')
        
        # Remove stop words
        filtered_tokens = self._remove_stop_words(tokens)
        
        # Extract features
        features = {
            'code_switching': self._analyze_code_switching(text),
            'script_type': self._detect_script_type(text),
            'darija_features': self._extract_darija_features(tokens),
            'morphological_features': self._extract_morphological_features(tokens),
            'semantic_features': self._extract_semantic_features(tokens),
            'named_entities': self._extract_named_entities(cleaned_text)
        }
        
        # Light stemming
        stemmed_tokens = self._light_stem(filtered_tokens)
        
        return {
            'original_text': text,
            'normalized_text': normalized_text,
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'filtered_tokens': filtered_tokens,
            'stemmed_tokens': stemmed_tokens,
            'features': features,
            'statistics': self._calculate_statistics(tokens, filtered_tokens, stemmed_tokens),
            'is_code_switched': is_code_switched
        }
    
    def _detect_code_switching(self, text: str) -> bool:
        """Detect if text contains code-switching"""
        for pattern in self.code_switch_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _normalize_script(self, text: str) -> str:
        """Normalize Darija script (Latin to Arabic)"""
        if self._is_latin_darija(text):
            return self._convert_latin_to_arabic(text)
        return text
    
    def _is_latin_darija(self, text: str) -> bool:
        """Check if text is Darija in Latin script"""
        # Check for Darija-specific patterns
        for pattern in self.code_switch_patterns:
            if re.search(pattern, text):
                return True
        
        # Check for Darija words in Latin script
        latin_words = text.lower().split()
        darija_count = 0
        
        for word in latin_words:
            # Check for Arabic numbers in Latin text
            if re.search(r'[3-7]', word):
                darija_count += 1
            # Check for Darija digraphs
            elif re.search(r'[ghkhsh]', word):
                darija_count += 1
        
        return darija_count > len(latin_words) * 0.1  # At least 10% Darija indicators
    
    def _convert_latin_to_arabic(self, text: str) -> str:
        """Convert Latin script Darija to Arabic script"""
        converted = text.lower()
        
        # Replace Latin Darija patterns with Arabic
        for latin, arabic in self.latin_to_arabic.items():
            converted = converted.replace(latin, arabic)
        
        # Note: This is a simplified conversion
        # Full conversion would require more sophisticated transliteration
        return converted
    
    def _detect_script_type(self, text: str) -> str:
        """Detect the script type of Darija text"""
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if arabic_chars > latin_chars:
            return 'arabic_script'
        elif latin_chars > arabic_chars:
            return 'latin_script'
        else:
            return 'mixed_script'
    
    def _analyze_code_switching(self, text: str) -> Dict:
        """Analyze code-switching patterns in text"""
        segments = []
        current_segment = {'text': '', 'script': 'unknown', 'start': 0}
        
        words = text.split()
        position = 0
        
        for i, word in enumerate(words):
            word_script = self._detect_word_script(word)
            
            if word_script != current_segment['script'] and i > 0:
                # Save current segment
                if current_segment['text'].strip():
                    current_segment['end'] = position - 1
                    segments.append(current_segment.copy())
                
                # Start new segment
                current_segment = {
                    'text': word + ' ',
                    'script': word_script,
                    'start': position
                }
            else:
                current_segment['text'] += word + ' '
                if current_segment['script'] == 'unknown':
                    current_segment['script'] = word_script
            
            position += len(word) + 1
        
        # Add final segment
        if current_segment['text'].strip():
            current_segment['end'] = len(text) - 1
            segments.append(current_segment)
        
        # Calculate statistics
        script_distribution = {}
        for segment in segments:
            script = segment['script']
            script_distribution[script] = script_distribution.get(script, 0) + 1
        
        return {
            'segments': segments,
            'script_distribution': script_distribution,
            'is_code_switched': len(script_distribution) > 1,
            'switch_count': len(segments) - 1
        }
    
    def _detect_word_script(self, word: str) -> str:
        """Detect script type of individual word"""
        if re.search(r'[\u0600-\u06FF]', word):
            return 'arabic'
        elif re.search(r'[a-zA-Z]', word):
            return 'latin'
        else:
            return 'unknown'
    
    def _remove_stop_words(self, tokens: List[str]) -> List[str]:
        """Remove Darija stop words"""
        return [token for token in tokens if token not in self.stop_words]
    
    def _extract_darija_features(self, tokens: List[str]) -> Dict:
        """Extract Darija-specific features"""
        features = {
            'darija_words_count': 0,
            'latin_darija_count': 0,
            'arabic_darija_count': 0,
            'darija_words_found': [],
            'has_code_switching': False,
            'dialect_markers': 0
        }
        
        for token in tokens:
            # Check for Darija vocabulary
            if token in self.darija_words:
                features['darija_words_count'] += 1
                features['darija_words_found'].append(token)
            
            # Check for Latin Darija patterns
            if re.search(r'[3-7]', token) or re.search(r'[ghkhsh]', token.lower()):
                features['latin_darija_count'] += 1
                features['has_code_switching'] = True
            
            # Check for Arabic Darija
            if re.search(r'[\u0600-\u06FF]', token):
                features['arabic_darija_count'] += 1
            
            # Check for dialect markers
            if re.search(r'(كا|ك|ن|ت|ي)', token):  # Darija prefixes
                features['dialect_markers'] += 1
        
        return features
    
    def _extract_morphological_features(self, tokens: List[str]) -> Dict:
        """Extract morphological features from Darija tokens"""
        features = {
            'prefix_count': 0,
            'suffix_count': 0,
            'average_token_length': 0,
            'has_darija_prefixes': 0,
            'has_darija_suffixes': 0,
            'verb_forms': 0,
            'noun_forms': 0
        }
        
        if not tokens:
            return features
        
        total_length = 0
        
        for token in tokens:
            total_length += len(token)
            
            # Check for Darija prefixes
            for prefix in self.prefixes:
                if token.startswith(prefix):
                    features['prefix_count'] += 1
                    features['has_darija_prefixes'] += 1
                    break
            
            # Check for Darija suffixes
            for suffix in self.suffixes:
                if token.endswith(suffix):
                    features['suffix_count'] += 1
                    features['has_darija_suffixes'] += 1
                    break
            
            # Simple verb/noun detection (very basic)
            if re.search(r'(ت|ن|ي|ك)\w+', token):  # Verb conjugation patterns
                features['verb_forms'] += 1
            else:
                features['noun_forms'] += 1
        
        features['average_token_length'] = total_length / len(tokens)
        
        return features
    
    def _extract_semantic_features(self, tokens: List[str]) -> Dict:
        """Extract semantic features from Darija tokens"""
        features = {
            'positive_words': 0,
            'negative_words': 0,
            'intensity_words': 0,
            'emotional_words': 0,
            'formal_words': 0,
            'informal_words': 0
        }
        
        # Darija sentiment words
        positive_words = ['زوين', 'حلو', 'جميل', 'رائع', 'ممتاز', 'به', 'باه', 'باهي']
        negative_words = ['كبير', 'سخيف', 'سيء', 'مر', 'خطر', 'معقد', 'صعب']
        intensity_words = ['بزاف', 'كتر', 'شحال', 'كبير', 'جدا', 'أكتر']
        emotional_words = ['حب', 'كره', 'فرح', 'حزن', 'غضب', 'خوف']
        informal_words = ['كاين', 'دار', 'فاش', 'شنو', 'واش', 'دابا']
        
        for token in tokens:
            if token in positive_words:
                features['positive_words'] += 1
            if token in negative_words:
                features['negative_words'] += 1
            if token in intensity_words:
                features['intensity_words'] += 1
            if token in emotional_words:
                features['emotional_words'] += 1
            if token in informal_words:
                features['informal_words'] += 1
        
        return features
    
    def _extract_named_entities(self, text: str) -> Dict:
        """Extract named entities specific to Darija context"""
        entities = {
            'persons': [],
            'locations': [],
            'organizations': []
        }
        
        # Use Darija-specific patterns
        for pattern in self.darija_person_patterns:
            matches = re.findall(pattern, text)
            entities['persons'].extend(matches)
        
        for pattern in self.darija_location_patterns:
            matches = re.findall(pattern, text)
            entities['locations'].extend(matches)
        
        # Remove duplicates
        for entity_type in entities:
            entities[entity_type] = list(set(entities[entity_type]))
        
        return entities
    
    def _light_stem(self, tokens: List[str]) -> List[str]:
        """Apply light stemming to Darija tokens"""
        stemmed_tokens = []
        
        for token in tokens:
            stemmed = token
            
            # Remove Darija prefixes
            for prefix in sorted(self.prefixes, key=len, reverse=True):
                if stemmed.startswith(prefix) and len(stemmed) > len(prefix) + 2:
                    stemmed = stemmed[len(prefix):]
                    break
            
            # Remove Darija suffixes
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
    
    def is_darija_text(self, text: str) -> bool:
        """Check if text is primarily Darija"""
        return self.language_detector.is_darija_text(text)
    
    def get_darija_vocabulary(self) -> Dict:
        """Get Darija vocabulary dictionary"""
        return self.darija_words.copy()
    
    def add_darija_words(self, words: Dict[str, str]):
        """Add custom Darija words to vocabulary"""
        self.darija_words.update(words)
    
    def get_darija_stop_words(self) -> set:
        """Get Darija stop words set"""
        return self.stop_words.copy()