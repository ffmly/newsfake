"""
Sentiment Feature Extraction Module
Extracts sentiment and emotion features for fake news detection
"""

import re
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import Counter
import json

class SentimentFeatures:
    """
    Sentiment feature extractor for Arabic, Darija, French, and English text
    Implements rule-based sentiment analysis with emotion detection
    """
    
    def __init__(self):
        """Initialize sentiment feature extractor"""
        
        # Arabic sentiment lexicon
        self.arabic_positive = [
            'جيد', 'ممتاز', 'رائع', 'جميل', 'سعيد', 'نجاح', 'تفوق', 'حب', 'سلام',
            'أمل', 'فرح', 'سعادة', 'تقدم', 'ازدهار', 'خير', 'بركة', 'مبارك',
            'مفيد', 'صحيح', 'صحيح', 'حقيقي', 'واقعي', 'مناسب', 'مثالي'
        ]
        
        self.arabic_negative = [
            'سيء', 'فشل', 'حزن', 'غضب', 'خوف', 'كره', 'حرب', 'موت', 'مرض',
            'فقر', 'بطالة', 'فساد', 'ظلم', 'عنف', 'إرهاب', 'كارثة', 'خطأ',
            'مشكلة', 'صعب', 'معقد', 'خطير', 'مخيف', 'مدمر', 'كاذب', 'زائف'
        ]
        
        # Darija sentiment lexicon
        self.darija_positive = [
            'زوين', 'حلو', 'جميل', 'رائع', 'ممتاز', 'به', 'باه', 'باهي', 'فرحان',
            'سعيد', 'مليح', 'عجيب', 'خو', 'لا بأس', 'مزيان', 'كبير', 'بزاف',
            'راسي', 'صح', 'صحيح', 'واخا', 'مرحبا', 'شكرا'
        ]
        
        self.darija_negative = [
            'كبير', 'سخيف', 'سيء', 'مر', 'خطر', 'معقد', 'صعب', 'كاذب', 'زور',
            'حرام', 'هدر', 'فاسد', 'مخرب', 'مدمر', 'كريه', 'بئيس', 'مزعج',
            'محزن', 'مؤلم', 'سيء', 'فاشل', 'خايب', 'ماشي مزيان'
        ]
        
        # French sentiment lexicon
        self.french_positive = [
            'bon', 'excellent', 'beau', 'joli', 'heureux', 'joie', 'amour', 'paix',
            'espoir', 'succès', 'progrès', 'bien', 'super', 'génial', 'parfait',
            'merveilleux', 'formidable', 'positif', 'agréable', 'content', 'satisfait'
        ]
        
        self.french_negative = [
            'mauvais', 'terrible', 'horrible', 'triste', 'colère', 'peur', 'haine',
            'guerre', 'mort', 'maladie', 'pauvreté', 'chômage', 'corruption',
            'injustice', 'violence', 'terrorisme', 'catastrophe', 'erreur', 'problème'
        ]
        
        # English sentiment lexicon
        self.english_positive = [
            'good', 'excellent', 'beautiful', 'happy', 'joy', 'love', 'peace',
            'hope', 'success', 'progress', 'great', 'wonderful', 'amazing',
            'perfect', 'positive', 'nice', 'fantastic', 'awesome', 'brilliant'
        ]
        
        self.english_negative = [
            'bad', 'terrible', 'horrible', 'sad', 'angry', 'fear', 'hate',
            'war', 'death', 'disease', 'poverty', 'unemployment', 'corruption',
            'injustice', 'violence', 'terrorism', 'disaster', 'error', 'problem'
        ]
        
        # Emotion words (multilingual)
        self.emotion_words = {
            'joy': {
                'arabic': ['فرح', 'سعادة', 'بهجة', 'سرور', 'ابتهاج'],
                'darija': ['فرح', 'سعيد', 'فرحان', 'مسرور', 'مبهج'],
                'french': ['joie', 'bonheur', 'gaieté', 'allégresse'],
                'english': ['joy', 'happiness', 'delight', 'glee']
            },
            'anger': {
                'arabic': ['غضب', 'سخط', 'حنق', 'غيظ', 'استياء'],
                'darija': ['غضبان', 'زعلان', 'مخرب', 'حنق', 'زعل'],
                'french': ['colère', 'rage', 'fureur', 'irritation'],
                'english': ['anger', 'rage', 'fury', 'irritation']
            },
            'fear': {
                'arabic': ['خوف', 'رهبة', 'فزع', 'هلع', 'ذعر'],
                'darija': ['خايف', 'مرعوب', 'مخيف', 'خوف', 'رهبة'],
                'french': ['peur', 'crainte', 'effroi', 'terreur'],
                'english': ['fear', 'terror', 'dread', 'horror']
            },
            'sadness': {
                'arabic': ['حزن', 'أسى', 'كآبة', 'يأس', 'تألم'],
                'darija': ['حزان', 'محزون', 'كئيب', 'تاعب', 'مصاب'],
                'french': ['tristesse', 'chagrin', 'peine', 'douleur'],
                'english': ['sadness', 'sorrow', 'grief', 'pain']
            },
            'surprise': {
                'arabic': ['مفاجأة', 'دهشة', ' astonishment', 'صدمة'],
                'darija': ['مفاجأة', 'مصدوم', 'متشاج', 'مستغرب'],
                'french': ['surprise', 'étonnement', 'stupeur', 'choc'],
                'english': ['surprise', 'astonishment', 'amazement', 'shock']
            }
        }
        
        # Intensity modifiers
        self.intensity_modifiers = {
            'arabic': ['جدا', 'كثير', 'بشكل', 'تماما', 'بكل', 'للغاية', 'غاية'],
            'darija': ['بزاف', 'كتر', 'جدا', 'بحال', 'كاين', 'مرا', 'أكتر'],
            'french': ['très', 'beaucoup', 'extrêmement', 'trop', 'énormément'],
            'english': ['very', 'extremely', 'really', 'quite', 'too', 'so']
        }
        
        # Negation words
        self.negation_words = {
            'arabic': ['لا', 'ليس', 'لم', 'لن', 'غير', 'بدون', 'لا', 'ما'],
            'darija': ['لا', 'ماكاينش', 'ماشي', 'مش', 'ما', 'غير', 'بدون'],
            'french': ['ne', 'pas', 'non', 'jamais', 'rien', 'personne'],
            'english': ['not', 'no', 'never', 'nothing', 'nobody', 'none']
        }
    
    def extract_sentiment_features(self, text: str, language: str = 'auto') -> Dict:
        """
        Extract comprehensive sentiment features
        
        Args:
            text: Input text to analyze
            language: Language of text ('arabic', 'darija', 'french', 'english', 'auto')
            
        Returns:
            Dictionary containing sentiment features
        """
        if not text or not text.strip():
            return self._empty_sentiment_features()
        
        # Detect language if auto
        if language == 'auto':
            language = self._detect_language(text)
        
        # Tokenize text
        tokens = self._tokenize(text)
        
        # Extract basic sentiment
        sentiment_scores = self._calculate_sentiment_scores(tokens, language)
        
        # Extract emotions
        emotion_scores = self._calculate_emotion_scores(tokens, language)
        
        # Extract intensity features
        intensity_features = self._calculate_intensity_features(tokens, language)
        
        # Extract polarity features
        polarity_features = self._calculate_polarity_features(tokens, language)
        
        # Combine all features
        all_features = {}
        all_features.update(sentiment_scores)
        all_features.update(emotion_scores)
        all_features.update(intensity_features)
        all_features.update(polarity_features)
        all_features['language'] = language
        all_features['token_count'] = len(tokens)
        
        return all_features
    
    def _empty_sentiment_features(self) -> Dict:
        """Return empty sentiment features dictionary"""
        return {
            'positive_score': 0.0,
            'negative_score': 0.0,
            'neutral_score': 1.0,
            'sentiment_polarity': 0.0,
            'sentiment_subjectivity': 0.0,
            'joy_score': 0.0,
            'anger_score': 0.0,
            'fear_score': 0.0,
            'sadness_score': 0.0,
            'surprise_score': 0.0,
            'intensity_score': 0.0,
            'negation_count': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'language': 'unknown',
            'token_count': 0
        }
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns"""
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if arabic_chars > latin_chars:
            # Check for Darija indicators
            darija_indicators = ['زوين', 'كاين', 'دار', 'شنو', 'واش', 'دابا']
            for indicator in darija_indicators:
                if indicator in text.lower():
                    return 'darija'
            return 'arabic'
        elif latin_chars > 0:
            # Simple heuristic for French vs English
            french_indicators = ['le', 'la', 'les', 'de', 'du', 'et', 'est', 'dans']
            english_indicators = ['the', 'and', 'or', 'but', 'in', 'on', 'at']
            
            french_count = sum(1 for word in french_indicators if word in text.lower())
            english_count = sum(1 for word in english_indicators if word in text.lower())
            
            return 'french' if french_count > english_count else 'english'
        
        return 'unknown'
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def _calculate_sentiment_scores(self, tokens: List[str], language: str) -> Dict:
        """Calculate basic sentiment scores"""
        # Get appropriate lexicon
        positive_words = self._get_positive_words(language)
        negative_words = self._get_negative_words(language)
        
        # Count sentiment words
        positive_count = sum(1 for token in tokens if token.lower() in positive_words)
        negative_count = sum(1 for token in tokens if token.lower() in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        total_tokens = len(tokens)
        
        # Calculate scores
        if total_tokens == 0:
            positive_score = negative_score = 0.0
        else:
            positive_score = positive_count / total_tokens
            negative_score = negative_count / total_tokens
        
        neutral_score = 1.0 - positive_score - negative_score
        neutral_score = max(0.0, neutral_score)
        
        # Calculate polarity and subjectivity
        polarity = positive_score - negative_score
        subjectivity = positive_score + negative_score
        
        return {
            'positive_score': positive_score,
            'negative_score': negative_score,
            'neutral_score': neutral_score,
            'sentiment_polarity': polarity,
            'sentiment_subjectivity': subjectivity,
            'positive_word_count': positive_count,
            'negative_word_count': negative_count,
            'sentiment_word_count': total_sentiment_words
        }
    
    def _calculate_emotion_scores(self, tokens: List[str], language: str) -> Dict:
        """Calculate emotion scores"""
        emotion_scores = {}
        
        for emotion, emotion_dict in self.emotion_words.items():
            emotion_words = emotion_dict.get(language, [])
            emotion_count = sum(1 for token in tokens if token.lower() in emotion_words)
            emotion_scores[f'{emotion}_score'] = emotion_count / len(tokens) if tokens else 0.0
            emotion_scores[f'{emotion}_count'] = emotion_count
        
        return emotion_scores
    
    def _calculate_intensity_features(self, tokens: List[str], language: str) -> Dict:
        """Calculate intensity-related features"""
        # Get intensity modifiers
        modifiers = self.intensity_modifiers.get(language, [])
        modifier_count = sum(1 for token in tokens if token.lower() in modifiers)
        
        # Count exclamation marks and question marks
        text = ' '.join(tokens)
        exclamation_count = text.count('!') + text.count('!')
        question_count = text.count('?') + text.count('؟')
        
        # Calculate intensity score
        intensity_score = (modifier_count + exclamation_count + question_count) / len(tokens) if tokens else 0.0
        
        return {
            'intensity_score': intensity_score,
            'modifier_count': modifier_count,
            'exclamation_count': exclamation_count,
            'question_count': question_count
        }
    
    def _calculate_polarity_features(self, tokens: List[str], language: str) -> Dict:
        """Calculate polarity-related features"""
        # Get negation words
        negation_words = self.negation_words.get(language, [])
        negation_count = sum(1 for token in tokens if token.lower() in negation_words)
        
        # Check for contrastive conjunctions
        contrastive_words = ['but', 'however', 'although', 'mais', 'mais', 'لكن', 'ولكن']
        contrastive_count = sum(1 for token in tokens if token.lower() in contrastive_words)
        
        return {
            'negation_count': negation_count,
            'contrastive_count': contrastive_count,
            'negation_ratio': negation_count / len(tokens) if tokens else 0.0
        }
    
    def _get_positive_words(self, language: str) -> List[str]:
        """Get positive words for given language"""
        positive_lexicon = {
            'arabic': self.arabic_positive,
            'darija': self.darija_positive,
            'french': self.french_positive,
            'english': self.english_positive
        }
        return positive_lexicon.get(language, [])
    
    def _get_negative_words(self, language: str) -> List[str]:
        """Get negative words for given language"""
        negative_lexicon = {
            'arabic': self.arabic_negative,
            'darija': self.darija_negative,
            'french': self.french_negative,
            'english': self.english_negative
        }
        return negative_lexicon.get(language, [])
    
    def get_sentiment_label(self, text: str, language: str = 'auto') -> str:
        """
        Get sentiment label for text
        
        Args:
            text: Input text
            language: Language of text
            
        Returns:
            Sentiment label ('positive', 'negative', 'neutral')
        """
        features = self.extract_sentiment_features(text, language)
        
        polarity = features['sentiment_polarity']
        subjectivity = features['sentiment_subjectivity']
        
        if subjectivity < 0.1:
            return 'neutral'
        elif polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def get_dominant_emotion(self, text: str, language: str = 'auto') -> str:
        """
        Get dominant emotion in text
        
        Args:
            text: Input text
            language: Language of text
            
        Returns:
            Dominant emotion label
        """
        features = self.extract_sentiment_features(text, language)
        
        emotions = ['joy', 'anger', 'fear', 'sadness', 'surprise']
        emotion_scores = [features[f'{emotion}_score'] for emotion in emotions]
        
        max_score = max(emotion_scores)
        if max_score == 0:
            return 'neutral'
        
        dominant_index = emotion_scores.index(max_score)
        return emotions[dominant_index]
    
    def analyze_sentiment_patterns(self, texts: List[str]) -> Dict:
        """
        Analyze sentiment patterns across multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary containing pattern analysis
        """
        all_features = [self.extract_sentiment_features(text) for text in texts]
        
        # Calculate aggregate statistics
        positive_scores = [f['positive_score'] for f in all_features]
        negative_scores = [f['negative_score'] for f in all_features]
        polarity_scores = [f['sentiment_polarity'] for f in all_features]
        
        return {
            'avg_positive_score': np.mean(positive_scores),
            'avg_negative_score': np.mean(negative_scores),
            'avg_polarity': np.mean(polarity_scores),
            'std_positive_score': np.std(positive_scores),
            'std_negative_score': np.std(negative_scores),
            'std_polarity': np.std(polarity_scores),
            'most_positive_text_index': np.argmax(positive_scores),
            'most_negative_text_index': np.argmax(negative_scores),
            'sentiment_distribution': {
                'positive': sum(1 for f in all_features if f['sentiment_polarity'] > 0.1),
                'negative': sum(1 for f in all_features if f['sentiment_polarity'] < -0.1),
                'neutral': sum(1 for f in all_features if abs(f['sentiment_polarity']) <= 0.1)
            }
        }
    
    def extract_batch_features(self, texts: List[str], languages: List[str] = None) -> List[Dict]:
        """
        Extract sentiment features from multiple texts
        
        Args:
            texts: List of texts to process
            languages: List of languages (auto-detected if None)
            
        Returns:
            List of sentiment feature dictionaries
        """
        if languages is None:
            languages = ['auto'] * len(texts)
        
        return [self.extract_sentiment_features(text, lang) for text, lang in zip(texts, languages)]
    
    def get_feature_names(self) -> List[str]:
        """Get list of all sentiment feature names"""
        empty_features = self._empty_sentiment_features()
        return list(empty_features.keys())
    
    def extract_feature_vector(self, text: str, language: str = 'auto') -> List[float]:
        """
        Extract sentiment features as a numeric vector
        
        Args:
            text: Input text
            language: Language of text
            
        Returns:
            List of feature values
        """
        features = self.extract_sentiment_features(text, language)
        feature_names = self.get_feature_names()
        return [features.get(name, 0.0) for name in feature_names]