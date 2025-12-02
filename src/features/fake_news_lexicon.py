"""
Fake News Lexicon Module
Contains lexicons and patterns for fake news detection
"""

import re
from typing import Dict, List, Tuple, Set
from collections import Counter

class FakeNewsLexicon:
    """
    Fake news lexicon and pattern matcher
    Contains indicators and patterns commonly found in fake news
    """
    
    def __init__(self):
        """Initialize fake news lexicon"""
        
        # Clickbait and sensational words (Arabic)
        self.arabic_clickbait = [
            'صادم', 'مذهل', 'مفاجأة', 'عاجل', 'خطير', 'كارثي', 'مدمر', 'مروع',
            'لا يصدق', 'صدمة', 'هائل', 'مرعب', 'فظيع', 'مخيف', 'غريب',
            'أول مرة', 'حصري', 'سري', 'مسرب', 'كشف', 'فضيحة', 'فضائح',
            'عنيف', 'دموي', 'مأساة', 'كارثة', 'شبح', 'رعب', 'هلع'
        ]
        
        # Clickbait and sensational words (Darija)
        self.darija_clickbait = [
            'صادم', 'مذهل', 'عاجل', 'خطير', 'كارثي', 'مدمر', 'مروع',
            'ما يصدقش', 'صدمة', 'هائل', 'مرعب', 'كبير', 'بزاف',
            'أول مرة', 'حصري', 'سري', 'فضيحة', 'فضائح', 'كاشف',
            'عنيف', 'دموي', 'مأساة', 'شبح', 'رعب', 'خطر'
        ]
        
        # Clickbait and sensational words (French)
        self.french_clickbait = [
            'choquant', 'incroyable', 'surprenant', 'urgent', 'dangereux', 'catastrophique',
            'effrayant', 'horrible', 'incroyable', 'incroyable', 'première fois', 'exclusif',
            'secret', 'fuité', 'révélé', 'scandale', 'scandales', 'violent', 'sanglant',
            'tragédie', 'catastrophe', 'spectre', 'terreur', 'effroi'
        ]
        
        # Clickbait and sensational words (English)
        self.english_clickbait = [
            'shocking', 'incredible', 'surprising', 'urgent', 'dangerous', 'catastrophic',
            'terrifying', 'horrible', 'unbelievable', 'amazing', 'first time', 'exclusive',
            'secret', 'leaked', 'revealed', 'scandal', 'scandals', 'violent', 'bloody',
            'tragedy', 'catastrophe', 'specter', 'terror', 'horror'
        ]
        
        # Uncertainty and speculation words (Arabic)
        self.arabic_uncertainty = [
            'ربما', 'قد', 'يمكن', 'احتمال', 'يزعم', 'يدعي', 'قالوا', 'أفادت',
            'بحسب', 'تقول مصادر', 'ذكرت', 'أشارت', 'يبدو', 'يبدو أن',
            'من الممكن', 'من المحتمل', 'لا يوجد دليل', 'غير مؤكد', 'مشكوك فيه'
        ]
        
        # Uncertainty and speculation words (Darija)
        self.darija_uncertainty = [
            'ربما', 'قد', 'يمكن', 'احتمال', 'يزعم', 'يدعي', 'قالو',
            'بحال', 'كاين لي كايقول', 'قالت مصادر', 'زعمو', 'كاين',
            'ممكن', 'محتمل', 'ماكاينش دليل', 'مشكوك فيه', 'مش متأكد'
        ]
        
        # Conspiracy and conspiracy theory indicators (Arabic)
        self.arabic_conspiracy = [
            'مؤامرة', 'مؤامرات', 'سرية', 'خفية', 'مخطط', 'مخططات',
            'إسرائيل', 'أمريكا', 'CIA', 'موساد', 'ماسونية', 'جماعات سرية',
            'نظام عالمي', 'حكومة خفية', 'أجندة خفية', 'تغطية', 'إخفاء',
            'حقيقة مخفية', 'ما لا يريدونك أن تعرفه', 'الحقيقة المرة'
        ]
        
        # Conspiracy and conspiracy theory indicators (Darija)
        self.darija_conspiracy = [
            'مؤامرة', 'مؤامرات', 'سرية', 'خفية', 'مخطط', 'مخططات',
            'إسرائيل', 'أمريكا', 'موساد', 'ماسونية', 'جماعات سرية',
            'نظام عالمي', 'حكومة خفية', 'أجندة خفية', 'تغطية', 'إخفاء',
            'حقيقة مخفية', 'الحقيقة المرة', 'ما كايبغيوك تعرف'
        ]
        
        # Political propaganda indicators (Arabic)
        self.arabic_propaganda = [
            'العدو', 'الخصم', 'الإرهاب', 'الإرهابيون', 'الخطر', 'التهديد',
            'الدفاع عن', 'حماية', 'أمن', 'استقرار', 'سيادة', 'كرامة',
            'مقاومة', 'تحرير', 'مصير', 'مصير الأمة', 'مستقبل', 'أجيال'
        ]
        
        # Political propaganda indicators (Darija)
        self.darija_propaganda = [
            'لعدو', 'لخصم', 'لإرهاب', 'لإرهابيين', 'لخطر', 'لتهديد',
            'لدفاع عن', 'حماية', 'أمن', 'استقرار', 'سيادة', 'كرامة',
            'مقاومة', 'تحرير', 'مصير', 'مصير الأمة', 'مستقبل', 'أجيال'
        ]
        
        # Medical misinformation indicators (Arabic)
        self.arabic_medical_misinfo = [
            'علاج سحري', 'علاج نهائي', 'شفاء تام', 'دواء معجزة', 'طب بديل',
            'أعشاب سحرية', 'علاج بالأعشاب', 'طب شعبي', 'وصفة جدتي',
            'كورونا', 'كوفيد', 'لقاح', 'تطعيم', 'آثار جانبية', 'خطيرة',
            'مؤامرة طبية', 'شركات الأدوية', 'صناعة الدواء'
        ]
        
        # Medical misinformation indicators (Darija)
        self.darija_medical_misinfo = [
            'علاج سحري', 'علاج نهائي', 'شفاء تام', 'دواء معجزة', 'طب بديل',
            'أعشاب سحرية', 'علاج بالأعشاب', 'طب شعبي', 'وصفة لجدة',
            'كورونا', 'كوفيد', 'لقاح', 'تطعيم', 'آثار جانبية', 'خطيرة',
            'مؤامرة طبية', 'شركات الدواء', 'صناعة الدواء'
        ]
        
        # Financial scam indicators (Arabic)
        self.arabic_financial_scam = [
            'ربح سريع', 'ثروة سريعة', 'مليونير', 'استثمار مضمون',
            'عملة رقمية', 'بيتكوين', 'كريبتو', 'تداول', 'أسهم',
            'ربح مضمون', 'فرصة ذهبية', 'لا تفوت', 'محدود الوقت',
            'بنك', 'قرض', 'فائدة', 'ضمان', 'مخاطرة'
        ]
        
        # Financial scam indicators (Darija)
        self.darija_financial_scam = [
            'ربح سريع', 'ثروة سريعة', 'مليونير', 'استثمار مضمون',
            'عملة رقمية', 'بيتكوين', 'كريبتو', 'تداول', 'أسهم',
            'ربح مضمون', 'فرصة ذهبية', 'لا تفوت', 'محدود الوقت',
            'بنك', 'قرض', 'فائدة', 'ضمان', 'مخاطرة'
        ]
        
        # Religious manipulation indicators (Arabic)
        self.arabic_religious_manipulation = [
            'معجزة', 'كرامة', 'ولي صالح', 'شيخ', 'داعية', 'فتوى',
            'حرام', 'حلال', 'جنة', 'نار', 'آخرة', 'دعاء', 'قرآن',
            'سنة', 'بدعة', 'كفر', 'إيمان', 'شرك', 'وثني'
        ]
        
        # Religious manipulation indicators (Darija)
        self.darija_religious_manipulation = [
            'معجزة', 'كرامة', 'ولي صالح', 'شيخ', 'داعية', 'فتوى',
            'حرام', 'حلال', 'جنة', 'نار', 'آخرة', 'دعاء', 'قرآن',
            'سنة', 'بدعة', 'كفر', 'إيمان', 'شرك', 'وثني'
        ]
        
        # Fake news patterns
        self.fake_news_patterns = [
            r'لن تصدق.*',
            r'.*صدمة.*',
            r'.*كارثة.*',
            r'.*فضيحة.*',
            r'.*عاجل.*',
            r'.*خطير.*',
            r'.*حصري.*',
            r'.*سري.*',
            r'.*مسرب.*',
            r'.*أول مرة.*',
            r'.*لا يصدق.*',
            r'.*مذهل.*',
            r'.*صادم.*',
            r'.*مرعب.*',
            r'.*مروع.*',
            r'.*مدمر.*',
            r'.*هائل.*',
            r'.*غريب.*',
            r'.*عجيب.*'
        ]
        
        # Source credibility indicators
        self.unreliable_sources = [
            'قالت مصادر مطلعة',
            'كشفت مصادر',
            'أفادت مصادر',
            'بحسب مصادر',
            'قال شهود عيان',
            'ذكر شهود',
            'كشفت وثائق مسربة',
            'وفق معلومات مؤكدة',
            'حصريا لـ',
            'نقلا عن مصادر مطلعة'
        ]
        
        # Emotional manipulation patterns
        self.emotional_manipulation = [
            r'.*بكاء.*',
            r'.*دموع.*',
            r'.*حزن.*',
            r'.*أسى.*',
            r'.*مأساة.*',
            r'.*مآسي.*',
            r'.*قلوب.*',
            r'.*مشاعر.*',
            r'.*عواطف.*',
            r'.*إحساس.*'
        ]
        
        # Call to action patterns
        self.call_to_action_patterns = [
            r'.*شارك.*',
            r'.*نشر.*',
            r'.*أرسل.*',
            r'.*أخبر أصدقاءك.*',
            r'.*لا تتردد.*',
            r'.*سارع.*',
            r'.*قبل فوات الأوان.*',
            r'.*الوقت محدود.*',
            r'.*فرصة لا تعوض.*'
        ]
    
    def extract_lexicon_features(self, text: str, language: str = 'auto') -> Dict:
        """
        Extract lexicon-based features for fake news detection
        
        Args:
            text: Input text to analyze
            language: Language of text ('arabic', 'darija', 'french', 'english', 'auto')
            
        Returns:
            Dictionary containing lexicon features
        """
        if not text or not text.strip():
            return self._empty_lexicon_features()
        
        # Detect language if auto
        if language == 'auto':
            language = self._detect_language(text)
        
        # Tokenize text
        tokens = self._tokenize(text)
        text_lower = text.lower()
        
        # Extract features
        clickbait_score = self._calculate_clickbait_score(text_lower, language)
        uncertainty_score = self._calculate_uncertainty_score(text_lower, language)
        conspiracy_score = self._calculate_conspiracy_score(text_lower, language)
        propaganda_score = self._calculate_propaganda_score(text_lower, language)
        medical_misinfo_score = self._calculate_medical_misinfo_score(text_lower, language)
        financial_scam_score = self._calculate_financial_scam_score(text_lower, language)
        religious_manipulation_score = self._calculate_religious_manipulation_score(text_lower, language)
        
        # Pattern matching
        pattern_score = self._calculate_pattern_score(text_lower)
        unreliable_source_score = self._calculate_unreliable_source_score(text_lower)
        emotional_manipulation_score = self._calculate_emotional_manipulation_score(text_lower)
        call_to_action_score = self._calculate_call_to_action_score(text_lower)
        
        # Overall fake news risk score
        overall_score = (
            clickbait_score * 0.15 +
            uncertainty_score * 0.10 +
            conspiracy_score * 0.15 +
            propaganda_score * 0.10 +
            medical_misinfo_score * 0.10 +
            financial_scam_score * 0.10 +
            religious_manipulation_score * 0.05 +
            pattern_score * 0.10 +
            unreliable_source_score * 0.10 +
            emotional_manipulation_score * 0.05
        )
        
        return {
            'clickbait_score': clickbait_score,
            'uncertainty_score': uncertainty_score,
            'conspiracy_score': conspiracy_score,
            'propaganda_score': propaganda_score,
            'medical_misinfo_score': medical_misinfo_score,
            'financial_scam_score': financial_scam_score,
            'religious_manipulation_score': religious_manipulation_score,
            'pattern_score': pattern_score,
            'unreliable_source_score': unreliable_source_score,
            'emotional_manipulation_score': emotional_manipulation_score,
            'call_to_action_score': call_to_action_score,
            'overall_fake_news_risk': overall_score,
            'language': language,
            'token_count': len(tokens)
        }
    
    def _empty_lexicon_features(self) -> Dict:
        """Return empty lexicon features dictionary"""
        return {
            'clickbait_score': 0.0,
            'uncertainty_score': 0.0,
            'conspiracy_score': 0.0,
            'propaganda_score': 0.0,
            'medical_misinfo_score': 0.0,
            'financial_scam_score': 0.0,
            'religious_manipulation_score': 0.0,
            'pattern_score': 0.0,
            'unreliable_source_score': 0.0,
            'emotional_manipulation_score': 0.0,
            'call_to_action_score': 0.0,
            'overall_fake_news_risk': 0.0,
            'language': 'unknown',
            'token_count': 0
        }
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
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
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def _calculate_clickbait_score(self, text: str, language: str) -> float:
        """Calculate clickbait score"""
        clickbait_lexicon = {
            'arabic': self.arabic_clickbait,
            'darija': self.darija_clickbait,
            'french': self.french_clickbait,
            'english': self.english_clickbait
        }
        
        words = clickbait_lexicon.get(language, [])
        matches = sum(1 for word in words if word in text)
        
        return matches / len(words) if words else 0.0
    
    def _calculate_uncertainty_score(self, text: str, language: str) -> float:
        """Calculate uncertainty score"""
        uncertainty_lexicon = {
            'arabic': self.arabic_uncertainty,
            'darija': self.darija_uncertainty,
            'french': [],  # Can be expanded
            'english': []   # Can be expanded
        }
        
        words = uncertainty_lexicon.get(language, [])
        matches = sum(1 for word in words if word in text)
        
        return matches / len(words) if words else 0.0
    
    def _calculate_conspiracy_score(self, text: str, language: str) -> float:
        """Calculate conspiracy score"""
        conspiracy_lexicon = {
            'arabic': self.arabic_conspiracy,
            'darija': self.darija_conspiracy,
            'french': [],  # Can be expanded
            'english': []   # Can be expanded
        }
        
        words = conspiracy_lexicon.get(language, [])
        matches = sum(1 for word in words if word in text)
        
        return matches / len(words) if words else 0.0
    
    def _calculate_propaganda_score(self, text: str, language: str) -> float:
        """Calculate propaganda score"""
        propaganda_lexicon = {
            'arabic': self.arabic_propaganda,
            'darija': self.darija_propaganda,
            'french': [],  # Can be expanded
            'english': []   # Can be expanded
        }
        
        words = propaganda_lexicon.get(language, [])
        matches = sum(1 for word in words if word in text)
        
        return matches / len(words) if words else 0.0
    
    def _calculate_medical_misinfo_score(self, text: str, language: str) -> float:
        """Calculate medical misinformation score"""
        medical_lexicon = {
            'arabic': self.arabic_medical_misinfo,
            'darija': self.darija_medical_misinfo,
            'french': [],  # Can be expanded
            'english': []   # Can be expanded
        }
        
        words = medical_lexicon.get(language, [])
        matches = sum(1 for word in words if word in text)
        
        return matches / len(words) if words else 0.0
    
    def _calculate_financial_scam_score(self, text: str, language: str) -> float:
        """Calculate financial scam score"""
        financial_lexicon = {
            'arabic': self.arabic_financial_scam,
            'darija': self.darija_financial_scam,
            'french': [],  # Can be expanded
            'english': []   # Can be expanded
        }
        
        words = financial_lexicon.get(language, [])
        matches = sum(1 for word in words if word in text)
        
        return matches / len(words) if words else 0.0
    
    def _calculate_religious_manipulation_score(self, text: str, language: str) -> float:
        """Calculate religious manipulation score"""
        religious_lexicon = {
            'arabic': self.arabic_religious_manipulation,
            'darija': self.darija_religious_manipulation,
            'french': [],  # Can be expanded
            'english': []   # Can be expanded
        }
        
        words = religious_lexicon.get(language, [])
        matches = sum(1 for word in words if word in text)
        
        return matches / len(words) if words else 0.0
    
    def _calculate_pattern_score(self, text: str) -> float:
        """Calculate pattern matching score"""
        matches = 0
        for pattern in self.fake_news_patterns:
            if re.search(pattern, text):
                matches += 1
        
        return matches / len(self.fake_news_patterns)
    
    def _calculate_unreliable_source_score(self, text: str) -> float:
        """Calculate unreliable source score"""
        matches = sum(1 for source in self.unreliable_sources if source in text)
        return matches / len(self.unreliable_sources)
    
    def _calculate_emotional_manipulation_score(self, text: str) -> float:
        """Calculate emotional manipulation score"""
        matches = 0
        for pattern in self.emotional_manipulation:
            if re.search(pattern, text):
                matches += 1
        
        return matches / len(self.emotional_manipulation)
    
    def _calculate_call_to_action_score(self, text: str) -> float:
        """Calculate call to action score"""
        matches = 0
        for pattern in self.call_to_action_patterns:
            if re.search(pattern, text):
                matches += 1
        
        return matches / len(self.call_to_action_patterns)
    
    def get_risk_level(self, text: str, language: str = 'auto') -> str:
        """
        Get risk level classification
        
        Args:
            text: Input text
            language: Language of text
            
        Returns:
            Risk level ('low', 'medium', 'high', 'very_high')
        """
        features = self.extract_lexicon_features(text, language)
        risk_score = features['overall_fake_news_risk']
        
        if risk_score < 0.1:
            return 'low'
        elif risk_score < 0.2:
            return 'medium'
        elif risk_score < 0.4:
            return 'high'
        else:
            return 'very_high'
    
    def get_dominant_indicators(self, text: str, language: str = 'auto', top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get dominant fake news indicators
        
        Args:
            text: Input text
            language: Language of text
            top_k: Number of top indicators to return
            
        Returns:
            List of (indicator, score) tuples
        """
        features = self.extract_lexicon_features(text, language)
        
        indicators = [
            ('clickbait', features['clickbait_score']),
            ('uncertainty', features['uncertainty_score']),
            ('conspiracy', features['conspiracy_score']),
            ('propaganda', features['propaganda_score']),
            ('medical_misinfo', features['medical_misinfo_score']),
            ('financial_scam', features['financial_scam_score']),
            ('religious_manipulation', features['religious_manipulation_score']),
            ('pattern_matching', features['pattern_score']),
            ('unreliable_source', features['unreliable_source_score']),
            ('emotional_manipulation', features['emotional_manipulation_score']),
            ('call_to_action', features['call_to_action_score'])
        ]
        
        # Sort by score and return top_k
        indicators.sort(key=lambda x: x[1], reverse=True)
        return indicators[:top_k]
    
    def explain_lexicon_analysis(self, text: str, language: str = 'auto') -> Dict:
        """
        Provide explanation for lexicon analysis
        
        Args:
            text: Input text
            language: Language of text
            
        Returns:
            Dictionary containing explanation
        """
        features = self.extract_lexicon_features(text, language)
        dominant_indicators = self.get_dominant_indicators(text, language, top_k=5)
        risk_level = self.get_risk_level(text, language)
        
        explanations = []
        for indicator, score in dominant_indicators:
            if score > 0:
                explanation = {
                    'indicator': indicator,
                    'score': score,
                    'severity': self._get_severity_level(score),
                    'description': self._get_indicator_description(indicator)
                }
                explanations.append(explanation)
        
        return {
            'text': text,
            'risk_level': risk_level,
            'overall_score': features['overall_fake_news_risk'],
            'dominant_indicators': explanations,
            'summary': f"Text shows {risk_level} risk of being fake news with {len(explanations)} indicators detected"
        }
    
    def _get_severity_level(self, score: float) -> str:
        """Get severity level for a score"""
        if score < 0.1:
            return 'low'
        elif score < 0.3:
            return 'medium'
        elif score < 0.6:
            return 'high'
        else:
            return 'very_high'
    
    def _get_indicator_description(self, indicator: str) -> str:
        """Get description for an indicator"""
        descriptions = {
            'clickbait': 'Sensational or exaggerated headlines designed to attract attention',
            'uncertainty': 'Language indicating speculation or lack of verification',
            'conspiracy': 'References to conspiracy theories or hidden agendas',
            'propaganda': 'Political or ideological manipulation techniques',
            'medical_misinfo': 'Potential medical misinformation or false health claims',
            'financial_scam': 'Indicators of financial scams or fraudulent schemes',
            'religious_manipulation': 'Use of religious language for manipulation',
            'pattern_matching': 'Text matches known fake news patterns',
            'unreliable_source': 'References to unreliable or anonymous sources',
            'emotional_manipulation': 'Attempts to manipulate emotions',
            'call_to_action': 'Urgent calls to share or act without verification'
        }
        
        return descriptions.get(indicator, 'Unknown indicator')
    
    def get_feature_names(self) -> List[str]:
        """Get list of all lexicon feature names"""
        empty_features = self._empty_lexicon_features()
        return list(empty_features.keys())
    
    def extract_feature_vector(self, text: str, language: str = 'auto') -> List[float]:
        """
        Extract lexicon features as a numeric vector
        
        Args:
            text: Input text
            language: Language of text
            
        Returns:
            List of feature values
        """
        features = self.extract_lexicon_features(text, language)
        feature_names = self.get_feature_names()
        return [features.get(name, 0.0) for name in feature_names]