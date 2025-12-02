"""
Prediction Engine Module
Orchestrates the complete fake news detection pipeline
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import logging

from ..api_client.haqiqa_client import HaqiqaClient
from ..preprocessing.language_detector import LanguageDetector
from ..preprocessing.text_cleaner import TextCleaner
from ..preprocessing.arabic_processor import ArabicProcessor
from ..preprocessing.darija_processor import DarijaProcessor
from ..features.text_features import TextFeatures
from ..features.tfidf_features import TfidfFeatures
from ..features.ngram_features import NgramFeatures
from ..features.sentiment_features import SentimentFeatures
from ..features.fake_news_lexicon import FakeNewsLexicon
from .risk_scorer import RiskScorer
from .feature_combiner import FeatureCombiner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionEngine:
    """
    Main prediction engine that orchestrates:
    - Text preprocessing
    - Feature extraction
    - Haqiqa API calls
    - Risk scoring
    - Result combination
    """
    
    def __init__(self, haqiqa_url: str = None, haqiqa_timeout: int = None):
        """
        Initialize prediction engine
        
        Args:
            haqiqa_url: Haqiqa API URL
            haqiqa_timeout: Request timeout
        """
        # Initialize components
        self.haqiqa_client = HaqiqaClient(haqiqa_url, haqiqa_timeout)
        self.language_detector = LanguageDetector()
        self.text_cleaner = TextCleaner()
        self.arabic_processor = ArabicProcessor()
        self.darija_processor = DarijaProcessor()
        
        # Feature extractors
        self.text_features = TextFeatures()
        self.tfidf_features = TfidfFeatures()
        self.ngram_features = NgramFeatures()
        self.sentiment_features = SentimentFeatures()
        self.lexicon_features = FakeNewsLexicon()
        
        # ML components
        self.risk_scorer = RiskScorer()
        self.feature_combiner = FeatureCombiner()
        
        # TF-IDF fitting status
        self.tfidf_fitted = False
        
        # Performance tracking
        self.prediction_history = []
    
    def predict_single(self, text: str, use_fallback: bool = True,
                      include_explanation: bool = True) -> Dict:
        """
        Predict fake news for a single text
        
        Args:
            text: Input text to analyze
            use_fallback: Whether to use fallback models
            include_explanation: Whether to include explanation
            
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        
        try:
            # Step 1: Language detection
            language_result = self.language_detector.detect_language(text)
            primary_language = language_result['primary_language']
            
            # Step 2: Text preprocessing
            clean_result = self.text_cleaner.clean_text(text, primary_language, level='standard')
            cleaned_text = clean_result['cleaned_text']
            
            # Step 3: Feature extraction
            feature_results = self._extract_all_features(text, cleaned_text, primary_language)
            
            # Step 4: Haqiqa API prediction
            haqiqa_result = self._get_haqiqa_prediction(cleaned_text, use_fallback)
            
            # Step 5: TF-IDF similarity (if fitted)
            tfidf_similarity = self._compute_tfidf_similarity(cleaned_text)
            
            # Step 6: Risk scoring
            risk_analysis = self.risk_scorer.compute_risk_score(
                haqiqa_result, feature_results, tfidf_similarity
            )
            
            # Step 7: Generate explanation
            explanation = None
            if include_explanation:
                explanation = self._generate_explanation(
                    text, language_result, feature_results, haqiqa_result, risk_analysis
                )
            
            # Step 8: Compile results
            processing_time = time.time() - start_time
            
            result = {
                'input_text': text,
                'language_analysis': language_result,
                'preprocessing': clean_result,
                'feature_analysis': feature_results,
                'haqiqa_prediction': haqiqa_result,
                'tfidf_similarity': tfidf_similarity,
                'risk_analysis': risk_analysis,
                'explanation': explanation,
                'processing_time': processing_time,
                'timestamp': time.time(),
                'success': True
            }
            
            # Track prediction
            self.prediction_history.append({
                'timestamp': result['timestamp'],
                'risk_score': risk_analysis['overall_risk_score'],
                'risk_level': risk_analysis['risk_level'],
                'language': primary_language,
                'processing_time': processing_time
            })
            
            logger.info(f"Prediction completed in {processing_time:.2f}s - Risk: {risk_analysis['risk_level']}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                'input_text': text,
                'error': str(e),
                'success': False,
                'timestamp': time.time(),
                'processing_time': time.time() - start_time
            }
    
    def predict_batch(self, texts: List[str], use_fallback: bool = True,
                     include_explanation: bool = False) -> List[Dict]:
        """
        Predict fake news for multiple texts
        
        Args:
            texts: List of input texts
            use_fallback: Whether to use fallback models
            include_explanation: Whether to include explanations
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Fit TF-IDF if not already fitted
        if not self.tfidf_fitted and texts:
            self._fit_tfidf(texts)
        
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            result = self.predict_single(text, use_fallback, include_explanation)
            result['batch_index'] = i
            results.append(result)
        
        return results
    
    def _extract_all_features(self, original_text: str, cleaned_text: str, 
                            language: str) -> Dict:
        """Extract all types of features"""
        feature_results = {}
        
        try:
            # Text features
            feature_results['text_features'] = self.text_features.extract_features(original_text)
        except Exception as e:
            logger.warning(f"Text feature extraction failed: {str(e)}")
            feature_results['text_features'] = {}
        
        try:
            # TF-IDF features
            if self.tfidf_fitted:
                feature_results['tfidf_features'] = self.tfidf_features.analyze_text_features(cleaned_text)
            else:
                feature_results['tfidf_features'] = {}
        except Exception as e:
            logger.warning(f"TF-IDF feature extraction failed: {str(e)}")
            feature_results['tfidf_features'] = {}
        
        try:
            # N-gram features
            feature_results['ngram_features'] = self.ngram_features.extract_ngrams(cleaned_text)
        except Exception as e:
            logger.warning(f"N-gram feature extraction failed: {str(e)}")
            feature_results['ngram_features'] = {}
        
        try:
            # Sentiment features
            feature_results['sentiment_features'] = self.sentiment_features.extract_sentiment_features(
                original_text, language
            )
        except Exception as e:
            logger.warning(f"Sentiment feature extraction failed: {str(e)}")
            feature_results['sentiment_features'] = {}
        
        try:
            # Lexicon features
            feature_results['lexicon_features'] = self.lexicon_features.extract_lexicon_features(
                original_text, language
            )
        except Exception as e:
            logger.warning(f"Lexicon feature extraction failed: {str(e)}")
            feature_results['lexicon_features'] = {}
        
        try:
            # Language features
            feature_results['language_features'] = self.language_detector.detect_language(original_text)
        except Exception as e:
            logger.warning(f"Language feature extraction failed: {str(e)}")
            feature_results['language_features'] = {}
        
        # Language-specific processing
        if language == 'arabic':
            try:
                arabic_result = self.arabic_processor.process_arabic_text(original_text)
                feature_results['arabic_features'] = arabic_result
            except Exception as e:
                logger.warning(f"Arabic processing failed: {str(e)}")
                feature_results['arabic_features'] = {}
        
        elif language == 'darija':
            try:
                darija_result = self.darija_processor.process_darija_text(original_text)
                feature_results['darija_features'] = darija_result
            except Exception as e:
                logger.warning(f"Darija processing failed: {str(e)}")
                feature_results['darija_features'] = {}
        
        return feature_results
    
    def _get_haqiqa_prediction(self, text: str, use_fallback: bool) -> Dict:
        """Get prediction from Haqiqa API"""
        try:
            # Try AraBERT first
            result = self.haqiqa_client.predict(text, 'arabert')
            
            if not result.get('error') or not use_fallback:
                return result
            
        except Exception as e:
            logger.warning(f"AraBERT prediction failed: {str(e)}")
        
        # Fallback to XGBoost
        if use_fallback:
            try:
                result = self.haqiqa_client.predict(text, 'xgboost')
                result['fallback_used'] = True
                return result
            except Exception as e:
                logger.error(f"XGBoost prediction also failed: {str(e)}")
        
        # Return error result
        return {
            'prediction': 'Unknown',
            'confidence': 0.0,
            'error': 'Both models failed',
            'success': False
        }
    
    def _compute_tfidf_similarity(self, text: str) -> Optional[float]:
        """Compute TF-IDF similarity to fake news patterns"""
        if not self.tfidf_fitted:
            return None
        
        try:
            # For now, return a placeholder
            # In a real implementation, this would compare against known fake news patterns
            return 0.5
        except Exception as e:
            logger.warning(f"TF-IDF similarity computation failed: {str(e)}")
            return None
    
    def _fit_tfidf(self, texts: List[str]):
        """Fit TF-IDF on training texts"""
        try:
            # Clean texts
            cleaned_texts = []
            for text in texts:
                clean_result = self.text_cleaner.preprocess_for_ml(text)
                cleaned_texts.append(clean_result)
            
            # Fit TF-IDF
            self.tfidf_features.fit(cleaned_texts)
            self.tfidf_fitted = True
            
            logger.info(f"TF-IDF fitted on {len(cleaned_texts)} texts")
            
        except Exception as e:
            logger.error(f"TF-IDF fitting failed: {str(e)}")
            self.tfidf_fitted = False
    
    def _generate_explanation(self, text: str, language_result: Dict,
                           feature_results: Dict, haqiqa_result: Dict,
                           risk_analysis: Dict) -> Dict:
        """Generate comprehensive explanation"""
        explanation = {
            'summary': '',
            'key_factors': [],
            'language_analysis': language_result,
            'feature_highlights': {},
            'recommendations': []
        }
        
        # Main summary
        risk_level = risk_analysis['risk_level']
        confidence = risk_analysis.get('confidence_interval', {})
        
        explanation['summary'] = (
            f"This text is classified as {risk_level} risk for being fake news. "
            f"The confidence interval is [{confidence.get('lower_bound', 0):.2f}, "
            f"{confidence.get('upper_bound', 1):.2f}]."
        )
        
        # Key risk factors
        risk_factors = risk_analysis.get('risk_factors', [])
        explanation['key_factors'] = risk_factors[:5]  # Top 5 factors
        
        # Feature highlights
        lexicon_features = feature_results.get('lexicon_features', {})
        if lexicon_features:
            dominant_indicators = self.lexicon_features.get_dominant_indicators(text)
            explanation['feature_highlights']['lexicon'] = dominant_indicators
        
        sentiment_features = feature_results.get('sentiment_features', {})
        if sentiment_features:
            sentiment_label = self.sentiment_features.get_sentiment_label(text)
            dominant_emotion = self.sentiment_features.get_dominant_emotion(text)
            explanation['feature_highlights']['sentiment'] = {
                'label': sentiment_label,
                'dominant_emotion': dominant_emotion
            }
        
        # Recommendations
        recommendations = risk_analysis.get('recommendation', '')
        explanation['recommendations'] = [recommendations]
        
        # Add language-specific insights
        primary_language = language_result.get('primary_language', 'unknown')
        if primary_language in ['arabic', 'darija']:
            explanation['language_insights'] = {
                'script_type': 'Arabic script' if primary_language == 'arabic' else 'Mixed script (Darija)',
                'code_switching': language_result.get('is_code_switched', False)
            }
        
        return explanation
    
    def fit_on_corpus(self, texts: List[str], labels: List[int] = None):
        """
        Fit the prediction engine on a corpus of texts
        
        Args:
            texts: Training texts
            labels: Optional labels for supervised learning
        """
        logger.info(f"Fitting prediction engine on {len(texts)} texts")
        
        # Fit TF-IDF
        self._fit_tfidf(texts)
        
        # Fit feature combiner
        if texts:
            # Extract features for all texts
            feature_dicts = []
            for text in texts:
                features = self._extract_all_features(text, text, 'auto')
                feature_dicts.append(features)
            
            # Fit feature combiner
            self.feature_combiner.fit(feature_dicts)
        
        logger.info("Prediction engine fitting completed")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.prediction_history:
            return {}
        
        processing_times = [p['processing_time'] for p in self.prediction_history]
        risk_scores = [p['risk_score'] for p in self.prediction_history]
        
        return {
            'total_predictions': len(self.prediction_history),
            'avg_processing_time': np.mean(processing_times),
            'median_processing_time': np.median(processing_times),
            'min_processing_time': np.min(processing_times),
            'max_processing_time': np.max(processing_times),
            'avg_risk_score': np.mean(risk_scores),
            'risk_distribution': {
                'very_low': sum(1 for p in self.prediction_history if p['risk_level'] == 'very_low'),
                'low': sum(1 for p in self.prediction_history if p['risk_level'] == 'low'),
                'medium': sum(1 for p in self.prediction_history if p['risk_level'] == 'medium'),
                'high': sum(1 for p in self.prediction_history if p['risk_level'] == 'high'),
                'very_high': sum(1 for p in self.prediction_history if p['risk_level'] == 'very_high')
            },
            'language_distribution': {
                lang: sum(1 for p in self.prediction_history if p['language'] == lang)
                for lang in set(p['language'] for p in self.prediction_history)
            }
        }
    
    def health_check(self) -> Dict:
        """Perform health check on all components"""
        health_status = {
            'overall_status': 'healthy',
            'components': {},
            'issues': []
        }
        
        # Check Haqiqa client
        try:
            haqiqa_health = self.haqiqa_client.health_check()
            health_status['components']['haqiqa_client'] = haqiqa_health
            if haqiqa_health['status'] != 'healthy':
                health_status['issues'].append('Haqiqa client unhealthy')
                health_status['overall_status'] = 'degraded'
        except Exception as e:
            health_status['components']['haqiqa_client'] = {'status': 'error', 'message': str(e)}
            health_status['issues'].append('Haqiqa client error')
            health_status['overall_status'] = 'unhealthy'
        
        # Check TF-IDF
        health_status['components']['tfidf_features'] = {
            'status': 'fitted' if self.tfidf_fitted else 'not_fitted'
        }
        
        # Check feature combiner
        health_status['components']['feature_combiner'] = {
            'status': 'fitted' if self.feature_combiner.is_fitted_status() else 'not_fitted'
        }
        
        return health_status
    
    def reset_history(self):
        """Reset prediction history"""
        self.prediction_history = []
        logger.info("Prediction history reset")
    
    def update_weights(self, haqiqa_weight: float = None, feature_weight: float = None):
        """Update risk scoring weights"""
        self.risk_scorer.update_weights(haqiqa_weight, feature_weight)
        logger.info(f"Updated weights - Haqiqa: {haqiqa_weight}, Features: {feature_weight}")