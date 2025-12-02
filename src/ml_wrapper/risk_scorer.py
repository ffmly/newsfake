"""
Risk Scorer Module
Combines multiple features to compute comprehensive fake news risk scores
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from ..utils.config import Config

class RiskScorer:
    """
    Comprehensive risk scorer that combines:
    - Haqiqa API predictions
    - TF-IDF similarity to fake news patterns
    - Lexicon-based risk indicators
    - Text feature analysis
    - Sentiment analysis
    """
    
    def __init__(self, haqiqa_weight: float = None, feature_weight: float = None):
        """
        Initialize risk scorer
        
        Args:
            haqiqa_weight: Weight for Haqiqa API prediction
            feature_weight: Weight for feature-based prediction
        """
        self.haqiqa_weight = haqiqa_weight or Config.HAQIQA_WEIGHT
        self.feature_weight = feature_weight or Config.FEATURE_WEIGHT
        
        # Risk thresholds
        self.risk_thresholds = {
            'very_low': 0.1,
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'very_high': 0.9
        }
        
        # Feature importance weights
        self.feature_importance = {
            'tfidf_similarity': 0.25,
            'lexicon_risk': 0.30,
            'sentiment_polarity': 0.15,
            'text_complexity': 0.10,
            'language_patterns': 0.10,
            'source_credibility': 0.10
        }
    
    def compute_risk_score(self, haqiqa_result: Dict, feature_results: Dict,
                         tfidf_similarity: float = None) -> Dict:
        """
        Compute comprehensive risk score
        
        Args:
            haqiqa_result: Result from Haqiqa API
            feature_results: Results from feature extraction
            tfidf_similarity: TF-IDF similarity to fake news patterns
            
        Returns:
            Dictionary containing risk analysis
        """
        # Extract Haqiqa confidence
        haqiqa_confidence = haqiqa_result.get('confidence', 0.0)
        haqiqa_prediction = haqiqa_result.get('prediction', 'Unknown')
        
        # Convert prediction to numeric score
        haqiqa_score = self._prediction_to_score(haqiqa_prediction, haqiqa_confidence)
        
        # Compute feature-based risk score
        feature_score = self._compute_feature_risk_score(feature_results, tfidf_similarity)
        
        # Combine scores using weighted average
        combined_score = (self.haqiqa_weight * haqiqa_score + 
                         self.feature_weight * feature_score)
        
        # Determine risk level
        risk_level = self._get_risk_level(combined_score)
        
        # Compute confidence intervals
        confidence_interval = self._compute_confidence_interval(
            haqiqa_score, feature_score, combined_score
        )
        
        # Analyze risk factors
        risk_factors = self._analyze_risk_factors(haqiqa_result, feature_results)
        
        return {
            'overall_risk_score': combined_score,
            'risk_level': risk_level,
            'haqiqa_score': haqiqa_score,
            'feature_score': feature_score,
            'confidence_interval': confidence_interval,
            'risk_factors': risk_factors,
            'recommendation': self._get_recommendation(risk_level, risk_factors),
            'weighting': {
                'haqiqa_weight': self.haqiqa_weight,
                'feature_weight': self.feature_weight
            }
        }
    
    def _prediction_to_score(self, prediction: str, confidence: float) -> float:
        """
        Convert Haqiqa prediction to numeric score
        
        Args:
            prediction: 'Real' or 'Fake'
            confidence: Confidence score (0-1)
            
        Returns:
            Numeric risk score (0-1, where 1 = high risk)
        """
        if prediction.lower() == 'fake':
            return confidence
        elif prediction.lower() == 'real':
            return 1.0 - confidence
        else:
            return 0.5  # Unknown prediction
    
    def _compute_feature_risk_score(self, feature_results: Dict, 
                                 tfidf_similarity: float = None) -> float:
        """
        Compute risk score from extracted features
        
        Args:
            feature_results: Dictionary containing all feature results
            tfidf_similarity: TF-IDF similarity score
            
        Returns:
            Feature-based risk score (0-1)
        """
        feature_scores = {}
        
        # TF-IDF similarity score
        if tfidf_similarity is not None:
            feature_scores['tfidf_similarity'] = tfidf_similarity
        else:
            feature_scores['tfidf_similarity'] = 0.5
        
        # Lexicon-based risk score
        lexicon_features = feature_results.get('lexicon_features', {})
        feature_scores['lexicon_risk'] = lexicon_features.get('overall_fake_news_risk', 0.0)
        
        # Sentiment-based risk
        sentiment_features = feature_results.get('sentiment_features', {})
        sentiment_polarity = sentiment_features.get('sentiment_polarity', 0.0)
        sentiment_subjectivity = sentiment_features.get('sentiment_subjectivity', 0.0)
        
        # High subjectivity and extreme polarity can indicate fake news
        sentiment_risk = abs(sentiment_polarity) * 0.5 + sentiment_subjectivity * 0.5
        feature_scores['sentiment_polarity'] = min(sentiment_risk, 1.0)
        
        # Text complexity features
        text_features = feature_results.get('text_features', {})
        complexity_score = self._compute_complexity_risk(text_features)
        feature_scores['text_complexity'] = complexity_score
        
        # Language patterns
        language_features = feature_results.get('language_features', {})
        pattern_risk = self._compute_pattern_risk(language_features)
        feature_scores['language_patterns'] = pattern_risk
        
        # Source credibility (if available)
        source_features = feature_results.get('source_features', {})
        feature_scores['source_credibility'] = source_features.get('credibility_risk', 0.5)
        
        # Compute weighted average
        weighted_score = 0.0
        total_weight = 0.0
        
        for feature, score in feature_scores.items():
            weight = self.feature_importance.get(feature, 0.1)
            weighted_score += weight * score
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.5
    
    def _compute_complexity_risk(self, text_features: Dict) -> float:
        """
        Compute risk score based on text complexity
        
        Args:
            text_features: Text feature dictionary
            
        Returns:
            Complexity risk score (0-1)
        """
        risk_score = 0.5  # Base score
        
        # Very short texts can be suspicious
        word_count = text_features.get('word_count', 0)
        if word_count < 10:
            risk_score += 0.2
        elif word_count > 1000:
            risk_score += 0.1
        
        # High punctuation ratio can indicate sensationalism
        punctuation_ratio = text_features.get('punctuation_ratio', 0.0)
        if punctuation_ratio > 0.15:
            risk_score += 0.15
        
        # High exclamation ratio
        exclamation_ratio = text_features.get('exclamation_ratio', 0.0)
        if exclamation_ratio > 0.05:
            risk_score += 0.2
        
        # High question ratio (clickbait)
        question_ratio = text_features.get('question_ratio', 0.0)
        if question_ratio > 0.05:
            risk_score += 0.15
        
        # URL and mention patterns
        url_ratio = text_features.get('url_ratio', 0.0)
        if url_ratio > 0.1:
            risk_score += 0.1
        
        return min(risk_score, 1.0)
    
    def _compute_pattern_risk(self, language_features: Dict) -> float:
        """
        Compute risk score based on language patterns
        
        Args:
            language_features: Language feature dictionary
            
        Returns:
            Pattern risk score (0-1)
        """
        risk_score = 0.5  # Base score
        
        # Code-switching can indicate manipulation
        is_code_switched = language_features.get('is_code_switched', False)
        if is_code_switched:
            risk_score += 0.1
        
        # Language distribution
        lang_distribution = language_features.get('language_distribution', {})
        if 'unknown' in lang_distribution and lang_distribution['unknown'] > 0.3:
            risk_score += 0.15
        
        # Low confidence in language detection
        confidence = language_features.get('confidence', 0.0)
        if confidence < 0.5:
            risk_score += 0.1
        
        return min(risk_score, 1.0)
    
    def _get_risk_level(self, score: float) -> str:
        """
        Get risk level from score
        
        Args:
            score: Risk score (0-1)
            
        Returns:
            Risk level string
        """
        if score <= self.risk_thresholds['very_low']:
            return 'very_low'
        elif score <= self.risk_thresholds['low']:
            return 'low'
        elif score <= self.risk_thresholds['medium']:
            return 'medium'
        elif score <= self.risk_thresholds['high']:
            return 'high'
        else:
            return 'very_high'
    
    def _compute_confidence_interval(self, haqiqa_score: float, feature_score: float,
                                  combined_score: float) -> Dict:
        """
        Compute confidence interval for risk score
        
        Args:
            haqiqa_score: Haqiqa-based score
            feature_score: Feature-based score
            combined_score: Combined risk score
            
        Returns:
            Dictionary containing confidence interval
        """
        # Calculate variance between different scoring methods
        variance = np.var([haqiqa_score, feature_score])
        
        # Standard error
        std_error = np.sqrt(variance / 2)  # 2 methods
        
        # 95% confidence interval
        margin_of_error = 1.96 * std_error
        
        return {
            'lower_bound': max(0.0, combined_score - margin_of_error),
            'upper_bound': min(1.0, combined_score + margin_of_error),
            'margin_of_error': margin_of_error,
            'confidence_level': 0.95,
            'variance': variance
        }
    
    def _analyze_risk_factors(self, haqiqa_result: Dict, feature_results: Dict) -> List[Dict]:
        """
        Analyze specific risk factors
        
        Args:
            haqiqa_result: Haqiqa API result
            feature_results: Feature extraction results
            
        Returns:
            List of risk factor dictionaries
        """
        risk_factors = []
        
        # Haqiqa-based factors
        haqiqa_confidence = haqiqa_result.get('confidence', 0.0)
        haqiqa_prediction = haqiqa_result.get('prediction', 'Unknown')
        
        if haqiqa_confidence < 0.6:
            risk_factors.append({
                'factor': 'low_haqiqa_confidence',
                'description': 'Low confidence from Haqiqa model',
                'severity': 'medium',
                'impact': 0.2
            })
        
        # Lexicon-based factors
        lexicon_features = feature_results.get('lexicon_features', {})
        lexicon_risk = lexicon_features.get('overall_fake_news_risk', 0.0)
        
        if lexicon_risk > 0.3:
            risk_factors.append({
                'factor': 'high_lexicon_risk',
                'description': 'Text contains fake news indicators',
                'severity': 'high' if lexicon_risk > 0.5 else 'medium',
                'impact': lexicon_risk
            })
        
        # Sentiment-based factors
        sentiment_features = feature_results.get('sentiment_features', {})
        sentiment_subjectivity = sentiment_features.get('sentiment_subjectivity', 0.0)
        
        if sentiment_subjectivity > 0.7:
            risk_factors.append({
                'factor': 'high_subjectivity',
                'description': 'Text shows high subjectivity',
                'severity': 'medium',
                'impact': 0.3
            })
        
        # Text-based factors
        text_features = feature_results.get('text_features', {})
        exclamation_ratio = text_features.get('exclamation_ratio', 0.0)
        
        if exclamation_ratio > 0.05:
            risk_factors.append({
                'factor': 'high_exclamation_usage',
                'description': 'Excessive use of exclamation marks',
                'severity': 'medium',
                'impact': 0.2
            })
        
        # Sort by impact
        risk_factors.sort(key=lambda x: x['impact'], reverse=True)
        
        return risk_factors
    
    def _get_recommendation(self, risk_level: str, risk_factors: List[Dict]) -> str:
        """
        Get recommendation based on risk level and factors
        
        Args:
            risk_level: Computed risk level
            risk_factors: List of risk factors
            
        Returns:
            Recommendation string
        """
        if risk_level in ['very_low', 'low']:
            return "Content appears to be reliable. No immediate action needed."
        
        elif risk_level == 'medium':
            return "Content shows some risk indicators. Verify with additional sources before sharing."
        
        elif risk_level == 'high':
            return "Content has significant risk indicators. Strongly recommend verification with multiple reliable sources."
        
        else:  # very_high
            return "Content shows very high risk of being fake news. Do not share without thorough verification from authoritative sources."
    
    def batch_compute_risk_scores(self, texts: List[str], haqiqa_results: List[Dict],
                                feature_results_list: List[Dict],
                                tfidf_similarities: List[float] = None) -> List[Dict]:
        """
        Compute risk scores for multiple texts
        
        Args:
            texts: List of input texts
            haqiqa_results: List of Haqiqa API results
            feature_results_list: List of feature extraction results
            tfidf_similarities: List of TF-IDF similarities
            
        Returns:
            List of risk analysis dictionaries
        """
        if tfidf_similarities is None:
            tfidf_similarities = [None] * len(texts)
        
        results = []
        for i, (text, haqiqa_result, features, tfidf_sim) in enumerate(
            zip(texts, haqiqa_results, feature_results_list, tfidf_similarities)
        ):
            risk_analysis = self.compute_risk_score(haqiqa_result, features, tfidf_sim)
            risk_analysis['text_index'] = i
            results.append(risk_analysis)
        
        return results
    
    def get_risk_statistics(self, risk_analyses: List[Dict]) -> Dict:
        """
        Compute statistics for multiple risk analyses
        
        Args:
            risk_analyses: List of risk analysis dictionaries
            
        Returns:
            Dictionary containing risk statistics
        """
        if not risk_analyses:
            return {}
        
        risk_scores = [analysis['overall_risk_score'] for analysis in risk_analyses]
        risk_levels = [analysis['risk_level'] for analysis in risk_analyses]
        
        # Risk level distribution
        level_counts = {}
        for level in risk_levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        return {
            'mean_risk_score': np.mean(risk_scores),
            'median_risk_score': np.median(risk_scores),
            'std_risk_score': np.std(risk_scores),
            'min_risk_score': np.min(risk_scores),
            'max_risk_score': np.max(risk_scores),
            'risk_level_distribution': level_counts,
            'high_risk_percentage': sum(1 for level in risk_levels if level in ['high', 'very_high']) / len(risk_levels),
            'total_analyzed': len(risk_analyses)
        }
    
    def update_weights(self, haqiqa_weight: float = None, feature_weight: float = None):
        """
        Update weighting for risk computation
        
        Args:
            haqiqa_weight: New weight for Haqiqa API
            feature_weight: New weight for feature-based analysis
        """
        if haqiqa_weight is not None:
            self.haqiqa_weight = haqiqa_weight
        
        if feature_weight is not None:
            self.feature_weight = feature_weight
        
        # Normalize weights to sum to 1
        total_weight = self.haqiqa_weight + self.feature_weight
        if total_weight > 0:
            self.haqiqa_weight /= total_weight
            self.feature_weight /= total_weight
    
    def get_feature_importance(self) -> Dict:
        """Get current feature importance weights"""
        return self.feature_importance.copy()
    
    def update_feature_importance(self, new_importance: Dict):
        """
        Update feature importance weights
        
        Args:
            new_importance: Dictionary of new importance weights
        """
        self.feature_importance.update(new_importance)
        
        # Normalize to sum to 1
        total_weight = sum(self.feature_importance.values())
        if total_weight > 0:
            for key in self.feature_importance:
                self.feature_importance[key] /= total_weight