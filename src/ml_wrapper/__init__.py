"""
ML Wrapper Module
Combines Haqiqa API predictions with custom features for enhanced fake news detection
"""

from .risk_scorer import RiskScorer
from .feature_combiner import FeatureCombiner
from .prediction_engine import PredictionEngine

__all__ = ['RiskScorer', 'FeatureCombiner', 'PredictionEngine']