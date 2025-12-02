"""
Feature Extraction Module for Fake News Detection
Implements TF-IDF, N-grams, and various text features
"""

from .text_features import TextFeatures
from .tfidf_features import TfidfFeatures
from .ngram_features import NgramFeatures
from .sentiment_features import SentimentFeatures
from .fake_news_lexicon import FakeNewsLexicon

__all__ = ['TextFeatures', 'TfidfFeatures', 'NgramFeatures', 'SentimentFeatures', 'FakeNewsLexicon']