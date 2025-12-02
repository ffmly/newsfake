"""
Test suite for feature extraction modules
"""

import pytest
import unittest
from unittest.mock import Mock, patch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.features.text_features import TextFeatures
from src.features.tfidf_features import TfidfFeatures
from src.features.ngram_features import NgramFeatures
from src.features.sentiment_features import SentimentFeatures
from src.features.fake_news_lexicon import FakeNewsLexicon

class TestTextFeatures(unittest.TestCase):
    """Test cases for TextFeatures class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.text_features = TextFeatures()
        self.sample_arabic_text = "هذا خبر عاجل جداً ويحتوي علامات ترقيم!"
        self.sample_darija_text = "كاين واحد كايقول ليك هاد الخبر صحيح ولا، داباا الزين ماشي مزيان."
        self.sample_english_text = "This is SHOCKING news with lots of punctuation!!!"
    
    def test_extract_features_arabic(self):
        """Test feature extraction for Arabic text"""
        features = self.text_features.extract_features(self.sample_arabic_text)
        
        # Test basic features
        self.assertIsInstance(features, dict)
        self.assertIn('text_length', features)
        self.assertIn('word_count', features)
        self.assertIn('punctuation_ratio', features)
        
        # Test Arabic-specific features
        self.assertGreater(features['arabic_char_ratio'], 0.5)
        self.assertLess(features['latin_char_ratio'], 0.5)
    
    def test_extract_features_darija(self):
        """Test feature extraction for Darija text"""
        features = self.text_features.extract_features(self.sample_darija_text)
        
        # Test basic features
        self.assertIsInstance(features, dict)
        self.assertGreater(features['word_count'], 0)
        
        # Should detect mixed script
        self.assertGreater(features['arabic_char_ratio'], 0.3)
        self.assertGreater(features['latin_char_ratio'], 0.1)
    
    def test_extract_features_english(self):
        """Test feature extraction for English text"""
        features = self.text_features.extract_features(self.sample_english_text)
        
        # Test punctuation features
        self.assertGreater(features['exclamation_ratio'], 0)
        self.assertGreater(features['punctuation_ratio'], 0)
        
        # Test length features
        self.assertGreater(features['text_length'], 10)
    
    def test_extract_features_empty_text(self):
        """Test feature extraction with empty text"""
        features = self.text_features.extract_features("")
        
        # Should return empty features
        self.assertEqual(features['text_length'], 0)
        self.assertEqual(features['word_count'], 0)
    
    def test_get_feature_names(self):
        """Test getting feature names"""
        feature_names = self.text_features.get_feature_names()
        
        self.assertIsInstance(feature_names, list)
        self.assertIn('text_length', feature_names)
        self.assertIn('word_count', feature_names)
        self.assertIn('punctuation_ratio', feature_names)
    
    def test_extract_feature_vector(self):
        """Test feature vector extraction"""
        vector = self.text_features.extract_feature_vector(self.sample_arabic_text)
        
        self.assertIsInstance(vector, list)
        self.assertEqual(len(vector), len(self.text_features.get_feature_names()))

class TestTfidfFeatures(unittest.TestCase):
    """Test cases for TfidfFeatures class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tfidf_features = TfidfFeatures(max_features=100)
        self.sample_texts = [
            "هذا خبر اختبار",
            "This is test news",
            "خبر ثالث للتجربة"
        ]
    
    def test_fit_transform(self):
        """Test fitting and transformation"""
        # Fit on sample texts
        self.tfidf_features.fit(self.sample_texts)
        
        # Transform texts
        vectors = self.tfidf_features.transform(self.sample_texts)
        
        # Check dimensions
        self.assertEqual(vectors.shape[0], len(self.sample_texts))
        self.assertLessEqual(vectors.shape[1], 100)  # max_features
    
    def test_analyze_text_features(self):
        """Test text feature analysis"""
        self.tfidf_features.fit(self.sample_texts)
        
        # Analyze single text
        analysis = self.tfidf_features.analyze_text_features("هذا خبر اختبار")
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('top_features', analysis)
        self.assertIn('statistics', analysis)
    
    def test_get_vocabulary(self):
        """Test vocabulary extraction"""
        self.tfidf_features.fit(self.sample_texts)
        
        vocab = self.tfidf_features.get_vocabulary()
        self.assertIsInstance(vocab, dict)
        self.assertGreater(len(vocab), 0)
    
    def test_similarity_calculation(self):
        """Test document similarity calculation"""
        self.tfidf_features.fit(self.sample_texts)
        
        # Calculate similarity
        similarity = self.tfidf_features.get_document_similarity(
            self.sample_texts[0], 
            self.sample_texts[1]
        )
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)

class TestNgramFeatures(unittest.TestCase):
    """Test cases for NgramFeatures class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ngram_features = NgramFeatures(ngram_range=(1, 2))
        self.sample_text = "هذا خبر اختبار للنظام"
    
    def test_extract_ngrams(self):
        """Test n-gram extraction"""
        ngrams = self.ngram_features.extract_ngrams(self.sample_text)
        
        self.assertIsInstance(ngrams, dict)
        self.assertIn('top_ngrams', ngrams)
        self.assertIn('total_ngrams', ngrams)
        self.assertIn('unique_ngrams', ngrams)
    
    def test_ngram_frequency(self):
        """Test n-gram frequency analysis"""
        self.ngram_features.fit([self.sample_text])
        
        freq = self.ngram_features.get_ngram_frequency([self.sample_text])
        
        self.assertIsInstance(freq, dict)
        self.assertGreater(len(freq), 0)
    
    def test_compare_texts(self):
        """Test text comparison using n-grams"""
        text1 = "هذا خبر اختبار"
        text2 = "هذا خبر آخر"
        
        comparison = self.ngram_features.compare_texts(text1, text2)
        
        self.assertIsInstance(comparison, dict)
        self.assertIn('jaccard_similarity', comparison)
        self.assertIn('common_ngrams', comparison)

class TestSentimentFeatures(unittest.TestCase):
    """Test cases for SentimentFeatures class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sentiment_features = SentimentFeatures()
        self.sample_positive_text = "هذا خبر ممتاز وجميل"
        self.sample_negative_text = "هذا خبر سيء ومؤلم جداً"
        self.sample_neutral_text = "هذا خبر عادي"
    
    def test_extract_sentiment_positive(self):
        """Test sentiment extraction for positive text"""
        features = self.sentiment_features.extract_sentiment_features(self.sample_positive_text)
        
        self.assertIsInstance(features, dict)
        self.assertGreater(features['positive_score'], 0)
        self.assertLess(features['negative_score'], 0.1)
    
    def test_extract_sentiment_negative(self):
        """Test sentiment extraction for negative text"""
        features = self.sentiment_features.extract_sentiment_features(self.sample_negative_text)
        
        self.assertIsInstance(features, dict)
        self.assertGreater(features['negative_score'], 0)
        self.assertLess(features['positive_score'], 0.1)
    
    def test_extract_sentiment_neutral(self):
        """Test sentiment extraction for neutral text"""
        features = self.sentiment_features.extract_sentiment_features(self.sample_neutral_text)
        
        self.assertIsInstance(features, dict)
        # Neutral text should have balanced scores
        self.assertAlmostEqual(features['positive_score'], features['negative_score'], delta=0.1)
    
    def test_get_sentiment_label(self):
        """Test sentiment label extraction"""
        positive_label = self.sentiment_features.get_sentiment_label(self.sample_positive_text)
        negative_label = self.sentiment_features.get_sentiment_label(self.sample_negative_text)
        neutral_label = self.sentiment_features.get_sentiment_label(self.sample_neutral_text)
        
        self.assertEqual(positive_label, 'positive')
        self.assertEqual(negative_label, 'negative')
        self.assertEqual(neutral_label, 'neutral')
    
    def test_get_dominant_emotion(self):
        """Test dominant emotion extraction"""
        joy_text = "أنا سعيد جداً اليوم"
        fear_text = "أنا خائف من هذا الأمر"
        
        joy_emotion = self.sentiment_features.get_dominant_emotion(joy_text)
        fear_emotion = self.sentiment_features.get_dominant_emotion(fear_text)
        
        self.assertEqual(joy_emotion, 'joy')
        self.assertEqual(fear_emotion, 'fear')

class TestFakeNewsLexicon(unittest.TestCase):
    """Test cases for FakeNewsLexicon class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.lexicon = FakeNewsLexicon()
        self.fake_news_text = "صدمة! كشف سر علاج سحري يقضي على السرطان!"
        self.normal_text = "هذا خبر عادي وموضوعي"
    
    def test_extract_lexicon_features_fake_news(self):
        """Test lexicon feature extraction for fake news"""
        features = self.lexicon.extract_lexicon_features(self.fake_news_text, 'arabic')
        
        self.assertIsInstance(features, dict)
        self.assertGreater(features['clickbait_score'], 0.1)
        self.assertGreater(features['overall_fake_news_risk'], 0.3)
    
    def test_extract_lexicon_features_normal(self):
        """Test lexicon feature extraction for normal text"""
        features = self.lexicon.extract_lexicon_features(self.normal_text, 'arabic')
        
        self.assertIsInstance(features, dict)
        self.assertLess(features['clickbait_score'], 0.1)
        self.assertLess(features['overall_fake_news_risk'], 0.2)
    
    def test_get_risk_level(self):
        """Test risk level classification"""
        very_low_text = "هذا خبر عادي"
        very_high_text = "صدمة! كشف مؤامرة خطيرة!"
        
        very_low_risk = self.lexicon.get_risk_level(very_low_text, 'arabic')
        very_high_risk = self.lexicon.get_risk_level(very_high_text, 'arabic')
        
        self.assertEqual(very_low_risk, 'low')  # Normal text might still have some risk
        self.assertEqual(very_high_risk, 'very_high')
    
    def test_get_dominant_indicators(self):
        """Test dominant indicator extraction"""
        indicators = self.lexicon.get_dominant_indicators(self.fake_news_text, 'arabic')
        
        self.assertIsInstance(indicators, list)
        self.assertGreater(len(indicators), 0)
        
        # Check that clickbait is likely a top indicator
        indicator_types = [ind[0] for ind in indicators]
        self.assertIn('clickbait', indicator_types)
    
    def test_explain_lexicon_analysis(self):
        """Test lexicon explanation generation"""
        explanation = self.lexicon.explain_lexicon_analysis(self.fake_news_text, 'arabic')
        
        self.assertIsInstance(explanation, dict)
        self.assertIn('risk_level', explanation)
        self.assertIn('dominant_indicators', explanation)
        self.assertIn('summary', explanation)

class TestFeatureIntegration(unittest.TestCase):
    """Test integration between different feature extractors"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.text_features = TextFeatures()
        self.sentiment_features = SentimentFeatures()
        self.lexicon = FakeNewsLexicon()
        
        self.sample_text = "هذا خبر صادم جداً ويحتوي مؤامرة!"
    
    def test_feature_integration(self):
        """Test integration between different feature types"""
        # Extract all feature types
        text_feats = self.text_features.extract_features(self.sample_text)
        sentiment_feats = self.sentiment_features.extract_sentiment_features(self.sample_text)
        lexicon_feats = self.lexicon.extract_lexicon_features(self.sample_text)
        
        # All should be dictionaries
        for feats in [text_feats, sentiment_feats, lexicon_feats]:
            self.assertIsInstance(feats, dict)
        
        # Check consistency
        self.assertEqual(text_feats['text_length'], len(self.sample_text))
        self.assertGreaterEqual(sentiment_feats['positive_score'], 0)
        self.assertGreaterEqual(lexicon_feats['overall_fake_news_risk'], 0)
    
    def test_multilingual_features(self):
        """Test feature extraction for multiple languages"""
        arabic_text = "هذا خبر بالعربية"
        english_text = "This is English news"
        mixed_text = "هذا خبر mixed English text"
        
        # Test each language
        arabic_feats = self.sentiment_features.extract_sentiment_features(arabic_text, 'arabic')
        english_feats = self.sentiment_features.extract_sentiment_features(english_text, 'english')
        auto_feats = self.sentiment_features.extract_sentiment_features(mixed_text, 'auto')
        
        # All should return valid features
        for feats in [arabic_feats, english_feats, auto_feats]:
            self.assertIsInstance(feats, dict)
            self.assertIn('positive_score', feats)
            self.assertIn('negative_score', feats)

if __name__ == '__main__':
    unittest.main()