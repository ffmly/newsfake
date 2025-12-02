"""
TF-IDF Feature Extraction Module
Computes TF-IDF vectors for text analysis and explainability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

class TfidfFeatures:
    """
    TF-IDF feature extractor with explainability capabilities
    Supports Arabic, Darija, French, and English text
    """
    
    def __init__(self, max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 2, max_df: float = 0.95):
        """
        Initialize TF-IDF feature extractor
        
        Args:
            max_features: Maximum number of features
            ngram_range: Range of n-grams to consider
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            stop_words=None,  # We'll handle stop words separately
            token_pattern=r'(?u)\b\w+\b',  # Unicode word boundaries
            sublinear_tf=True,
            norm='l2'
        )
        
        self.is_fitted = False
        self.feature_names = []
        self.vocabulary = {}
        
    def fit(self, texts: List[str]) -> 'TfidfFeatures':
        """
        Fit TF-IDF vectorizer on training texts
        
        Args:
            texts: List of training texts
            
        Returns:
            Self for method chaining
        """
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Fit vectorizer
        self.vectorizer.fit(processed_texts)
        
        # Store feature information
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.vocabulary = self.vectorizer.vocabulary_
        self.is_fitted = True
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF vectors
        
        Args:
            texts: List of texts to transform
            
        Returns:
            TF-IDF matrix
        """
        if not self.is_fitted:
            raise ValueError("TF-IDF vectorizer must be fitted before transformation")
        
        processed_texts = [self._preprocess_text(text) for text in texts]
        return self.vectorizer.transform(processed_texts)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform texts in one step
        
        Args:
            texts: List of texts to process
            
        Returns:
            TF-IDF matrix
        """
        return self.fit(texts).transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (vocabulary)"""
        return self.feature_names.tolist() if hasattr(self.feature_names, 'tolist') else list(self.feature_names)
    
    def get_vocabulary(self) -> Dict[str, int]:
        """Get vocabulary dictionary"""
        return self.vocabulary.copy()
    
    def get_top_features(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top TF-IDF features for a single text
        
        Args:
            text: Input text
            top_k: Number of top features to return
            
        Returns:
            List of (feature, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("TF-IDF vectorizer must be fitted first")
        
        # Transform text
        tfidf_vector = self.transform([text])
        
        # Get non-zero features
        feature_indices = tfidf_vector.nonzero()[1]
        scores = tfidf_vector.data
        
        # Create list of (feature, score) pairs
        features_scores = []
        for idx, score in zip(feature_indices, scores):
            feature_name = self.feature_names[idx]
            features_scores.append((feature_name, score))
        
        # Sort by score and return top_k
        features_scores.sort(key=lambda x: x[1], reverse=True)
        return features_scores[:top_k]
    
    def get_document_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score
        """
        if not self.is_fitted:
            raise ValueError("TF-IDF vectorizer must be fitted first")
        
        # Transform both texts
        vectors = self.transform([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        return float(similarity)
    
    def get_feature_importance(self, texts: List[str], labels: List[int] = None) -> Dict[str, float]:
        """
        Calculate feature importance based on TF-IDF scores
        
        Args:
            texts: List of texts
            labels: Optional labels for supervised importance
            
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("TF-IDF vectorizer must be fitted first")
        
        # Transform texts
        tfidf_matrix = self.transform(texts)
        
        # Calculate average TF-IDF scores for each feature
        feature_scores = np.mean(tfidf_matrix, axis=0)
        
        # Create importance dictionary
        importance = {}
        for i, feature_name in enumerate(self.feature_names):
            importance[feature_name] = float(feature_scores[0, i])
        
        return importance
    
    def analyze_text_features(self, text: str) -> Dict:
        """
        Comprehensive analysis of text features using TF-IDF
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing feature analysis
        """
        if not self.is_fitted:
            raise ValueError("TF-IDF vectorizer must be fitted first")
        
        # Get TF-IDF vector
        tfidf_vector = self.transform([text])
        
        # Get top features
        top_features = self.get_top_features(text, top_k=20)
        
        # Calculate statistics
        non_zero_features = tfidf_vector.nonzero()[1]
        feature_count = len(non_zero_features)
        total_features = len(self.feature_names)
        
        # Calculate sparsity
        sparsity = 1 - (feature_count / total_features)
        
        # Calculate average TF-IDF score
        avg_score = np.mean(tfidf_vector.data) if tfidf_vector.nnz > 0 else 0
        
        # Calculate max TF-IDF score
        max_score = np.max(tfidf_vector.data) if tfidf_vector.nnz > 0 else 0
        
        return {
            'top_features': top_features,
            'feature_count': feature_count,
            'total_features': total_features,
            'sparsity': sparsity,
            'avg_tfidf_score': float(avg_score),
            'max_tfidf_score': float(max_score),
            'feature_density': feature_count / total_features,
            'tfidf_vector': tfidf_vector.toarray().tolist()[0]
        }
    
    def compare_texts(self, text1: str, text2: str) -> Dict:
        """
        Compare two texts using TF-IDF features
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary containing comparison results
        """
        if not self.is_fitted:
            raise ValueError("TF-IDF vectorizer must be fitted first")
        
        # Get features for both texts
        features1 = self.get_top_features(text1, top_k=10)
        features2 = self.get_top_features(text2, top_k=10)
        
        # Calculate similarity
        similarity = self.get_document_similarity(text1, text2)
        
        # Find common features
        features1_set = set([f[0] for f in features1])
        features2_set = set([f[0] for f in features2])
        common_features = features1_set.intersection(features2_set)
        unique_features1 = features1_set - features2_set
        unique_features2 = features2_set - features1_set
        
        return {
            'similarity': similarity,
            'text1_features': features1,
            'text2_features': features2,
            'common_features': list(common_features),
            'unique_features_text1': list(unique_features1),
            'unique_features_text2': list(unique_features2),
            'common_features_count': len(common_features),
            'jaccard_similarity': len(common_features) / len(features1_set.union(features2_set)) if features1_set.union(features2_set) else 0
        }
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for TF-IDF
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Basic preprocessing
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        return text.strip()
    
    def get_feature_statistics(self) -> Dict:
        """
        Get statistics about the fitted TF-IDF features
        
        Returns:
            Dictionary containing feature statistics
        """
        if not self.is_fitted:
            raise ValueError("TF-IDF vectorizer must be fitted first")
        
        # Feature length statistics
        feature_lengths = [len(feature.split()) for feature in self.feature_names]
        
        return {
            'total_features': len(self.feature_names),
            'vocabulary_size': len(self.vocabulary),
            'avg_feature_length': np.mean(feature_lengths),
            'min_feature_length': np.min(feature_lengths),
            'max_feature_length': np.max(feature_lengths),
            'ngram_range': self.ngram_range,
            'max_features': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df
        }
    
    def save_vocabulary(self, filepath: str):
        """
        Save vocabulary to file
        
        Args:
            filepath: Path to save vocabulary
        """
        if not self.is_fitted:
            raise ValueError("TF-IDF vectorizer must be fitted first")
        
        vocab_df = pd.DataFrame(list(self.vocabulary.items()), columns=['feature', 'index'])
        vocab_df.to_csv(filepath, index=False)
    
    def load_vocabulary(self, filepath: str):
        """
        Load vocabulary from file
        
        Args:
            filepath: Path to load vocabulary from
        """
        vocab_df = pd.read_csv(filepath)
        self.vocabulary = dict(zip(vocab_df['feature'], vocab_df['index']))
        self.feature_names = list(self.vocabulary.keys())
        self.vectorizer.vocabulary_ = self.vocabulary
    
    def explain_prediction(self, text: str, top_k: int = 10) -> Dict:
        """
        Explain TF-IDF features for a text (for interpretability)
        
        Args:
            text: Input text
            top_k: Number of top features to explain
            
        Returns:
            Dictionary containing explanation
        """
        if not self.is_fitted:
            raise ValueError("TF-IDF vectorizer must be fitted first")
        
        # Get top features
        top_features = self.get_top_features(text, top_k)
        
        # Analyze each feature
        explanations = []
        for feature, score in top_features:
            # Get feature statistics
            feature_idx = self.vocabulary.get(feature, -1)
            
            explanation = {
                'feature': feature,
                'tfidf_score': score,
                'feature_type': self._classify_feature(feature),
                'importance_rank': len(explanations) + 1
            }
            
            explanations.append(explanation)
        
        return {
            'text': text,
            'top_features': explanations,
            'explanation_summary': f"Text is characterized by {len(explanations)} main features",
            'dominant_theme': explanations[0]['feature'] if explanations else None
        }
    
    def _classify_feature(self, feature: str) -> str:
        """
        Classify feature type (unigram, bigram, etc.)
        
        Args:
            feature: Feature string
            
        Returns:
            Feature type
        """
        word_count = len(feature.split())
        
        if word_count == 1:
            return 'unigram'
        elif word_count == 2:
            return 'bigram'
        elif word_count == 3:
            return 'trigram'
        else:
            return f'{word_count}-gram'