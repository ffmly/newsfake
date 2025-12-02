"""
N-gram Feature Extraction Module
Extracts unigrams and bigrams for fake news detection
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class NgramFeatures:
    """
    N-gram feature extractor for text analysis
    Supports unigrams and bigrams with various preprocessing options
    """
    
    def __init__(self, ngram_range: Tuple[int, int] = (1, 2), 
                 max_features: int = 10000, min_freq: int = 2):
        """
        Initialize N-gram feature extractor
        
        Args:
            ngram_range: Range of n-grams to extract (1,2) for unigrams+bigrams
            max_features: Maximum number of features to keep
            min_freq: Minimum frequency for n-grams to be included
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_freq = min_freq
        
        # Initialize CountVectorizer
        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_freq,
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b',
            stop_words=None
        )
        
        self.is_fitted = False
        self.vocabulary = {}
        self.feature_names = []
        self.ngram_stats = {}
    
    def fit(self, texts: List[str]) -> 'NgramFeatures':
        """
        Fit N-gram extractor on training texts
        
        Args:
            texts: List of training texts
            
        Returns:
            Self for method chaining
        """
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Fit vectorizer
        self.vectorizer.fit(processed_texts)
        
        # Store vocabulary and feature names
        self.vocabulary = self.vectorizer.vocabulary_
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Calculate n-gram statistics
        self._calculate_ngram_statistics(processed_texts)
        
        self.is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to n-gram count vectors
        
        Args:
            texts: List of texts to transform
            
        Returns:
            N-gram count matrix
        """
        if not self.is_fitted:
            raise ValueError("N-gram extractor must be fitted before transformation")
        
        processed_texts = [self._preprocess_text(text) for text in texts]
        return self.vectorizer.transform(processed_texts)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform texts in one step
        
        Args:
            texts: List of texts to process
            
        Returns:
            N-gram count matrix
        """
        return self.fit(texts).transform(texts)
    
    def extract_ngrams(self, text: str, top_k: int = 20) -> Dict:
        """
        Extract top n-grams from a single text
        
        Args:
            text: Input text
            top_k: Number of top n-grams to return
            
        Returns:
            Dictionary containing n-gram analysis
        """
        if not self.is_fitted:
            raise ValueError("N-gram extractor must be fitted first")
        
        # Transform text
        ngram_vector = self.transform([text])
        
        # Get non-zero n-grams
        ngram_indices = ngram_vector.nonzero()[1]
        counts = ngram_vector.data
        
        # Create list of (ngram, count) pairs
        ngrams_counts = []
        for idx, count in zip(ngram_indices, counts):
            ngram = self.feature_names[idx]
            ngrams_counts.append((ngram, int(count)))
        
        # Sort by count and return top_k
        ngrams_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Separate unigrams and bigrams
        unigrams = [(ng, count) for ng, count in ngrams_counts if len(ng.split()) == 1]
        bigrams = [(ng, count) for ng, count in ngrams_counts if len(ng.split()) == 2]
        
        return {
            'top_ngrams': ngrams_counts[:top_k],
            'top_unigrams': unigrams[:top_k//2],
            'top_bigrams': bigrams[:top_k//2],
            'total_ngrams': len(ngrams_counts),
            'unique_ngrams': len(set([ng for ng, _ in ngrams_counts])),
            'ngram_vector': ngram_vector.toarray().tolist()[0]
        }
    
    def get_ngram_frequency(self, texts: List[str]) -> Dict:
        """
        Calculate n-gram frequency distribution
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary containing frequency distribution
        """
        if not self.is_fitted:
            raise ValueError("N-gram extractor must be fitted first")
        
        # Transform all texts
        ngram_matrix = self.transform(texts)
        
        # Calculate total counts for each n-gram
        total_counts = np.sum(ngram_matrix, axis=0)
        
        # Create frequency dictionary
        frequency_dict = {}
        for i, ngram in enumerate(self.feature_names):
            frequency_dict[ngram] = int(total_counts[0, i])
        
        # Sort by frequency
        sorted_frequency = sorted(frequency_dict.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'frequency_distribution': dict(sorted_frequency),
            'total_ngram_occurrences': int(np.sum(total_counts)),
            'unique_ngrams': len(frequency_dict),
            'most_frequent': sorted_frequency[:10],
            'least_frequent': sorted_frequency[-10:] if len(sorted_frequency) > 10 else sorted_frequency
        }
    
    def analyze_ngram_patterns(self, texts: List[str]) -> Dict:
        """
        Analyze n-gram patterns in the corpus
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary containing pattern analysis
        """
        if not self.is_fitted:
            raise ValueError("N-gram extractor must be fitted first")
        
        # Separate unigrams and bigrams
        unigrams = [ng for ng in self.feature_names if len(ng.split()) == 1]
        bigrams = [bg for bg in self.feature_names if len(bg.split()) == 2]
        
        # Analyze unigram patterns
        unigram_lengths = [len(ng) for ng in unigrams]
        unigram_stats = {
            'count': len(unigrams),
            'avg_length': np.mean(unigram_lengths) if unigram_lengths else 0,
            'min_length': np.min(unigram_lengths) if unigram_lengths else 0,
            'max_length': np.max(unigram_lengths) if unigram_lengths else 0
        }
        
        # Analyze bigram patterns
        bigram_first_words = [bg.split()[0] for bg in bigrams if len(bg.split()) == 2]
        bigram_second_words = [bg.split()[1] for bg in bigrams if len(bg.split()) == 2]
        
        first_word_freq = Counter(bigram_first_words)
        second_word_freq = Counter(bigram_second_words)
        
        bigram_stats = {
            'count': len(bigrams),
            'most_common_first_words': first_word_freq.most_common(10),
            'most_common_second_words': second_word_freq.most_common(10),
            'avg_first_word_length': np.mean([len(w) for w in bigram_first_words]) if bigram_first_words else 0,
            'avg_second_word_length': np.mean([len(w) for w in bigram_second_words]) if bigram_second_words else 0
        }
        
        return {
            'unigram_analysis': unigram_stats,
            'bigram_analysis': bigram_stats,
            'total_features': len(self.feature_names),
            'ngram_range': self.ngram_range
        }
    
    def compare_texts_ngrams(self, text1: str, text2: str, top_k: int = 15) -> Dict:
        """
        Compare n-grams between two texts
        
        Args:
            text1: First text
            text2: Second text
            top_k: Number of top n-grams to compare
            
        Returns:
            Dictionary containing comparison results
        """
        if not self.is_fitted:
            raise ValueError("N-gram extractor must be fitted first")
        
        # Extract n-grams from both texts
        ngrams1 = self.extract_ngrams(text1, top_k)
        ngrams2 = self.extract_ngrams(text2, top_k)
        
        # Get top n-grams
        top_ngrams1 = set([ng for ng, _ in ngrams1['top_ngrams']])
        top_ngrams2 = set([ng for ng, _ in ngrams2['top_ngrams']])
        
        # Calculate similarity metrics
        common_ngrams = top_ngrams1.intersection(top_ngrams2)
        unique_ngrams1 = top_ngrams1 - top_ngrams2
        unique_ngrams2 = top_ngrams2 - top_ngrams1
        
        # Jaccard similarity
        union_ngrams = top_ngrams1.union(top_ngrams2)
        jaccard_similarity = len(common_ngrams) / len(union_ngrams) if union_ngrams else 0
        
        # Overlap coefficient
        overlap_coefficient = len(common_ngrams) / min(len(top_ngrams1), len(top_ngrams2)) if top_ngrams1 and top_ngrams2 else 0
        
        return {
            'text1_ngrams': ngrams1['top_ngrams'],
            'text2_ngrams': ngrams2['top_ngrams'],
            'common_ngrams': list(common_ngrams),
            'unique_ngrams_text1': list(unique_ngrams1),
            'unique_ngrams_text2': list(unique_ngrams2),
            'jaccard_similarity': jaccard_similarity,
            'overlap_coefficient': overlap_coefficient,
            'common_count': len(common_ngrams),
            'total_unique_ngrams': len(union_ngrams)
        }
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (n-grams)"""
        return self.feature_names.tolist() if hasattr(self.feature_names, 'tolist') else list(self.feature_names)
    
    def get_vocabulary(self) -> Dict[str, int]:
        """Get vocabulary dictionary"""
        return self.vocabulary.copy()
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for n-gram extraction
        
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
        
        # Remove punctuation but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        return text.strip()
    
    def _calculate_ngram_statistics(self, texts: List[str]):
        """Calculate internal n-gram statistics"""
        # Transform texts
        ngram_matrix = self.transform(texts)
        
        # Calculate statistics for each n-gram
        doc_freq = np.sum(ngram_matrix > 0, axis=0)  # Document frequency
        total_freq = np.sum(ngram_matrix, axis=0)     # Total frequency
        
        self.ngram_stats = {}
        for i, ngram in enumerate(self.feature_names):
            self.ngram_stats[ngram] = {
                'document_frequency': int(doc_freq[0, i]),
                'total_frequency': int(total_freq[0, i]),
                'avg_frequency_per_doc': float(total_freq[0, i] / doc_freq[0, i]) if doc_freq[0, i] > 0 else 0
            }
    
    def get_ngram_statistics(self) -> Dict:
        """Get n-gram statistics"""
        if not self.is_fitted:
            raise ValueError("N-gram extractor must be fitted first")
        
        return self.ngram_stats.copy()
    
    def find_discriminative_ngrams(self, texts1: List[str], texts2: List[str], 
                                 top_k: int = 20) -> Dict:
        """
        Find n-grams that discriminate between two text collections
        
        Args:
            texts1: First collection of texts
            texts2: Second collection of texts
            top_k: Number of top discriminative n-grams
            
        Returns:
            Dictionary containing discriminative n-grams
        """
        if not self.is_fitted:
            raise ValueError("N-gram extractor must be fitted first")
        
        # Transform both collections
        matrix1 = self.transform(texts1)
        matrix2 = self.transform(texts2)
        
        # Calculate frequencies
        freq1 = np.sum(matrix1, axis=0) / len(texts1)
        freq2 = np.sum(matrix2, axis=0) / len(texts2)
        
        # Calculate discriminative scores (ratio of frequencies)
        discriminative_scores = []
        for i, ngram in enumerate(self.feature_names):
            score1 = freq1[0, i]
            score2 = freq2[0, i]
            
            # Avoid division by zero
            if score1 + score2 == 0:
                ratio = 0
            else:
                ratio = abs(score1 - score2) / (score1 + score2)
            
            discriminative_scores.append((ngram, float(ratio), float(score1), float(score2)))
        
        # Sort by discriminative score
        discriminative_scores.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'discriminative_ngrams': discriminative_scores[:top_k],
            'collection1_size': len(texts1),
            'collection2_size': len(texts2),
            'total_ngrams_analyzed': len(self.feature_names)
        }
    
    def explain_ngrams(self, text: str, top_k: int = 10) -> Dict:
        """
        Explain n-grams in a text for interpretability
        
        Args:
            text: Input text
            top_k: Number of n-grams to explain
            
        Returns:
            Dictionary containing explanation
        """
        if not self.is_fitted:
            raise ValueError("N-gram extractor must be fitted first")
        
        # Extract n-grams
        ngram_analysis = self.extract_ngrams(text, top_k)
        
        # Create explanations
        explanations = []
        for ngram, count in ngram_analysis['top_ngrams']:
            ngram_type = 'unigram' if len(ngram.split()) == 1 else 'bigram'
            
            explanation = {
                'ngram': ngram,
                'count': count,
                'type': ngram_type,
                'importance_rank': len(explanations) + 1,
                'frequency_category': self._categorize_frequency(count),
                'statistics': self.ngram_stats.get(ngram, {})
            }
            
            explanations.append(explanation)
        
        return {
            'text': text,
            'ngram_explanations': explanations,
            'summary': f"Text contains {len(ngram_analysis['total_ngrams'])} unique n-grams",
            'dominant_pattern': explanations[0]['ngram'] if explanations else None
        }
    
    def _categorize_frequency(self, count: int) -> str:
        """Categorize n-gram frequency"""
        if count == 1:
            return 'rare'
        elif count <= 3:
            return 'uncommon'
        elif count <= 10:
            return 'common'
        else:
            return 'frequent'