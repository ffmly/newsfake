"""
Feature Combiner Module
Combines and normalizes features from different extractors
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd

class FeatureCombiner:
    """
    Combines features from multiple extractors:
    - Text features (length, punctuation, etc.)
    - TF-IDF features
    - N-gram features
    - Sentiment features
    - Lexicon features
    - Language features
    """
    
    def __init__(self, normalize: bool = True, reduce_dimensions: bool = False):
        """
        Initialize feature combiner
        
        Args:
            normalize: Whether to normalize features
            reduce_dimensions: Whether to apply dimensionality reduction
        """
        self.normalize = normalize
        self.reduce_dimensions = reduce_dimensions
        
        # Scalers for different feature types
        self.scalers = {}
        self.feature_types = {
            'text_features': 'numerical',
            'tfidf_features': 'sparse',
            'ngram_features': 'sparse',
            'sentiment_features': 'numerical',
            'lexicon_features': 'numerical',
            'language_features': 'mixed'
        }
        
        # Dimensionality reduction
        self.pca = None
        self.n_components = 50  # For PCA
        
        # Feature mapping
        self.feature_mapping = {}
        self.is_fitted = False
    
    def combine_features(self, text_features: Dict = None,
                      tfidf_features: Dict = None,
                      ngram_features: Dict = None,
                      sentiment_features: Dict = None,
                      lexicon_features: Dict = None,
                      language_features: Dict = None) -> Dict:
        """
        Combine features from all extractors
        
        Args:
            text_features: Text-level features
            tfidf_features: TF-IDF features
            ngram_features: N-gram features
            sentiment_features: Sentiment analysis features
            lexicon_features: Lexicon-based features
            language_features: Language detection features
            
        Returns:
            Combined feature dictionary
        """
        # Initialize combined features
        combined = {}
        
        # Add text features
        if text_features:
            combined.update(self._flatten_dict(text_features, 'text_'))
        
        # Add TF-IDF features
        if tfidf_features:
            tfidf_flat = self._flatten_tfidf_features(tfidf_features)
            combined.update(tfidf_flat)
        
        # Add N-gram features
        if ngram_features:
            ngram_flat = self._flatten_dict(ngram_features, 'ngram_')
            combined.update(ngram_flat)
        
        # Add sentiment features
        if sentiment_features:
            combined.update(self._flatten_dict(sentiment_features, 'sentiment_'))
        
        # Add lexicon features
        if lexicon_features:
            combined.update(self._flatten_dict(lexicon_features, 'lexicon_'))
        
        # Add language features
        if language_features:
            lang_flat = self._flatten_language_features(language_features)
            combined.update(lang_flat)
        
        # Handle missing values
        combined = self._handle_missing_values(combined)
        
        return combined
    
    def combine_features_to_vector(self, **feature_dicts) -> np.ndarray:
        """
        Combine features into a numeric vector
        
        Args:
            **feature_dicts: Keyword arguments for different feature types
            
        Returns:
            Combined feature vector
        """
        # Combine features
        combined_dict = self.combine_features(**feature_dicts)
        
        # Convert to vector
        feature_vector = []
        feature_names = []
        
        # Sort features by name for consistency
        sorted_features = sorted(combined_dict.items())
        
        for name, value in sorted_features:
            # Convert to numeric
            numeric_value = self._to_numeric(value)
            feature_vector.append(numeric_value)
            feature_names.append(name)
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Normalize if required
        if self.normalize and self.is_fitted:
            feature_vector = self._normalize_vector(feature_vector)
        
        # Apply dimensionality reduction if required
        if self.reduce_dimensions and self.is_fitted and self.pca:
            feature_vector = self.pca.transform(feature_vector)
        
        return feature_vector.flatten(), feature_names
    
    def fit(self, feature_vectors_list: List[Dict]) -> 'FeatureCombiner':
        """
        Fit the combiner on training data
        
        Args:
            feature_vectors_list: List of combined feature dictionaries
            
        Returns:
            Self for method chaining
        """
        if not feature_vectors_list:
            return self
        
        # Convert all feature dictionaries to vectors
        vectors = []
        all_feature_names = set()
        
        for features in feature_vectors_list:
            vector, names = self.combine_features_to_vector(**features)
            vectors.append(vector)
            all_feature_names.update(names)
        
        # Stack vectors
        X = np.vstack(vectors)
        
        # Fit scalers
        if self.normalize:
            self._fit_scalers(X)
        
        # Fit PCA if required
        if self.reduce_dimensions:
            self.pca = PCA(n_components=min(self.n_components, X.shape[1]))
            self.pca.fit(X)
        
        # Store feature mapping
        self._create_feature_mapping(all_feature_names)
        
        self.is_fitted = True
        return self
    
    def transform(self, feature_dicts: List[Dict]) -> List[np.ndarray]:
        """
        Transform feature dictionaries using fitted combiner
        
        Args:
            feature_dicts: List of feature dictionaries
            
        Returns:
            List of transformed feature vectors
        """
        if not self.is_fitted:
            raise ValueError("FeatureCombiner must be fitted before transformation")
        
        transformed = []
        for features in feature_dicts:
            vector, _ = self.combine_features_to_vector(**features)
            transformed.append(vector)
        
        return transformed
    
    def fit_transform(self, feature_dicts_list: List[Dict]) -> List[np.ndarray]:
        """
        Fit and transform in one step
        
        Args:
            feature_dicts_list: List of feature dictionaries
            
        Returns:
            List of transformed feature vectors
        """
        return self.fit(feature_dicts_list).transform(feature_dicts_list)
    
    def _flatten_dict(self, d: Dict, prefix: str = '') -> Dict:
        """Flatten nested dictionary with prefix"""
        flattened = {}
        
        for key, value in d.items():
            new_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                nested = self._flatten_dict(value, f"{new_key}_")
                flattened.update(nested)
            elif isinstance(value, (list, tuple)):
                # Convert lists to multiple features
                for i, item in enumerate(value):
                    if isinstance(item, (int, float, str)):
                        flattened[f"{new_key}_{i}"] = item
                    else:
                        # Handle complex objects
                        flattened[f"{new_key}_{i}_type"] = type(item).__name__
            else:
                flattened[new_key] = value
        
        return flattened
    
    def _flatten_tfidf_features(self, tfidf_features: Dict) -> Dict:
        """Flatten TF-IDF features"""
        flattened = {}
        
        # Handle top features
        top_features = tfidf_features.get('top_features', [])
        for i, (feature, score) in enumerate(top_features):
            flattened[f"tfidf_top_{i}_feature"] = feature
            flattened[f"tfidf_top_{i}_score"] = score
        
        # Handle statistics
        stats = tfidf_features.get('statistics', {})
        for key, value in stats.items():
            flattened[f"tfidf_{key}"] = value
        
        return flattened
    
    def _flatten_language_features(self, language_features: Dict) -> Dict:
        """Flatten language features"""
        flattened = {}
        
        for key, value in language_features.items():
            if key == 'language_distribution':
                # Flatten language distribution
                for lang, ratio in value.items():
                    flattened[f"lang_{lang}_ratio"] = ratio
            elif key == 'segments':
                # Handle segments (convert to count)
                flattened[f"lang_segments_count"] = len(value)
                flattened[f"lang_is_code_switched"] = int(language_features.get('is_code_switched', False))
            else:
                flattened[f"lang_{key}"] = value
        
        return flattened
    
    def _handle_missing_values(self, features: Dict) -> Dict:
        """Handle missing values in features"""
        for key, value in features.items():
            if value is None:
                features[key] = 0.0
            elif isinstance(value, str):
                # Convert string to numeric if possible
                try:
                    features[key] = float(value)
                except ValueError:
                    # Handle categorical strings
                    features[key] = self._encode_categorical(value)
            elif isinstance(value, bool):
                features[key] = int(value)
        
        return features
    
    def _to_numeric(self, value: Any) -> float:
        """Convert value to numeric"""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return self._encode_categorical(value)
        elif isinstance(value, bool):
            return float(int(value))
        else:
            return 0.0
    
    def _encode_categorical(self, value: str) -> float:
        """Encode categorical string to numeric"""
        # Simple encoding for common categories
        encoding_map = {
            'arabic': 1.0,
            'darija': 2.0,
            'french': 3.0,
            'english': 4.0,
            'unknown': 0.0,
            'real': 0.0,
            'fake': 1.0,
            'low': 0.25,
            'medium': 0.5,
            'high': 0.75,
            'very_high': 1.0,
            'very_low': 0.0
        }
        
        return encoding_map.get(value.lower(), 0.0)
    
    def _fit_scalers(self, X: np.ndarray):
        """Fit normalization scalers"""
        if self.normalize:
            self.scalers['standard'] = StandardScaler()
            self.scalers['minmax'] = MinMaxScaler()
            
            # Fit scalers
            self.scalers['standard'].fit(X)
            self.scalers['minmax'].fit(X)
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize feature vector"""
        if self.normalize and 'minmax' in self.scalers:
            return self.scalers['minmax'].transform(vector.reshape(1, -1))
        return vector
    
    def _create_feature_mapping(self, feature_names: set):
        """Create mapping of feature names to indices"""
        self.feature_mapping = {name: idx for idx, name in enumerate(sorted(feature_names))}
    
    def get_feature_importance(self, feature_names: List[str], 
                           importance_scores: List[float]) -> Dict:
        """
        Map importance scores to feature names
        
        Args:
            feature_names: List of feature names
            importance_scores: List of importance scores
            
        Returns:
            Dictionary mapping features to importance
        """
        if len(feature_names) != len(importance_scores):
            raise ValueError("Feature names and importance scores must have same length")
        
        return dict(zip(feature_names, importance_scores))
    
    def get_top_features(self, feature_dict: Dict, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top features by value
        
        Args:
            feature_dict: Feature dictionary
            top_k: Number of top features to return
            
        Returns:
            List of (feature_name, value) tuples
        """
        # Filter numeric features
        numeric_features = {}
        for key, value in feature_dict.items():
            try:
                numeric_value = float(value)
                numeric_features[key] = numeric_value
            except (ValueError, TypeError):
                continue
        
        # Sort by value
        sorted_features = sorted(numeric_features.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_features[:top_k]
    
    def get_feature_statistics(self, feature_dicts: List[Dict]) -> Dict:
        """
        Get statistics for features across multiple samples
        
        Args:
            feature_dicts: List of feature dictionaries
            
        Returns:
            Dictionary containing feature statistics
        """
        if not feature_dicts:
            return {}
        
        # Combine all features
        all_features = {}
        for features in feature_dicts:
            for key, value in features.items():
                if key not in all_features:
                    all_features[key] = []
                try:
                    numeric_value = float(value)
                    all_features[key].append(numeric_value)
                except (ValueError, TypeError):
                    continue
        
        # Calculate statistics
        stats = {}
        for feature_name, values in all_features.items():
            if values:
                stats[feature_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'count': len(values)
                }
        
        return stats
    
    def export_features_to_dataframe(self, feature_dicts: List[Dict]) -> pd.DataFrame:
        """
        Export features to pandas DataFrame
        
        Args:
            feature_dicts: List of feature dictionaries
            
        Returns:
            pandas DataFrame with features
        """
        # Combine all features
        combined_features = []
        for features in feature_dicts:
            combined = self.combine_features(**features)
            combined_features.append(combined)
        
        # Create DataFrame
        df = pd.DataFrame(combined_features)
        
        # Fill missing values
        df = df.fillna(0)
        
        return df
    
    def get_feature_types(self) -> Dict:
        """Get feature types mapping"""
        return self.feature_types.copy()
    
    def is_fitted_status(self) -> bool:
        """Check if combiner is fitted"""
        return self.is_fitted