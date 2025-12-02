"""
LIME-like Explainer Module
Provides local interpretability for fake news detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import random
import re

class LimeExplainer:
    """
    LIME-like explainer for fake news detection
    Provides local explanations by analyzing feature importance
    """
    
    def __init__(self, num_samples: int = 5000, num_features: int = 10):
        """
        Initialize LIME explainer
        
        Args:
            num_samples: Number of samples to generate for explanation
            num_features: Number of top features to include in explanation
        """
        self.num_samples = num_samples
        self.num_features = num_features
        
        # Feature importance tracking
        self.feature_importance = {}
        self.explanation_history = []
    
    def explain_prediction(self, text: str, prediction_result: Dict,
                         feature_results: Dict, predict_fn: callable) -> Dict:
        """
        Generate LIME-like explanation for a prediction
        
        Args:
            text: Original text
            prediction_result: Prediction result from the model
            feature_results: Extracted features
            predict_fn: Prediction function that takes text and returns score
            
        Returns:
            Dictionary containing explanation
        """
        # Tokenize text
        tokens = self._tokenize_text(text)
        
        if not tokens:
            return {
                'explanation': 'No tokens to explain',
                'feature_importance': [],
                'success': False
            }
        
        # Generate perturbed samples
        samples, predictions = self._generate_samples(text, tokens, predict_fn)
        
        # Compute feature importance
        importance_scores = self._compute_feature_importance(
            tokens, samples, predictions
        )
        
        # Select top features
        top_features = self._select_top_features(importance_scores)
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            text, top_features, prediction_result
        )
        
        # Create explanation
        explanation = {
            'original_text': text,
            'prediction': prediction_result,
            'feature_importance': top_features,
            'explanation_text': explanation_text,
            'tokens': tokens,
            'num_samples': len(samples),
            'confidence': self._compute_explanation_confidence(importance_scores),
            'intercept': self._compute_intercept(predictions),
            'success': True
        }
        
        # Store explanation
        self.explanation_history.append(explanation)
        
        return explanation
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for explanation"""
        # Simple tokenization - split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _generate_samples(self, text: str, tokens: List[str], 
                         predict_fn: callable) -> Tuple[List[Dict], List[float]]:
        """Generate perturbed samples for LIME"""
        samples = []
        predictions = []
        
        # Original prediction
        original_prediction = predict_fn(text)
        
        for i in range(self.num_samples):
            # Randomly select subset of tokens to keep
            num_tokens_to_keep = random.randint(1, len(tokens))
            tokens_to_keep = random.sample(tokens, num_tokens_to_keep)
            
            # Create perturbed text
            perturbed_text = ' '.join(tokens_to_keep)
            
            # Get prediction for perturbed text
            try:
                perturbed_prediction = predict_fn(perturbed_text)
            except:
                # If prediction fails, use neutral score
                perturbed_prediction = 0.5
            
            # Create sample record
            sample = {
                'text': perturbed_text,
                'tokens_present': set(tokens_to_keep),
                'num_tokens_present': num_tokens_to_keep,
                'prediction': perturbed_prediction
            }
            
            samples.append(sample)
            predictions.append(perturbed_prediction)
        
        return samples, predictions
    
    def _compute_feature_importance(self, tokens: List[str], 
                                 samples: List[Dict], 
                                 predictions: List[float]) -> Dict[str, float]:
        """Compute importance scores for each token"""
        # Initialize importance scores
        importance_scores = defaultdict(list)
        
        # For each sample, compute contribution of each token
        for sample, prediction in zip(samples, predictions):
            tokens_present = sample['tokens_present']
            
            for token in tokens:
                if token in tokens_present:
                    # Token is present in this sample
                    importance_scores[token].append(prediction)
                else:
                    # Token is absent, add neutral prediction
                    importance_scores[token].append(0.5)
        
        # Compute final importance scores
        final_importance = {}
        
        for token in tokens:
            scores = importance_scores[token]
            if not scores:
                final_importance[token] = 0.0
            else:
                # Compute importance as deviation from mean
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                # Higher importance if token presence significantly changes prediction
                importance = abs(mean_score - 0.5) * (1 + std_score)
                final_importance[token] = importance
        
        return final_importance
    
    def _select_top_features(self, importance_scores: Dict[str, float]) -> List[Dict]:
        """Select top features for explanation"""
        # Sort by importance
        sorted_features = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top features
        top_features = []
        for i, (token, importance) in enumerate(sorted_features[:self.num_features]):
            # Determine direction (positive/negative contribution)
            direction = 'positive' if importance > 0 else 'negative'
            
            feature_info = {
                'feature': token,
                'importance': importance,
                'rank': i + 1,
                'direction': direction,
                'weight': abs(importance)
            }
            
            top_features.append(feature_info)
        
        return top_features
    
    def _generate_explanation_text(self, text: str, top_features: List[Dict],
                                prediction_result: Dict) -> str:
        """Generate human-readable explanation text"""
        if not top_features:
            return "No significant features found for explanation."
        
        # Get prediction label
        prediction_label = prediction_result.get('prediction', 'Unknown')
        confidence = prediction_result.get('confidence', 0.0)
        
        # Separate positive and negative features
        positive_features = [f for f in top_features if f['direction'] == 'positive']
        negative_features = [f for f in top_features if f['direction'] == 'negative']
        
        # Generate explanation
        explanation_parts = []
        
        # Main prediction statement
        explanation_parts.append(
            f"The text is classified as '{prediction_label}' with {confidence:.1%} confidence."
        )
        
        # Positive contributors
        if positive_features:
            pos_words = [f['feature'] for f in positive_features[:3]]
            explanation_parts.append(
                f"Words that suggest this classification: {', '.join(pos_words)}."
            )
        
        # Negative contributors
        if negative_features:
            neg_words = [f['feature'] for f in negative_features[:3]]
            explanation_parts.append(
                f"Words that contradict this classification: {', '.join(neg_words)}."
            )
        
        # Most important feature
        if top_features:
            most_important = top_features[0]
            explanation_parts.append(
                f"The most influential word is '{most_important['feature']}' "
                f"with importance score {most_important['importance']:.3f}."
            )
        
        return ' '.join(explanation_parts)
    
    def _compute_explanation_confidence(self, importance_scores: Dict[str, float]) -> float:
        """Compute confidence in the explanation"""
        if not importance_scores:
            return 0.0
        
        scores = list(importance_scores.values())
        
        # Higher confidence if importance scores are well-distributed
        std_score = np.std(scores)
        mean_score = np.mean(scores)
        
        # Normalize confidence
        confidence = min(1.0, (std_score / (mean_score + 1e-6)))
        return confidence
    
    def _compute_intercept(self, predictions: List[float]) -> float:
        """Compute intercept (baseline prediction)"""
        return np.mean(predictions)
    
    def explain_batch(self, texts: List[str], prediction_results: List[Dict],
                    feature_results_list: List[Dict], 
                    predict_fn: callable) -> List[Dict]:
        """Generate explanations for multiple texts"""
        explanations = []
        
        for i, (text, pred_result, features) in enumerate(
            zip(texts, prediction_results, feature_results_list)
        ):
            explanation = self.explain_prediction(text, pred_result, features, predict_fn)
            explanation['batch_index'] = i
            explanations.append(explanation)
        
        return explanations
    
    def get_feature_statistics(self) -> Dict:
        """Get statistics about feature importance across explanations"""
        if not self.explanation_history:
            return {}
        
        # Collect all feature importances
        all_importances = defaultdict(list)
        
        for explanation in self.explanation_history:
            for feature_info in explanation.get('feature_importance', []):
                feature = feature_info['feature']
                importance = feature_info['importance']
                all_importances[feature].append(importance)
        
        # Compute statistics
        feature_stats = {}
        
        for feature, importances in all_importances.items():
            feature_stats[feature] = {
                'mean_importance': np.mean(importances),
                'std_importance': np.std(importances),
                'max_importance': np.max(importances),
                'min_importance': np.min(importances),
                'appearance_count': len(importances),
                'avg_rank': np.mean([
                    exp['feature_importance'].index(
                        next((f for f in exp['feature_importance'] if f['feature'] == feature), None)
                    ) + 1
                    for exp in self.explanation_history
                    if any(f['feature'] == feature for f in exp['feature_importance'])
                ])
            }
        
        return feature_stats
    
    def get_global_feature_importance(self) -> List[Dict]:
        """Get global feature importance across all explanations"""
        feature_stats = self.get_feature_statistics()
        
        # Sort by mean importance
        sorted_features = sorted(
            feature_stats.items(),
            key=lambda x: x[1]['mean_importance'],
            reverse=True
        )
        
        global_importance = []
        for i, (feature, stats) in enumerate(sorted_features):
            importance_info = {
                'feature': feature,
                'global_importance': stats['mean_importance'],
                'rank': i + 1,
                'appearance_count': stats['appearance_count'],
                'std_importance': stats['std_importance'],
                'avg_rank': stats['avg_rank']
            }
            global_importance.append(importance_info)
        
        return global_importance
    
    def visualize_explanation(self, explanation: Dict) -> Dict:
        """Create visualization data for explanation"""
        feature_importance = explanation.get('feature_importance', [])
        
        # Prepare data for visualization
        visualization_data = {
            'features': [],
            'importances': [],
            'directions': [],
            'colors': []
        }
        
        for feature_info in feature_importance:
            feature = feature_info['feature']
            importance = feature_info['importance']
            direction = feature_info['direction']
            
            visualization_data['features'].append(feature)
            visualization_data['importances'].append(abs(importance))
            visualization_data['directions'].append(direction)
            
            # Color based on direction
            color = '#ff4444' if direction == 'negative' else '#44ff44'
            visualization_data['colors'].append(color)
        
        return {
            'type': 'bar_chart',
            'data': visualization_data,
            'title': 'Feature Importance for Fake News Detection',
            'x_label': 'Features',
            'y_label': 'Importance Score',
            'explanation_text': explanation.get('explanation_text', '')
        }
    
    def compare_explanations(self, explanations: List[Dict]) -> Dict:
        """Compare multiple explanations"""
        if not explanations:
            return {}
        
        # Extract features from all explanations
        all_features = set()
        explanation_features = []
        
        for exp in explanations:
            features = set(f['feature'] for f in exp.get('feature_importance', []))
            all_features.update(features)
            explanation_features.append(features)
        
        # Compute overlap
        common_features = set.intersection(*explanation_features) if explanation_features else set()
        unique_features = [feature_set - common_features for feature_set in explanation_features]
        
        # Compute similarity matrix
        similarity_matrix = []
        for i, exp1 in enumerate(explanation_features):
            row = []
            for j, exp2 in enumerate(explanation_features):
                if i == j:
                    row.append(1.0)
                else:
                    intersection = len(exp1.intersection(exp2))
                    union = len(exp1.union(exp2))
                    similarity = intersection / union if union > 0 else 0.0
                    row.append(similarity)
            similarity_matrix.append(row)
        
        return {
            'total_explanations': len(explanations),
            'total_unique_features': len(all_features),
            'common_features': list(common_features),
            'common_feature_count': len(common_features),
            'unique_features_per_explanation': [list(uniq) for uniq in unique_features],
            'similarity_matrix': similarity_matrix,
            'avg_similarity': np.mean(similarity_matrix) if similarity_matrix else 0.0
        }
    
    def reset_history(self):
        """Reset explanation history"""
        self.explanation_history = []
    
    def get_explanation_history(self) -> List[Dict]:
        """Get all explanation history"""
        return self.explanation_history.copy()
    
    def update_parameters(self, num_samples: int = None, num_features: int = None):
        """Update explainer parameters"""
        if num_samples is not None:
            self.num_samples = num_samples
        
        if num_features is not None:
            self.num_features = num_features