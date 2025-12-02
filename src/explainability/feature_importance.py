"""
Feature Importance Module
Analyzes and visualizes feature importance for fake news detection
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pandas as pd

# Set Arabic font support
plt.rcParams['font.family'] = 'DejaVu Sans'

class FeatureImportance:
    """
    Analyzes feature importance from multiple sources:
    - Risk scoring weights
    - Feature correlations
    - Statistical significance
    - Visual importance representations
    """
    
    def __init__(self):
        """Initialize feature importance analyzer"""
        self.importance_history = []
        self.feature_categories = {
            'text_features': [
                'text_length', 'word_count', 'punctuation_ratio',
                'exclamation_ratio', 'question_ratio', 'url_ratio'
            ],
            'sentiment_features': [
                'positive_score', 'negative_score', 'sentiment_polarity',
                'sentiment_subjectivity', 'intensity_score'
            ],
            'lexicon_features': [
                'clickbait_score', 'uncertainty_score', 'conspiracy_score',
                'propaganda_score', 'overall_fake_news_risk'
            ],
            'language_features': [
                'arabic_char_ratio', 'latin_char_ratio', 'is_code_switched',
                'language_confidence'
            ],
            'tfidf_features': [
                'avg_tfidf_score', 'max_tfidf_score', 'feature_density',
                'sparsity'
            ],
            'ngram_features': [
                'total_ngrams', 'unique_ngrams', 'avg_frequency',
                'lexical_diversity'
            ]
        }
    
    def analyze_global_importance(self, risk_analyses: List[Dict],
                              feature_results_list: List[Dict]) -> Dict:
        """
        Analyze global feature importance across multiple predictions
        
        Args:
            risk_analyses: List of risk analysis results
            feature_results_list: List of feature extraction results
            
        Returns:
            Dictionary containing global importance analysis
        """
        if not risk_analyses or not feature_results_list:
            return {}
        
        # Extract risk factors
        all_risk_factors = []
        for analysis in risk_analyses:
            risk_factors = analysis.get('risk_factors', [])
            all_risk_factors.extend(risk_factors)
        
        # Analyze risk factor importance
        factor_importance = self._analyze_risk_factors(all_risk_factors)
        
        # Analyze feature correlations with risk scores
        feature_correlations = self._analyze_feature_risk_correlations(
            feature_results_list, risk_analyses
        )
        
        # Analyze statistical significance
        statistical_significance = self._analyze_statistical_significance(
            feature_results_list, risk_analyses
        )
        
        # Compute category importance
        category_importance = self._compute_category_importance(
            feature_correlations, statistical_significance
        )
        
        return {
            'risk_factor_importance': factor_importance,
            'feature_correlations': feature_correlations,
            'statistical_significance': statistical_significance,
            'category_importance': category_importance,
            'total_analyses': len(risk_analyses),
            'analysis_summary': self._generate_importance_summary(
                factor_importance, category_importance
            )
        }
    
    def _analyze_risk_factors(self, risk_factors: List[Dict]) -> Dict:
        """Analyze importance of different risk factors"""
        factor_counts = defaultdict(int)
        factor_impacts = defaultdict(list)
        
        for factor in risk_factors:
            factor_name = factor.get('factor', 'unknown')
            impact = factor.get('impact', 0.0)
            
            factor_counts[factor_name] += 1
            factor_impacts[factor_name].append(impact)
        
        # Compute statistics for each factor
        factor_stats = {}
        
        for factor_name, count in factor_counts.items():
            impacts = factor_impacts[factor_name]
            
            stats = {
                'occurrence_count': count,
                'total_impact': sum(impacts),
                'avg_impact': np.mean(impacts),
                'max_impact': np.max(impacts),
                'min_impact': np.min(impacts),
                'std_impact': np.std(impacts),
                'frequency': count / len(risk_factors) if risk_factors else 0.0
            }
            
            factor_stats[factor_name] = stats
        
        # Sort by average impact
        sorted_factors = sorted(
            factor_stats.items(),
            key=lambda x: x[1]['avg_impact'],
            reverse=True
        )
        
        return {
            'factor_statistics': dict(sorted_factors),
            'most_common_factor': max(factor_counts.items(), key=lambda x: x[1])[0] if factor_counts else None,
            'highest_impact_factor': max(factor_impacts.items(), key=lambda x: np.mean(x[1]))[0] if factor_impacts else None,
            'total_unique_factors': len(factor_counts)
        }
    
    def _analyze_feature_risk_correlations(self, feature_results_list: List[Dict],
                                       risk_analyses: List[Dict]) -> Dict:
        """Analyze correlations between features and risk scores"""
        # Extract feature vectors and risk scores
        feature_vectors = []
        risk_scores = []
        
        for features, risk_analysis in zip(feature_results_list, risk_analyses):
            # Flatten features
            flat_features = self._flatten_features(features)
            feature_vectors.append(flat_features)
            
            # Get risk score
            risk_score = risk_analysis.get('overall_risk_score', 0.0)
            risk_scores.append(risk_score)
        
        if not feature_vectors:
            return {}
        
        # Convert to DataFrame for easier analysis
        feature_df = pd.DataFrame(feature_vectors)
        feature_df['risk_score'] = risk_scores
        
        # Compute correlations
        correlations = {}
        
        for column in feature_df.columns:
            if column != 'risk_score':
                correlation = feature_df[column].corr(feature_df['risk_score'])
                if not np.isnan(correlation):
                    correlations[column] = abs(correlation)
        
        # Sort by correlation strength
        sorted_correlations = sorted(
            correlations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'correlations': dict(sorted_correlations),
            'top_correlated_features': sorted_correlations[:10],
            'correlation_summary': self._summarize_correlations(correlations),
            'feature_count': len(correlations)
        }
    
    def _analyze_statistical_significance(self, feature_results_list: List[Dict],
                                     risk_analyses: List[Dict]) -> Dict:
        """Analyze statistical significance of features"""
        # Separate high and low risk samples
        high_risk_features = []
        low_risk_features = []
        
        for features, risk_analysis in zip(feature_results_list, risk_analyses):
            flat_features = self._flatten_features(features)
            risk_score = risk_analysis.get('overall_risk_score', 0.0)
            
            if risk_score > 0.6:  # High risk threshold
                high_risk_features.append(flat_features)
            else:
                low_risk_features.append(flat_features)
        
        if not high_risk_features or not low_risk_features:
            return {}
        
        # Convert to DataFrames
        high_risk_df = pd.DataFrame(high_risk_features)
        low_risk_df = pd.DataFrame(low_risk_features)
        
        # Compute statistical tests
        significance_tests = {}
        
        for column in high_risk_df.columns:
            if column in low_risk_df.columns:
                high_values = high_risk_df[column].dropna()
                low_values = low_risk_df[column].dropna()
                
                if len(high_values) > 0 and len(low_values) > 0:
                    # Simple t-test approximation
                    mean_diff = np.mean(high_values) - np.mean(low_values)
                    pooled_std = np.sqrt(
                        (np.var(high_values) * (len(high_values) - 1) +
                         np.var(low_values) * (len(low_values) - 1)) /
                        (len(high_values) + len(low_values) - 2)
                    )
                    
                    if pooled_std > 0:
                        t_statistic = mean_diff / (pooled_std * np.sqrt(1/len(high_values) + 1/len(low_values)))
                        significance_tests[column] = {
                            't_statistic': t_statistic,
                            'mean_difference': mean_diff,
                            'high_risk_mean': np.mean(high_values),
                            'low_risk_mean': np.mean(low_values),
                            'effect_size': abs(mean_diff) / pooled_std
                        }
        
        # Sort by effect size
        sorted_significance = sorted(
            significance_tests.items(),
            key=lambda x: abs(x[1]['effect_size']),
            reverse=True
        )
        
        return {
            'significance_tests': dict(sorted_significance),
            'most_significant_features': sorted_significance[:10],
            'high_risk_sample_count': len(high_risk_features),
            'low_risk_sample_count': len(low_risk_features)
        }
    
    def _compute_category_importance(self, feature_correlations: Dict,
                                   statistical_significance: Dict) -> Dict:
        """Compute importance for each feature category"""
        category_importance = {}
        
        for category, features in self.feature_categories.items():
            category_scores = []
            category_effects = []
            
            for feature in features:
                # Get correlation
                correlation = feature_correlations.get(feature, 0.0)
                category_scores.append(correlation)
                
                # Get effect size from significance tests
                significance = statistical_significance.get(feature, {})
                effect_size = significance.get('effect_size', 0.0)
                category_effects.append(effect_size)
            
            # Compute category importance
            if category_scores:
                avg_correlation = np.mean(category_scores)
                max_correlation = np.max(category_scores)
            else:
                avg_correlation = max_correlation = 0.0
            
            if category_effects:
                avg_effect_size = np.mean(category_effects)
                max_effect_size = np.max(category_effects)
            else:
                avg_effect_size = max_effect_size = 0.0
            
            # Combined importance score
            importance_score = (avg_correlation * 0.5 + avg_effect_size * 0.5)
            
            category_importance[category] = {
                'importance_score': importance_score,
                'avg_correlation': avg_correlation,
                'max_correlation': max_correlation,
                'avg_effect_size': avg_effect_size,
                'max_effect_size': max_effect_size,
                'feature_count': len(features),
                'top_features': self._get_top_category_features(
                    features, feature_correlations, statistical_significance
                )
            }
        
        # Sort by importance score
        sorted_categories = sorted(
            category_importance.items(),
            key=lambda x: x[1]['importance_score'],
            reverse=True
        )
        
        return dict(sorted_categories)
    
    def _flatten_features(self, features: Dict) -> Dict:
        """Flatten nested feature dictionary"""
        flattened = {}
        
        def _flatten(d, parent_key=''):
            for key, value in d.items():
                new_key = f"{parent_key}_{key}" if parent_key else key
                
                if isinstance(value, dict):
                    _flatten(value, new_key)
                elif isinstance(value, (int, float)):
                    flattened[new_key] = float(value)
                elif isinstance(value, bool):
                    flattened[new_key] = float(int(value))
        
        _flatten(features)
        return flattened
    
    def _summarize_correlations(self, correlations: Dict) -> Dict:
        """Summarize correlation statistics"""
        if not correlations:
            return {}
        
        values = list(correlations.values())
        
        return {
            'mean_correlation': np.mean(values),
            'median_correlation': np.median(values),
            'std_correlation': np.std(values),
            'max_correlation': np.max(values),
            'min_correlation': np.min(values),
            'correlation_range': np.max(values) - np.min(values)
        }
    
    def _get_top_category_features(self, category_features: List[str],
                                 feature_correlations: Dict,
                                 statistical_significance: Dict) -> List[Dict]:
        """Get top features in a category"""
        feature_scores = []
        
        for feature in category_features:
            correlation = feature_correlations.get(feature, 0.0)
            significance = statistical_significance.get(feature, {})
            effect_size = significance.get('effect_size', 0.0)
            
            combined_score = correlation * 0.5 + effect_size * 0.5
            
            feature_scores.append({
                'feature': feature,
                'combined_score': combined_score,
                'correlation': correlation,
                'effect_size': effect_size
            })
        
        # Sort and return top 5
        feature_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        return feature_scores[:5]
    
    def _generate_importance_summary(self, factor_importance: Dict,
                                  category_importance: Dict) -> str:
        """Generate text summary of importance analysis"""
        summary_parts = []
        
        # Top risk factors
        if factor_importance.get('factor_statistics'):
            top_factors = list(factor_importance['factor_statistics'].keys())[:3]
            summary_parts.append(f"Top risk factors: {', '.join(top_factors)}")
        
        # Top categories
        if category_importance:
            top_categories = list(category_importance.keys())[:3]
            summary_parts.append(f"Most important feature categories: {', '.join(top_categories)}")
        
        return '. '.join(summary_parts)
    
    def create_importance_visualization(self, importance_data: Dict,
                                   chart_type: str = 'bar') -> Dict:
        """
        Create visualization data for feature importance
        
        Args:
            importance_data: Dictionary containing importance analysis
            chart_type: Type of chart ('bar', 'heatmap', 'radar')
            
        Returns:
            Dictionary containing visualization data
        """
        if chart_type == 'bar':
            return self._create_bar_chart(importance_data)
        elif chart_type == 'heatmap':
            return self._create_heatmap(importance_data)
        elif chart_type == 'radar':
            return self._create_radar_chart(importance_data)
        else:
            return {'error': f'Unsupported chart type: {chart_type}'}
    
    def _create_bar_chart(self, importance_data: Dict) -> Dict:
        """Create bar chart visualization data"""
        correlations = importance_data.get('feature_correlations', {})
        
        if not correlations:
            return {'error': 'No correlation data available'}
        
        # Get top 15 features
        top_features = sorted(
            correlations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]
        
        features = [f[0] for f in top_features]
        values = [f[1] for f in top_features]
        colors = ['#ff6b6b' if v > 0.5 else '#4ecdc4' for v in values]
        
        return {
            'type': 'bar_chart',
            'data': {
                'features': features,
                'values': values,
                'colors': colors
            },
            'layout': {
                'title': 'Top 15 Feature Correlations with Risk Score',
                'x_label': 'Features',
                'y_label': 'Correlation Strength',
                'orientation': 'h'  # Horizontal for better readability
            }
        }
    
    def _create_heatmap(self, importance_data: Dict) -> Dict:
        """Create heatmap visualization data"""
        category_importance = importance_data.get('category_importance', {})
        
        if not category_importance:
            return {'error': 'No category data available'}
        
        # Create matrix data
        categories = list(category_importance.keys())
        metrics = ['importance_score', 'avg_correlation', 'avg_effect_size']
        
        matrix_data = []
        for category in categories:
            row = []
            cat_data = category_importance[category]
            for metric in metrics:
                value = cat_data.get(metric, 0.0)
                row.append(value)
            matrix_data.append(row)
        
        return {
            'type': 'heatmap',
            'data': {
                'matrix': matrix_data,
                'categories': categories,
                'metrics': metrics
            },
            'layout': {
                'title': 'Feature Category Importance Heatmap',
                'x_label': 'Metrics',
                'y_label': 'Categories'
            }
        }
    
    def _create_radar_chart(self, importance_data: Dict) -> Dict:
        """Create radar chart visualization data"""
        category_importance = importance_data.get('category_importance', {})
        
        if not category_importance:
            return {'error': 'No category data available'}
        
        # Prepare radar data
        categories = list(category_importance.keys())
        values = [category_importance[cat].get('importance_score', 0.0) for cat in categories]
        
        # Close the radar chart
        values.append(values[0])
        categories.append(categories[0])
        
        return {
            'type': 'radar_chart',
            'data': {
                'categories': categories,
                'values': values
            },
            'layout': {
                'title': 'Feature Category Importance Radar Chart',
                'radial_axis': {
                    'range': [0, 1]
                }
            }
        }
    
    def export_importance_report(self, importance_data: Dict,
                              output_path: str = 'importance_report.html') -> str:
        """
        Export comprehensive importance report as HTML
        
        Args:
            importance_data: Dictionary containing importance analysis
            output_path: Path to save the report
            
        Returns:
            Path to saved report
        """
        html_content = self._generate_html_report(importance_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_html_report(self, importance_data: Dict) -> str:
        """Generate HTML report for importance analysis"""
        html_template = """
        <!DOCTYPE html>
        <html dir="rtl" lang="ar">
        <head>
            <meta charset="UTF-8">
            <title>Feature Importance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; direction: rtl; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
                .section { margin-bottom: 30px; }
                .metric { background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }
                table { border-collapse: collapse; width: 100%; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
                th { background-color: #f2f2f2; }
                .high-importance { color: #d32f2f; }
                .medium-importance { color: #f57c00; }
                .low-importance { color: #388e3c; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>تقرير أهمية الميزات</h1>
                <p>Feature Importance Analysis for Fake News Detection</p>
            </div>
            
            <div class="section">
                <h2>ملخص التحليل</h2>
                <div class="metric">Total Analyses: {total_analyses}</div>
                <div class="metric">Analysis Date: {date}</div>
            </div>
            
            <div class="section">
                <h2>عوامل الخطر الأكثر أهمية</h2>
                {risk_factors_table}
            </div>
            
            <div class="section">
                <h2>أهمية فئات الميزات</h2>
                {categories_table}
            </div>
            
            <div class="section">
                <h2>أهم الميزات المتعلقة بالخطر</h2>
                {correlations_table}
            </div>
        </body>
        </html>
        """
        
        # Format data
        risk_factors = importance_data.get('risk_factor_importance', {})
        categories = importance_data.get('category_importance', {})
        correlations = importance_data.get('feature_correlations', {})
        
        # Generate tables
        risk_factors_table = self._generate_risk_factors_table(risk_factors)
        categories_table = self._generate_categories_table(categories)
        correlations_table = self._generate_correlations_table(correlations)
        
        # Fill template
        return html_template.format(
            total_analyses=importance_data.get('total_analyses', 0),
            date=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            risk_factors_table=risk_factors_table,
            categories_table=categories_table,
            correlations_table=correlations_table
        )
    
    def _generate_risk_factors_table(self, risk_factors: Dict) -> str:
        """Generate HTML table for risk factors"""
        if not risk_factors.get('factor_statistics'):
            return "<p>No risk factor data available</p>"
        
        stats = risk_factors['factor_statistics']
        rows = []
        
        for factor, data in stats.items():
            importance_class = 'high-importance' if data['avg_impact'] > 0.7 else 'medium-importance' if data['avg_impact'] > 0.4 else 'low-importance'
            
            row = f"""
            <tr>
                <td>{factor}</td>
                <td>{data['occurrence_count']}</td>
                <td>{data['avg_impact']:.3f}</td>
                <td>{data['frequency']:.2%}</td>
                <td class="{importance_class}">{data['avg_impact']:.3f}</td>
            </tr>
            """
            rows.append(row)
        
        return f"""
        <table>
            <tr>
                <th>عامل الخطر</th>
                <th>عدد التكرارات</th>
                <th>متوسط التأثير</th>
                <th>التكرار</th>
                <th>الأهمية</th>
            </tr>
            {''.join(rows)}
        </table>
        """
    
    def _generate_categories_table(self, categories: Dict) -> str:
        """Generate HTML table for feature categories"""
        if not categories:
            return "<p>No category data available</p>"
        
        rows = []
        for category, data in categories.items():
            importance_class = 'high-importance' if data['importance_score'] > 0.7 else 'medium-importance' if data['importance_score'] > 0.4 else 'low-importance'
            
            row = f"""
            <tr>
                <td>{category}</td>
                <td>{data['feature_count']}</td>
                <td>{data['importance_score']:.3f}</td>
                <td>{data['avg_correlation']:.3f}</td>
                <td class="{importance_class}">{data['importance_score']:.3f}</td>
            </tr>
            """
            rows.append(row)
        
        return f"""
        <table>
            <tr>
                <th>فئة الميزات</th>
                <th>عدد الميزات</th>
                <th>درجة الأهمية</th>
                <th>متوسط الارتباط</th>
                <th>الأهمية الإجمالية</th>
            </tr>
            {''.join(rows)}
        </table>
        """
    
    def _generate_correlations_table(self, correlations: Dict) -> str:
        """Generate HTML table for feature correlations"""
        if not correlations:
            return "<p>No correlation data available</p>"
        
        # Get top 10 correlations
        top_correlations = sorted(
            correlations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        rows = []
        for feature, correlation in top_correlations:
            importance_class = 'high-importance' if correlation > 0.7 else 'medium-importance' if correlation > 0.4 else 'low-importance'
            
            row = f"""
            <tr>
                <td>{feature}</td>
                <td class="{importance_class}">{correlation:.3f}</td>
                <td>{correlation:.1%}</td>
            </tr>
            """
            rows.append(row)
        
        return f"""
        <table>
            <tr>
                <th>الميزة</th>
                <th>قوة الارتباط</th>
                <th>النسبة المئوية</th>
            </tr>
            {''.join(rows)}
        </table>
        """