"""
Test suite for Haqiqa API Client
"""

import pytest
import unittest
from unittest.mock import Mock, patch
import requests
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import directly from the module
from api_client.haqiqa_client import HaqiqaClient

class TestHaqiqaClient(unittest.TestCase):
    """Test cases for HaqiqaClient class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.client = HaqiqaClient()
        self.sample_text = "هذا خبر اختبار للكشف عن الأخبار الكاذبة"
        self.sample_text_en = "This is a test news for fake news detection"
    
    def test_client_initialization(self):
        """Test client initialization"""
        client = HaqiqaClient()
        self.assertIsNotNone(client)
        self.assertEqual(client.api_url, "https://walidalsafadi-haqiqa-arabic-fake-news-detector.hf.space/api/predict")
        self.assertEqual(client.timeout, 30)
    
    def test_client_initialization_custom_params(self):
        """Test client initialization with custom parameters"""
        custom_url = "https://custom-api.example.com/predict"
        custom_timeout = 60
        
        client = HaqiqaClient(api_url=custom_url, timeout=custom_timeout)
        self.assertEqual(client.api_url, custom_url)
        self.assertEqual(client.timeout, custom_timeout)
    
    @patch('src.api_client.haqiqa_client.requests.Session.post')
    def test_predict_success(self, mock_post):
        """Test successful prediction"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "prediction": "Fake",
            "confidence": 0.85,
            "real_prob": 0.15,
            "fake_prob": 0.85
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        client = HaqiqaClient()
        result = client.predict(self.sample_text, "arabert")
        
        # Verify the result
        self.assertTrue(result['success'])
        self.assertEqual(result['prediction'], 'Fake')
        self.assertEqual(result['confidence'], 0.85)
        self.assertEqual(result['real_probability'], 0.15)
        self.assertEqual(result['fake_probability'], 0.85)
        self.assertEqual(result['model_used'], 'arabert')
        
        # Verify API was called correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertIn('{"data": [self.sample_text, "arabert"]}', str(call_args))
    
    @patch('src.api_client.haqiqa_client.requests.Session.post')
    def test_predict_api_error(self, mock_post):
        """Test API error handling"""
        # Mock API error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error")
        mock_post.return_value = mock_response
        
        client = HaqiqaClient()
        result = client.predict(self.sample_text, "arabert")
        
        # Verify error handling
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'API_ERROR')
        self.assertIn('API_TIMEOUT' not in result['error'], result)
    
    @patch('src.api_client.haqiqa_client.requests.Session.post')
    def test_predict_timeout(self, mock_post):
        """Test timeout handling"""
        # Mock timeout
        mock_post.side_effect = requests.exceptions.Timeout("Request timeout")
        
        client = HaqiqaClient(timeout=1)
        result = client.predict(self.sample_text, "arabert")
        
        # Verify timeout handling
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'API_TIMEOUT')
        self.assertIn('timeout', result['message'].lower())
    
    def test_predict_empty_text(self):
        """Test prediction with empty text"""
        client = HaqiqaClient()
        result = client.predict("", "arabert")
        
        # Should handle empty text gracefully
        self.assertTrue(result['success'])
        # The actual API call will handle empty text
    
    def test_predict_with_fallback_success(self):
        """Test prediction with fallback mechanism"""
        # This test would require mocking both primary and fallback calls
        client = HaqiqaClient()
        
        # Test that fallback logic exists
        self.assertTrue(hasattr(client, 'predict_with_fallback'))
    
    @patch('src.api_client.haqiqa_client.requests.Session.post')
    def test_batch_predict(self, mock_post):
        """Test batch prediction"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"prediction": "Fake", "confidence": 0.85},
            {"prediction": "Real", "confidence": 0.75}
        ]
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        client = HaqiqaClient()
        texts = [self.sample_text, self.sample_text_en]
        results = client.batch_predict(texts, "arabert")
        
        # Verify batch results
        self.assertEqual(len(results), 2)
        self.assertTrue(all('success' in result for result in results))
    
    def test_get_supported_models(self):
        """Test getting supported models"""
        client = HaqiqaClient()
        models = client.get_supported_models()
        
        expected_models = ["arabert", "xgboost"]
        self.assertEqual(sorted(models), sorted(expected_models))
    
    @patch('src.api_client.haqiqa_client.requests.Session.get')
    def test_health_check_success(self, mock_get):
        """Test successful health check"""
        # Mock successful health check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "prediction": "Real",
            "confidence": 0.95
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        client = HaqiqaClient()
        health = client.health_check()
        
        # Verify health check
        self.assertEqual(health['status'], 'healthy')
        self.assertIn('test_prediction', health)
    
    @patch('src.api_client.haqiqa_client.requests.Session.get')
    def test_health_check_failure(self, mock_get):
        """Test health check failure"""
        # Mock failed health check
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")
        
        client = HaqiqaClient()
        health = client.health_check()
        
        # Verify error handling
        self.assertEqual(health['status'], 'unhealthy')
        self.assertIn('Connection error', health['message'])
    
    def test_context_manager(self):
        """Test client context manager"""
        with HaqiqaClient() as client:
            self.assertIsNotNone(client)
        
        # Context manager should clean up properly
        # This test mainly ensures no exceptions are raised
    
    def test_standardize_response_format(self):
        """Test response standardization"""
        client = HaqiqaClient()
        
        # Test with list response (Gradio format)
        list_response = [{"prediction": "Fake", "confidence": 0.85}]
        standardized = client._standardize_response(list_response, "arabert")
        
        self.assertTrue(standardized['success'])
        self.assertEqual(standardized['prediction'], 'Fake')
        self.assertEqual(standardized['confidence'], 0.85)
        
        # Test with dict response
        dict_response = {"prediction": "Real", "confidence": 0.75}
        standardized = client._standardize_response(dict_response, "xgboost")
        
        self.assertTrue(standardized['success'])
        self.assertEqual(standardized['prediction'], 'Real')
        self.assertEqual(standardized['confidence'], 0.75)
        self.assertEqual(standardized['model_used'], 'xgboost')
    
    def test_error_handling_invalid_response(self):
        """Test error handling with invalid response"""
        client = HaqiqaClient()
        
        # Test with None response
        standardized = client._standardize_response(None, "arabert")
        
        self.assertFalse(standardized['success'])
        self.assertEqual(standardized['prediction'], 'Unknown')
        self.assertEqual(standardized['confidence'], 0.0)
        
        # Test with invalid JSON
        with patch('json.loads', side_effect=json.JSONDecodeError("Invalid JSON")):
            with patch.object(client.session, 'post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON")
                mock_post.return_value = mock_response
                
                result = client.predict(self.sample_text, "arabert")
                
                self.assertFalse(result['success'])
                self.assertEqual(result['error'], 'JSON_PARSE_ERROR')

class TestHaqiqaClientIntegration(unittest.TestCase):
    """Integration tests for HaqiqaClient (requires actual API)"""
    
    @pytest.mark.integration
    def test_real_api_call(self):
        """Test actual API call (marked as integration test)"""
        # This test requires actual API access
        # Mark with pytest -m "not integration" to skip during unit testing
        
        client = HaqiqaClient()
        
        # Test with a simple Arabic text
        result = client.predict("هذا خبر اختبار", "arabert")
        
        # Basic validations
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        
        if result['success']:
            self.assertIn('prediction', result)
            self.assertIn('confidence', result)
            self.assertIn('model_used', result)

if __name__ == '__main__':
    unittest.main()