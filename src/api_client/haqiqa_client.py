"""
Haqiqa API Client for Arabic Fake News Detection
Integrates with the Haqiqa HuggingFace Space API using Gradio Client
"""

import json
import logging
import sys
import os
from typing import Dict, Optional, Tuple

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HaqiqaClient:
    """
    Client for interacting with the Haqiqa Arabic Fake News Detection API
    Using Gradio Client: WalidAlsafadi/Haqiqa-Arabic-Fake-News-Detector
    """
    
    def __init__(self, space_name: str = None, timeout: int = None):
        """
        Initialize Haqiqa API client using Gradio Client
        
        Args:
            space_name: HuggingFace Space name
            timeout: Request timeout in seconds
        """
        self.space_name = space_name or "WalidAlsafadi/Haqiqa-Arabic-Fake-News-Detector"
        self.timeout = timeout or Config.REQUEST_TIMEOUT
        self.client = None
        
        # Initialize Gradio client
        try:
            from gradio_client import Client
            self.client = Client(self.space_name)
            logger.info(f"Successfully initialized Gradio client for {self.space_name}")
        except ImportError:
            logger.error("gradio_client not installed. Please install with: pip install gradio_client")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize Gradio client: {str(e)}")
            self.client = None
    
    def predict(self, text: str, model: str = "arabert") -> Dict:
        """
        Predict fake news using Haqiqa API with Gradio Client
        
        Args:
            text: Arabic text to analyze
            model: Model to use ('arabert' or 'xgboost')
            
        Returns:
            Dictionary containing prediction results
        """
        if not self.client:
            return {
                "error": "CLIENT_NOT_INITIALIZED",
                "message": "Gradio client not initialized. Please install gradio_client",
                "prediction": "Unknown",
                "confidence": 0.0
            }
        
        try:
            logger.info(f"Sending request to Haqiqa API with model: {model}")
            
            # Make API request using Gradio client
            result = self.client.predict(
                text=text,
                model_name=model,
                api_name="/predict"
            )
            
            # Standardize response format
            standardized_result = self._standardize_response(result, model)
            
            logger.info(f"Successfully received prediction: {standardized_result['prediction']}")
            return standardized_result
            
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            return {
                "error": "API_ERROR",
                "message": str(e),
                "prediction": "Unknown",
                "confidence": 0.0
            }
    
    def _standardize_response(self, api_response: Dict, model: str) -> Dict:
        """
        Standardize API response format
        
        Args:
            api_response: Raw response from Haqiqa API
            model: Model name used for prediction
            
        Returns:
            Standardized response dictionary
        """
        # Handle different response formats from Haqiqa
        if isinstance(api_response, list) and len(api_response) > 0:
            # Gradio API returns a list
            response_data = api_response[0]
        elif isinstance(api_response, dict):
            response_data = api_response
        else:
            response_data = api_response
        
        # Extract prediction information
        if isinstance(response_data, dict):
            prediction = response_data.get('prediction', 'Unknown')
            confidence = response_data.get('confidence', 0.0)
            real_prob = response_data.get('real_prob', 0.0)
            fake_prob = response_data.get('fake_prob', 0.0)
        else:
            # Handle string responses
            prediction = str(response_data)
            confidence = 0.5
            real_prob = 0.5
            fake_prob = 0.5
        
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "real_probability": float(real_prob),
            "fake_probability": float(fake_prob),
            "model_used": model,
            "api_response": response_data,
            "success": True
        }
    
    def predict_with_fallback(self, text: str, primary_model: str = "arabert", 
                            fallback_model: str = "xgboost") -> Dict:
        """
        Predict with fallback model if primary fails
        
        Args:
            text: Arabic text to analyze
            primary_model: Primary model to use
            fallback_model: Fallback model if primary fails
            
        Returns:
            Dictionary containing prediction results
        """
        # Try primary model first
        result = self.predict(text, primary_model)
        
        # If primary model fails, try fallback
        if result.get("error") and primary_model != fallback_model:
            logger.warning(f"Primary model {primary_model} failed, trying fallback {fallback_model}")
            result = self.predict(text, fallback_model)
            if not result.get("error"):
                result["fallback_used"] = True
        
        return result
    
    def batch_predict(self, texts: list, model: str = "arabert") -> list:
        """
        Predict multiple texts
        
        Args:
            texts: List of Arabic texts to analyze
            model: Model to use
            
        Returns:
            List of prediction results
        """
        results = []
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            result = self.predict(text, model)
            results.append(result)
        
        return results
    
    def health_check(self) -> Dict:
        """
        Check API health status
        
        Returns:
            Dictionary containing health status
        """
        if not self.client:
            return {
                "status": "unhealthy",
                "message": "Gradio client not initialized",
                "space_name": self.space_name
            }
        
        try:
            # Send a simple test request
            test_text = "اختبار"
            response = self.predict(test_text, "arabert")
            
            if response.get("error"):
                return {
                    "status": "unhealthy",
                    "message": response.get("message", "API returned error"),
                    "space_name": self.space_name
                }
            else:
                return {
                    "status": "healthy",
                    "message": "API is responding correctly",
                    "space_name": self.space_name,
                    "test_prediction": response.get("prediction")
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": str(e),
                "space_name": self.space_name
            }
    
    def get_supported_models(self) -> list:
        """
        Get list of supported models
        
        Returns:
            List of supported model names
        """
        return ["arabert", "xgboost"]
    
    def close(self):
        """Close the client connection"""
        if self.client:
            try:
                # Gradio client doesn't have an explicit close method
                # Just set to None for cleanup
                self.client = None
                logger.info("Gradio client connection closed")
            except Exception as e:
                logger.error(f"Error closing client: {str(e)}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()