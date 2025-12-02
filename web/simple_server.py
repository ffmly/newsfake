"""
Simple HTTP server for Arabic Fake News Detector web interface
"""

import os
import sys
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.api_client.haqiqa_client import HaqiqaClient
from src.social_media.scraper import SocialMediaScraper

app = Flask(__name__)
CORS(app)

# Initialize Haqiqa client and social media scraper with hardcoded configuration
haqiqa_client = HaqiqaClient()
social_scraper = SocialMediaScraper()

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'simple.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze text for fake news detection"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Handle both direct text and social media URL analysis
        if 'text' in data:
            text = data['text']
            model = data.get('model', 'arabert')
            
            # Use the Haqiqa client to analyze text
            result = haqiqa_client.predict(text, model)
            
            if result.get('error'):
                return jsonify({'error': result.get('message', 'Analysis failed')}), 500
            
            # Return prediction result
            return jsonify({
                'prediction': result.get('prediction', 'Unknown'),
                'confidence': result.get('confidence', 0.0),
                'model_used': result.get('model_used', model),
                'input_type': 'text'
            })
        
        elif 'url' in data:
            url = data['url']
            model = data.get('model', 'arabert')
            
            # Extract content from social media URL
            social_result = social_scraper.extract_content_from_url(url)
            
            if not social_result.get('success'):
                return jsonify({'error': social_result.get('error', 'Failed to extract content from URL')}), 400
            
            # Analyze extracted content
            text = social_result.get('content', '')
            result = haqiqa_client.predict(text, model)
            
            if result.get('error'):
                return jsonify({'error': result.get('message', 'Analysis failed')}), 500
            
            # Return prediction result with social media info
            return jsonify({
                'prediction': result.get('prediction', 'Unknown'),
                'confidence': result.get('confidence', 0.0),
                'model_used': result.get('model_used', model),
                'input_type': 'social_media',
                'social_media': {
                    'platform': social_result.get('platform'),
                    'url': url,
                    'extracted_content': text
                }
            })
        
        else:
            return jsonify({'error': 'No text or URL provided'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/validate-url', methods=['POST'])
def validate_url():
    """Validate social media URL"""
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({'error': 'No URL provided'}), 400
        
        url = data['url']
        validation = social_scraper.validate_url(url)
        
        return jsonify(validation)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    health = haqiqa_client.health_check()
    return jsonify(health)

if __name__ == '__main__':
    print("Starting HaqiqaByUnibyte web server...")
    print("Open http://localhost:8081 in your browser")
    print("Make sure to API server is running on http://localhost:5000")
    app.run(host='127.0.0.1', port=8081, debug=False)