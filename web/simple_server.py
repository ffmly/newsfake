"""
Simple HTTP server for the Arabic Fake News Detector web interface
"""

import os
import sys
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.api_client.haqiqa_client import HaqiqaClient

app = Flask(__name__)
CORS(app)

# Initialize the Haqiqa client
haqiqa_client = HaqiqaClient()

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'simple.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze text for fake news detection"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        model = data.get('model', 'arabert')
        
        # Use the Haqiqa client to analyze the text
        result = haqiqa_client.predict(text, model)
        
        if result.get('error'):
            return jsonify({'error': result.get('message', 'Analysis failed')}), 500
        
        # Return the prediction result
        return jsonify({
            'prediction': result.get('prediction', 'Unknown'),
            'confidence': result.get('confidence', 0.0),
            'model_used': result.get('model_used', model)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    health = haqiqa_client.health_check()
    return jsonify(health)

if __name__ == '__main__':
    print("Starting Arabic Fake News Detector web server...")
    print("Open http://localhost:8081 in your browser")
    print("Make sure to API server is running on http://localhost:5000")
    app.run(host='127.0.0.1', port=8081, debug=False)