"""
Flask Backend API for Arabic/Darija Fake News Detection
Provides RESTful endpoints for text analysis and fake news detection
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import logging
import time
import traceback
from datetime import datetime
import os

# Import our modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ml_wrapper.prediction_engine import PredictionEngine
from src.utils.config import Config

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize prediction engine
prediction_engine = PredictionEngine()

# HTML templates for documentation
HOME_TEMPLATE = """
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arabic/Darija Fake News Detection API</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f5f5f5; 
            direction: rtl; 
            text-align: right;
        }
        .container { 
            max-width: 800px; 
            margin: 0 auto; 
            background-color: white; 
            padding: 30px; 
            border-radius: 10px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
        h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .endpoint { 
            background-color: #ecf0f1; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 5px; 
            border-left: 4px solid #3498db; 
        }
        .method { 
            display: inline-block; 
            padding: 3px 8px; 
            border-radius: 3px; 
            font-weight: bold; 
            margin-left: 10px; 
        }
        .get { background-color: #2ecc71; color: white; }
        .post { background-color: #3498db; color: white; }
        .put { background-color: #f39c12; color: white; }
        .delete { background-color: #e74c3c; color: white; }
        code { 
            background-color: #f8f9fa; 
            padding: 2px 4px; 
            border-radius: 3px; 
            font-family: 'Courier New', monospace; 
        }
        .example { 
            background-color: #e8f5e8; 
            padding: 10px; 
            border-radius: 5px; 
            margin: 10px 0; 
        }
        .stats { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 20px; 
            margin: 20px 0; 
        }
        .stat-card { 
            background-color: #3498db; 
            color: white; 
            padding: 20px; 
            border-radius: 5px; 
            text-align: center; 
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Arabic/Darija Fake News Detection API</h1>
        
        <div class="stats">
            <div class="stat-card">
                <h3>التحليلات اليوم</h3>
                <p style="font-size: 2em; margin: 0;">{{ daily_analyses }}</p>
            </div>
            <div class="stat-card">
                <h3>متوسط وقت المعالجة</h3>
                <p style="font-size: 2em; margin: 0;">{{ avg_time }}ms</p>
            </div>
            <div class="stat-card">
                <h3>معدل النجاح</h3>
                <p style="font-size: 2em; margin: 0;">{{ success_rate }}%</p>
            </div>
        </div>
        
        <h2>📡 نقاط النهاية (Endpoints)</h2>
        
        <div class="endpoint">
            <h3>تحليل النص الرئيسي <span class="method post">POST</span></h3>
            <p><strong>/analyze</strong> - نقطة النهاية الرئيسية لتحليل النصوص</p>
            
            <h4>الطلب (Request):</h4>
            <div class="example">
                <code>
POST /analyze<br>
Content-Type: application/json<br>
{<br>
  "text": "النص المراد تحليله",<br>
  "include_explanation": true,<br>
  "use_fallback": true<br>
}
                </code>
            </div>
            
            <h4>الاستجابة (Response):</h4>
            <div class="example">
                <code>
{<br>
  "success": true,<br>
  "risk_analysis": {<br>
    "overall_risk_score": 0.75,<br>
    "risk_level": "high",<br>
    "haqiqa_score": 0.8,<br>
    "feature_score": 0.7,<br>
    "recommendation": "Content shows significant risk indicators..."<br>
  },<br>
  "explanation": { ... },<br>
  "processing_time": 1.23<br>
}
                </code>
            </div>
        </div>
        
        <div class="endpoint">
            <h3>التحليل المجمّع <span class="method post">POST</span></h3>
            <p><strong>/analyze/batch</strong> - تحليل多个 نصوص في وقت واحد</p>
            
            <h4>الطلب (Request):</h4>
            <div class="example">
                <code>
POST /analyze/batch<br>
Content-Type: application/json<br>
{<br>
  "texts": ["نص الأول", "نص الثاني"],<br>
  "include_explanation": false<br>
}
                </code>
            </div>
        </div>
        
        <div class="endpoint">
            <h3>تحليل وسائل التواصل الاجتماعي <span class="method post">POST</span></h3>
            <p><strong>/analyze/social-media</strong> - تحليل منشورات وسائل التواصل الاجتماعي</p>
            
            <h4>الطلب (Request):</h4>
            <div class="example">
                <code>
POST /analyze/social-media<br>
Content-Type: application/json<br>
{<br>
  "url": "https://twitter.com/user/status/123456789",<br>
  "text": "Optional text content if already extracted",<br>
  "include_explanation": true,<br>
  "use_fallback": true<br>
}
                </code>
            </div>
            
            <h4>الاستجابة (Response):</h4>
            <div class="example">
                <code>
{<br>
  "success": true,<br>
  "result": {<br>
    "social_media_processing": { ... },<br>
    "risk_analysis": {<br>
      "overall_risk_score": 0.75,<br>
      "risk_level": "high",<br>
      "haqiqa_score": 0.8,<br>
      "feature_score": 0.7,<br>
      "recommendation": "Social media content shows significant risk indicators..."<br>
    },<br>
    "explanation": { ... },<br>
    "processing_time": 1.23<br>
  }<br>
}
                </code>
            </div>
        </div>
        
        <div class="endpoint">
            <h3>التحليل المجمّع لوسائل التواصل الاجتماعي <span class="method post">POST</span></h3>
            <p><strong>/analyze/social-media/batch</strong> - تحليل عدة منشورات من وسائل التواصل الاجتماعي</p>
            
            <h4>الطلب (Request):</h4>
            <div class="example">
                <code>
POST /analyze/social-media/batch<br>
Content-Type: application/json<br>
{<br>
  "posts": [<br>
    {"url": "https://twitter.com/user/status/123456789"},<br>
    {"url": "https://www.instagram.com/p/CX123456789/"}<br>
  ],<br>
  "include_explanation": false<br>
}
                </code>
            </div>
        </div>
        
        <div class="endpoint">
            <h3>فحص النظام <span class="method get">GET</span></h3>
            <p><strong>/health</strong> - فحص حالة النظام والمكونات</p>
        </div>
        
        <div class="endpoint">
            <h3>إحصائئ النظام <span class="method get">GET</span></h3>
            <p><strong>/stats</strong> - الحصول على إحصائئ الأداء</p>
        </div>
        
        <div class="endpoint">
            <h3>معلومات الميزات <span class="method get">GET</span></h3>
            <p><strong>/features</strong> - معلومات حول الميزات المستخدمة</p>
        </div>
        
        <h2>🔧 أمثلة الاستخدام</h2>
        
        <div class="example">
            <h4>مثال باستخدام curl:</h4>
            <code>
curl -X POST http://localhost:5000/analyze \<br>
  -H "Content-Type: application/json" \<br>
  -d '{<br>
    "text": "هذا خبر عاجل جدا",<br>
    "include_explanation": true<br>
  }'
            </code>
        </div>
        
        <div class="example">
            <h4>مثال باستخدام Python:</h4>
            <code>
import requests<br>
<br>
response = requests.post(<br>
    'http://localhost:5000/analyze',<br>
    json={<br>
        'text': 'هذا خبر عاجل جدا',<br>
        'include_explanation': True<br>
    }<br>
)<br>
result = response.json()
            </code>
        </div>
        
        <h2>📊 مستويات المخاطرة</h2>
        <ul>
            <li><strong>very_low:</strong> 0.0 - 0.1 - منخفض جداً</li>
            <li><strong>low:</strong> 0.1 - 0.3 - منخفض</li>
            <li><strong>medium:</strong> 0.3 - 0.5 - متوسط</li>
            <li><strong>high:</strong> 0.5 - 0.7 - مرتفع</li>
            <li><strong>very_high:</strong> 0.7 - 1.0 - مرتفع جداً</li>
        </ul>
        
        <h2>🌐 الدعم اللغوي</h2>
        <p>يدعم النظام اللغات التالية:</p>
        <ul>
            <li>العربية الفصحى (Modern Standard Arabic)</li>
            <li>الدارجة (Moroccan Darija)</li>
            <li>الفرنسية (French)</li>
            <li>الإنجليزية (English)</li>
        </ul>
        <p>يدعم النظام أيضاً النصوص المختلطة (code-switching) بين هذه اللغات.</p>
        
        <h2>⚡ الميزات الرئيسية</h2>
        <ul>
            <li>تحليل متقدم للنصوص العربية والدارجة</li>
            <li>استخدام Haqiqa API للتنبؤ بالأخبار الكاذبة</li>
            <li>استخلاص ميزات متعددة (TF-IDF, N-grams, sentiment, etc.)</li>
            <li>تقييم شامل للمخاطر مع شرح مفصل</li>
            <li>دعم RTL وواجهة عربية كاملة</li>
            <li>شرح القرارات باستخدام تقنيات LIME-like</li>
            <li>تحليل محتوى وسائل التواصل الاجتماعي (Twitter, Instagram, Facebook)</li>
            <li>إزالة الرموز التعبيرية والروابط والمحتوى غير المرغوب فيه</li>
            <li>استخراج النص النقي من منشورات وسائل التواصل الاجتماعي</li>
        </ul>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    """Home page with API documentation"""
    try:
        stats = prediction_engine.get_performance_stats()
        return render_template_string(HOME_TEMPLATE, 
            daily_analyses=stats.get('total_predictions', 0),
            avg_time=f"{stats.get('avg_processing_time', 0)*1000:.0f}",
            success_rate="95"  # Placeholder
        )
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        return render_template_string(HOME_TEMPLATE,
            daily_analyses=0, avg_time="0", success_rate="95"
        )

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """
    Analyze a single text for fake news detection
    
    Expected JSON payload:
    {
        "text": "text to analyze",
        "include_explanation": true/false (default: true),
        "use_fallback": true/false (default: true)
    }
    """
    start_time = time.time()
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'error_code': 'INVALID_REQUEST'
            }), 400
        
        # Validate required fields
        text = data.get('text', '').strip()
        if not text:
            return jsonify({
                'success': False,
                'error': 'Text field is required and cannot be empty',
                'error_code': 'MISSING_TEXT'
            }), 400
        
        # Get optional parameters
        include_explanation = data.get('include_explanation', True)
        use_fallback = data.get('use_fallback', True)
        
        # Validate text length
        if len(text) < Config.MIN_TEXT_LENGTH:
            return jsonify({
                'success': False,
                'error': f'Text must be at least {Config.MIN_TEXT_LENGTH} characters long',
                'error_code': 'TEXT_TOO_SHORT'
            }), 400
        
        if len(text) > Config.MAX_TEXT_LENGTH:
            return jsonify({
                'success': False,
                'error': f'Text cannot exceed {Config.MAX_TEXT_LENGTH} characters',
                'error_code': 'TEXT_TOO_LONG'
            }), 400
        
        # Perform analysis
        logger.info(f"Analyzing text of length {len(text)}")
        result = prediction_engine.predict_single(
            text, use_fallback, include_explanation
        )
        
        # Add processing time
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        
        # Log the analysis
        logger.info(f"Analysis completed in {processing_time:.2f}s - Risk: {result.get('risk_analysis', {}).get('risk_level', 'unknown')}")
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        logger.error(f"Analysis failed after {processing_time:.2f}s: {error_msg}")
        logger.error(f"Error trace: {error_trace}")
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'error_code': 'ANALYSIS_FAILED',
            'processing_time': processing_time,
            'traceback': error_trace if app.debug else None
        }), 500

@app.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    """
    Analyze multiple texts for fake news detection
    
    Expected JSON payload:
    {
        "texts": ["text1", "text2", ...],
        "include_explanation": true/false (default: false)
    }
    """
    start_time = time.time()
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'error_code': 'INVALID_REQUEST'
            }), 400
        
        # Validate required fields
        texts = data.get('texts', [])
        if not texts or not isinstance(texts, list):
            return jsonify({
                'success': False,
                'error': 'texts field is required and must be an array',
                'error_code': 'INVALID_TEXTS'
            }), 400
        
        # Validate batch size
        if len(texts) > 100:  # Limit batch size
            return jsonify({
                'success': False,
                'error': 'Batch size cannot exceed 100 texts',
                'error_code': 'BATCH_TOO_LARGE'
            }), 400
        
        # Validate each text
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                return jsonify({
                    'success': False,
                    'error': f'Text at index {i} is invalid or empty',
                    'error_code': 'INVALID_TEXT_IN_BATCH'
                }), 400
            
            if len(text) < Config.MIN_TEXT_LENGTH:
                return jsonify({
                    'success': False,
                    'error': f'Text at index {i} is too short (min {Config.MIN_TEXT_LENGTH} chars)',
                    'error_code': 'TEXT_TOO_SHORT'
                }), 400
            
            if len(text) > Config.MAX_TEXT_LENGTH:
                return jsonify({
                    'success': False,
                    'error': f'Text at index {i} is too long (max {Config.MAX_TEXT_LENGTH} chars)',
                    'error_code': 'TEXT_TOO_LONG'
                }), 400
        
        # Get optional parameters
        include_explanation = data.get('include_explanation', False)
        use_fallback = data.get('use_fallback', True)
        
        # Perform batch analysis
        logger.info(f"Analyzing batch of {len(texts)} texts")
        results = prediction_engine.predict_batch(
            texts, use_fallback, include_explanation
        )
        
        # Add processing time
        processing_time = time.time() - start_time
        
        # Create response
        response = {
            'success': True,
            'batch_size': len(texts),
            'processing_time': processing_time,
            'results': results,
            'summary': {
                'total_processed': len(results),
                'successful_analyses': sum(1 for r in results if r.get('success', False)),
                'failed_analyses': sum(1 for r in results if not r.get('success', False)),
                'avg_processing_time': processing_time / len(texts)
            }
        }
        
        logger.info(f"Batch analysis completed in {processing_time:.2f}s - {len(texts)} texts processed")
        
        return jsonify(response)
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        logger.error(f"Batch analysis failed after {processing_time:.2f}s: {error_msg}")
        logger.error(f"Error trace: {error_trace}")
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'error_code': 'BATCH_ANALYSIS_FAILED',
            'processing_time': processing_time,
            'traceback': error_trace if app.debug else None
        }), 500

@app.route('/analyze/social-media', methods=['POST'])
def analyze_social_media():
    """
    Analyze social media content for fake news detection
    
    Expected JSON payload:
    {
        "url": "social media post URL",
        "text": "optional text content if already extracted",
        "include_explanation": true/false (default: true),
        "use_fallback": true/false (default: true)
    }
    """
    start_time = time.time()
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'error_code': 'INVALID_REQUEST'
            }), 400
        
        # Validate required fields
        url = data.get('url', '').strip()
        if not url:
            return jsonify({
                'success': False,
                'error': 'URL field is required and cannot be empty',
                'error_code': 'MISSING_URL'
            }), 400
        
        # Get optional parameters
        text = data.get('text', '').strip()
        include_explanation = data.get('include_explanation', True)
        use_fallback = data.get('use_fallback', True)
        
        # Validate URL format
        if not (url.startswith('http://') or url.startswith('https://')):
            return jsonify({
                'success': False,
                'error': 'URL must start with http:// or https://',
                'error_code': 'INVALID_URL_FORMAT'
            }), 400
        
        # Validate URL length
        if len(url) > 2048:  # Standard URL length limit
            return jsonify({
                'success': False,
                'error': 'URL cannot exceed 2048 characters',
                'error_code': 'URL_TOO_LONG'
            }), 400
        
        # Validate text length if provided
        if text and len(text) > Config.MAX_TEXT_LENGTH:
            return jsonify({
                'success': False,
                'error': f'Text cannot exceed {Config.MAX_TEXT_LENGTH} characters',
                'error_code': 'TEXT_TOO_LONG'
            }), 400
        
        # Prepare social media data
        social_media_data = {
            'url': url
        }
        if text:
            social_media_data['text'] = text
        
        # Perform social media analysis
        logger.info(f"Analyzing social media content from {url}")
        result = prediction_engine.predict_social_media(
            social_media_data, use_fallback, include_explanation
        )
        
        # Add processing time
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        
        # Log the analysis
        logger.info(f"Social media analysis completed in {processing_time:.2f}s - Risk: {result.get('risk_analysis', {}).get('risk_level', 'unknown')}")
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        logger.error(f"Social media analysis failed after {processing_time:.2f}s: {error_msg}")
        logger.error(f"Error trace: {error_trace}")
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'error_code': 'SOCIAL_MEDIA_ANALYSIS_FAILED',
            'processing_time': processing_time,
            'traceback': error_trace if app.debug else None
        }), 500

@app.route('/analyze/social-media/batch', methods=['POST'])
def analyze_social_media_batch():
    """
    Analyze multiple social media posts for fake news detection
    
    Expected JSON payload:
    {
        "posts": [
            {"url": "url1", "text": "optional text1"},
            {"url": "url2", "text": "optional text2"}
        ],
        "include_explanation": true/false (default: false)
    }
    """
    start_time = time.time()
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'error_code': 'INVALID_REQUEST'
            }), 400
        
        # Validate required fields
        posts = data.get('posts', [])
        if not posts or not isinstance(posts, list):
            return jsonify({
                'success': False,
                'error': 'posts field is required and must be an array',
                'error_code': 'INVALID_POSTS'
            }), 400
        
        # Validate batch size
        if len(posts) > 50:  # Limit batch size for social media
            return jsonify({
                'success': False,
                'error': 'Batch size cannot exceed 50 social media posts',
                'error_code': 'BATCH_TOO_LARGE'
            }), 400
        
        # Validate each post
        for i, post in enumerate(posts):
            if not isinstance(post, dict):
                return jsonify({
                    'success': False,
                    'error': f'Post at index {i} must be an object',
                    'error_code': 'INVALID_POST_FORMAT'
                }), 400
            
            url = post.get('url', '').strip()
            if not url:
                return jsonify({
                    'success': False,
                    'error': f'Post at index {i} must have a URL',
                    'error_code': 'MISSING_URL_IN_POST'
                }), 400
            
            if not (url.startswith('http://') or url.startswith('https://')):
                return jsonify({
                    'success': False,
                    'error': f'URL at index {i} must start with http:// or https://',
                    'error_code': 'INVALID_URL_FORMAT'
                }), 400
            
            if len(url) > 2048:
                return jsonify({
                    'success': False,
                    'error': f'URL at index {i} cannot exceed 2048 characters',
                    'error_code': 'URL_TOO_LONG'
                }), 400
        
        # Get optional parameters
        include_explanation = data.get('include_explanation', False)
        use_fallback = data.get('use_fallback', True)
        
        # Perform batch social media analysis
        logger.info(f"Analyzing batch of {len(posts)} social media posts")
        results = []
        
        for i, post in enumerate(posts):
            logger.info(f"Processing social media post {i+1}/{len(posts)}")
            result = prediction_engine.predict_social_media(
                post, use_fallback, include_explanation
            )
            result['batch_index'] = i
            results.append(result)
        
        # Add processing time
        processing_time = time.time() - start_time
        
        # Create response
        response = {
            'success': True,
            'batch_size': len(posts),
            'processing_time': processing_time,
            'results': results,
            'summary': {
                'total_processed': len(results),
                'successful_analyses': sum(1 for r in results if r.get('success', False)),
                'failed_analyses': sum(1 for r in results if not r.get('success', False)),
                'avg_processing_time': processing_time / len(posts)
            }
        }
        
        logger.info(f"Batch social media analysis completed in {processing_time:.2f}s - {len(posts)} posts processed")
        
        return jsonify(response)
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        logger.error(f"Batch social media analysis failed after {processing_time:.2f}s: {error_msg}")
        logger.error(f"Error trace: {error_trace}")
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'error_code': 'BATCH_SOCIAL_MEDIA_ANALYSIS_FAILED',
            'processing_time': processing_time,
            'traceback': error_trace if app.debug else None
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    Returns system health status and component status
    """
    try:
        # Perform health check
        health_status = prediction_engine.health_check()
        
        # Add system information
        health_status['system'] = {
            'timestamp': datetime.now().isoformat(),
            'uptime': 'N/A',  # Could be calculated
            'version': '1.0.0',
            'environment': 'development' if app.debug else 'production'
        }
        
        # Add performance stats
        stats = prediction_engine.get_performance_stats()
        health_status['performance'] = {
            'total_predictions': stats.get('total_predictions', 0),
            'avg_processing_time': stats.get('avg_processing_time', 0),
            'success_rate': 'N/A'  # Could be calculated
        }
        
        # Determine overall status
        if health_status['overall_status'] == 'healthy':
            status_code = 200
        elif health_status['overall_status'] == 'degraded':
            status_code = 200  # Still serve but warn
        else:
            status_code = 503  # Service unavailable
        
        return jsonify(health_status), status_code
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Health check failed: {error_msg}")
        
        return jsonify({
            'overall_status': 'unhealthy',
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/stats', methods=['GET'])
def get_stats():
    """
    Get performance statistics
    Returns detailed performance metrics
    """
    try:
        stats = prediction_engine.get_performance_stats()
        
        # Add additional metadata
        stats['timestamp'] = datetime.now().isoformat()
        stats['api_version'] = '1.0.0'
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Stats retrieval failed: {error_msg}")
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'error_code': 'STATS_RETRIEVAL_FAILED'
        }), 500

@app.route('/features', methods=['GET'])
def get_features():
    """
    Get information about features used in the system
    Returns feature descriptions and importance
    """
    try:
        # Feature information
        feature_info = {
            'text_features': {
                'description': 'Basic text-level features (length, punctuation, etc.)',
                'features': [
                    'text_length', 'word_count', 'sentence_count',
                    'punctuation_ratio', 'exclamation_ratio',
                    'question_ratio', 'url_ratio', 'digit_ratio'
                ]
            },
            'sentiment_features': {
                'description': 'Sentiment analysis features',
                'features': [
                    'positive_score', 'negative_score', 'sentiment_polarity',
                    'sentiment_subjectivity', 'intensity_score'
                ]
            },
            'lexicon_features': {
                'description': 'Fake news lexicon-based features',
                'features': [
                    'clickbait_score', 'uncertainty_score', 'conspiracy_score',
                    'propaganda_score', 'overall_fake_news_risk'
                ]
            },
            'language_features': {
                'description': 'Language detection and code-switching features',
                'features': [
                    'primary_language', 'is_code_switched', 'language_distribution',
                    'arabic_char_ratio', 'latin_char_ratio'
                ]
            },
            'tfidf_features': {
                'description': 'TF-IDF vector features for similarity analysis',
                'features': ['tfidf_vector', 'top_features', 'similarity_scores']
            },
            'ngram_features': {
                'description': 'N-gram features (unigrams, bigrams)',
                'features': ['ngram_counts', 'top_ngrams', 'ngram_patterns']
            }
        }
        
        # Risk scoring information
        risk_info = {
            'description': 'Risk scoring combines Haqiqa API with custom features',
            'weights': {
                'haqiqa_weight': Config.HAQIQA_WEIGHT,
                'feature_weight': Config.FEATURE_WEIGHT
            },
            'risk_levels': {
                'very_low': '0.0 - 0.1',
                'low': '0.1 - 0.3',
                'medium': '0.3 - 0.5',
                'high': '0.5 - 0.7',
                'very_high': '0.7 - 1.0'
            }
        }
        
        return jsonify({
            'success': True,
            'features': feature_info,
            'risk_scoring': risk_info,
            'supported_languages': ['arabic', 'darija', 'french', 'english'],
            'api_capabilities': [
                'single_text_analysis',
                'batch_analysis',
                'explanation_generation',
                'risk_assessment',
                'language_detection',
                'code_switching_detection'
            ]
        })
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Features retrieval failed: {error_msg}")
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'error_code': 'FEATURES_RETRIEVAL_FAILED'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'error_code': 'NOT_FOUND',
        'available_endpoints': [
            'GET /',
            'POST /analyze',
            'POST /analyze/batch',
            'POST /analyze/social-media',
            'POST /analyze/social-media/batch',
            'GET /health',
            'GET /stats',
            'GET /features'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'error_code': 'INTERNAL_ERROR'
    }), 500

@app.errorhandler(400)
def bad_request(error):
    """Handle 400 errors"""
    return jsonify({
        'success': False,
        'error': 'Bad request',
        'error_code': 'BAD_REQUEST'
    }), 400

if __name__ == '__main__':
    # Validate configuration
    config_errors = Config.validate_config()
    if config_errors:
        logger.error("Configuration errors found:")
        for error in config_errors:
            logger.error(f"  - {error}")
        logger.error("Please fix configuration errors before starting the server")
        exit(1)
    
    # Start the Flask app
    logger.info("Starting Arabic/Darija Fake News Detection API")
    logger.info(f"Debug mode: {app.debug}")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.debug
    )