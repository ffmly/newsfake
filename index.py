"""
Vercel serverless function entry point for the Arabic/Darija Fake News Detection API
This file acts as a wrapper around the Flask app to make it compatible with Vercel's serverless functions
"""

import sys
import os
from flask import Flask, jsonify

# Add the parent directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the Flask app
from api.app import app

# Vercel serverless function handler
def handler(request):
    """
    Vercel serverless function handler
    Converts Vercel's request format to Flask's WSGI format
    """
    try:
        # Convert Vercel request to WSGI environ
        environ = {
            'REQUEST_METHOD': request.method,
            'PATH_INFO': request.path,
            'SERVER_NAME': 'vercel.app',
            'SERVER_PORT': '443',
            'wsgi.version': (1, 0),
            'wsgi.input': request.body,
            'wsgi.errors': sys.stderr,
            'wsgi.multithread': False,
            'wsgi.multiprocess': False,
            'wsgi.run_once': False,
            'wsgi.url_scheme': 'https',
        }
        
        # Add headers
        for key, value in request.headers.items():
            environ[f'HTTP_{key.upper().replace("-", "_")}'] = value
        
        # Add query string
        if request.query_string:
            environ['QUERY_STRING'] = request.query_string.decode('utf-8')
        
        # Collect response
        response_data = {}
        status_code = 200
        headers = {}
        
        def start_response(status, response_headers):
            nonlocal status_code, headers
            status_code = int(status.split(' ')[0])
            headers = dict(response_headers)
        
        # Get response from Flask app
        app_iter = app(environ, start_response)
        response_body = b''.join(app_iter)
        
        # Parse response
        if response_body:
            try:
                response_data = response_body.decode('utf-8')
                # Try to parse as JSON
                import json
                response_data = json.loads(response_data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # If not JSON, return as string
                response_data = response_body.decode('utf-8', errors='ignore')
        
        return {
            'statusCode': status_code,
            'headers': headers,
            'body': response_data if isinstance(response_data, str) else json.dumps(response_data)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'success': False,
                'error': f'Vercel handler error: {str(e)}',
                'error_code': 'VERCEL_HANDLER_ERROR'
            })
        }

# Alternative handler for Vercel's newer format
async def lambda_handler(event, context):
    """
    AWS Lambda style handler for Vercel compatibility
    """
    class MockRequest:
        def __init__(self, event):
            self.method = event.get('httpMethod', 'GET')
            self.path = event.get('path', '/')
            self.headers = event.get('headers', {})
            self.query_string = event.get('queryStringParameters', '')
            self.body = event.get('body', '')
            
            if self.query_string:
                self.query_string = '&'.join([f"{k}={v}" for k, v in self.query_string.items()])
    
    request = MockRequest(event)
    return handler(request)

# For Vercel's Python runtime, we need to export the handler
handler_module = handler