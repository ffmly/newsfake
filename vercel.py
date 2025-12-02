"""
Vercel serverless function handler for Arabic/Darija Fake News Detection API
This follows Vercel's recommended pattern for Python serverless functions
"""

import sys
import os
import json
from flask import Flask, request, jsonify

# Add the parent directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the Flask app
from api.app import app

# Vercel handler function
def handler(event, context):
    """
    Vercel serverless function handler
    """
    try:
        # Parse the event
        method = event.get('httpMethod', 'GET')
        path = event.get('path', '/')
        headers = event.get('headers', {})
        query_params = event.get('queryStringParameters', {}) or {}
        body = event.get('body', '')
        
        # Create Flask request context
        with app.test_request_context(
            path=path,
            method=method,
            headers=headers,
            query_string=query_params,
            data=body if body else None,
            content_type=headers.get('content-type', 'application/json')
        ):
            # Process the request
            response = app.full_dispatch_request()
            
            # Get response data
            response_data = response.get_data(as_text=True)
            
            # Try to parse as JSON for validation
            try:
                json.loads(response_data)
                content_type = 'application/json'
            except json.JSONDecodeError:
                content_type = 'text/html'
            
            return {
                'statusCode': response.status_code,
                'headers': {
                    'Content-Type': content_type,
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization'
                },
                'body': response_data
            }
            
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': f'Vercel handler error: {str(e)}',
                'error_code': 'VERCEL_HANDLER_ERROR'
            })
        }

# Export for Vercel
lambda_handler = handler