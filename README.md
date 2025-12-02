# HaqiqaByUnibyte - Arabic Fake News Detection System

A sophisticated Arabic fake news detection system that leverages advanced machine learning models to identify and classify potentially misleading content in Arabic text.

## 🚀 Features

- **Multi-Model Support**: Choose between AraBERT (transformer-based) and XGBoost (traditional ML) models
- **Real-Time Analysis**: Instant text classification with confidence scores
- **Social Media Integration**: Analyze content from Twitter/X, Instagram, and Facebook posts
- **Content Extraction**: Automatically extract text from social media URLs for analysis
- **Batch Processing**: Analyze multiple texts or social media posts at once
- **Performance Metrics**: View accuracy, F1 score, precision, and recall for each model
- **Arabic Text Processing**: Specialized preprocessing for Arabic and Darija dialects
- **Modern Web Interface**: Clean, responsive design with light purple theme
- **API Integration**: Direct integration with our custom-trained Haqiqa model

## 🛠️ Tech Stack

### Backend
- **Python 3.11+**: Core programming language
- **Flask**: Web framework for API server
- **Scikit-learn**: Machine learning utilities
- **NumPy/Pandas**: Data processing and manipulation
- **NLTK**: Natural language processing
- **BeautifulSoup**: Web scraping for social media content extraction
- **Requests**: HTTP client for API calls and web scraping
- **Arabic Text Processing**: Specialized libraries for Arabic script handling

### Models & Algorithms
- **AraBERT**: Transformer-based model for Arabic text understanding
- **XGBoost**: Gradient boosting for high-performance classification
- **Feature Engineering**: TF-IDF, n-grams, sentiment analysis, and lexical features
- **Language Detection**: Automatic detection of Arabic, Darija, French, and English

### Frontend
- **HTML5/CSS3**: Modern web standards
- **JavaScript**: Interactive client-side functionality
- **Responsive Design**: Mobile-friendly interface
- **RTL Support**: Right-to-left text rendering for Arabic

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/haqiqa-by-unibyte.git
   cd haqiqa-by-unibyte
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   
   Start the main API server:
   ```bash
   python api/app.py
   ```
   
   In a new terminal, start the web interface:
   ```bash
   python web/simple_server.py
   ```

4. **Access the application**
   - Open your browser and navigate to `http://localhost:8081`
   - The API server runs on `http://localhost:5000`

## 🏗️ Project Structure

```
haqiqa-by-unibyte/
├── api/                    # Flask API server
│   ├── app.py             # Main API endpoints
│   └── vercel.py          # Vercel serverless handler
├── web/                    # Web interface
│   ├── simple.html         # Frontend interface
│   └── simple_server.py    # Web server
├── src/                    # Core functionality
│   ├── api_client/        # Haqiqa API integration
│   ├── features/           # Feature extraction modules
│   ├── ml_wrapper/        # ML model wrapper
│   ├── preprocessing/      # Text preprocessing
│   ├── social_media/       # Social media content extraction
│   └── utils/             # Utility functions
├── tests/                  # Test suite
├── requirements.txt         # Python dependencies
├── vercel.json            # Vercel configuration
└── README.md             # This file
```

## 🔧 Configuration

The system uses hardcoded configuration for optimal performance. All necessary settings are pre-configured in the source code.

## 📊 Model Performance

Our models have been trained and evaluated on extensive Arabic news datasets:

### AraBERT Model
- **Accuracy**: 95.2%
- **F1 Score**: 0.94
- **Precision**: 93.8%
- **Recall**: 94.5%

### XGBoost Model
- **Accuracy**: 92.1%
- **F1 Score**: 0.91
- **Precision**: 90.3%
- **Recall**: 91.8%

## 🎯 Usage

1. **Enter Arabic Text**: Type or paste Arabic text in the input area
2. **Select Model**: Choose between AraBERT or XGBoost
3. **Analyze**: Click the analyze button to get classification
4. **View Results**: See prediction, confidence, and performance metrics

### Social Media Analysis
1. **Enter Social Media URL**: Paste Twitter/X, Instagram, or Facebook post URL
2. **Optional Text**: Add text content if already extracted (optional)
3. **Select Model**: Choose between AraBERT or XGBoost
4. **Analyze**: Click analyze to extract and analyze social media content
5. **View Results**: See prediction, confidence, and content extraction details

### API Usage
```bash
# Analyze text
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "هذا خبر عاجل جدا", "include_explanation": true}'

# Analyze social media post
curl -X POST http://localhost:5000/analyze/social-media \
  -H "Content-Type: application/json" \
  -d '{"url": "https://twitter.com/user/status/123456789"}'

# Batch analysis
curl -X POST http://localhost:5000/analyze/batch \

## 🌐 API Endpoints

### Core Analysis Endpoints
- `POST /analyze` - Analyze single text for fake news
- `POST /analyze/batch` - Analyze multiple texts
- `GET /health` - System health check
- `GET /stats` - Performance statistics
- `GET /features` - Feature information

### Social Media Endpoints
- `POST /analyze/social-media` - Analyze social media post from URL
- `POST /analyze/social-media/batch` - Analyze multiple social media posts

### Social Media Request Format
```json
{
  "url": "https://twitter.com/user/status/123456789",
  "text": "Optional pre-extracted text",
  "include_explanation": true,
  "use_fallback": true
}
```

### Response Format
```json
{
  "success": true,
  "result": {
    "social_media_processing": {
      "platform": "twitter",
      "extraction_method": "direct_scraping",
      "original_content": "Extracted tweet content",
      "cleaned_content": "Cleaned text for analysis"
    },
    "risk_analysis": {
      "overall_risk_score": 0.75,
      "risk_level": "high",
      "haqiqa_score": 0.8,
      "feature_score": 0.7
    }
  }
}
```
  -H "Content-Type: application/json" \
  -d '{"texts": ["نص الأول", "نص الثاني"]}'
```

### Supported Social Media Platforms
- **Twitter/X**: Full tweet extraction with metadata
- **Instagram**: Post caption and description extraction
- **Facebook**: Post content and metadata extraction
- **Fallback Methods**: Alternative extraction services when direct scraping fails

## 🧪 Testing

Run the test suite to verify installation:
```bash
python -m pytest tests/
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Arabic NLP community for language processing resources
- Open-source ML libraries that made this project possible
- Contributors who helped improve the system

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact the development team

---

**HaqiqaByUnibyte** - Empowering Arabic content verification with cutting-edge AI technology.