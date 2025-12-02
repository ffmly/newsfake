# Troubleshooting Guide
## Arabic/Darija Fake News Detection System

This guide helps resolve common installation and runtime issues.

## 🚨 Quick Emergency Fix

**If you just ran `python setup.py` and got numpy/pandas/scipy errors:**

```bash
# Run this command immediately (RECOMMENDED):
python comprehensive_fix.py

# If that doesn't work, try manual fix:
pip uninstall numpy pandas scipy scikit-learn streamlit matplotlib seaborn -y
pip install numpy==1.25.2 scipy==1.11.4 scikit-learn==1.3.2 pandas==2.0.3 streamlit==1.28.1 matplotlib==3.7.2 seaborn==0.12.2
```

This should resolve ALL dependency conflicts immediately.

## 🚨 Common Dependency Issues

### 1. NumPy/Pandas Compatibility Error
**Error**: `ValueError: numpy.dtype size changed, may indicate binary incompatibility`

**Solution**:
```bash
# Option 1: Use our automated fix script
python fix_dependencies.py

# Option 2: Manual fix
pip uninstall numpy pandas streamlit -y
pip install numpy==1.23.5 pandas==1.5.3 streamlit==1.28.1

# Option 3: Use our setup script
python setup.py

# Option 4: Quick fix commands
pip uninstall numpy pandas -y && pip install numpy==1.23.5 pandas==1.5.3
```

**Why this happens**: NumPy 2.x is incompatible with older pandas versions. The setup script may install incompatible versions due to dependency conflicts with other packages on your system.

### 2. Python-dotenv Version Error
**Error**: `Could not find a version that satisfies the requirement python-dotenv==1.0.0`

**Solution**:
```bash
# Install a compatible version
pip install python-dotenv>=0.19.0,<1.1.0

# Or let setup.py handle it
python setup.py
```

### 3. Streamlit Not Recognized
**Error**: `streamlit : The term 'streamlit' is not recognized`

**Solution**:
```bash
# Use python module syntax
python -m streamlit run web/app.py

# Or reinstall streamlit
pip install streamlit>=1.27.0
```

### 4. Arabic Text Processing Issues
**Error**: Import errors for arabic-reshaper or python-bidi

**Solution**:
```bash
# Install Arabic text processing packages
pip install arabic-reshaper>=3.0.0 python-bidi>=0.4.2

# If still failing, try without them (system will work with limited Arabic display)
```

### 5. NLTK Data Download Issues
**Error**: NLTK data download failures

**Solution**:
```bash
# Download NLTK data manually
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

## 🔧 Installation Solutions

### Option 1: Automated Setup (Recommended)
```bash
python setup.py
```
This handles most dependency issues automatically.

### Option 2: Manual Installation
```bash
# Install basic requirements
pip install flask>=2.3.0 streamlit>=1.27.0 requests>=2.30.0

# Install compatible ML packages
pip install numpy==1.23.5 pandas==1.5.3 scikit-learn>=1.2.0

# Install NLP packages
pip install nltk>=3.8.0 langdetect>=1.0.9

# Install visualization
pip install plotly>=5.15.0 matplotlib>=3.7.0

# Install utilities
pip install python-dotenv>=0.19.0 tqdm>=4.66.0
```

### Option 3: Virtual Environment (Cleanest)
```bash
# Create virtual environment
python -m venv fake_news_env

# Activate (Windows)
fake_news_env\Scripts\activate

# Activate (Linux/Mac)
source fake_news_env/bin/activate

# Install requirements
pip install -r requirements.txt

# Run setup
python setup.py
```

## 🌐 Runtime Issues

### 1. Port Already in Use
**Error**: Port 5000 or 8501 already in use

**Solution**:
```bash
# Find process using the port (Windows)
netstat -ano | findstr :5000

# Kill the process
taskkill /PID <PID> /F

# Or change ports in .env file
FLASK_PORT=5001
STREAMLIT_PORT=8502
```

### 2. API Connection Issues
**Error**: Cannot connect to Haqiqa API

**Solution**:
1. Check internet connection
2. Verify API credentials in `.env` file
3. Test API endpoint manually:
```bash
curl -X GET https://haqiqa-api.example.com/health
```

### 3. Memory Issues
**Error**: Out of memory errors with large texts

**Solution**:
1. Reduce text input size
2. Limit batch processing size
3. Close unused applications

## 🖥️ Platform-Specific Issues

### Windows
```powershell
# If PowerShell execution policy blocks scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# If PATH issues occur, use full Python path
C:\Python311\python.exe setup.py
```

### Linux/Mac
```bash
# If permission issues occur
chmod +x setup.py quick_start.py

# If Python 3 not default
python3 setup.py
```

## 🧪 Testing Your Installation

### Basic Test
```bash
# Test imports
python -c "
import flask, streamlit, requests, numpy, pandas, sklearn, nltk, langdetect, plotly, matplotlib
print('✅ All core modules imported successfully')
"
```

### API Test
```bash
# Start API server
python api/app.py

# In another terminal, test API
curl -X POST http://127.0.0.1:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "test text"}'
```

### Web Interface Test
```bash
# Start web interface
python -m streamlit run web/app.py

# Open browser to http://127.0.0.1:8501
```

## 📞 Getting Help

### 1. Check Logs
- API server logs: Check terminal where `python api/app.py` is running
- Web interface logs: Check terminal where `streamlit run web/app.py` is running
- Error logs: Check for error messages in terminals

### 2. Verify Files
```bash
# Check required files exist
ls -la api/app.py web/app.py src/ requirements.txt .env.example

# Check Python path
which python
python --version
```

### 3. Clean Reinstall
```bash
# Remove all packages and reinstall
pip freeze | xargs pip uninstall -y
pip install -r requirements.txt
python setup.py
```

## 🔄 Alternative Setup Methods

### Docker (Advanced)
```bash
# Build Docker image
docker build -t arabic-fake-news .

# Run container
docker run -p 5000:5000 -p 8501:8501 arabic-fake-news
```

### Conda (Alternative to pip)
```bash
# Create conda environment
conda create -n fake_news python=3.9

# Activate environment
conda activate fake_news

# Install packages
conda install flask requests numpy pandas scikit-learn nltk plotly matplotlib

# Install remaining packages with pip
pip install streamlit langdetect python-dotenv arabic-reshaper python-bidi
```

## 📋 Minimum Working Setup

If you're having trouble with full installation, try this minimal setup:

```bash
# Install only essential packages
pip install flask==2.3.3 requests==2.31.0 numpy==1.23.5 pandas==1.5.3

# Test basic API functionality
python api/app.py

# This will give you a working API (without web interface)
```

## 🎯 Quick Fix Commands

```bash
# Fix common numpy/pandas issues
pip uninstall numpy pandas -y && pip install numpy==1.23.5 pandas==1.5.3

# Fix streamlit issues
pip uninstall streamlit -y && pip install streamlit==1.28.1

# Fix python-dotenv issues
pip uninstall python-dotenv -y && pip install python-dotenv==0.19.0

# Complete reinstall
pip uninstall -r requirements.txt -y && pip install -r requirements.txt
```

## 📝 Still Having Issues?

1. **Check Python Version**: Must be Python 3.8+
2. **Check Disk Space**: At least 2GB free space
3. **Check Internet**: Required for downloading NLTK data and API calls
4. **Check Permissions**: Write permissions in project directory
5. **Check Antivirus**: Sometimes blocks package installation

If issues persist, try the minimal setup or contact support with:
- Python version: `python --version`
- Operating system: `uname -a` or system info
- Error messages: Full error output
- Steps taken: What you've already tried