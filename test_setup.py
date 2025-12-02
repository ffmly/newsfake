#!/usr/bin/env python3
"""
Simple test script to verify basic installation and functionality
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if core modules can be imported"""
    print("🧪 Testing core module imports...")
    
    required_modules = [
        ("flask", "Flask"),
        ("streamlit", "Streamlit"),
        ("requests", "Requests"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("nltk", "NLTK"),
        ("langdetect", "Language Detection"),
        ("plotly", "Plotly"),
        ("matplotlib", "Matplotlib")
    ]
    
    failed_imports = []
    
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError as e:
            print(f"❌ {display_name}: {e}")
            failed_imports.append(display_name)
    
    if failed_imports:
        print(f"\n⚠️  {len(failed_imports)} modules failed to import")
        print("   Try running: python setup.py")
        return False
    else:
        print("\n✅ All core modules imported successfully")
        return True

def test_project_structure():
    """Test if project structure is correct"""
    print("\n🏗️  Testing project structure...")
    
    required_files = [
        ("src/utils/config.py", "Configuration module"),
        ("src/api_client/haqiqa_client.py", "API client"),
        ("src/preprocessing/language_detector.py", "Language detector"),
        ("src/features/text_features.py", "Feature extraction"),
        ("api/app.py", "Flask API"),
        ("web/app.py", "Streamlit web interface"),
        ("requirements.txt", "Dependencies"),
        (".env.example", "Environment template")
    ]
    
    missing_files = []
    
    for file_path, description in required_files:
        if Path(file_path).exists():
            print(f"✅ {description}")
        else:
            print(f"❌ {description}: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  {len(missing_files)} files are missing")
        return False
    else:
        print("\n✅ Project structure is correct")
        return True

def test_basic_functionality():
    """Test basic functionality of core components"""
    print("\n🔬 Testing basic functionality...")
    
    try:
        # Test configuration
        sys.path.insert(0, 'src')
        from utils.config import Config
        config = Config()
        print("✅ Configuration loading")
        
        # Test language detector
        from preprocessing.language_detector import LanguageDetector
        detector = LanguageDetector()
        result = detector.detect_language("هذا نص عربي")
        if result and 'primary_language' in result:
            print("✅ Language detection")
        else:
            print("⚠️  Language detection issue")
        
        # Test text cleaner
        from preprocessing.text_cleaner import TextCleaner
        cleaner = TextCleaner()
        cleaned = cleaner.clean_text("هذا نص   بمسافات  زائدة")
        if cleaned and len(cleaned) > 0:
            print("✅ Text cleaning")
        else:
            print("⚠️  Text cleaning issue")
        
        # Test feature extraction
        from features.text_features import TextFeatures
        extractor = TextFeatures()
        features = extractor.extract_features("هذا نص اختبار")
        if features and len(features) > 0:
            print("✅ Feature extraction")
        else:
            print("⚠️  Feature extraction issue")
        
        # Test language utils
        from utils.language_utils import is_arabic_text, detect_script_type
        if is_arabic_text("هذا نص عربي"):
            print("✅ Language utilities")
        else:
            print("⚠️  Language utilities issue")
        
        print("\n✅ Basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic functionality test failed: {e}")
        return False

def test_api_client():
    """Test API client functionality"""
    print("\n🌐 Testing API client...")
    
    try:
        sys.path.insert(0, 'src')
        from api_client.haqiqa_client import HaqiqaClient
        client = HaqiqaClient()
        
        # Test health check (will fail without real API, but should not crash)
        try:
            health = client.health_check()
            print("✅ API client health check")
        except Exception:
            print("⚠️  API client health check failed (expected without real API)")
        
        print("✅ API client initialization")
        return True
        
    except Exception as e:
        print(f"❌ API client test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Arabic/Darija Fake News Detection System - Setup Test")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Basic Functionality", test_basic_functionality),
        ("API Client", test_api_client)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Tests...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Run: python quick_start.py")
        print("2. Open: http://127.0.0.1:8501")
        print("3. Start analyzing Arabic/Darija text!")
    else:
        print(f"\n⚠️  {total - passed} test suites failed.")
        print("\nSuggested fixes:")
        print("1. Run: python setup.py")
        print("2. Check: TROUBLESHOOTING.md")
        print("3. Try: pip install -r requirements.txt")
        print("4. Verify: Python 3.8+ is installed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)