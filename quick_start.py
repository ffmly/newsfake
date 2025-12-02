#!/usr/bin/env python3
"""
Quick Start Script for Arabic/Darija Fake News Detection System
Automated setup and launch for hackathon demonstration
"""

import os
import sys
import subprocess
import time
import threading
import webbrowser
from pathlib import Path

def run_command(cmd, description, check=True, capture_output=False):
    """Run a command with proper error handling"""
    print(f"\n🔄 {description}...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        if capture_output:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
            return result.stdout.strip(), result.stderr.strip()
        else:
            subprocess.run(cmd, check=check)
            return None, None
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description}")
        print(f"   Return code: {e.returncode}")
        if capture_output and e.stdout:
            print(f"   STDOUT: {e.stdout.strip()}")
        if capture_output and e.stderr:
            print(f"   STDERR: {e.stderr.strip()}")
        return None, str(e) if capture_output else None

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
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
    
    missing_modules = []
    
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name}")
            missing_modules.append(display_name)
    
    if missing_modules:
        print(f"\n⚠️  {len(missing_modules)} required modules are missing")
        return False
    else:
        print("\n✅ All dependencies are available")
        return True

def install_dependencies():
    """Install dependencies using the setup script"""
    print("\n📦 Installing dependencies...")
    
    # First try the setup script
    setup_script = Path("setup.py")
    if setup_script.exists():
        cmd = [sys.executable, "setup.py"]
        stdout, stderr = run_command(cmd, "Running setup script", check=False, capture_output=True)
        
        if stderr and "error" in stderr.lower():
            print("⚠️  Setup script had issues, trying manual installation...")
            return manual_install()
        else:
            print("✅ Setup completed successfully")
            return True
    else:
        print("⚠️  setup.py not found, trying manual installation...")
        return manual_install()

def manual_install():
    """Manual dependency installation as fallback"""
    print("\n🔧 Manual dependency installation...")
    
    # Try installing from requirements.txt
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        stdout, stderr = run_command(cmd, "Installing from requirements.txt", check=False, capture_output=True)
        
        if stderr and "error" in stderr.lower():
            print("⚠️  Requirements installation had issues, trying basic packages...")
            return install_basic_packages()
        else:
            print("✅ Requirements installed successfully")
            return True
    else:
        return install_basic_packages()

def install_basic_packages():
    """Install basic packages as last resort"""
    print("\n📦 Installing basic packages...")
    
    basic_packages = [
        "flask>=2.3.0",
        "streamlit>=1.27.0", 
        "requests>=2.30.0",
        "numpy>=1.21.0,<1.25.0",
        "pandas>=1.5.0,<2.1.0",
        "scikit-learn>=1.2.0,<1.4.0",
        "nltk>=3.8.0",
        "langdetect>=1.0.9",
        "plotly>=5.15.0",
        "matplotlib>=3.7.0",
        "python-dotenv>=0.19.0"
    ]
    
    success_count = 0
    for package in basic_packages:
        cmd = [sys.executable, "-m", "pip", "install", package]
        stdout, stderr = run_command(cmd, f"Installing {package}", check=False, capture_output=True)
        
        if stderr and "error" in stderr.lower():
            print(f"⚠️  Failed to install {package}")
        else:
            print(f"✅ {package} installed")
            success_count += 1
    
    print(f"\n📊 Installation summary: {success_count}/{len(basic_packages)} packages installed")
    return success_count >= len(basic_packages) * 0.8  # At least 80% success rate

def setup_environment():
    """Set up environment configuration"""
    print("\n⚙️  Setting up environment...")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            try:
                with open(env_example, 'r', encoding='utf-8') as f:
                    content = f.read()
                with open(env_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print("✅ .env file created from template")
            except Exception as e:
                print(f"⚠️  Warning: Failed to create .env file: {e}")
        else:
            # Create a basic .env file
            basic_env = """# Haqiqa API Configuration
HAQIQA_API_URL=https://haqiqa-api.example.com
HAQIQA_API_KEY=your_api_key_here

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_HOST=127.0.0.1
FLASK_PORT=5000

# Streamlit Configuration
STREAMLIT_HOST=127.0.0.1
STREAMLIT_PORT=8501

# Feature Extraction
MAX_FEATURES=1000
N_GRAM_RANGE=1,2
"""
            try:
                with open(env_file, 'w', encoding='utf-8') as f:
                    f.write(basic_env)
                print("✅ Basic .env file created")
            except Exception as e:
                print(f"⚠️  Warning: Failed to create .env file: {e}")

def setup_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Setting up NLTK data...")
    try:
        import nltk
        nltk_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        
        for data in nltk_data:
            try:
                nltk.download(data, quiet=True)
                print(f"✅ NLTK {data} downloaded")
            except Exception as e:
                print(f"⚠️  Warning: Failed to download NLTK {data}: {e}")
    except ImportError:
        print("⚠️  NLTK not available, skipping data download")

def test_api_connectivity():
    """Test connectivity to Haqiqa API"""
    print("\n🌐 Testing API connectivity...")
    
    # This is a mock test since we don't have real API credentials
    try:
        import requests
        
        # Test with a simple HTTP request
        response = requests.get("https://httpbin.org/get", timeout=5)
        if response.status_code == 200:
            print("✅ Internet connectivity confirmed")
            print("⚠️  Haqiqa API connectivity will be tested when you provide API credentials")
            return True
        else:
            print("⚠️  Internet connectivity issues detected")
            return False
    except Exception as e:
        print(f"⚠️  Connectivity test failed: {e}")
        return False

def start_api_server():
    """Start the Flask API server"""
    print("\n🚀 Starting Flask API server...")
    
    api_file = Path("api/app.py")
    if not api_file.exists():
        print("❌ API server file not found: api/app.py")
        return None
    
    try:
        # Start Flask server in a subprocess
        cmd = [sys.executable, "api/app.py"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Give it a moment to start
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ Flask API server started successfully")
            print("   API URL: http://127.0.0.1:5000")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Flask API server failed to start")
            if stdout:
                print(f"   STDOUT: {stdout}")
            if stderr:
                print(f"   STDERR: {stderr}")
            return None
    except Exception as e:
        print(f"❌ Failed to start Flask API server: {e}")
        return None

def start_web_interface():
    """Start the Streamlit web interface"""
    print("\n🚀 Starting Streamlit web interface...")
    
    web_file = Path("web/app.py")
    if not web_file.exists():
        print("❌ Web interface file not found: web/app.py")
        return None
    
    try:
        # Start Streamlit in a subprocess
        cmd = [sys.executable, "-m", "streamlit", "run", "web/app.py", "--server.headless=true"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Give it a moment to start
        time.sleep(5)
        
        if process.poll() is None:
            print("✅ Streamlit web interface started successfully")
            print("   Web URL: http://127.0.0.1:8501")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Streamlit web interface failed to start")
            if stdout:
                print(f"   STDOUT: {stdout}")
            if stderr:
                print(f"   STDERR: {stderr}")
            return None
    except Exception as e:
        print(f"❌ Failed to start Streamlit web interface: {e}")
        return None

def open_browser():
    """Open browser with the web interface"""
    print("\n🌐 Opening web browser...")
    try:
        webbrowser.open("http://127.0.0.1:8501")
        print("✅ Browser opened with web interface")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("   Please manually open: http://127.0.0.1:8501")

def display_usage_instructions():
    """Display usage instructions"""
    print("\n" + "=" * 60)
    print("🎉 Arabic/Darija Fake News Detection System is Ready!")
    print("=" * 60)
    
    print("\n📋 Usage Instructions:")
    print("1. Web Interface: http://127.0.0.1:8501")
    print("   - Enter Arabic/Darija text to analyze")
    print("   - View real-time risk assessment")
    print("   - Explore feature explanations")
    
    print("\n2. API Endpoint: http://127.0.0.1:5000")
    print("   - POST /analyze - Single text analysis")
    print("   - POST /analyze/batch - Batch analysis")
    print("   - GET /health - System health check")
    
    print("\n3. Sample API Usage:")
    print("   curl -X POST http://127.0.0.1:5000/analyze \\")
    print("     -H \"Content-Type: application/json\" \\")
    print("     -d '{\"text\": \"هذا خبر اختبار\"}'")
    
    print("\n🔧 Configuration:")
    print("- Edit .env file to configure API settings")
    print("- Add Haqiqa API credentials for full functionality")
    
    print("\n📚 Documentation:")
    print("- README.md - Complete documentation")
    print("- DEPLOYMENT.md - Deployment instructions")
    
    print("\n🛑 To stop the services:")
    print("- Press Ctrl+C in this terminal")
    print("- Or close the terminal windows")

def main():
    """Main quick start function"""
    print("🚀 Arabic/Darija Fake News Detection System - Quick Start")
    print("=" * 60)
    
    # Check if dependencies are installed
    if not check_dependencies():
        print("\n📦 Dependencies missing, installing...")
        if not install_dependencies():
            print("\n❌ Failed to install dependencies")
            print("   Please try running: python setup.py")
            print("   Or install manually: pip install -r requirements.txt")
            sys.exit(1)
    
    # Set up environment
    setup_environment()
    
    # Set up NLTK data
    setup_nltk_data()
    
    # Test connectivity
    test_api_connectivity()
    
    # Start services
    api_process = start_api_server()
    web_process = start_web_interface()
    
    if api_process and web_process:
        # Open browser
        open_browser()
        
        # Display instructions
        display_usage_instructions()
        
        try:
            # Keep the script running
            print("\n⏳ Services are running. Press Ctrl+C to stop...")
            while True:
                time.sleep(1)
                
                # Check if processes are still running
                if api_process.poll() is not None:
                    print("⚠️  API server stopped")
                    api_process = None
                
                if web_process.poll() is not None:
                    print("⚠️  Web interface stopped")
                    web_process = None
                
                if not api_process and not web_process:
                    print("\n🛑 All services have stopped")
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping services...")
            
        finally:
            # Clean up processes
            if api_process:
                api_process.terminate()
                print("✅ API server stopped")
            if web_process:
                web_process.terminate()
                print("✅ Web interface stopped")
    else:
        print("\n❌ Failed to start services")
        print("   Please check the error messages above")
        print("   You can try starting services manually:")
        print("   - API: python api/app.py")
        print("   - Web: streamlit run web/app.py")

if __name__ == "__main__":
    main()