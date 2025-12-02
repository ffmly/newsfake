#!/usr/bin/env python3
"""
Robust setup script for Arabic/Darija Fake News Detection System
Handles dependency installation and environment setup with error recovery
"""

import os
import sys
import subprocess
import platform
import json
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

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def upgrade_pip():
    """Upgrade pip to latest version"""
    print("\n📦 Upgrading pip...")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
    stdout, stderr = run_command(cmd, "Upgrading pip", check=False, capture_output=True)
    if stderr and "error" in stderr.lower():
        print(f"⚠️  Warning: pip upgrade failed, but continuing...")
    else:
        print("✅ pip upgraded successfully")

def install_basic_requirements():
    """Install basic requirements first"""
    print("\n📦 Installing basic requirements...")
    
    # First, install numpy and pandas in specific compatible versions
    print("   Installing numpy and pandas first (critical compatibility)...")
    critical_packages = [
        ("numpy==1.23.5", "NumPy"),
        ("pandas==1.5.3", "Pandas")
    ]
    
    for package, display_name in critical_packages:
        cmd = [sys.executable, "-m", "pip", "install", package]
        stdout, stderr = run_command(cmd, f"Installing {display_name}", check=False, capture_output=True)
        
        if stderr and "error" in stderr.lower():
            print(f"⚠️  Warning: Failed to install {display_name}")
            print("   This is a critical dependency - trying alternative approach...")
            # Try uninstalling first, then reinstalling
            run_command([sys.executable, "-m", "pip", "uninstall", package.split("==")[0], "-y"],
                       f"Uninstalling existing {display_name}", check=False)
            run_command([sys.executable, "-m", "pip", "install", package],
                       f"Reinstalling {display_name}", check=False)
        else:
            print(f"✅ {display_name} installed successfully")
    
    # Then install other basic packages
    other_packages = [
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "streamlit>=1.27.0",
        "requests>=2.30.0",
        "python-dotenv>=0.19.0"
    ]
    
    for package in other_packages:
        cmd = [sys.executable, "-m", "pip", "install", package]
        stdout, stderr = run_command(cmd, f"Installing {package}", check=False, capture_output=True)
        
        if stderr and "error" in stderr.lower():
            print(f"⚠️  Warning: Failed to install {package}")
            if "numpy.dtype size changed" in stderr:
                print("   This is a common numpy/pandas compatibility issue")
                print("   Please run: python fix_dependencies.py")
        else:
            print(f"✅ {package} installed successfully")

def install_nlp_requirements():
    """Install NLP requirements"""
    print("\n📦 Installing NLP requirements...")
    nlp_packages = [
        "scikit-learn>=1.2.0,<1.4.0",
        "nltk>=3.8.0",
        "langdetect>=1.0.9",
        "textblob>=0.17.0"
    ]
    
    for package in nlp_packages:
        cmd = [sys.executable, "-m", "pip", "install", package]
        run_command(cmd, f"Installing {package}", check=False)

def install_arabic_requirements():
    """Install Arabic text processing requirements"""
    print("\n📦 Installing Arabic text processing requirements...")
    arabic_packages = [
        "arabic-reshaper>=3.0.0",
        "python-bidi>=0.4.2"
    ]
    
    for package in arabic_packages:
        cmd = [sys.executable, "-m", "pip", "install", package]
        run_command(cmd, f"Installing {package}", check=False)

def install_visualization_requirements():
    """Install visualization requirements"""
    print("\n📦 Installing visualization requirements...")
    viz_packages = [
        "plotly>=5.15.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0"
    ]
    
    for package in viz_packages:
        cmd = [sys.executable, "-m", "pip", "install", package]
        run_command(cmd, f"Installing {package}", check=False)

def install_optional_ml():
    """Install optional ML dependencies"""
    print("\n📦 Installing optional ML dependencies...")
    print("   (These may take longer to install)")
    
    optional_packages = [
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "lime>=0.2.0.1"
    ]
    
    for package in optional_packages:
        cmd = [sys.executable, "-m", "pip", "install", package]
        stdout, stderr = run_command(cmd, f"Installing {package}", check=False, capture_output=True)
        
        if stderr and "error" in stderr.lower():
            print(f"⚠️  Warning: Failed to install {package}")
            print("   This is optional - the system will work without it")
        else:
            print(f"✅ {package} installed successfully")

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

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("\n📝 Creating .env file from template...")
        try:
            with open(env_example, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ .env file created")
        except Exception as e:
            print(f"⚠️  Warning: Failed to create .env file: {e}")

def test_imports():
    """Test if key modules can be imported"""
    print("\n🧪 Testing imports...")
    
    test_modules = [
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
    
    for module_name, display_name in test_modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError as e:
            print(f"❌ {display_name}: {e}")
            failed_imports.append(display_name)
    
    if failed_imports:
        print(f"\n⚠️  Warning: {len(failed_imports)} modules failed to import")
        print("   The system may still work with limited functionality")
    else:
        print("\n✅ All core modules imported successfully")

def main():
    """Main setup function"""
    print("🚀 Arabic/Darija Fake News Detection System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Upgrade pip
    upgrade_pip()
    
    # Install requirements in phases
    install_basic_requirements()
    install_nlp_requirements()
    install_arabic_requirements()
    install_visualization_requirements()
    
    # Install optional ML dependencies
    try:
        install_optional_ml()
    except KeyboardInterrupt:
        print("\n⚠️  Optional ML installation interrupted")
    
    # Setup NLTK data
    setup_nltk_data()
    
    # Create environment file
    create_env_file()
    
    # Test imports
    test_imports()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Review and update .env file if needed")
    print("2. Start the API server: python api/app.py")
    print("3. Start the web interface: streamlit run web/app.py")
    print("4. Or use the quick start: python quick_start.py")
    print("\nIf you encounter issues:")
    print("- Try running: python setup.py")
    print("- Check the troubleshooting section in README.md")

if __name__ == "__main__":
    main()