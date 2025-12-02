#!/usr/bin/env python3
"""
Comprehensive fix for all dependency conflicts
Addresses numpy/pandas/scipy/scikit-learn compatibility
"""

import subprocess
import sys

def run_command(cmd, description, check=True):
    """Run a command with proper error handling"""
    print(f"\n🔄 {description}...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=check)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description}")
        print(f"   Return code: {e.returncode}")
        return False

def comprehensive_dependency_fix():
    """Fix all dependency conflicts comprehensively"""
    print("🔧 Comprehensive Dependency Fix")
    print("=" * 50)
    print("This will resolve numpy/pandas/scipy/scikit-learn conflicts")
    
    # Step 1: Uninstall all conflicting packages
    print("\n📦 Step 1: Uninstalling conflicting packages...")
    packages_to_uninstall = [
        "numpy", "pandas", "scipy", "scikit-learn", 
        "streamlit", "matplotlib", "seaborn"
    ]
    
    for package in packages_to_uninstall:
        run_command([sys.executable, "-m", "pip", "uninstall", package, "-y"], 
                  f"Uninstalling {package}", check=False)
    
    # Step 2: Install compatible versions in correct order
    print("\n📦 Step 2: Installing compatible versions...")
    
    # Install NumPy first (version that works with everything)
    install_order = [
        ("numpy==1.25.2", "NumPy 1.25.2 (compatible with SciPy/scikit-learn)"),
        ("scipy==1.11.4", "SciPy 1.11.4 (compatible with NumPy 1.25.2)"),
        ("scikit-learn==1.3.2", "Scikit-learn 1.3.2 (compatible with NumPy 1.25.2)"),
        ("pandas==2.0.3", "Pandas 2.0.3 (compatible with NumPy 1.25.2)"),
        ("matplotlib==3.7.2", "Matplotlib 3.7.2"),
        ("seaborn==0.12.2", "Seaborn 0.12.2"),
        ("streamlit==1.28.1", "Streamlit 1.28.1"),
    ]
    
    for package, description in install_order:
        success = run_command([sys.executable, "-m", "pip", "install", package], 
                           f"Installing {description}", check=False)
        if not success:
            print(f"⚠️  Failed to install {package}")
            return False
    
    return True

def test_critical_imports():
    """Test if all critical imports work"""
    print("\n🧪 Testing critical imports...")
    
    critical_modules = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-learn"),
        ("streamlit", "Streamlit"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
    ]
    
    failed_imports = []
    
    for module_name, display_name in critical_modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError as e:
            print(f"❌ {display_name}: {e}")
            failed_imports.append(display_name)
        except Exception as e:
            print(f"⚠️  {display_name}: {e}")
            failed_imports.append(display_name)
    
    if failed_imports:
        print(f"\n❌ {len(failed_imports)} critical modules failed to import")
        return False
    else:
        print("\n✅ All critical imports successful!")
        return True

def test_our_system():
    """Test our specific system components"""
    print("\n🧪 Testing our system components...")
    
    try:
        # Test basic imports
        sys.path.insert(0, 'src')
        
        from utils.config import Config
        print("✅ Configuration module")
        
        from preprocessing.language_detector import LanguageDetector
        print("✅ Language detector")
        
        from features.text_features import TextFeatures
        print("✅ Feature extractor")
        
        from api_client.haqiqa_client import HaqiqaClient
        print("✅ API client")
        
        print("\n✅ All system components working!")
        return True
        
    except Exception as e:
        print(f"\n❌ System component test failed: {e}")
        return False

def main():
    """Main fix function"""
    print("🚀 Comprehensive Dependency Fix for Arabic/Darija Fake News Detection")
    print("=" * 70)
    
    # Step 1: Fix dependencies
    if comprehensive_dependency_fix():
        # Step 2: Test imports
        if test_critical_imports():
            # Step 3: Test our system
            if test_our_system():
                print("\n🎉 Comprehensive fix completed successfully!")
                print("\nNext steps:")
                print("1. Run: python quick_start.py")
                print("2. Open: http://127.0.0.1:8501")
                print("3. Start analyzing Arabic/Darija text!")
                return True
            else:
                print("\n⚠️  System components have issues, but basic imports work")
                print("Try running: python test_setup.py")
        else:
            print("\n❌ Critical imports still failing")
    
    print("\n❌ Fix failed. Alternative solutions:")
    print("1. Try virtual environment:")
    print("   python -m venv fake_news_env")
    print("   fake_news_env\\Scripts\\activate  # Windows")
    print("   pip install -r requirements.txt")
    print("\n2. Try conda environment:")
    print("   conda create -n fake_news python=3.11")
    print("   conda activate fake_news")
    print("   pip install numpy==1.24.4 pandas==2.0.3 scipy==1.11.4 scikit-learn==1.3.2")
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)