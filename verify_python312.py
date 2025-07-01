#!/usr/bin/env python3
"""
Python 3.12 Compatibility Verification Script for BVC
Checks that the system is properly configured for Python 3.12
"""

import sys
import os
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    print("1. Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 8):
        print("   ❌ ERROR: Python 3.8+ required")
        return False
    elif version >= (3, 12):
        print("   ✅ Python 3.12+ detected - using threading mode for SocketIO")
        return True
    else:
        print("   ✅ Compatible Python version")
        return True

def check_ssl_compatibility():
    """Check SSL module compatibility"""
    print("\n2. Checking SSL compatibility...")
    try:
        import ssl
        if hasattr(ssl, 'wrap_socket'):
            print("   ⚠️  ssl.wrap_socket present (older Python version)")
            print("   Note: EventLet would work but we use threading for consistency")
        else:
            print("   ✅ ssl.wrap_socket not present (Python 3.12+ compatible)")
        return True
    except Exception as e:
        print(f"   ❌ SSL check failed: {e}")
        return False

def check_eventlet_removed():
    """Check that eventlet is not installed"""
    print("\n3. Checking eventlet status...")
    try:
        import eventlet
        print("   ⚠️  EventLet is installed - this may cause issues on Python 3.12+")
        print("   Recommendation: pip uninstall eventlet")
        return False
    except ImportError:
        print("   ✅ EventLet not installed (good for Python 3.12+)")
        return True

def check_requirements_file():
    """Check requirements.txt format"""
    print("\n4. Checking requirements.txt...")
    req_path = os.path.join(os.path.dirname(__file__), 'web', 'api', 'requirements.txt')
    
    if not os.path.exists(req_path):
        print(f"   ❌ Requirements file not found: {req_path}")
        return False
    
    with open(req_path, 'r') as f:
        content = f.read()
    
    # Check for eventlet as a requirement (not just in comments)
    lines = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
    eventlet_lines = [line for line in lines if line.lower().startswith('eventlet')]
    
    if eventlet_lines:
        print(f"   ❌ EventLet found as requirement: {eventlet_lines}")
        return False
    
    if 'Flask==' in content and 'Flask-SocketIO==' in content:
        print("   ✅ Requirements file looks good")
        return True
    else:
        print("   ❌ Required packages not found in requirements.txt")
        return False

def check_socketio_config():
    """Check SocketIO configuration in training.py"""
    print("\n5. Checking SocketIO configuration...")
    training_path = os.path.join(os.path.dirname(__file__), 'web', 'api', 'training.py')
    
    if not os.path.exists(training_path):
        print(f"   ❌ Training file not found: {training_path}")
        return False
    
    with open(training_path, 'r') as f:
        content = f.read()
    
    if "async_mode='threading'" in content:
        print("   ✅ SocketIO configured with threading mode")
        return True
    elif "async_mode" in content:
        print("   ⚠️  SocketIO has async_mode but not threading")
        return True
    else:
        print("   ❌ SocketIO not explicitly configured with async_mode")
        return False

def check_syntax():
    """Check Python syntax of main files"""
    print("\n6. Checking Python syntax...")
    files_to_check = [
        'web/api/training.py',
        'setup.sh'  # bash syntax
    ]
    
    success = True
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            continue
            
        if file_path.endswith('.py'):
            try:
                result = subprocess.run([sys.executable, '-m', 'py_compile', file_path], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"   ✅ {file_path} syntax OK")
                else:
                    print(f"   ❌ {file_path} syntax error: {result.stderr}")
                    success = False
            except Exception as e:
                print(f"   ❌ Failed to check {file_path}: {e}")
                success = False
        elif file_path.endswith('.sh'):
            try:
                result = subprocess.run(['bash', '-n', file_path], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"   ✅ {file_path} syntax OK")
                else:
                    print(f"   ❌ {file_path} syntax error: {result.stderr}")
                    success = False
            except Exception as e:
                print(f"   ❌ Failed to check {file_path}: {e}")
                success = False
    
    return success

def main():
    print("BVC Python 3.12 Compatibility Verification")
    print("=" * 50)
    
    checks = [
        check_python_version,
        check_ssl_compatibility,
        check_eventlet_removed,
        check_requirements_file,
        check_socketio_config,
        check_syntax
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Check failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} checks passed!")
        print("\nThe system should be compatible with Python 3.12.")
        print("\nNext steps:")
        print("1. Install dependencies: cd web/api && pip install -r requirements.txt")
        print("2. Start server: python3 training.py")
        return 0
    else:
        print(f"❌ {total - passed} out of {total} checks failed.")
        print("\nPlease fix the issues above before running the system.")
        return 1

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())