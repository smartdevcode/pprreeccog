#!/usr/bin/env python3
"""
Test script to verify all custom miners are properly configured.
This script checks that all miner modules can be imported and have the required forward function.
"""

import sys
import importlib
from typing import Dict, List, Tuple

def test_miner_module(module_name: str) -> Tuple[bool, str]:
    """Test if a miner module can be imported and has the required forward function."""
    try:
        # Import the module
        module = importlib.import_module(f"precog.miners.{module_name}")
        
        # Check if forward function exists
        if not hasattr(module, 'forward'):
            return False, f"Module {module_name} missing 'forward' function"
        
        # Check if forward function is callable
        if not callable(getattr(module, 'forward')):
            return False, f"Module {module_name} 'forward' is not callable"
        
        return True, f"Module {module_name} is properly configured"
        
    except ImportError as e:
        return False, f"Failed to import {module_name}: {e}"
    except Exception as e:
        return False, f"Error testing {module_name}: {e}"

def main():
    """Test all miner modules."""
    print("🧪 Testing Custom Miners Configuration")
    print("=" * 50)
    
    # List of all miner modules to test
    miner_modules = [
        'base_miner',
        'ml_miner', 
        'technical_analysis_miner',
        'ensemble_miner',
        'lstm_miner',
        'sentiment_miner',
        'advanced_ensemble_miner'
    ]
    
    results = {}
    
    for module_name in miner_modules:
        print(f"Testing {module_name}...", end=" ")
        success, message = test_miner_module(module_name)
        results[module_name] = (success, message)
        
        if success:
            print("✅ PASS")
        else:
            print("❌ FAIL")
            print(f"   {message}")
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for success, _ in results.values() if success)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All miners are properly configured!")
        print("\n🚀 You can now run any of these miners:")
        print("   make miner_lstm ENV_FILE=.env.miner")
        print("   make miner_sentiment ENV_FILE=.env.miner")
        print("   make miner_advanced_ensemble ENV_FILE=.env.miner")
        print("   make miner_ensemble ENV_FILE=.env.miner")
        print("   make miner_ml ENV_FILE=.env.miner")
        print("   make miner_technical ENV_FILE=.env.miner")
        return 0
    else:
        print("❌ Some miners have configuration issues:")
        for module_name, (success, message) in results.items():
            if not success:
                print(f"   - {module_name}: {message}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
