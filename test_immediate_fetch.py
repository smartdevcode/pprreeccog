#!/usr/bin/env python3
"""
Test script to verify immediate price fetching functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from precog.utils.real_time_prices import force_fresh_price_fetch, get_current_market_prices
import time

def test_immediate_fetch():
    """Test immediate price fetching."""
    print("🧪 Testing Immediate Price Fetching")
    print("="*50)
    
    assets = ['btc', 'eth', 'tao_bittensor']
    
    print("📡 Testing immediate price fetch...")
    
    # Force immediate fetch
    print("🔄 Forcing immediate fresh price fetch...")
    force_fresh_price_fetch()
    
    # Wait a moment
    time.sleep(2)
    
    # Get prices
    print("📊 Getting current prices...")
    try:
        prices = get_current_market_prices(assets)
        print(f"✅ Current prices: {prices}")
        
        # Check if all assets have prices
        missing_assets = [asset for asset in assets if asset not in prices]
        if missing_assets:
            print(f"❌ Missing prices for: {missing_assets}")
        else:
            print("✅ All assets have prices!")
            
    except Exception as e:
        print(f"❌ Failed to get prices: {e}")
    
    print("\n🎯 Expected Results:")
    print("- Should fetch fresh prices for BTC, ETH, and TAO")
    print("- All assets should have current market prices")
    print("- No more cached price issues")

if __name__ == "__main__":
    test_immediate_fetch()
