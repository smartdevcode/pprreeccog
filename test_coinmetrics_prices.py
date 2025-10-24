#!/usr/bin/env python3
"""
Test script to verify CoinMetrics price fetching functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from precog.utils.real_time_prices import force_fresh_price_fetch, get_current_market_prices
import time

def test_coinmetrics_prices():
    """Test CoinMetrics price fetching."""
    print("🧪 Testing CoinMetrics Price Fetching")
    print("="*50)
    
    assets = ['btc', 'eth', 'tao_bittensor']
    
    print("📡 Testing CoinMetrics price fetching...")
    
    # Test individual asset fetching
    for asset in assets:
        print(f"\n🔄 Testing {asset.upper()} from CoinMetrics...")
        try:
            prices = get_current_market_prices([asset])
            if asset in prices:
                print(f"✅ {asset.upper()}: ${prices[asset]:.2f}")
            else:
                print(f"❌ {asset.upper()}: No price data")
        except Exception as e:
            print(f"❌ {asset.upper()}: Failed - {e}")
        
        time.sleep(1)  # Small delay between tests
    
    print("\n🔄 Testing all assets together...")
    try:
        prices = get_current_market_prices(assets)
        print(f"✅ All assets: {prices}")
        
        # Check if all assets have prices
        missing_assets = [asset for asset in assets if asset not in prices]
        if missing_assets:
            print(f"❌ Missing prices for: {missing_assets}")
        else:
            print("✅ All assets have prices from CoinMetrics!")
            
    except Exception as e:
        print(f"❌ Failed to get all prices: {e}")
    
    print("\n🎯 Expected Results:")
    print("- Should fetch prices from CoinMetrics (institutional-grade data)")
    print("- More reliable than Binance API")
    print("- Better data quality and consistency")
    print("- No rate limiting issues")

if __name__ == "__main__":
    test_coinmetrics_prices()
