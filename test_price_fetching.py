#!/usr/bin/env python3
"""
Test script to verify the improved price fetching functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from precog.utils.real_time_prices import get_current_market_prices
import time

def test_price_fetching():
    """Test the improved price fetching functionality."""
    print("🧪 Testing Improved Price Fetching")
    print("="*50)
    
    assets = ['btc', 'eth', 'tao_bittensor']
    
    print(f"Testing price fetching for: {assets}")
    print("\n📡 Fetching prices...")
    
    # Test multiple fetches to see caching in action
    for i in range(3):
        print(f"\n--- Fetch #{i+1} ---")
        start_time = time.time()
        
        try:
            prices = get_current_market_prices(assets)
            fetch_time = time.time() - start_time
            
            print(f"✅ Fetch completed in {fetch_time:.3f} seconds")
            for asset, price in prices.items():
                print(f"   {asset.upper()}: ${price:.2f}")
                
        except Exception as e:
            print(f"❌ Fetch failed: {e}")
        
        # Wait 2 seconds between fetches
        if i < 2:
            print("⏳ Waiting 2 seconds before next fetch...")
            time.sleep(2)
    
    print("\n🎯 Expected Results:")
    print("- First fetch: Should use Binance API")
    print("- Subsequent fetches: Should use cached prices (faster)")
    print("- No more CoinGecko 429 errors")
    print("- More stable and reliable price fetching")

if __name__ == "__main__":
    test_price_fetching()
