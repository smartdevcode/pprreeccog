#!/usr/bin/env python3
"""
Test script to verify background price fetching functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from precog.utils.real_time_prices import start_background_price_fetching, get_current_market_prices, stop_background_price_fetching
import time

def test_background_fetching():
    """Test the background price fetching functionality."""
    print("🧪 Testing Background Price Fetching")
    print("="*50)
    
    # Start background fetching
    print("🚀 Starting background price fetching...")
    start_background_price_fetching()
    
    # Wait a bit for initial fetch
    print("⏳ Waiting 5 seconds for initial fetch...")
    time.sleep(5)
    
    # Test getting prices multiple times
    assets = ['btc', 'eth', 'tao_bittensor']
    
    for i in range(3):
        print(f"\n--- Test #{i+1} ---")
        try:
            prices = get_current_market_prices(assets)
            print(f"✅ Got prices: {prices}")
        except Exception as e:
            print(f"❌ Failed to get prices: {e}")
        
        # Wait 10 seconds between tests
        if i < 2:
            print("⏳ Waiting 10 seconds...")
            time.sleep(10)
    
    # Stop background fetching
    print("\n⏹️ Stopping background price fetching...")
    stop_background_price_fetching()
    
    print("\n🎯 Expected Results:")
    print("- Background fetching should start automatically")
    print("- Prices should be fetched every 30 seconds")
    print("- All assets (BTC, ETH, TAO) should be fetched")
    print("- No more CoinGecko 429 errors")

if __name__ == "__main__":
    test_background_fetching()
