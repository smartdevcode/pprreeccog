#!/usr/bin/env python3
"""
Debug script to check what data sources are actually available
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from precog.utils.cm_data import CMData
from precog.utils.timestamp import get_before, to_datetime, to_str
from datetime import datetime
import pandas as pd

def debug_data_sources():
    """Debug what data is actually available from CoinMetrics API"""
    
    print("🔍 Debugging CoinMetrics Data Sources...")
    
    # Initialize CMData
    cm = CMData()
    
    # Test assets
    assets_to_test = ['btc', 'eth', 'tao_bittensor']
    
    # Current time
    current_time = datetime.now()
    start_time = get_before(current_time, hours=1, minutes=0, seconds=0)
    
    print(f"📅 Testing data from {start_time} to {current_time}")
    print("=" * 60)
    
    for asset in assets_to_test:
        print(f"\n🔍 Testing {asset.upper()}:")
        print("-" * 30)
        
        try:
            # Fetch data
            data = cm.get_CM_ReferenceRate(
                assets=[asset],
                start=to_str(start_time),
                end=to_str(current_time),
                frequency="1s"
            )
            
            if data.empty:
                print(f"❌ No data available for {asset}")
            else:
                print(f"✅ Data available for {asset}")
                print(f"   - Rows: {len(data)}")
                print(f"   - Columns: {list(data.columns)}")
                if 'ReferenceRateUSD' in data.columns:
                    latest_price = data['ReferenceRateUSD'].iloc[-1]
                    price_range = f"{data['ReferenceRateUSD'].min():.2f} - {data['ReferenceRateUSD'].max():.2f}"
                    print(f"   - Latest price: ${latest_price:,.2f}")
                    print(f"   - Price range: ${price_range}")
                else:
                    print(f"   - No ReferenceRateUSD column found")
                    print(f"   - Available columns: {list(data.columns)}")
                
                # Show sample data
                if len(data) > 0:
                    print(f"   - Sample data:")
                    print(data.head(3).to_string())
                    
        except Exception as e:
            print(f"❌ Error fetching {asset}: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Summary:")
    print("   - If BTC/ETH show 'No data available', that's why you're getting TAO prices")
    print("   - If they show data but wrong prices, there's a mapping issue")
    print("   - If all show the same price, there's a data source problem")

if __name__ == "__main__":
    debug_data_sources()
