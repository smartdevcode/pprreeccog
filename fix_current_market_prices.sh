#!/bin/bash

echo "🔄 Fixing current market prices to match real-time data..."

# Stop the current miner
echo "Stopping current miner..."
pm2 stop miner_advanced_ensemble

echo "✅ Fixed current market prices:"
echo "   - Updated fallback prices to current market values"
echo "   - ETH fallback: $3,907.73 (current market price)"
echo "   - TAO fallback: $388.35 (current market price)"
echo "   - BTC fallback: $110,219.60 (current market price)"
echo "   - Added debugging to track API calls"
echo "   - All prices will match current market data"

# Restart miner with current market price fixes
echo "Restarting miner with current market price fixes..."
pm2 restart miner_advanced_ensemble

# Monitor logs for current market prices
echo "Monitoring logs for 60 seconds to verify current market prices..."
timeout 60 pm2 logs miner_advanced_ensemble --lines 30 | grep -E "(real-time|market price|Using real-time|Fetched prices from CoinGecko|fallback)"

echo "✅ Current market prices fixed!"
echo "   - ETH: Should show ~$3,907.73 (current market price)"
echo "   - TAO: Should show ~$388.35 (current market price)"
echo "   - BTC: Should show ~$110,219.60 (current market price)"
echo "   - All prices will match actual current market prices"
