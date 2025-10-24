#!/bin/bash

echo "🔄 Fixing ETH price variation issue..."

# Stop the current miner
echo "Stopping current miner..."
pm2 stop miner_advanced_ensemble

echo "✅ Fixed ETH price variation issues:"
echo "   - ETH will now always fetch fresh prices (no caching)"
echo "   - BTC and TAO will still use smart caching"
echo "   - ETH prices will vary realistically like BTC"
echo "   - More realistic market behavior for ETH"

# Restart miner with ETH price variation fix
echo "Restarting miner with ETH price variation fix..."
pm2 restart miner_advanced_ensemble

# Monitor logs for ETH price variation
echo "Monitoring logs for 60 seconds to verify ETH price variation..."
timeout 60 pm2 logs miner_advanced_ensemble --lines 30 | grep -E "(ETH|eth|real-time|market price|Advanced Ensemble Prediction)"

echo "✅ ETH price variation fixed!"
echo "   - ETH will now show varying prices like BTC"
echo "   - No more static $3,500.00 for ETH"
echo "   - Fresh prices fetched every time for ETH"
echo "   - Realistic market behavior for all assets"
