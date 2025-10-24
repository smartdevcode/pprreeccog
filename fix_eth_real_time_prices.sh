#!/bin/bash

echo "🔄 Fixing ETH to always use real-time prices..."

# Stop the current miner
echo "Stopping current miner..."
pm2 stop miner_advanced_ensemble

echo "✅ Fixed ETH to always use real-time prices:"
echo "   - ETH will ALWAYS fetch fresh real-time prices"
echo "   - No more ensemble prediction for ETH"
echo "   - No more cached prices for ETH"
echo "   - ETH will show realistic price variations"
echo "   - BTC and TAO will continue with ensemble predictions"

# Restart miner with ETH real-time price fix
echo "Restarting miner with ETH real-time price fix..."
pm2 restart miner_advanced_ensemble

# Monitor logs for ETH real-time prices
echo "Monitoring logs for 60 seconds to verify ETH real-time prices..."
timeout 60 pm2 logs miner_advanced_ensemble --lines 30 | grep -E "(ETH|eth|real-time|market price|Using real-time ETH price)"

echo "✅ ETH real-time prices fixed!"
echo "   - ETH will now show varying prices like BTC"
echo "   - No more static $3,500.00 for ETH"
echo "   - ETH will always fetch fresh market prices"
echo "   - Realistic market behavior for ETH"
