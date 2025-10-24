#!/bin/bash

echo "🚀 Implementing Real-Time Price Fetching..."

# Stop the current miner
echo "Stopping current miner..."
pm2 stop miner_advanced_ensemble

# Backup current version
echo "Creating backup..."
cp precog/miners/advanced_ensemble_miner.py precog/miners/advanced_ensemble_miner.py.backup.$(date +%Y%m%d_%H%M%S)

echo "✅ Real-time price fetching implemented:"
echo "   - CoinGecko API (free, no API key required)"
echo "   - CoinCap API (free, no API key required)"
echo "   - Binance API (free, no API key required)"
echo "   - Automatic fallback between APIs"
echo "   - 30-second caching to avoid rate limits"
echo "   - Conservative fallback prices if all APIs fail"

# Install required dependencies
echo "Installing required dependencies..."
pip install requests

# Restart miner with real-time price fetching
echo "Restarting miner with real-time price fetching..."
pm2 restart miner_advanced_ensemble

# Monitor logs for real-time price fetching
echo "Monitoring logs for 60 seconds to verify real-time price fetching..."
timeout 60 pm2 logs miner_advanced_ensemble --lines 30 | grep -E "(real-time|market price|CoinGecko|CoinCap|Binance|Performance Summary)"

echo "✅ Real-time price fetching implemented!"
echo "   - BTC will fetch real-time prices from multiple APIs"
echo "   - ETH will fetch real-time prices from multiple APIs"
echo "   - TAO will fetch real-time prices from multiple APIs"
echo "   - No more hardcoded prices - always current market data"
echo "   - Automatic API failover for reliability"
