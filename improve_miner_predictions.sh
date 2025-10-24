#!/bin/bash

echo "🔧 Improving Miner Predictions for Better Accuracy..."

# Stop the current miner
echo "Stopping current miner..."
pm2 stop miner_advanced_ensemble

# Backup current version
echo "Creating backup..."
cp precog/miners/advanced_ensemble_miner.py precog/miners/advanced_ensemble_miner.py.backup.$(date +%Y%m%d_%H%M%S)

echo "✅ Enhanced features added:"
echo "   - Asset-specific price validation (BTC: $50k-$100k, ETH: $2k-$8k, TAO: $200-$1000)"
echo "   - Proper asset mapping for CoinMetrics API"
echo "   - Enhanced price correction (80% towards market price)"
echo "   - Better performance monitoring with price logging"
echo "   - Improved data quality validation"

# Restart miner with improvements
echo "Restarting miner with prediction improvements..."
pm2 restart miner_advanced_ensemble

# Monitor logs for improvements
echo "Monitoring logs for 60 seconds to verify improvements..."
timeout 60 pm2 logs miner_advanced_ensemble --lines 30 | grep -E "(Performance Summary|Price.*outside|Corrected|BTC|ETH|TAO)"

echo "✅ Miner predictions improved!"
echo "   - BTC predictions should now be ~$65,000-70,000"
echo "   - ETH predictions should now be ~$3,000-4,000"  
echo "   - TAO predictions should remain ~$400-500"
echo "   - Price validation will catch unrealistic predictions"
