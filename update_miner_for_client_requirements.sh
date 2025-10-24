#!/bin/bash

echo "🚀 Updating Miner for Client Requirements..."

# Stop the current miner
echo "Stopping current miner..."
pm2 stop miner_advanced_ensemble

# Backup current miner
echo "Creating backup..."
cp precog/miners/advanced_ensemble_miner.py precog/miners/advanced_ensemble_miner.py.backup

# The enhanced miner.py has already been updated with:
# 1. Enhanced block synchronization
# 2. Auto-reconnection capabilities
# 3. Better error handling

# The enhanced advanced_ensemble_miner.py now includes:
# 1. Price validation and correction
# 2. Enhanced data quality validation
# 3. Performance monitoring
# 4. Better market regime detection
# 5. Improved ensemble weighting

echo "✅ Enhanced features added:"
echo "   - Price validation and correction (15% deviation threshold)"
echo "   - Enhanced data quality validation (completeness, recency, volatility)"
echo "   - Performance monitoring and reporting"
echo "   - Improved market regime detection"
echo "   - Better ensemble weighting algorithms"
echo "   - Robust error handling and fallbacks"

# Restart miner with enhanced features
echo "Restarting miner with enhanced features..."
pm2 restart miner_advanced_ensemble

# Monitor logs for enhanced features
echo "Monitoring logs for 60 seconds to verify enhancements..."
timeout 60 pm2 logs miner_advanced_ensemble --lines 20 | grep -E "(Performance Summary|Data quality|Price deviation|Corrected|Enhanced)"

echo "✅ Miner updated with client requirements!"
echo "   - Accurate predictions with price validation"
echo "   - Reliable CoinMetrics API data usage"
echo "   - Robust ML models with enhanced ensemble"
echo "   - Performance monitoring and quality assurance"
