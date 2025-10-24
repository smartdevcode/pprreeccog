#!/bin/bash

echo "🔧 Fixing Miner Block Sync Issues..."

# Stop the miner
echo "Stopping miner..."
pm2 stop miner_advanced_ensemble

# Check network connectivity
echo "Checking network connectivity..."
ping -c 3 entrypoint-finney.opentensor.ai

# Test chain endpoint
echo "Testing chain endpoint..."
curl -I https://entrypoint-finney.opentensor.ai:443

# Check current block from command line
echo "Checking current block..."
btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 | head -5

# Restart miner with improved block sync
echo "Restarting miner with improved block sync..."
pm2 restart miner_advanced_ensemble

# Monitor logs for block updates
echo "Monitoring logs for 60 seconds..."
timeout 60 pm2 logs miner_advanced_ensemble --lines 20 | grep -E "(Block:|Syncing|reconnect|update)"

echo "✅ Miner restart complete. Monitor logs to ensure blocks are updating."
