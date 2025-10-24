# Precog Subnet Debugging Guide

## Overview

This comprehensive debugging guide covers common issues, troubleshooting strategies, and debugging techniques for the Precog Subnet.

## 🔍 **Debugging Tools and Setup**

### 1. Logging Configuration

#### Enable Debug Logging
```bash
# Set debug level in .env files
LOGGING_LEVEL=debug

# Or trace level for maximum detail
LOGGING_LEVEL=trace
```

#### Monitor Logs in Real-Time
```bash
# Monitor all PM2 processes
pm2 logs

# Monitor specific process
pm2 logs miner
pm2 logs validator

# Follow logs with timestamps
pm2 logs --timestamp

# Monitor with line limits
pm2 logs --lines 100
```

### 2. System Monitoring

#### Process Status
```bash
# Check PM2 status
pm2 status

# Detailed process info
pm2 show miner
pm2 show validator

# Monitor system resources
pm2 monit
```

#### Network Diagnostics
```bash
# Check network connectivity
ping 8.8.8.8

# Check Bittensor network
btcli subnet metagraph --netuid 55

# Check wallet status
btcli wallet overview
```

## 🐛 **Common Issues and Solutions**

### 1. Miner Issues

#### Issue: Miner Not Responding to Validators
**Symptoms:**
- Validator logs show "No response from miner"
- Miner shows no incoming requests
- PM2 status shows miner as "online" but no activity

**Debugging Steps:**
```bash
# 1. Check miner logs
pm2 logs miner --lines 50

# 2. Verify miner registration
btcli subnet list --netuid 55 | grep your_hotkey

# 3. Check network connectivity
netstat -tulpn | grep 8092

# 4. Test miner manually
curl -X POST http://localhost:8092/health
```

**Solutions:**
```bash
# Restart miner
pm2 restart miner

# Check firewall settings
sudo ufw status
sudo ufw allow 8092

# Verify port binding
lsof -i :8092
```

#### Issue: Miner Predictions Always Empty
**Symptoms:**
- Miner responds but predictions are None
- Logs show "No predictions for this request"
- Validator receives empty responses

**Debugging Steps:**
```python
# Add debug logging to your miner
async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    bt.logging.debug(f"Received request: {synapse.timestamp}")
    bt.logging.debug(f"Assets: {synapse.assets}")
    
    # Check data availability
    data = cm.get_CM_ReferenceRate(...)
    bt.logging.debug(f"Data shape: {data.shape}")
    bt.logging.debug(f"Data columns: {data.columns.tolist()}")
    
    # Your prediction logic here
    # ...
    
    bt.logging.debug(f"Predictions: {predictions}")
    bt.logging.debug(f"Intervals: {intervals}")
    
    return synapse
```

**Solutions:**
```python
# Ensure predictions are set
if not predictions:
    bt.logging.warning("No predictions generated, using fallback")
    latest_price = float(data['ReferenceRateUSD'].iloc[-1])
    predictions = {asset: latest_price for asset in synapse.assets}
    intervals = {asset: [latest_price * 0.95, latest_price * 1.05] for asset in synapse.assets}

synapse.predictions = predictions
synapse.intervals = intervals
```

#### Issue: High Memory Usage
**Symptoms:**
- System becomes slow
- PM2 shows high memory usage
- Out of memory errors

**Debugging Steps:**
```bash
# Check memory usage
htop
pm2 monit

# Check specific process memory
ps aux | grep python
```

**Solutions:**
```python
# Implement memory management in your miner
import gc

async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    try:
        # Your prediction logic
        pass
    finally:
        # Clean up memory
        gc.collect()
```

### 2. Validator Issues

#### Issue: Validator Not Querying Miners
**Symptoms:**
- Validator logs show "No miners available"
- No prediction requests being sent
- Validator running but inactive

**Debugging Steps:**
```bash
# Check validator logs
pm2 logs validator --lines 100

# Check metagraph sync
btcli subnet metagraph --netuid 55

# Check available UIDs
btcli subnet list --netuid 55
```

**Solutions:**
```bash
# Restart validator
pm2 restart validator

# Check network configuration
cat .env.validator | grep NETWORK

# Verify subtensor connection
btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443
```

#### Issue: Weight Setting Failures
**Symptoms:**
- "Failed to set weights" messages
- Validator not updating weights on blockchain
- High block count since last update

**Debugging Steps:**
```bash
# Check validator logs for weight setting
pm2 logs validator | grep -i weight

# Check blockchain connection
btcli subnet metagraph --netuid 55

# Check wallet balance
btcli wallet overview
```

**Solutions:**
```bash
# Check wallet has sufficient TAO
btcli wallet overview

# Verify network connection
ping entrypoint-finney.opentensor.ai

# Check rate limiting
# Weight setting is rate-limited by subnet hyperparameters
```

#### Issue: Reward Calculation Errors
**Symptoms:**
- "Failed to calculate rewards" errors
- Validator crashes during reward calculation
- Invalid reward values

**Debugging Steps:**
```python
# Add debug logging to reward calculation
def calc_rewards(self, responses: List[Challenge]) -> np.ndarray:
    bt.logging.debug(f"Processing {len(responses)} responses")
    
    for i, response in enumerate(responses):
        bt.logging.debug(f"Response {i}: {response.predictions}")
        bt.logging.debug(f"Response {i} intervals: {response.intervals}")
    
    # Your reward calculation logic
    # ...
```

**Solutions:**
```python
# Add error handling
try:
    rewards = calc_rewards(self, responses=responses)
except Exception as e:
    bt.logging.error(f"Reward calculation failed: {e}")
    # Return equal weights as fallback
    rewards = np.ones(len(self.available_uids)) / len(self.available_uids)
```

### 3. Network and Connectivity Issues

#### Issue: Cannot Connect to Bittensor Network
**Symptoms:**
- "Connection refused" errors
- Timeout errors
- Network unreachable

**Debugging Steps:**
```bash
# Test network connectivity
ping entrypoint-finney.opentensor.ai
telnet entrypoint-finney.opentensor.ai 443

# Check DNS resolution
nslookup entrypoint-finney.opentensor.ai

# Test with different endpoints
btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443
```

**Solutions:**
```bash
# Try different network endpoints
# Finney (mainnet)
NETWORK=finney

# Testnet
NETWORK=testnet

# Local network
NETWORK=localnet
LOCALNET=ws://127.0.0.1:9945
```

#### Issue: Wallet Connection Problems
**Symptoms:**
- "Wallet not registered" errors
- Cannot access wallet
- Permission denied errors

**Debugging Steps:**
```bash
# Check wallet files
ls -la ~/.bittensor/wallets/

# Test wallet access
btcli wallet overview

# Check wallet permissions
chmod 600 ~/.bittensor/wallets/*/coldkey
chmod 600 ~/.bittensor/wallets/*/hotkey
```

**Solutions:**
```bash
# Recreate wallet if corrupted
btcli wallet new_coldkey --wallet.name backup
btcli wallet new_hotkey --wallet.name backup --wallet.hotkey backup

# Restore from backup
cp ~/backup/wallets/* ~/.bittensor/wallets/
```

## 🔧 **Advanced Debugging Techniques**

### 1. Custom Debug Logging

#### Add Comprehensive Logging
```python
import logging
import traceback

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    bt.logging.debug(f"=== MINER DEBUG START ===")
    bt.logging.debug(f"Timestamp: {synapse.timestamp}")
    bt.logging.debug(f"Assets: {synapse.assets}")
    bt.logging.debug(f"Dendrite: {synapse.dendrite}")
    
    try:
        # Your prediction logic
        bt.logging.debug(f"Data fetch successful")
        bt.logging.debug(f"Predictions: {predictions}")
        bt.logging.debug(f"Intervals: {intervals}")
        
    except Exception as e:
        bt.logging.error(f"Prediction failed: {e}")
        bt.logging.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    bt.logging.debug(f"=== MINER DEBUG END ===")
    return synapse
```

### 2. Performance Profiling

#### Profile Miner Performance
```python
import cProfile
import pstats
from io import StringIO

async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    # Profile the forward function
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        # Your prediction logic
        result = await your_prediction_logic(synapse, cm)
        
    finally:
        profiler.disable()
        
        # Print profiling results
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        bt.logging.debug(f"Profiling results:\n{s.getvalue()}")
    
    return result
```

#### Memory Profiling
```python
import tracemalloc

async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    # Start memory tracing
    tracemalloc.start()
    
    try:
        # Your prediction logic
        result = await your_prediction_logic(synapse, cm)
        
    finally:
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        bt.logging.debug(f"Current memory: {current / 1024 / 1024:.2f} MB")
        bt.logging.debug(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
        
        # Stop tracing
        tracemalloc.stop()
    
    return result
```

### 3. Data Validation

#### Validate Input Data
```python
def validate_synapse(synapse: Challenge) -> bool:
    """Validate incoming synapse data."""
    if not synapse.timestamp:
        bt.logging.error("Missing timestamp")
        return False
    
    if not synapse.assets:
        bt.logging.error("Missing assets")
        return False
    
    try:
        # Validate timestamp format
        to_datetime(synapse.timestamp)
    except Exception as e:
        bt.logging.error(f"Invalid timestamp format: {e}")
        return False
    
    return True

async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    if not validate_synapse(synapse):
        # Return empty predictions for invalid requests
        return Challenge(
            timestamp=synapse.timestamp,
            assets=synapse.assets,
            predictions={},
            intervals={}
        )
    
    # Your prediction logic
    # ...
```

#### Validate Output Data
```python
def validate_predictions(predictions: Dict, intervals: Dict, assets: List[str]) -> bool:
    """Validate prediction output."""
    for asset in assets:
        if asset not in predictions:
            bt.logging.error(f"Missing prediction for {asset}")
            return False
        
        if asset not in intervals:
            bt.logging.error(f"Missing interval for {asset}")
            return False
        
        # Validate prediction value
        pred = predictions[asset]
        if not isinstance(pred, (int, float)) or pred <= 0:
            bt.logging.error(f"Invalid prediction for {asset}: {pred}")
            return False
        
        # Validate interval
        interval = intervals[asset]
        if not isinstance(interval, list) or len(interval) != 2:
            bt.logging.error(f"Invalid interval for {asset}: {interval}")
            return False
        
        lower, upper = interval
        if not all(isinstance(x, (int, float)) for x in [lower, upper]):
            bt.logging.error(f"Invalid interval values for {asset}")
            return False
        
        if lower >= upper:
            bt.logging.error(f"Invalid interval bounds for {asset}: {lower} >= {upper}")
            return False
    
    return True
```

## 🚨 **Emergency Recovery Procedures**

### 1. Complete System Reset
```bash
# Stop all processes
pm2 delete all

# Clear PM2 logs
pm2 flush

# Restart from scratch
pm2 start ecosystem.config.js
```

### 2. State Recovery
```bash
# Backup current state
cp -r ~/.bittensor/logs/ ~/backup/state-$(date +%Y%m%d)

# Reset validator state
rm ~/.bittensor/logs/*/validator/state.pt

# Restart validator (will create new state)
pm2 restart validator
```

### 3. Network Recovery
```bash
# Reset network configuration
rm ~/.bittensor/logs/*/netuid55/

# Re-sync metagraph
btcli subnet metagraph --netuid 55

# Restart processes
pm2 restart all
```

## 📊 **Monitoring and Alerting**

### 1. Health Check Script
```bash
#!/bin/bash
# health_check.sh

# Check PM2 processes
pm2 status | grep -q "online" || echo "PM2 processes down"

# Check network connectivity
ping -c 1 8.8.8.8 > /dev/null || echo "Network connectivity issues"

# Check disk space
df -h | awk '$5 > 90 {print "Disk space low: " $0}'

# Check memory usage
free -h | awk 'NR==2{printf "Memory usage: %.2f%%\n", $3*100/$2}'
```

### 2. Automated Monitoring
```python
# monitoring.py
import psutil
import time
import bittensor as bt

def monitor_system():
    """Monitor system resources and log warnings."""
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    if cpu_percent > 80:
        bt.logging.warning(f"High CPU usage: {cpu_percent}%")
    
    # Memory usage
    memory = psutil.virtual_memory()
    if memory.percent > 80:
        bt.logging.warning(f"High memory usage: {memory.percent}%")
    
    # Disk usage
    disk = psutil.disk_usage('/')
    if disk.percent > 90:
        bt.logging.warning(f"High disk usage: {disk.percent}%")

# Run monitoring in background
while True:
    monitor_system()
    time.sleep(60)  # Check every minute
```

This comprehensive debugging guide provides all the tools and techniques needed to diagnose and resolve issues in the Precog Subnet system.

