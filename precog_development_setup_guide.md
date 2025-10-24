# Precog Subnet Development Environment Setup Guide

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 50GB free space
- **CPU**: 2+ cores (4+ cores recommended)

### Required Software

#### 1. Python Environment
```bash
# Install Python 3.9, 3.10, or 3.11
# Using pyenv (recommended)
curl https://pyenv.run | bash
pyenv install 3.11.0
pyenv global 3.11.0

# Verify installation
python --version  # Should show 3.11.0
```

#### 2. Node.js and NPM
```bash
# Install Node.js and NPM
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation
node --version
npm --version
```

#### 3. PM2 Process Manager
```bash
# Install PM2 globally
sudo npm install pm2@latest -g

# Verify installation
pm2 --version
```

#### 4. Git
```bash
# Install Git
sudo apt update
sudo apt install git

# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Installation Steps

### 1. Clone the Repository
```bash
# Clone the Precog repository
git clone https://github.com/coinmetrics/precog.git
cd precog

# Check out the latest release
git checkout v3.0.0
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Poetry
pip install poetry

# Install dependencies
poetry install

# Verify installation
poetry run python -c "import bittensor; print('Bittensor installed successfully')"
```

### 3. Configure Environment Files

#### For Validators
```bash
# Copy validator environment template
cp .env.validator.example .env.validator

# Edit configuration
nano .env.validator
```

**Validator Configuration**:
```bash
# Network Configuration
NETWORK=finney  # or testnet, localnet

# Wallet Configuration
COLDKEY=your_validator_coldkey
VALIDATOR_HOTKEY=your_validator_hotkey

# Node Configuration
VALIDATOR_NAME=validator
VALIDATOR_PORT=8091

# Auto Update Configuration
AUTO_UPDATE=1
AUTO_UPDATE_PROC_NAME=auto-updater
SCRIPT_LOCATION=./precog/validators/validator.py

# Logging
LOGGING_LEVEL=debug

# WandB Setup (Required for validators)
WANDB_API_KEY=your_wandb_api_key
```

#### For Miners
```bash
# Copy miner environment template
cp .env.miner.example .env.miner

# Edit configuration
nano .env.miner
```

**Miner Configuration**:
```bash
# Network Configuration
NETWORK=finney  # or testnet, localnet

# Wallet Configuration
COLDKEY=your_miner_coldkey
MINER_HOTKEY=your_miner_hotkey

# Node Configuration
MINER_NAME=miner
MINER_PORT=8092

# Miner Settings
TIMEOUT=16
VPERMIT_TAO_LIMIT=2
FORWARD_FUNCTION=base_miner  # or your custom miner

# Logging
LOGGING_LEVEL=debug
```

### 4. Set Up WandB (Validators Only)

#### Create WandB Account
1. Go to [https://wandb.ai](https://wandb.ai)
2. Sign up for a free account
3. Go to Settings → API Keys
4. Copy your API key

#### Configure WandB
```bash
# Login to WandB
wandb login

# Paste your API key when prompted
```

### 5. Set Up Bittensor Wallets

#### Create Wallets
```bash
# Create validator wallet
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey validator

# Create miner wallet
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey miner
```

#### Register on Subnet
```bash
# Register validator (requires TAO stake)
make register ENV_FILE=.env.validator

# Register miner
make register ENV_FILE=.env.miner
```

## Development Workflow

### 1. Local Development Setup

#### Install Development Dependencies
```bash
# Install development tools
poetry install --with dev

# Install pre-commit hooks
pre-commit install
```

#### Run Tests
```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_package.py

# Run with coverage
poetry run pytest --cov=precog
```

### 2. Custom Miner Development

#### Create Custom Miner
```python
# File: precog/miners/my_custom_miner.py

import bittensor as bt
from precog.protocol import Challenge
from precog.utils.cm_data import CMData

async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    """Custom prediction logic"""
    
    # Your prediction algorithm here
    predictions = {}
    intervals = {}
    
    # Example: Simple moving average strategy
    for asset in synapse.assets:
        # Get historical data
        data = cm.get_CM_ReferenceRate(
            assets=[asset],
            start=synapse.timestamp,
            end=synapse.timestamp,
            frequency="1s"
        )
        
        if not data.empty:
            # Calculate SMA
            sma = data['ReferenceRateUSD'].mean()
            volatility = data['ReferenceRateUSD'].std()
            
            # Set predictions
            predictions[asset] = float(sma)
            intervals[asset] = [
                float(sma - 2 * volatility),
                float(sma + 2 * volatility)
            ]
    
    # Set the synapse fields
    synapse.predictions = predictions
    synapse.intervals = intervals
    
    return synapse
```

#### Test Custom Miner
```bash
# Update .env.miner
FORWARD_FUNCTION=my_custom_miner

# Run miner
make miner ENV_FILE=.env.miner
```

### 3. Testing and Debugging

#### Local Testing
```bash
# Run on localnet
NETWORK=localnet make miner ENV_FILE=.env.miner
NETWORK=localnet make validator ENV_FILE=.env.validator
```

#### Debug Mode
```bash
# Enable debug logging
LOGGING_LEVEL=debug make miner ENV_FILE=.env.miner
```

#### Monitor Logs
```bash
# View PM2 logs
pm2 logs

# View specific process logs
pm2 logs miner
pm2 logs validator
```

### 4. Production Deployment

#### Validator Deployment
```bash
# Start validator
make validator ENV_FILE=.env.validator

# Check status
pm2 status

# Monitor logs
pm2 logs validator
```

#### Miner Deployment
```bash
# Start miner
make miner ENV_FILE=.env.miner

# Check status
pm2 status

# Monitor logs
pm2 logs miner
```

## Troubleshooting

### Common Issues

#### 1. Python Version Issues
```bash
# Check Python version
python --version

# If wrong version, use pyenv
pyenv install 3.11.0
pyenv local 3.11.0
```

#### 2. Dependency Issues
```bash
# Clean install
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
poetry install
```

#### 3. Wallet Issues
```bash
# Check wallet status
btcli wallet overview

# Check registration
btcli subnet list --netuid 55
```

#### 4. Network Issues
```bash
# Check network connectivity
ping 8.8.8.8

# Check Bittensor network
btcli subnet metagraph --netuid 55
```

#### 5. PM2 Issues
```bash
# Restart PM2
pm2 restart all

# Clear PM2 logs
pm2 flush

# Delete all processes
pm2 delete all
```

### Performance Optimization

#### 1. System Optimization
```bash
# Increase file descriptor limit
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Optimize network settings
echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf
sysctl -p
```

#### 2. Memory Management
```bash
# Monitor memory usage
htop

# Check PM2 memory usage
pm2 monit
```

#### 3. Log Management
```bash
# Rotate logs
pm2 install pm2-logrotate
pm2 set pm2-logrotate:max_size 10M
pm2 set pm2-logrotate:retain 7
```

## Monitoring and Maintenance

### 1. Health Checks
```bash
# Check process status
pm2 status

# Check network status
btcli subnet metagraph --netuid 55

# Check wallet balance
btcli wallet overview
```

### 2. Log Monitoring
```bash
# Monitor logs in real-time
pm2 logs --lines 100

# Monitor specific process
pm2 logs miner --lines 50
```

### 3. Performance Monitoring
```bash
# Monitor system resources
htop

# Monitor PM2 processes
pm2 monit

# Check disk usage
df -h
```

### 4. Backup and Recovery
```bash
# Backup state files
cp -r ~/.bittensor/logs/ ~/backup/bittensor-logs-$(date +%Y%m%d)

# Backup configuration
cp .env.validator ~/backup/
cp .env.miner ~/backup/
```

## Advanced Configuration

### 1. Custom Network Settings
```bash
# Custom subtensor endpoint
LOCALNET=ws://your-subtensor-node:9945
```

### 2. Advanced Miner Configuration
```bash
# Custom timeout settings
TIMEOUT=30

# Custom vpermit settings
VPERMIT_TAO_LIMIT=1024
```

### 3. Validator Optimization
```bash
# Disable auto-updates for custom code
AUTO_UPDATE=0

# Custom logging level
LOGGING_LEVEL=trace
```

## Security Best Practices

### 1. Wallet Security
- Use strong passwords for wallet encryption
- Store cold keys offline
- Use hardware wallets for production
- Regularly backup wallet files

### 2. Network Security
- Use firewall rules to restrict access
- Monitor for suspicious activity
- Keep software updated
- Use VPN for remote access

### 3. System Security
- Run as non-root user
- Use systemd for service management
- Monitor system logs
- Implement intrusion detection

This comprehensive setup guide provides everything needed to develop, test, and deploy on the Precog Subnet, from basic installation to advanced production configurations.

