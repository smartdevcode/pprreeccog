# Precog Subnet Architecture Analysis

## System Overview

The Precog Subnet is a sophisticated Bittensor-based system for Bitcoin price prediction that operates on high-frequency intervals (every 5 minutes) with short resolution times (every hour). The system supports multiple assets (BTC, ETH, TAO) and uses a dual prediction mechanism.

## Core Architecture Components

### 1. Protocol Layer (`precog/protocol.py`)

**Challenge Synapse:**
- **Purpose**: Defines the communication protocol between validators and miners
- **Key Fields**:
  - `timestamp`: Prediction timestamp (ISO 8601 format)
  - `assets`: List of assets to predict (default: ["btc"])
  - `predictions`: Point estimates for each asset (filled by miners)
  - `intervals`: Price range predictions [min, max] (filled by miners)

**Data Flow:**
```
Validator → Challenge(timestamp, assets) → Miner
Miner → Challenge(predictions, intervals) → Validator
```

### 2. Miner System (`precog/miners/`)

**Base Miner (`base_miner.py`):**
- **Data Source**: CoinMetrics API for real-time price data
- **Prediction Strategy**: 
  - Point estimate: Latest price from CM data
  - Interval estimate: Historical volatility-based range calculation
- **Key Features**:
  - Caching system for 24-hour data
  - Volatility calculation using 1-hour historical data
  - Fallback mechanisms for insufficient data

**Custom Miner Support:**
- Modular design allows custom prediction strategies
- Forward function can be specified via `--forward_function` parameter
- Example: `--forward_function my_custom_miner`

**Miner Architecture:**
```
Miner Class:
├── CMData (CoinMetrics integration)
├── Forward Function (prediction logic)
├── Blacklist Function (security)
├── Priority Function (request ordering)
└── Axon (Bittensor communication)
```

### 3. Validator System (`precog/validators/`)

**Weight Setter (`weight_setter.py`):**
- **Core Function**: Manages validator operations and weight setting
- **Key Responsibilities**:
  - Query miners every 5 minutes
  - Calculate rewards based on prediction accuracy
  - Update moving average scores
  - Set weights on Bittensor blockchain
  - Manage miner history and state

**Validator Main Loop:**
```
1. Scheduled Prediction Request (every 5 minutes)
   ├── Query all available miners
   ├── Collect predictions and intervals
   └── Calculate rewards
2. Weight Setting (based on rate limit)
   ├── Update moving averages
   ├── Convert to uint16 weights
   └── Submit to blockchain
3. Metagraph Sync (every 10 minutes)
   ├── Update available UIDs
   ├── Handle new/removed miners
   └── Maintain state consistency
```

### 4. Reward System (`precog/validators/reward.py`)

**Dual Scoring Mechanism:**

**Point Prediction Scoring:**
- **Metric**: Absolute percentage error from actual price
- **Formula**: `|predicted - actual| / actual`
- **Ranking**: Lower error = higher rank

**Interval Prediction Scoring:**
- **Metrics**: 
  - Inclusion factor (f_i): Percentage of actual prices within predicted range
  - Width factor (f_w): Effective overlap between predicted and actual ranges
- **Formula**: `f_i × f_w`
- **Ranking**: Higher score = higher rank

**Task Weighting:**
```python
TASK_WEIGHTS = {
    "btc": {"point": 0.166, "interval": 0.166},
    "eth": {"point": 0.166, "interval": 0.166}, 
    "tao_bittensor": {"point": 0.166, "interval": 0.166}
}
```

### 5. Data Management

**CoinMetrics Integration (`precog/utils/cm_data.py`):**
- **API Client**: CoinMetrics Python client
- **Caching**: 24-hour rolling cache for performance
- **Data Types**: Reference rates, candles, open interest, funding rates
- **Frequency**: 1-second resolution for high-frequency data

**Timestamp Management (`precog/utils/timestamp.py`):**
- **Timezone**: UTC for consistency
- **Rounding**: 5-minute intervals
- **Functions**: Conversion, rounding, epoch detection

## System Data Flow

### 1. Prediction Request Cycle (Every 5 Minutes)

```
1. Validator detects new epoch
2. Creates Challenge synapse with timestamp and assets
3. Queries all available miners concurrently
4. Collects responses (predictions + intervals)
5. Calculates rewards based on historical data
6. Updates moving average scores
7. Logs to WandB (if enabled)
```

### 2. Weight Setting Cycle (Based on Rate Limit)

```
1. Check if enough blocks have passed since last update
2. Convert moving average scores to weights
3. Normalize weights (sum to 1.0)
4. Submit weights to Bittensor blockchain
5. Reset block counter
```

### 3. State Management

**MinerHistory Class:**
- **Purpose**: Track miner predictions over time
- **Data**: Predictions and intervals by timestamp
- **Cleanup**: Remove data older than 24 hours
- **Format**: `{timestamp: {asset: value}}`

**State Persistence:**
- **File**: `state.pt` (pickle format)
- **Contents**: scores, moving_average_scores, MinerHistory
- **Location**: `~/.bittensor/logs/{wallet}/{hotkey}/netuid{netuid}/validator/`

## Key Configuration Parameters

### Network Configuration
- **Mainnet UID**: 55
- **Testnet UID**: 256
- **Prediction Interval**: 5 minutes
- **Evaluation Window**: 6 hours
- **Prediction Horizon**: 1 hour

### Performance Settings
- **Cache Size**: 100MB max, 500K rows max
- **Timeout**: 20 seconds (validators), 16 seconds (miners)
- **Moving Average Alpha**: 0.05
- **Ranking Decay**: 0.8

### Supported Assets
- **Primary**: BTC, ETH, TAO (Bittensor)
- **Equal Weighting**: Each asset gets 1/3 of total weight
- **Dual Tasks**: Point + Interval predictions per asset

## Security & Performance Features

### Security
- **Blacklisting**: Non-registered entities, insufficient stake
- **Validator Permits**: Force validator-only requests
- **Hotkey Validation**: Verify sender identity

### Performance
- **Caching**: 24-hour data cache for fast responses
- **Concurrent Processing**: Parallel miner queries
- **State Management**: Efficient memory usage
- **Auto-updates**: Automatic code updates for validators

## Development & Deployment

### Environment Setup
- **Python**: 3.9, 3.10, or 3.11
- **Dependencies**: Poetry for package management
- **Process Management**: PM2 for production
- **Monitoring**: WandB integration

### Customization Points
1. **Custom Miners**: Implement custom forward functions
2. **Reward Mechanisms**: Modify scoring algorithms
3. **Data Sources**: Integrate additional data providers
4. **Prediction Strategies**: Develop advanced ML models

## Current Status (v3.0.0)

- **Multi-asset Support**: BTC, ETH, TAO with equal weighting
- **High-frequency Operations**: 5-minute prediction cycles
- **Robust State Management**: Persistent state with cleanup
- **Performance Optimized**: Caching and concurrent processing
- **Production Ready**: Auto-updates and monitoring

This architecture provides a robust foundation for high-frequency cryptocurrency price prediction while maintaining the decentralized and competitive nature of the Bittensor ecosystem.

