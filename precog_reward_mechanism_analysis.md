# Precog Subnet Reward Mechanism Deep Dive

## Overview

The Precog Subnet uses a sophisticated dual-task reward system that evaluates miners on both point predictions and interval predictions across multiple assets (BTC, ETH, TAO). The system is designed to reward both accuracy and uncertainty quantification.

## Core Reward Components

### 1. Point Prediction Scoring

**Metric**: Absolute Percentage Error (APE)
```python
point_error = abs(predicted_price - actual_price) / actual_price
```

**Characteristics**:
- Lower error = higher reward
- Penalizes both overestimation and underestimation equally
- Normalized by actual price for fair comparison across different price levels

**Ranking**: Miners are ranked by error (ascending), with lower errors receiving higher ranks.

### 2. Interval Prediction Scoring

**Dual Metrics**:

#### Inclusion Factor (f_i)
```python
inclusion_factor = prices_in_bounds / total_prices
```
- Measures what percentage of actual prices fall within the predicted range
- Range: [0, 1] where 1 = perfect inclusion
- Rewards miners who capture the actual price movement within their predicted bounds

#### Width Factor (f_w)
```python
effective_top = min(predicted_max, observed_max)
effective_bottom = max(predicted_min, observed_min)
width_factor = (effective_top - effective_bottom) / (predicted_max - predicted_min)
```
- Measures the effective overlap between predicted and observed ranges
- Range: [0, 1] where 1 = perfect overlap
- Rewards miners whose predicted ranges align well with actual price ranges

#### Combined Interval Score
```python
interval_score = inclusion_factor × width_factor
```

**Ranking**: Miners are ranked by interval score (descending), with higher scores receiving higher ranks.

## Task Weighting System

### Asset-Level Weighting
```python
TASK_WEIGHTS = {
    "btc": {"point": 0.166, "interval": 0.166},
    "eth": {"point": 0.166, "interval": 0.166}, 
    "tao_bittensor": {"point": 0.166, "interval": 0.166}
}
```

**Total Weight Distribution**:
- 6 tasks total (3 assets × 2 tasks each)
- Each task gets 1/6 = 0.166 weight
- Equal weighting ensures no single asset dominates the reward

### Ranking and Weight Conversion

#### Ranking Algorithm
```python
def rank(vector):
    # Returns ranks for tied values
    # Ties get the same rank (e.g., [1, 1, 3, 4] for values [0.1, 0.1, 0.3, 0.4])
```

#### Weight Calculation with Ties
```python
def get_average_weights_for_ties(ranks, decay=0.8):
    # Base weights: [1.0, 0.8, 0.64, 0.512, ...]
    base_weights = decay ** np.arange(n)
    
    # For tied ranks, average the corresponding base weights
    # Example: If ranks [0, 0, 2, 3] for 4 miners
    # First two get average of weights[0] and weights[1]
```

**Decay Factor**: 0.8
- Creates exponential decay in weights: [1.0, 0.8, 0.64, 0.512, ...]
- Ensures significant differentiation between ranks
- Prevents winner-take-all scenarios

## Final Reward Calculation

### Step-by-Step Process

1. **Data Collection**
   ```python
   # For each asset and each miner
   asset_point_errors[asset] = [error1, error2, ...]
   asset_interval_scores[asset] = [score1, score2, ...]
   ```

2. **Independent Task Ranking**
   ```python
   # Point prediction ranking (lower error = higher rank)
   point_ranks = rank(np.array(asset_point_errors[asset]))
   point_weights = get_average_weights_for_ties(point_ranks, decay=0.8)
   
   # Interval prediction ranking (higher score = higher rank)
   interval_ranks = rank(-np.array(asset_interval_scores[asset]))  # Flip for higher=better
   interval_weights = get_average_weights_for_ties(interval_ranks, decay=0.8)
   ```

3. **Task Weight Application**
   ```python
   final_rewards = np.zeros(len(available_uids))
   
   for asset in assets:
       point_weight = TASK_WEIGHTS[asset]["point"]  # 0.166
       interval_weight = TASK_WEIGHTS[asset]["interval"]  # 0.166
       
       final_rewards += point_weight * point_weights
       final_rewards += interval_weight * interval_weights
   ```

### Example Calculation

**Scenario**: 4 miners predicting BTC price

**Point Predictions**: [50000, 51000, 49000, 50500] (actual: 50000)
**Point Errors**: [0.0, 0.02, 0.02, 0.01]
**Point Ranks**: [0, 2, 2, 1] (tied at rank 2)
**Point Weights**: [1.0, 0.72, 0.72, 0.8] (averaged for ties)

**Interval Scores**: [0.8, 0.6, 0.9, 0.7]
**Interval Ranks**: [1, 3, 0, 2]
**Interval Weights**: [0.8, 0.512, 1.0, 0.64]

**Final BTC Reward**: 0.166 × (point_weights + interval_weights)

## Moving Average Integration

### Exponential Moving Average
```python
new_score = (1 - alpha) × old_score + alpha × current_reward
```

**Alpha Parameter**: 0.05 (configurable)
- **Half-life**: ~14 iterations (2 hours at 5-minute intervals)
- **Purpose**: Smooth out short-term fluctuations
- **Effect**: Recent performance weighted more heavily than historical

### State Persistence
- **Storage**: Pickle file (`state.pt`)
- **Contents**: `moving_average_scores`, `MinerHistory`
- **Recovery**: Automatic state restoration on restart
- **Cleanup**: Old predictions removed after 24 hours

## Evaluation Timeline

### Prediction Cycle
1. **T+0**: Miners make predictions for T+1 hour
2. **T+1 hour**: Validators evaluate predictions against actual data
3. **T+1 hour**: Rewards calculated and moving averages updated
4. **T+1 hour**: Weights set on blockchain (if rate limit allows)

### Data Requirements
- **Historical Data**: 1 hour of 1-second resolution price data
- **Evaluation Window**: 6 hours of stored predictions
- **Data Source**: CoinMetrics Reference Rate API

## Advanced Features

### Error Handling
- **Missing Predictions**: Assigned `np.inf` error (lowest rank)
- **Missing Data**: Assigned score of 0 (lowest rank)
- **API Failures**: Graceful degradation with warnings

### Performance Optimizations
- **Batch Processing**: All assets queried in single API call
- **Caching**: 24-hour rolling cache for fast data access
- **Concurrent Evaluation**: Parallel processing of multiple miners

### Monitoring and Logging
- **WandB Integration**: Real-time performance tracking
- **Debug Logging**: Detailed per-miner, per-asset scoring
- **State Persistence**: Automatic backup and recovery

## Reward Mechanism Strengths

1. **Dual Task Design**: Rewards both accuracy and uncertainty quantification
2. **Fair Weighting**: Equal treatment of all assets and tasks
3. **Tie Handling**: Proper averaging for tied performances
4. **Smooth Integration**: Moving averages prevent volatility
5. **Robust Evaluation**: Handles missing data gracefully

## Potential Improvements

1. **Dynamic Weighting**: Adjust weights based on asset volatility
2. **Confidence Intervals**: Reward appropriate interval sizing
3. **Multi-timeframe**: Evaluate predictions at multiple horizons
4. **Market Regime**: Adjust scoring based on market conditions

This reward mechanism creates a competitive environment where miners are incentivized to provide both accurate point estimates and well-calibrated uncertainty bounds, leading to more useful predictions for the broader community.

