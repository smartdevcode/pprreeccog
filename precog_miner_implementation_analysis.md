# Precog Subnet Miner Implementation Analysis

## Overview

The Precog Subnet miner system is designed for high-performance, low-latency Bitcoin price prediction. It supports both a base implementation and custom miner strategies, with robust data handling and prediction mechanisms.

## Base Miner Implementation (`base_miner.py`)

### Core Strategy

**Point Prediction**: Latest price from CoinMetrics data
**Interval Prediction**: Historical volatility-based range calculation

### Prediction Algorithm

#### 1. Data Collection
```python
# Fetch 1 hour of 1-second resolution data
all_data = cm.get_CM_ReferenceRate(
    assets=assets,
    start=start_timestamp,
    end=provided_timestamp,
    frequency="1s"
)
```

#### 2. Point Estimate Calculation
```python
# Use latest available price as point estimate
point_estimate = float(asset_data["ReferenceRateUSD"].iloc[-1])
```

#### 3. Interval Calculation Algorithm

**Step 1: Calculate Returns**
```python
hourly_returns = historical_prices.pct_change().dropna()
```

**Step 2: Outlier Removal**
```python
# Remove extreme outliers (beyond 3 standard deviations)
returns_std = hourly_returns.std()
returns_mean = hourly_returns.mean()
outlier_mask = abs(hourly_returns - returns_mean) <= 3 * returns_std
clean_returns = hourly_returns[outlier_mask]
```

**Step 3: Volatility Calculation**
```python
hourly_vol = float(clean_returns.std())
```

**Step 4: Confidence Interval**
```python
# 2.58 standard deviations = 99% confidence interval
margin = point_estimate * hourly_vol * 2.58
```

**Step 5: Bounds Adjustment**
```python
# Apply min/max constraints
max_margin = point_estimate * 0.30  # Cap at ±30%
min_margin = point_estimate * 0.02  # Minimum ±2%
margin = max(min_margin, min(margin, max_margin))
```

### Fallback Mechanisms

#### Data Insufficiency
- **< 100 data points**: ±10% interval
- **< 12 clean returns**: ±10% interval
- **API failure**: ±15% interval

#### Error Handling
```python
try:
    # Main calculation logic
except Exception as e:
    bt.logging.error(f"Error calculating interval for {asset}: {e}")
    # Emergency fallback: ±15% interval
    margin = point_estimate * 0.15
    return point_estimate - margin, point_estimate + margin
```

## Miner Architecture (`miner.py`)

### Core Components

#### 1. Miner Class Structure
```python
class Miner:
    def __init__(self, config):
        self.forward_module = importlib.import_module(f"precog.miners.{config.forward_function}")
        self.cm = CMData()
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority
        )
```

#### 2. Request Processing Pipeline
```
1. Blacklist Check → 2. Priority Assignment → 3. Forward Function → 4. Response
```

#### 3. Security Features

**Blacklist Function**:
```python
async def blacklist(self, synapse: Challenge) -> Tuple[bool, str]:
    # Check for missing dendrite/hotkey
    if synapse.dendrite is None or synapse.dendrite.hotkey is None:
        return True, "Missing dendrite or hotkey"
    
    # Check registration status
    if not self.config.blacklist.allow_non_registered:
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            return True, "Unrecognized hotkey"
    
    # Check validator permit requirement
    if self.config.blacklist.force_validator_permit:
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if not self.metagraph.validator_permit[uid]:
            return True, "Non-validator hotkey"
```

**Priority Function**:
```python
async def priority(self, synapse: Challenge) -> float:
    # Priority based on caller's stake
    caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
    priority = float(self.metagraph.S[caller_uid])
    return priority
```

### Performance Optimizations

#### 1. Caching System
- **24-hour rolling cache** for CoinMetrics data
- **Automatic cleanup** of old data
- **Memory monitoring** with size limits

#### 2. Concurrent Processing
- **Async/await** pattern for non-blocking operations
- **Lock mechanisms** for thread safety
- **Timeout handling** for API calls

#### 3. State Management
- **Metagraph synchronization** every 10 minutes
- **Block tracking** for network updates
- **Error recovery** with retry mechanisms

## Custom Miner Development

### Implementation Pattern

#### 1. Create Custom Forward Function
```python
# File: precog/miners/my_custom_miner.py

async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    """Custom prediction logic"""
    
    # Your prediction algorithm here
    predictions = {}
    intervals = {}
    
    # Set the synapse fields
    synapse.predictions = predictions
    synapse.intervals = intervals
    
    return synapse
```

#### 2. Configuration
```bash
# Use custom miner
make miner ENV_FILE=.env.miner FORWARD_FUNCTION=my_custom_miner
```

#### 3. Required Interface
- **Input**: `synapse: Challenge`, `cm: CMData`
- **Output**: `Challenge` with populated `predictions` and `intervals`
- **Format**: 
  - `predictions`: `{asset: float}`
  - `intervals`: `{asset: [min, max]}`

### Custom Miner Strategies

#### 1. Machine Learning Approach
```python
async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    # Load your trained model
    model = load_model("my_model.pkl")
    
    # Get historical data
    data = cm.get_CM_ReferenceRate(...)
    
    # Feature engineering
    features = extract_features(data)
    
    # Make predictions
    point_pred = model.predict(features)
    interval_pred = model.predict_interval(features, confidence=0.99)
    
    return Challenge(predictions={asset: point_pred}, intervals={asset: interval_pred})
```

#### 2. Technical Analysis Approach
```python
async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    # Get price data
    data = cm.get_CM_ReferenceRate(...)
    
    # Calculate technical indicators
    sma_20 = data['price'].rolling(20).mean()
    rsi = calculate_rsi(data['price'])
    bollinger = calculate_bollinger_bands(data['price'])
    
    # Generate predictions based on indicators
    if rsi < 30:  # Oversold
        prediction = data['price'].iloc[-1] * 1.02  # Bullish
    elif rsi > 70:  # Overbought
        prediction = data['price'].iloc[-1] * 0.98  # Bearish
    else:
        prediction = data['price'].iloc[-1]  # Neutral
    
    return Challenge(predictions={asset: prediction}, intervals={asset: bollinger})
```

#### 3. Multi-Asset Correlation Approach
```python
async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    # Get data for all assets
    btc_data = cm.get_CM_ReferenceRate(assets=["btc"], ...)
    eth_data = cm.get_CM_ReferenceRate(assets=["eth"], ...)
    
    # Calculate correlations
    correlation = btc_data['price'].corr(eth_data['price'])
    
    # Use correlation for predictions
    if correlation > 0.8:
        # High correlation - use similar predictions
        btc_pred = btc_data['price'].iloc[-1]
        eth_pred = eth_data['price'].iloc[-1]
    else:
        # Low correlation - independent predictions
        btc_pred = calculate_independent_prediction(btc_data)
        eth_pred = calculate_independent_prediction(eth_data)
    
    return Challenge(
        predictions={"btc": btc_pred, "eth": eth_pred},
        intervals={"btc": btc_interval, "eth": eth_interval}
    )
```

## Data Integration

### CoinMetrics API Integration

#### Available Data Types
1. **Reference Rates**: High-frequency price data
2. **Candles**: OHLCV data for technical analysis
3. **Open Interest**: Derivatives market data
4. **Funding Rates**: Perpetual swap data

#### Usage Examples
```python
# Reference rates (most common)
data = cm.get_CM_ReferenceRate(assets=["btc"], start=start, end=end, frequency="1s")

# Candles for technical analysis
candles = cm.get_pair_candles(pairs=["btc-usd"], start=start, end=end, frequency="1h")

# Open interest for derivatives
oi_catalog = cm.get_open_interest_catalog(base="btc", quote="usd", market_type="future")
oi_data = cm.get_market_open_interest(markets=markets)
```

### Caching Strategy

#### Cache Management
```python
# Automatic cache management
- 24-hour rolling window
- Memory limit: 100MB
- Row limit: 500,000
- Automatic cleanup of old data
```

#### Performance Benefits
- **Reduced API calls**: Cached data for repeated requests
- **Faster responses**: Local data access vs API calls
- **Cost efficiency**: Reduced API usage costs

## Error Handling and Resilience

### Robust Error Handling
```python
try:
    # Main prediction logic
    predictions = calculate_predictions(data)
except InsufficientDataError:
    # Fallback to simple strategy
    predictions = fallback_predictions(data)
except APIError:
    # Use cached data
    predictions = cached_predictions()
except Exception as e:
    # Log error and return empty predictions
    bt.logging.error(f"Prediction failed: {e}")
    return Challenge(predictions={}, intervals={})
```

### Performance Monitoring
```python
# Built-in performance tracking
total_time = time.perf_counter() - start_time
bt.logging.debug(f"⏱️ Total forward call took: {total_time:.3f} seconds")

# Cache statistics
if hasattr(self.cm, "log_cache_stats"):
    self.cm.log_cache_stats()
```

## Best Practices for Custom Miners

### 1. Performance Optimization
- **Minimize API calls**: Use caching effectively
- **Async operations**: Use async/await for I/O
- **Error handling**: Implement robust fallbacks
- **Logging**: Add detailed logging for debugging

### 2. Prediction Quality
- **Calibrated intervals**: Ensure intervals are realistic
- **Multi-asset consistency**: Consider correlations
- **Market regime awareness**: Adapt to different conditions
- **Uncertainty quantification**: Provide meaningful intervals

### 3. Integration
- **Standard interface**: Follow the required function signature
- **Data validation**: Check input data quality
- **Response format**: Ensure correct output format
- **Testing**: Test with various market conditions

## Development Workflow

### 1. Local Development
```bash
# Set up environment
poetry install
cp .env.miner.example .env.miner

# Run custom miner
make miner ENV_FILE=.env.miner FORWARD_FUNCTION=my_custom_miner
```

### 2. Testing Strategy
```python
# Test with historical data
def test_miner_with_historical_data():
    cm = CMData()
    synapse = Challenge(timestamp="2024-01-01T12:00:00.000Z", assets=["btc"])
    
    result = await forward(synapse, cm)
    
    assert result.predictions is not None
    assert result.intervals is not None
    assert "btc" in result.predictions
```

### 3. Deployment
```bash
# Production deployment
pm2 start --name my_miner python3 -- precog/miners/miner.py \
    --forward_function my_custom_miner \
    --wallet.name my_wallet \
    --wallet.hotkey my_hotkey
```

This comprehensive miner implementation provides a solid foundation for both simple and sophisticated prediction strategies, with robust error handling, performance optimization, and easy customization capabilities.

