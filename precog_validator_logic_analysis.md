# Precog Subnet Validator Logic Analysis

## Overview

The Precog Subnet validator system is a sophisticated orchestration layer that manages miner discovery, querying, evaluation, and weight setting. It operates on a high-frequency schedule with robust state management and error handling.

## Core Validator Architecture

### 1. Main Validator Class (`validator.py`)

**Purpose**: Entry point and lifecycle management
**Key Responsibilities**:
- Initialize and manage the weight_setter
- Handle graceful shutdown
- Coordinate async operations

```python
class Validator:
    async def main(self):
        self.weight_setter = await weight_setter.create(config=self.config, loop=loop)
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            # Clean shutdown logic
```

### 2. Weight Setter (`weight_setter.py`)

**Core Orchestrator**: Manages all validator operations
**Key Components**:
- Metagraph synchronization
- Miner discovery and management
- Prediction querying
- Reward calculation
- Weight setting
- State persistence

## Metagraph Synchronization

### Synchronization Process

#### 1. Initial Setup
```python
async def initialize(self):
    setup_bittensor_objects(self)
    self.hyperparameters = func_with_retry(self.subtensor.get_subnet_hyperparameters, netuid=self.config.netuid)
    self.available_uids = await self.get_available_uids()
    self.hotkeys = {uid: value for uid, value in enumerate(self.metagraph.hotkeys)}
```

#### 2. Periodic Resync (Every 10 minutes)
```python
async def resync_metagraph(self):
    # Sync with blockchain
    self.metagraph.sync(subtensor=self.subtensor)
    
    # Update available UIDs
    self.available_uids = await self.get_available_uids()
    
    # Process hotkey changes
    for uid, hotkey in enumerate(self.metagraph.hotkeys):
        new_miner = uid in new_uids and uid not in old_uids
        replaced_miner = self.hotkeys.get(uid, "") != hotkey
        
        if new_miner or replaced_miner:
            self.moving_average_scores[uid] = 0
            self.MinerHistory[uid] = MinerHistory(uid, timezone=self.timezone)
```

### Miner Discovery Logic

#### UID Availability Check
```python
def check_uid_availability(metagraph, uid: int, vpermit_tao_limit: Optional[int] = None) -> bool:
    # Filter non-serving axons
    if not metagraph.axons[uid].is_serving:
        return False
    
    # Note: Validator permit filtering temporarily disabled
    # Available otherwise
    return True
```

#### Available UIDs Discovery
```python
async def get_available_uids(self):
    miner_uids = []
    for uid in range(len(self.metagraph.S)):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        if uid_is_available:
            miner_uids.append(uid)
    return miner_uids
```

## Prediction Querying System

### Query Scheduling

#### Epoch Detection
```python
def is_query_time(prediction_interval: int, timestamp: str, tolerance: int = 120) -> bool:
    now = get_now()
    provided_timestamp = to_datetime(timestamp)
    
    # Check if enough time has passed since last query
    been_long_enough = elapsed_seconds(now, provided_timestamp) > tolerance
    
    # Check if we're at the beginning of a new epoch
    midnight = get_midnight()
    sec_since_open = elapsed_seconds(now, midnight)
    sec_since_epoch_start = sec_since_open % (prediction_interval * 60)
    beginning_of_epoch = sec_since_epoch_start < tolerance
    
    return been_long_enough and beginning_of_epoch
```

#### Query Execution
```python
async def query_miners(self):
    timestamp = to_str(round_to_interval(get_now(), interval_minutes=5))
    synapse = Challenge(timestamp=timestamp, assets=constants.SUPPORTED_ASSETS)
    
    responses = await self.dendrite.forward(
        axons=[self.metagraph.axons[uid] for uid in self.available_uids],
        synapse=synapse,
        deserialize=False,
        timeout=self.config.neuron.timeout,
    )
    return responses, timestamp
```

### Concurrent Processing

#### Configuration Parameters
- **Timeout**: 20 seconds per miner
- **Concurrent Forwards**: 1 (configurable)
- **Sample Size**: 50 miners (configurable)
- **Assets**: BTC, ETH, TAO (all queried simultaneously)

## Weight Setting System

### Weight Calculation Process

#### 1. Moving Average Update
```python
# Update moving average scores
for i, value in zip(self.available_uids, rewards):
    self.moving_average_scores[i] = (
        1 - self.config.neuron.moving_average_alpha
    ) * self.moving_average_scores[i] + self.config.neuron.moving_average_alpha * value
```

#### 2. Weight Conversion
```python
async def set_weights(self):
    # Check rate limit
    if self.blocks_since_last_update >= self.hyperparameters.weights_rate_limit:
        uids = array(self.available_uids)
        weights = [self.moving_average_scores[uid] for uid in self.available_uids]
        
        # Handle zero weights
        if sum(weights) == 0:
            weights = [1] * len(weights)
        
        # Convert to uint16 for blockchain
        uint_uids, uint_weights = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=uids, weights=array(weights)
        )
        
        # Submit to blockchain
        result, msg = self.subtensor.set_weights(
            netuid=self.config.netuid,
            wallet=self.wallet,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_inclusion=True,
            version_key=__spec_version__,
        )
```

### Rate Limiting

#### Block-based Rate Limiting
- **Rate Limit**: Set by subnet hyperparameters
- **Check**: `blocks_since_last_update >= weights_rate_limit`
- **Reset**: After successful weight setting

#### Moving Average Parameters
- **Alpha**: 0.05 (configurable)
- **Half-life**: ~14 iterations (2 hours at 5-minute intervals)
- **Purpose**: Smooth out short-term fluctuations

## State Management

### State Persistence

#### Save State
```python
def save_state(self):
    state_path = os.path.join(self.config.full_path, "state.pt")
    state = {
        "scores": self.scores,
        "moving_average_scores": self.moving_average_scores,
        "MinerHistory": self.MinerHistory,
    }
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
```

#### Load State
```python
def load_state(self) -> None:
    try:
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        self.scores = state["scores"]
        self.moving_average_scores = state["moving_average_scores"]
        self.MinerHistory = state.get("MinerHistory", {})
    except Exception as e:
        # Initialize default state
        self.scores = [0.0] * len(self.metagraph.S)
        self.moving_average_scores = {uid: 0 for uid in self.metagraph.uids}
        self.MinerHistory = {uid: MinerHistory(uid, timezone=self.timezone) for uid in range(len(self.metagraph.S))}
```

### Memory Management

#### Periodic Cleanup
```python
async def clear_old_miner_histories(self):
    """Periodically clears old predictions (>24h) from all MinerHistory objects."""
    for uid in self.MinerHistory:
        self.MinerHistory[uid].clear_old_predictions()
    self.save_state()
```

#### Cache Limits
- **Max Cache Size**: 100MB
- **Max Cache Rows**: 500,000
- **Cleanup Frequency**: Every hour

## Error Handling and Resilience

### Network Resilience

#### Subtensor Reconnection
```python
try:
    self.metagraph.sync(subtensor=self.subtensor)
except Exception as e:
    bt.logging.debug(f"Failed to sync metagraph: {e}")
    bt.logging.debug("Instantiating new subtensor")
    self.subtensor = bt.subtensor(config=self.config, network=self.config.subtensor.chain_endpoint)
    self.metagraph.sync(subtensor=self.subtensor)
```

#### Retry Mechanisms
```python
def func_with_retry(func: Callable, max_attempts: int = 3, delay: float = 1, *args, **kwargs) -> Any:
    attempt = 0
    while attempt < max_attempts:
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            attempt += 1
            if attempt == max_attempts:
                raise
            else:
                time.sleep(delay)
```

### Graceful Shutdown

#### Task Cancellation
```python
async def main(self):
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        # Clean shutdown: cancel all tasks except current task
        current_task = asyncio.current_task()
        tasks = [t for t in asyncio.all_tasks() if t is not current_task]
        
        # Cancel all other tasks
        for task in tasks:
            task.cancel()
        
        # Wait for all tasks to complete cancellation
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
```

## Monitoring and Logging

### WandB Integration

#### Setup
```python
if not self.config.wandb.off:
    setup_wandb(self)
```

#### Logging
```python
if not self.config.wandb.off:
    log_wandb(responses, rewards, self.available_uids, self.hotkeys, self.moving_average_scores)
```

### Performance Monitoring

#### Status Logging
```python
def print_info(self) -> None:
    weight_timing = self.hyperparameters.weights_rate_limit - self.blocks_since_last_update
    log = (
        "Validator | "
        f"UID:{self.my_uid} | "
        f"Block:{self.current_block} | "
        f"Stake:{self.metagraph.S[self.my_uid]:.3f} | "
        f"VTrust:{self.metagraph.Tv[self.my_uid]:.3f} | "
        f"Dividend:{self.metagraph.D[self.my_uid]:.3f} | "
        f"Emission:{self.metagraph.E[self.my_uid]:.3f} | "
        f"Setting weights in {weight_timing} blocks"
    )
```

## Configuration Parameters

### Network Configuration
- **Mainnet UID**: 55
- **Testnet UID**: 256
- **Timeout**: 20 seconds
- **Concurrent Forwards**: 1
- **Sample Size**: 50

### Performance Tuning
- **Moving Average Alpha**: 0.05
- **Resync Rate**: 600 seconds (10 minutes)
- **Cleanup Rate**: 3600 seconds (1 hour)
- **Print Cadence**: 12 seconds

### Security Settings
- **Vpermit TAO Limit**: 4096
- **Force Validator Permit**: True
- **Allow Non-registered**: False

## Operational Workflow

### 1. Startup Sequence
```
1. Load configuration
2. Initialize Bittensor objects
3. Sync metagraph
4. Load state from disk
5. Start background tasks
6. Begin main loop
```

### 2. Runtime Operations
```
Every 5 minutes:
├── Check if query time
├── Query all available miners
├── Calculate rewards
├── Update moving averages
└── Log to WandB

Every 10 minutes:
├── Sync metagraph
├── Update available UIDs
├── Handle new/removed miners
└── Save state

Every hour:
├── Clear old predictions
└── Save state

When rate limit reached:
├── Convert scores to weights
├── Submit to blockchain
└── Reset counter
```

### 3. Shutdown Sequence
```
1. Cancel all background tasks
2. Wait for task completion
3. Save final state
4. Clean up resources
```

This comprehensive validator system provides robust, high-performance orchestration of the Precog Subnet, with sophisticated state management, error handling, and monitoring capabilities.

