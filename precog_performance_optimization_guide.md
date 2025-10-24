# Precog Subnet Performance Optimization Guide

## Overview

This guide provides comprehensive performance optimization techniques for the Precog Subnet, covering system-level optimizations, code-level improvements, and monitoring strategies.

## 🚀 **System-Level Optimizations**

### 1. Operating System Tuning

#### Linux Kernel Optimizations
```bash
# Increase file descriptor limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Network optimizations
echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.rmem_default = 65536' >> /etc/sysctl.conf
echo 'net.core.wmem_default = 65536' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 65536 134217728' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 134217728' >> /etc/sysctl.conf

# Apply changes
sysctl -p
```

#### CPU and Memory Optimizations
```bash
# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable swap for better performance (if you have enough RAM)
sudo swapoff -a

# Optimize memory allocation
echo 'vm.swappiness = 10' >> /etc/sysctl.conf
echo 'vm.vfs_cache_pressure = 50' >> /etc/sysctl.conf
```

### 2. Python Environment Optimization

#### Python Performance Settings
```bash
# Set Python optimization flags
export PYTHONOPTIMIZE=1
export PYTHONUNBUFFERED=1

# Use optimized Python builds
# Consider using PyPy for CPU-intensive tasks
pip install pypy3
```

#### Virtual Environment Optimization
```bash
# Use faster package manager
pip install --upgrade pip
pip install wheel setuptools

# Optimize package installation
pip install --no-cache-dir --compile package_name
```

## ⚡ **Code-Level Optimizations**

### 1. Data Processing Optimizations

#### Efficient Data Handling
```python
# Use pandas optimizations
import pandas as pd
import numpy as np

# Optimize data types
def optimize_dtypes(df):
    """Optimize DataFrame dtypes for memory efficiency."""
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to category if low cardinality
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        elif df[col].dtype == 'int64':
            # Downcast integers
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
                elif df[col].max() < 4294967295:
                    df[col] = df[col].astype('uint32')
        elif df[col].dtype == 'float64':
            # Downcast floats
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

# Use chunked processing for large datasets
def process_large_data(data, chunk_size=10000):
    """Process large datasets in chunks."""
    results = []
    for chunk in pd.read_csv(data, chunksize=chunk_size):
        # Process chunk
        processed_chunk = process_chunk(chunk)
        results.append(processed_chunk)
    
    return pd.concat(results, ignore_index=True)
```

#### Caching Strategies
```python
from functools import lru_cache
import hashlib
import pickle
import os

class SmartCache:
    """Intelligent caching system for miner predictions."""
    
    def __init__(self, cache_dir="cache", max_size_mb=100):
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, timestamp, assets):
        """Generate cache key for request."""
        key_data = f"{timestamp}_{sorted(assets)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, timestamp, assets):
        """Get cached prediction."""
        cache_key = self._get_cache_key(timestamp, assets)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                os.remove(cache_file)
        
        return None
    
    def set(self, timestamp, assets, predictions, intervals, ttl=300):
        """Cache prediction with TTL."""
        cache_key = self._get_cache_key(timestamp, assets)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        cache_data = {
            'timestamp': timestamp,
            'assets': assets,
            'predictions': predictions,
            'intervals': intervals,
            'created_at': time.time(),
            'ttl': ttl
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            bt.logging.error(f"Cache write failed: {e}")
    
    def cleanup(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.cache_dir, filename)
                try:
                    with open(filepath, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    if current_time - cache_data['created_at'] > cache_data['ttl']:
                        os.remove(filepath)
                except Exception:
                    os.remove(filepath)
```

### 2. Async and Concurrent Processing

#### Optimized Async Patterns
```python
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time

class OptimizedMiner:
    """Performance-optimized miner with async processing."""
    
    def __init__(self):
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.cache = SmartCache()
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        self.executor.shutdown(wait=True)
    
    async def fetch_data_async(self, assets, start_time, end_time):
        """Fetch data asynchronously."""
        tasks = []
        for asset in assets:
            task = self._fetch_asset_data(asset, start_time, end_time)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def _fetch_asset_data(self, asset, start_time, end_time):
        """Fetch data for single asset."""
        # Check cache first
        cached_data = self.cache.get(start_time, [asset])
        if cached_data:
            return cached_data
        
        # Fetch from API
        data = await self._api_call(asset, start_time, end_time)
        
        # Cache result
        self.cache.set(start_time, [asset], data, ttl=300)
        
        return data
    
    async def _api_call(self, asset, start_time, end_time):
        """Make API call with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Your API call logic here
                return await self._make_request(asset, start_time, end_time)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def _make_request(self, asset, start_time, end_time):
        """Make actual HTTP request."""
        # Implementation depends on your data source
        pass
```

#### Parallel Processing
```python
from multiprocessing import Pool, cpu_count
import numpy as np

class ParallelProcessor:
    """Parallel processing for CPU-intensive tasks."""
    
    def __init__(self, n_processes=None):
        self.n_processes = n_processes or min(cpu_count(), 4)
        self.pool = None
    
    def __enter__(self):
        self.pool = Pool(self.n_processes)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            self.pool.close()
            self.pool.join()
    
    def parallel_predict(self, data_chunks):
        """Process data chunks in parallel."""
        results = self.pool.map(self._process_chunk, data_chunks)
        return results
    
    def _process_chunk(self, chunk):
        """Process single data chunk."""
        # Your processing logic here
        return processed_chunk

# Usage example
def optimize_prediction_processing(data):
    """Optimize prediction processing with parallel execution."""
    # Split data into chunks
    chunk_size = len(data) // 4
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    
    with ParallelProcessor() as processor:
        results = processor.parallel_predict(chunks)
    
    return np.concatenate(results)
```

### 3. Memory Optimization

#### Memory-Efficient Data Structures
```python
import gc
from typing import Dict, List
import weakref

class MemoryEfficientMiner:
    """Memory-optimized miner implementation."""
    
    def __init__(self):
        self.data_cache = {}
        self.prediction_cache = weakref.WeakValueDictionary()
        self.max_cache_size = 1000
    
    def optimize_memory_usage(self):
        """Optimize memory usage by cleaning up unused data."""
        # Force garbage collection
        gc.collect()
        
        # Clear old cache entries
        if len(self.data_cache) > self.max_cache_size:
            # Remove oldest entries
            oldest_keys = list(self.data_cache.keys())[:len(self.data_cache) - self.max_cache_size]
            for key in oldest_keys:
                del self.data_cache[key]
        
        # Clear prediction cache
        self.prediction_cache.clear()
    
    def process_data_efficiently(self, data):
        """Process data with minimal memory allocation."""
        # Use generators instead of lists
        def data_generator():
            for row in data.itertuples():
                yield process_row(row)
        
        # Process in chunks to avoid memory spikes
        chunk_size = 1000
        results = []
        
        for chunk in self._chunk_data(data, chunk_size):
            processed_chunk = self._process_chunk(chunk)
            results.append(processed_chunk)
            
            # Clean up chunk
            del chunk
            gc.collect()
        
        return results
    
    def _chunk_data(self, data, chunk_size):
        """Split data into chunks."""
        for i in range(0, len(data), chunk_size):
            yield data.iloc[i:i+chunk_size]
    
    def _process_chunk(self, chunk):
        """Process single chunk."""
        # Your processing logic here
        return processed_chunk
```

## 📊 **Performance Monitoring**

### 1. Real-Time Performance Metrics

#### System Metrics Monitor
```python
import psutil
import time
import threading
from collections import deque

class PerformanceMonitor:
    """Real-time performance monitoring."""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics = {
            'cpu': deque(maxlen=window_size),
            'memory': deque(maxlen=window_size),
            'disk_io': deque(maxlen=window_size),
            'network_io': deque(maxlen=window_size)
        }
        self.running = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics['cpu'].append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics['memory'].append(memory.percent)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.metrics['disk_io'].append(disk_io.read_bytes + disk_io.write_bytes)
                
                # Network I/O
                net_io = psutil.net_io_counters()
                if net_io:
                    self.metrics['network_io'].append(net_io.bytes_sent + net_io.bytes_recv)
                
                # Log warnings
                self._check_thresholds()
                
            except Exception as e:
                bt.logging.error(f"Monitoring error: {e}")
            
            time.sleep(5)  # Check every 5 seconds
    
    def _check_thresholds(self):
        """Check performance thresholds and log warnings."""
        if len(self.metrics['cpu']) > 0:
            avg_cpu = sum(self.metrics['cpu']) / len(self.metrics['cpu'])
            if avg_cpu > 80:
                bt.logging.warning(f"High CPU usage: {avg_cpu:.1f}%")
        
        if len(self.metrics['memory']) > 0:
            avg_memory = sum(self.metrics['memory']) / len(self.metrics['memory'])
            if avg_memory > 80:
                bt.logging.warning(f"High memory usage: {avg_memory:.1f}%")
    
    def get_performance_summary(self):
        """Get performance summary."""
        summary = {}
        for metric, values in self.metrics.items():
            if values:
                summary[metric] = {
                    'current': values[-1],
                    'average': sum(values) / len(values),
                    'max': max(values),
                    'min': min(values)
                }
        return summary
```

#### Application-Specific Metrics
```python
import time
from functools import wraps

class MinerMetrics:
    """Miner-specific performance metrics."""
    
    def __init__(self):
        self.prediction_times = []
        self.api_call_times = []
        self.cache_hit_rate = 0.0
        self.error_count = 0
        self.success_count = 0
    
    def time_prediction(self, func):
        """Decorator to time prediction functions."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                self.success_count += 1
                return result
            except Exception as e:
                self.error_count += 1
                raise
            finally:
                end_time = time.perf_counter()
                self.prediction_times.append(end_time - start_time)
        return wrapper
    
    def time_api_call(self, func):
        """Decorator to time API calls."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            end_time = time.perf_counter()
            self.api_call_times.append(end_time - start_time)
            return result
        return wrapper
    
    def get_metrics(self):
        """Get current metrics."""
        return {
            'avg_prediction_time': np.mean(self.prediction_times) if self.prediction_times else 0,
            'avg_api_call_time': np.mean(self.api_call_times) if self.api_call_times else 0,
            'cache_hit_rate': self.cache_hit_rate,
            'error_rate': self.error_count / (self.success_count + self.error_count) if (self.success_count + self.error_count) > 0 else 0,
            'total_predictions': self.success_count + self.error_count
        }
```

### 2. Automated Performance Optimization

#### Dynamic Resource Allocation
```python
class AdaptiveResourceManager:
    """Dynamically adjust resources based on performance."""
    
    def __init__(self):
        self.performance_history = deque(maxlen=100)
        self.current_workers = 4
        self.max_workers = 8
        self.min_workers = 2
    
    def adjust_workers(self, current_performance):
        """Adjust worker count based on performance."""
        self.performance_history.append(current_performance)
        
        if len(self.performance_history) < 10:
            return  # Need more data
        
        recent_avg = sum(list(self.performance_history)[-10:]) / 10
        
        if recent_avg > 0.8 and self.current_workers < self.max_workers:
            # High load, increase workers
            self.current_workers = min(self.current_workers + 1, self.max_workers)
            bt.logging.info(f"Increased workers to {self.current_workers}")
        
        elif recent_avg < 0.3 and self.current_workers > self.min_workers:
            # Low load, decrease workers
            self.current_workers = max(self.current_workers - 1, self.min_workers)
            bt.logging.info(f"Decreased workers to {self.current_workers}")
```

#### Cache Optimization
```python
class AdaptiveCache:
    """Adaptive caching with automatic optimization."""
    
    def __init__(self, initial_size=1000):
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.max_size = initial_size
    
    def get(self, key):
        """Get value from cache."""
        if key in self.cache:
            self.hit_count += 1
            self.access_times[key] = time.time()
            return self.cache[key]
        else:
            self.miss_count += 1
            return None
    
    def set(self, key, value):
        """Set value in cache."""
        if len(self.cache) >= self.max_size:
            self._evict_least_recent()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def _evict_least_recent(self):
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def optimize_size(self):
        """Optimize cache size based on hit rate."""
        total_requests = self.hit_count + self.miss_count
        if total_requests < 100:
            return  # Need more data
        
        hit_rate = self.hit_count / total_requests
        
        if hit_rate > 0.8 and self.max_size < 10000:
            # High hit rate, increase cache size
            self.max_size = min(self.max_size * 2, 10000)
            bt.logging.info(f"Increased cache size to {self.max_size}")
        
        elif hit_rate < 0.5 and self.max_size > 100:
            # Low hit rate, decrease cache size
            self.max_size = max(self.max_size // 2, 100)
            bt.logging.info(f"Decreased cache size to {self.max_size}")
```

## 🔧 **Production Optimization Checklist**

### 1. Pre-Deployment Optimization
- [ ] Enable Python optimizations (`PYTHONOPTIMIZE=1`)
- [ ] Configure system limits (file descriptors, memory)
- [ ] Set up monitoring and alerting
- [ ] Implement caching strategies
- [ ] Optimize database connections
- [ ] Configure load balancing (if applicable)

### 2. Runtime Optimization
- [ ] Monitor CPU and memory usage
- [ ] Track prediction latency
- [ ] Monitor cache hit rates
- [ ] Log performance metrics
- [ ] Implement automatic scaling
- [ ] Regular cleanup of old data

### 3. Maintenance Optimization
- [ ] Regular performance reviews
- [ ] Database optimization
- [ ] Log rotation and cleanup
- [ ] Security updates
- [ ] Backup and recovery testing

This comprehensive performance optimization guide provides all the tools and techniques needed to maximize the performance of your Precog Subnet implementation.

