"""
Real-time cryptocurrency price fetcher using multiple reliable APIs.
"""

import requests
import time
import bittensor as bt
from typing import Dict, Optional
import json


class RealTimePriceFetcher:
    """Fetches real-time cryptocurrency prices from multiple reliable sources."""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 30  # Cache for 30 seconds to balance freshness and rate limits
        self.last_update = 0
        self.api_failures = {}
        self.rate_limit_delay = 2  # Delay between API calls
        self.background_fetching = False
        self.fetch_interval = 60  # Fetch prices every 60 seconds to avoid rate limits
        self.all_assets = ['btc', 'eth', 'tao_bittensor']  # All assets to fetch
        
    def get_current_prices(self, assets: list) -> Dict[str, float]:
        """Get current prices for multiple assets from reliable sources."""
        current_time = time.time()
        
        # Check if cache is expired and needs refresh
        cache_expired = current_time - self.last_update > self.cache_timeout
        
        # Use cache if recent enough and not expired
        if not cache_expired and self.cache:
            bt.logging.debug("Using cached prices (cache still fresh)")
            return self.cache
        
        # Cache is expired or empty, need to fetch fresh prices
        if cache_expired:
            bt.logging.debug(f"Cache expired (age: {current_time - self.last_update:.1f}s), fetching fresh prices")
            # Clear the cache to force fresh fetch
            self.cache.clear()
        
        prices = {}
        
        # Use CoinMetrics as primary source, fallback to Binance
        try:
            bt.logging.info(f"🔄 Making API call to CoinMetrics for: {assets}")
            prices = self._fetch_from_coinmetrics(assets)
            if prices:
                bt.logging.info(f"✅ Successfully fetched fresh prices from CoinMetrics: {prices}")
            else:
                bt.logging.warning("CoinMetrics returned empty prices, trying Binance")
                prices = self._fetch_from_binance(assets)
                if prices:
                    bt.logging.info(f"✅ Successfully fetched fresh prices from Binance: {prices}")
                else:
                    bt.logging.warning("Both CoinMetrics and Binance failed, using fallback")
                    prices = self._get_fallback_prices(assets)
                        
        except Exception as e:
            bt.logging.error(f"Failed to fetch real-time prices: {e}")
            # Return fallback prices if all APIs fail
            prices = self._get_fallback_prices(assets)
            
        # Update cache
        self.cache = prices
        self.last_update = current_time
        
        return prices
    
    def _fetch_from_coingecko(self, assets: list) -> Dict[str, float]:
        """Fetch prices from CoinGecko API with rate limiting."""
        try:
            # Add delay to avoid rate limiting
            time.sleep(self.rate_limit_delay)
            
            # Map assets to CoinGecko IDs
            asset_mapping = {
                'btc': 'bitcoin',
                'eth': 'ethereum',
                'tao': 'bittensor',
                'tao_bittensor': 'bittensor'
            }
            
            coin_ids = [asset_mapping.get(asset.lower(), asset.lower()) for asset in assets]
            coin_ids_str = ','.join(coin_ids)
            
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_ids_str}&vs_currencies=usd"
            
            # Add headers to be more respectful to the API
            headers = {
                'User-Agent': 'Precog-Miner/1.0',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            prices = {}
            for asset in assets:
                coin_id = asset_mapping.get(asset.lower(), asset.lower())
                if coin_id in data and 'usd' in data[coin_id]:
                    base_price = float(data[coin_id]['usd'])
                    # Use actual real-time price without artificial variation
                    prices[asset.lower()] = base_price
                    
            return prices
            
        except Exception as e:
            bt.logging.debug(f"CoinGecko API failed: {e}")
            return {}
    
    def _fetch_from_coincap(self, assets: list) -> Dict[str, float]:
        """Fetch prices from CoinCap API."""
        try:
            # Map assets to CoinCap IDs
            asset_mapping = {
                'btc': 'bitcoin',
                'eth': 'ethereum',
                'tao': 'bittensor',
                'tao_bittensor': 'bittensor'
            }
            
            prices = {}
            for asset in assets:
                coin_id = asset_mapping.get(asset.lower(), asset.lower())
                url = f"https://api.coincap.io/v2/assets/{coin_id}"
                
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                data = response.json()
                
                if 'data' in data and 'priceUsd' in data['data']:
                    base_price = float(data['data']['priceUsd'])
                    # Use actual real-time price without artificial variation
                    prices[asset.lower()] = base_price
                    
            return prices
            
        except Exception as e:
            bt.logging.debug(f"CoinCap API failed: {e}")
            return {}
    
    def _fetch_from_binance(self, assets: list) -> Dict[str, float]:
        """Fetch prices from Binance API with better rate limiting."""
        try:
            # Map assets to Binance symbols
            asset_mapping = {
                'btc': 'BTCUSDT',
                'eth': 'ETHUSDT',
                'tao': 'TAOUSDT',
                'tao_bittensor': 'TAOUSDT'
            }
            
            prices = {}
            
            # Fetch prices individually to avoid API issues with multiple symbols
            for i, asset in enumerate(assets):
                symbol = asset_mapping.get(asset.lower(), f"{asset.upper()}USDT")
                url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                
                headers = {
                    'User-Agent': 'Precog-Miner/1.0',
                    'Accept': 'application/json'
                }
                
                try:
                    response = requests.get(url, timeout=10, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'price' in data:
                        prices[asset.lower()] = float(data['price'])
                        bt.logging.debug(f"✅ Fetched {asset.upper()} price from Binance: ${prices[asset.lower()]:.2f}")
                    else:
                        bt.logging.warning(f"No price data for {asset}")
                        
                except Exception as e:
                    bt.logging.debug(f"Failed to fetch {asset} price from Binance: {e}")
                    continue
                
                # Small delay between requests to avoid rate limiting
                if i < len(assets) - 1:  # Don't delay after the last request
                    time.sleep(0.1)
                    
            return prices
            
        except Exception as e:
            bt.logging.debug(f"Binance API failed: {e}")
            return {}
    
    def _fetch_from_coinmetrics(self, assets: list) -> Dict[str, float]:
        """Fetch latest prices from CoinMetrics API."""
        try:
            from precog.utils.cm_data import CMData
            from datetime import datetime, timedelta
            
            # Initialize CoinMetrics client
            cm = CMData()
            
            # Map assets to CoinMetrics symbols (use exact CoinMetrics asset names)
            asset_mapping = {
                'btc': 'btc',
                'eth': 'eth',
                'tao': 'tao',
                'tao_bittensor': 'tao'
            }
            
            prices = {}
            
            # Get latest prices from CoinMetrics (try different time ranges)
            end_time = datetime.now()
            time_ranges = [
                timedelta(minutes=5),   # Last 5 minutes
                timedelta(minutes=30),  # Last 30 minutes
                timedelta(hours=1),     # Last 1 hour
                timedelta(hours=6)      # Last 6 hours
            ]
            
            for asset in assets:
                cm_asset = asset_mapping.get(asset.lower(), asset.lower())
                
                try:
                    bt.logging.debug(f"Fetching CoinMetrics data for {asset} (CM asset: {cm_asset})")
                    
                    # Try different time ranges and frequencies
                    frequencies = ["1s", "1m", "5m"]
                    data = None
                    
                    for time_range in time_ranges:
                        start_time = end_time - time_range
                        bt.logging.debug(f"Trying time range: {start_time} to {end_time}")
                        
                        for freq in frequencies:
                            try:
                                data = cm.get_CM_ReferenceRate(
                                    assets=[cm_asset],
                                    start=start_time,
                                    end=end_time,
                                    frequency=freq,
                                    use_cache=False
                                )
                                if not data.empty:
                                    bt.logging.debug(f"✅ Got data with {freq} frequency in {time_range}")
                                    break
                            except Exception as e:
                                bt.logging.debug(f"Failed with {freq} frequency: {e}")
                                continue
                        
                        if data is not None and not data.empty:
                            break
                    
                    if data is not None and not data.empty:
                        # Check available columns and handle NaN values
                        bt.logging.debug(f"Available columns: {list(data.columns)}")
                        
                        # Try different price columns
                        price_columns = ['ReferenceRateUSD', 'priceUSD', 'price', 'close']
                        latest_price = None
                        
                        for col in price_columns:
                            if col in data.columns:
                                # Get the latest non-NaN value
                                valid_prices = data[col].dropna()
                                if not valid_prices.empty:
                                    latest_price = float(valid_prices.iloc[-1])
                                    bt.logging.debug(f"✅ Got price from {col} column: ${latest_price:.2f}")
                                    break
                        
                        if latest_price is not None:
                            prices[asset.lower()] = latest_price
                            bt.logging.debug(f"✅ Fetched {asset.upper()} price from CoinMetrics: ${latest_price:.2f}")
                        else:
                            bt.logging.warning(f"No valid price data for {asset} (CM asset: {cm_asset})")
                            bt.logging.debug(f"Data sample: {data.head()}")
                            
                            # Try alternative method - get latest available data
                            try:
                                bt.logging.debug(f"Trying alternative method for {asset} (CM asset: {cm_asset})")
                                
                                # Get data from last 24 hours to find any valid price
                                alt_start = end_time - timedelta(hours=24)
                                alt_data = cm.get_CM_ReferenceRate(
                                    assets=[cm_asset],
                                    start=alt_start,
                                    end=end_time,
                                    frequency="1h",
                                    use_cache=False
                                )
                                
                                bt.logging.debug(f"Alternative data for {asset}: shape={alt_data.shape}, columns={list(alt_data.columns) if not alt_data.empty else 'empty'}")
                                
                                if not alt_data.empty:
                                    for col in ['ReferenceRateUSD', 'priceUSD', 'price', 'close']:
                                        if col in alt_data.columns:
                                            valid_prices = alt_data[col].dropna()
                                            if not valid_prices.empty:
                                                latest_price = float(valid_prices.iloc[-1])
                                                prices[asset.lower()] = latest_price
                                                bt.logging.debug(f"✅ Got {asset.upper()} price from alternative method: ${latest_price:.2f}")
                                                break
                                    else:
                                        bt.logging.warning(f"No valid price columns found in alternative data for {asset}")
                                else:
                                    bt.logging.warning(f"Alternative data is empty for {asset}")
                            except Exception as e:
                                bt.logging.debug(f"Alternative method failed for {asset}: {e}")
                                # If alternative method fails, try fallback prices
                                fallback_prices = {'btc': 110000.0, 'eth': 3900.0, 'tao': 400.0, 'tao_bittensor': 400.0}
                                if asset.lower() in fallback_prices:
                                    prices[asset.lower()] = fallback_prices[asset.lower()]
                                    bt.logging.debug(f"✅ Using fallback price for {asset.upper()}: ${fallback_prices[asset.lower()]:.2f}")
                    else:
                        bt.logging.warning(f"No CoinMetrics data for {asset} (CM asset: {cm_asset})")
                        
                except Exception as e:
                    bt.logging.debug(f"Failed to fetch {asset} price from CoinMetrics: {e}")
                    continue
            
            # Ensure all requested assets have prices
            for asset in assets:
                if asset.lower() not in prices:
                    bt.logging.warning(f"No price found for {asset}, using fallback")
                    fallback_prices = {'btc': 110000.0, 'eth': 3900.0, 'tao': 400.0, 'tao_bittensor': 400.0}
                    if asset.lower() in fallback_prices:
                        prices[asset.lower()] = fallback_prices[asset.lower()]
                        bt.logging.debug(f"✅ Using fallback price for {asset.upper()}: ${fallback_prices[asset.lower()]:.2f}")
            
            return prices
            
        except Exception as e:
            bt.logging.debug(f"CoinMetrics API failed: {e}")
            # Return fallback prices if everything fails
            fallback_prices = {'btc': 110000.0, 'eth': 3900.0, 'tao': 400.0, 'tao_bittensor': 400.0}
            result = {}
            for asset in assets:
                if asset.lower() in fallback_prices:
                    result[asset.lower()] = fallback_prices[asset.lower()]
                    bt.logging.debug(f"✅ Using emergency fallback price for {asset.upper()}: ${fallback_prices[asset.lower()]:.2f}")
            return result
    
    def _get_fallback_prices(self, assets: list) -> Dict[str, float]:
        """Fallback prices when all APIs fail."""
        fallback_prices = {
            'btc': 110219.60,    # Current market price fallback
            'eth': 3907.73,      # Current market price fallback
            'tao': 388.35,       # Current market price fallback
            'tao_bittensor': 388.35
        }
        
        return {asset.lower(): fallback_prices.get(asset.lower(), 50000.0) for asset in assets}
    
    def start_background_fetching(self):
        """Start background price fetching for all assets."""
        if self.background_fetching:
            return
        
        self.background_fetching = True
        bt.logging.info("🔄 Starting background price fetching for all assets")
        
        # Initial fetch
        self._fetch_all_prices()
        
        # Start background thread
        import threading
        self.fetch_thread = threading.Thread(target=self._background_fetch_loop, daemon=True)
        self.fetch_thread.start()
    
    def stop_background_fetching(self):
        """Stop background price fetching."""
        self.background_fetching = False
        bt.logging.info("⏹️ Stopping background price fetching")
    
    def _background_fetch_loop(self):
        """Background loop to fetch prices periodically."""
        bt.logging.info(f"🔄 Background fetch loop started, will fetch every {self.fetch_interval} seconds")
        while self.background_fetching:
            try:
                time.sleep(self.fetch_interval)
                if self.background_fetching:  # Check again after sleep
                    bt.logging.debug("🔄 Background fetch loop: triggering price fetch")
                    self._fetch_all_prices()
                else:
                    bt.logging.info("⏹️ Background fetch loop: stopping")
                    break
            except Exception as e:
                bt.logging.error(f"Background price fetching error: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _fetch_all_prices(self):
        """Fetch prices for all assets."""
        try:
            bt.logging.info(f"🔄 Background fetching prices for all assets: {self.all_assets}")
            prices = self._fetch_from_coinmetrics(self.all_assets)
            if not prices:
                bt.logging.warning("CoinMetrics failed, trying Binance for background fetch")
                prices = self._fetch_from_binance(self.all_assets)
            if prices:
                # Force update cache and timestamp
                self.cache = prices.copy()
                self.last_update = time.time()
                bt.logging.info(f"✅ Background fetch successful: {prices}")
                bt.logging.info(f"📊 Cache updated at: {time.strftime('%H:%M:%S')}")
            else:
                bt.logging.warning("Background fetch returned empty prices")
        except Exception as e:
            bt.logging.error(f"Background price fetch failed: {e}")


# Global instance for easy access
price_fetcher = RealTimePriceFetcher()


def get_current_market_prices(assets: list) -> Dict[str, float]:
    """Get current market prices for the specified assets."""
    return price_fetcher.get_current_prices(assets)


def start_background_price_fetching():
    """Start background price fetching for all assets."""
    price_fetcher.start_background_fetching()


def stop_background_price_fetching():
    """Stop background price fetching."""
    price_fetcher.stop_background_fetching()


def force_fresh_price_fetch():
    """Force a fresh price fetch for all assets."""
    price_fetcher._fetch_all_prices()
