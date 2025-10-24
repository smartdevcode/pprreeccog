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
        
        # Use only Binance API to avoid rate limiting issues
        try:
            bt.logging.info(f"🔄 Making API call to Binance for: {assets}")
            prices = self._fetch_from_binance(assets)
            if prices:
                bt.logging.info(f"✅ Successfully fetched fresh prices from Binance: {prices}")
            else:
                bt.logging.warning("Binance returned empty prices, using fallback")
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
            
            # Fetch all prices in a single request to avoid rate limits
            symbols = []
            for asset in assets:
                symbol = asset_mapping.get(asset.lower(), f"{asset.upper()}USDT")
                symbols.append(symbol)
            
            # Single API call for all symbols
            url = f"https://api.binance.com/api/v3/ticker/price"
            params = {'symbols': json.dumps(symbols)}
            
            headers = {
                'User-Agent': 'Precog-Miner/1.0',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, params=params, timeout=10, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Parse the response
            if isinstance(data, list):
                for item in data:
                    if 'symbol' in item and 'price' in item:
                        symbol = item['symbol']
                        price = float(item['price'])
                        
                        # Map back to asset names
                        for asset in assets:
                            expected_symbol = asset_mapping.get(asset.lower(), f"{asset.upper()}USDT")
                            if symbol == expected_symbol:
                                prices[asset.lower()] = price
                                break
            else:
                # Single symbol response
                for asset in assets:
                    symbol = asset_mapping.get(asset.lower(), f"{asset.upper()}USDT")
                    if 'symbol' in data and data['symbol'] == symbol and 'price' in data:
                        prices[asset.lower()] = float(data['price'])
                    
            return prices
            
        except Exception as e:
            bt.logging.debug(f"Binance API failed: {e}")
            return {}
    
    def _get_fallback_prices(self, assets: list) -> Dict[str, float]:
        """Fallback prices when all APIs fail."""
        fallback_prices = {
            'btc': 110219.60,    # Current market price fallback
            'eth': 3907.73,      # Current market price fallback
            'tao': 388.35,       # Current market price fallback
            'tao_bittensor': 388.35
        }
        
        return {asset.lower(): fallback_prices.get(asset.lower(), 50000.0) for asset in assets}


# Global instance for easy access
price_fetcher = RealTimePriceFetcher()


def get_current_market_prices(assets: list) -> Dict[str, float]:
    """Get current market prices for the specified assets."""
    return price_fetcher.get_current_prices(assets)
