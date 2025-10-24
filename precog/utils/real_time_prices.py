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
        self.cache_timeout = 30  # Cache prices for 30 seconds
        self.last_update = 0
        
    def get_current_prices(self, assets: list) -> Dict[str, float]:
        """Get current prices for multiple assets from reliable sources."""
        current_time = time.time()
        
        # Use cache if recent enough
        if current_time - self.last_update < self.cache_timeout and self.cache:
            bt.logging.debug("Using cached prices")
            return self.cache
        
        prices = {}
        
        # Try multiple sources for reliability
        try:
            # Method 1: CoinGecko API (free, no API key)
            prices = self._fetch_from_coingecko(assets)
            if prices:
                bt.logging.info(f"✅ Fetched prices from CoinGecko: {prices}")
            else:
                # Method 2: CoinCap API (free, no API key)
                prices = self._fetch_from_coincap(assets)
                if prices:
                    bt.logging.info(f"✅ Fetched prices from CoinCap: {prices}")
                else:
                    # Method 3: Binance API (free, no API key)
                    prices = self._fetch_from_binance(assets)
                    if prices:
                        bt.logging.info(f"✅ Fetched prices from Binance: {prices}")
                        
        except Exception as e:
            bt.logging.error(f"Failed to fetch real-time prices: {e}")
            # Return fallback prices if all APIs fail
            prices = self._get_fallback_prices(assets)
            
        # Update cache
        self.cache = prices
        self.last_update = current_time
        
        return prices
    
    def _fetch_from_coingecko(self, assets: list) -> Dict[str, float]:
        """Fetch prices from CoinGecko API."""
        try:
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
            
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            prices = {}
            for asset in assets:
                coin_id = asset_mapping.get(asset.lower(), asset.lower())
                if coin_id in data and 'usd' in data[coin_id]:
                    prices[asset.lower()] = float(data[coin_id]['usd'])
                    
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
                    prices[asset.lower()] = float(data['data']['priceUsd'])
                    
            return prices
            
        except Exception as e:
            bt.logging.debug(f"CoinCap API failed: {e}")
            return {}
    
    def _fetch_from_binance(self, assets: list) -> Dict[str, float]:
        """Fetch prices from Binance API."""
        try:
            # Map assets to Binance symbols
            asset_mapping = {
                'btc': 'BTCUSDT',
                'eth': 'ETHUSDT',
                'tao': 'TAOUSDT',
                'tao_bittensor': 'TAOUSDT'
            }
            
            prices = {}
            for asset in assets:
                symbol = asset_mapping.get(asset.lower(), f"{asset.upper()}USDT")
                url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                data = response.json()
                
                if 'price' in data:
                    prices[asset.lower()] = float(data['price'])
                    
            return prices
            
        except Exception as e:
            bt.logging.debug(f"Binance API failed: {e}")
            return {}
    
    def _get_fallback_prices(self, assets: list) -> Dict[str, float]:
        """Fallback prices when all APIs fail."""
        fallback_prices = {
            'btc': 65000.0,      # Conservative fallback
            'eth': 3500.0,       # Conservative fallback
            'tao': 400.0,        # Conservative fallback
            'tao_bittensor': 400.0
        }
        
        return {asset.lower(): fallback_prices.get(asset.lower(), 50000.0) for asset in assets}


# Global instance for easy access
price_fetcher = RealTimePriceFetcher()


def get_current_market_prices(assets: list) -> Dict[str, float]:
    """Get current market prices for the specified assets."""
    return price_fetcher.get_current_prices(assets)
