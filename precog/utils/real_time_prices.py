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
        self.cache_timeout = 0  # Disable caching - always fetch fresh prices
        self.last_update = 0
        
    def get_current_prices(self, assets: list) -> Dict[str, float]:
        """Get current prices for multiple assets from reliable sources."""
        current_time = time.time()
        
        # Always fetch fresh prices for ETH and TAO to ensure variation
        if 'eth' in [asset.lower() for asset in assets]:
            bt.logging.debug("Forcing fresh price fetch for ETH")
            self.cache.pop('eth', None)  # Clear ETH from cache
        if 'tao' in [asset.lower() for asset in assets] or 'tao_bittensor' in [asset.lower() for asset in assets]:
            bt.logging.debug("Forcing fresh price fetch for TAO")
            self.cache.pop('tao', None)  # Clear TAO from cache
            self.cache.pop('tao_bittensor', None)  # Clear TAO_BITTENSOR from cache
        
        # Always fetch fresh prices for better accuracy (disable caching)
        # This ensures we get the most current market prices for predictions
        bt.logging.debug("Fetching fresh prices for maximum accuracy")
        
        prices = {}
        
        # Try multiple sources for reliability
        try:
            # Method 1: CoinGecko API (free, no API key)
            bt.logging.debug(f"Attempting to fetch prices from CoinGecko for: {assets}")
            prices = self._fetch_from_coingecko(assets)
            if prices:
                bt.logging.info(f"✅ Fetched prices from CoinGecko: {prices}")
            else:
                bt.logging.debug("CoinGecko returned empty prices, trying CoinCap")
                # Method 2: CoinCap API (free, no API key)
                prices = self._fetch_from_coincap(assets)
                if prices:
                    bt.logging.info(f"✅ Fetched prices from CoinCap: {prices}")
                else:
                    bt.logging.debug("CoinCap returned empty prices, trying Binance")
                    # Method 3: Binance API (free, no API key)
                    prices = self._fetch_from_binance(assets)
                    if prices:
                        bt.logging.info(f"✅ Fetched prices from Binance: {prices}")
                    else:
                        bt.logging.warning("All APIs returned empty prices, using fallback")
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
                    base_price = float(data['price'])
                    # Use actual real-time price without artificial variation
                    prices[asset.lower()] = base_price
                    
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
