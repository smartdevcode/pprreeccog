"""
Technical Analysis miner using advanced chart patterns and indicators.
This miner implements comprehensive technical analysis for cryptocurrency prediction.
"""

import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import bittensor as bt
from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import get_before, to_datetime, to_str


class TechnicalAnalysisMiner:
    """Advanced technical analysis miner with multiple indicators."""
    
    def __init__(self):
        self.lookback_periods = [5, 10, 20, 50, 100]
        self.confidence_levels = [0.95, 0.99]
    
    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return prices.rolling(window=period).mean()
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return prices.ewm(span=period).mean()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)."""
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        sma = self.calculate_sma(prices, period)
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def detect_support_resistance(self, prices: pd.Series, window: int = 20) -> Tuple[float, float]:
        """Detect support and resistance levels."""
        # Find local minima and maxima
        rolling_min = prices.rolling(window=window, center=True).min()
        rolling_max = prices.rolling(window=window, center=True).max()
        
        # Support level (local minima)
        support_candidates = prices[rolling_min == prices].dropna()
        support_level = support_candidates.mean() if not support_candidates.empty else prices.min()
        
        # Resistance level (local maxima)
        resistance_candidates = prices[rolling_max == prices].dropna()
        resistance_level = resistance_candidates.mean() if not resistance_candidates.empty else prices.max()
        
        return support_level, resistance_level
    
    def analyze_trend(self, prices: pd.Series) -> Dict[str, float]:
        """Comprehensive trend analysis."""
        if len(prices) < 50:
            return {"trend": "neutral", "strength": 0.0, "direction": 0.0}
        
        # Multiple timeframe analysis
        sma_20 = self.calculate_sma(prices, 20)
        sma_50 = self.calculate_sma(prices, 50)
        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)
        
        # Trend direction
        current_price = prices.iloc[-1]
        sma_20_current = sma_20.iloc[-1]
        sma_50_current = sma_50.iloc[-1]
        
        # Bullish signals
        bullish_signals = 0
        if current_price > sma_20_current:
            bullish_signals += 1
        if sma_20_current > sma_50_current:
            bullish_signals += 1
        if ema_12.iloc[-1] > ema_26.iloc[-1]:
            bullish_signals += 1
        
        # Bearish signals
        bearish_signals = 0
        if current_price < sma_20_current:
            bearish_signals += 1
        if sma_20_current < sma_50_current:
            bearish_signals += 1
        if ema_12.iloc[-1] < ema_26.iloc[-1]:
            bearish_signals += 1
        
        # Determine trend
        if bullish_signals > bearish_signals:
            trend = "bullish"
            strength = bullish_signals / 3.0
            direction = 1.0
        elif bearish_signals > bullish_signals:
            trend = "bearish"
            strength = bearish_signals / 3.0
            direction = -1.0
        else:
            trend = "neutral"
            strength = 0.5
            direction = 0.0
        
        return {"trend": trend, "strength": strength, "direction": direction}
    
    def calculate_volatility_forecast(self, prices: pd.Series, horizon: int = 1) -> float:
        """Calculate volatility forecast using GARCH-like approach."""
        returns = prices.pct_change().dropna()
        
        if len(returns) < 20:
            return 0.02  # Default 2% volatility
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=20).std()
        recent_vol = rolling_vol.iloc[-1]
        
        # Volatility clustering (high vol tends to persist)
        vol_trend = rolling_vol.diff().iloc[-5:].mean()
        
        # Forecast volatility
        forecast_vol = recent_vol + (vol_trend * horizon)
        
        # Cap volatility between 0.5% and 10%
        forecast_vol = max(0.005, min(0.10, forecast_vol))
        
        return forecast_vol
    
    def generate_prediction(self, data: pd.DataFrame) -> Tuple[float, float, float]:
        """Generate price prediction using technical analysis."""
        if data.empty or len(data) < 50:
            latest_price = float(data['ReferenceRateUSD'].iloc[-1]) if not data.empty else 50000.0
            return latest_price, latest_price * 0.95, latest_price * 1.05
        
        prices = data['ReferenceRateUSD']
        current_price = float(prices.iloc[-1])
        
        # Technical indicators
        rsi = self.calculate_rsi(prices, 14)
        macd, signal, histogram = self.calculate_macd(prices)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(prices, 20)
        
        # Trend analysis
        trend_analysis = self.analyze_trend(prices)
        
        # Support and resistance
        support, resistance = self.detect_support_resistance(prices)
        
        # Volatility forecast
        volatility = self.calculate_volatility_forecast(prices)
        
        # Prediction logic based on technical signals
        prediction_adjustment = 0.0
        
        # RSI signals
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50
        if current_rsi < 30:  # Oversold
            prediction_adjustment += 0.02  # Bullish
        elif current_rsi > 70:  # Overbought
            prediction_adjustment -= 0.02  # Bearish
        
        # MACD signals
        current_macd = macd.iloc[-1] if not macd.empty else 0
        current_signal = signal.iloc[-1] if not signal.empty else 0
        if current_macd > current_signal:  # Bullish MACD
            prediction_adjustment += 0.01
        else:  # Bearish MACD
            prediction_adjustment -= 0.01
        
        # Bollinger Bands position
        current_bb_upper = bb_upper.iloc[-1] if not bb_upper.empty else current_price * 1.02
        current_bb_lower = bb_lower.iloc[-1] if not bb_lower.empty else current_price * 0.98
        bb_position = (current_price - current_bb_lower) / (current_bb_upper - current_bb_lower)
        
        if bb_position < 0.2:  # Near lower band
            prediction_adjustment += 0.015
        elif bb_position > 0.8:  # Near upper band
            prediction_adjustment -= 0.015
        
        # Trend-based adjustment
        if trend_analysis["trend"] == "bullish":
            prediction_adjustment += trend_analysis["strength"] * 0.02
        elif trend_analysis["trend"] == "bearish":
            prediction_adjustment -= trend_analysis["strength"] * 0.02
        
        # Support/Resistance adjustment
        if current_price < support * 1.01:  # Near support
            prediction_adjustment += 0.01
        elif current_price > resistance * 0.99:  # Near resistance
            prediction_adjustment -= 0.01
        
        # Calculate final prediction
        point_estimate = current_price * (1 + prediction_adjustment)
        
        # Calculate confidence interval based on volatility
        confidence_multiplier = 2.0  # 95% confidence
        margin = point_estimate * volatility * confidence_multiplier
        
        lower_bound = point_estimate - margin
        upper_bound = point_estimate + margin
        
        # Ensure bounds are reasonable
        lower_bound = max(lower_bound, point_estimate * 0.9)  # Min 10% down
        upper_bound = min(upper_bound, point_estimate * 1.1)  # Max 10% up
        
        return point_estimate, lower_bound, upper_bound


async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    """Technical Analysis-based forward function."""
    start_time = time.perf_counter()
    
    # Get assets to predict
    assets = [asset.lower() for asset in synapse.assets] if synapse.assets else ["btc"]
    
    bt.logging.info(f"📊 Technical Analysis Miner: Analyzing {assets} at {synapse.timestamp}")
    
    predictions = {}
    intervals = {}
    
    # Initialize technical analysis miner
    ta_miner = TechnicalAnalysisMiner()
    
    for asset in assets:
        try:
            # Get historical data (2 hours for analysis)
            end_time = to_datetime(synapse.timestamp)
            start_time_data = get_before(synapse.timestamp, hours=2, minutes=0, seconds=0)
            
            # Fetch data
            data = cm.get_CM_ReferenceRate(
                assets=[asset],
                start=to_str(start_time_data),
                end=to_str(end_time),
                frequency="1s"
            )
            
            if data.empty or len(data) < 50:
                bt.logging.warning(f"Insufficient data for {asset}, using fallback")
                latest_price = float(data['ReferenceRateUSD'].iloc[-1]) if not data.empty else 50000.0
                predictions[asset] = latest_price
                intervals[asset] = [latest_price * 0.95, latest_price * 1.05]
                continue
            
            # Generate technical analysis prediction
            point_estimate, lower_bound, upper_bound = ta_miner.generate_prediction(data)
            
            predictions[asset] = point_estimate
            intervals[asset] = [lower_bound, upper_bound]
            
            bt.logging.info(
                f"📊 {asset}: TA Prediction=${point_estimate:.2f} | "
                f"Interval=[${lower_bound:.2f}, ${upper_bound:.2f}]"
            )
            
        except Exception as e:
            bt.logging.error(f"Technical analysis failed for {asset}: {e}")
            # Fallback to latest price
            if not data.empty:
                latest_price = float(data['ReferenceRateUSD'].iloc[-1])
                predictions[asset] = latest_price
                intervals[asset] = [latest_price * 0.95, latest_price * 1.05]
    
    # Set synapse results
    synapse.predictions = predictions
    synapse.intervals = intervals
    
    total_time = time.perf_counter() - start_time
    bt.logging.debug(f"⏱️ Technical Analysis Miner took: {total_time:.3f} seconds")
    
    return synapse

