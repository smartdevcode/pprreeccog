"""
Sentiment Analysis miner using news and social media data.
This miner implements sentiment analysis for cryptocurrency price prediction.
"""

import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import bittensor as bt
from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import get_before, to_datetime, to_str


class SentimentAnalyzer:
    """Advanced sentiment analysis for cryptocurrency markets."""
    
    def __init__(self):
        self.sentiment_weights = {
            'news': 0.4,
            'social': 0.3,
            'fear_greed': 0.2,
            'technical': 0.1
        }
        self.sentiment_history = []
        self.max_history = 100
    
    def analyze_news_sentiment(self, asset: str) -> float:
        """Analyze news sentiment for the given asset."""
        try:
            # This would integrate with news APIs like NewsAPI, Alpha Vantage, etc.
            # For now, we'll simulate sentiment analysis
            # In production, you would:
            # 1. Fetch news articles about the asset
            # 2. Use NLP models (BERT, RoBERTa) for sentiment analysis
            # 3. Aggregate sentiment scores
            
            # Simulate news sentiment based on price volatility
            # In reality, you'd fetch actual news and analyze sentiment
            sentiment_score = np.random.normal(0, 0.3)  # Simulated sentiment
            sentiment_score = max(-1, min(1, sentiment_score))  # Clamp to [-1, 1]
            
            return sentiment_score
            
        except Exception as e:
            bt.logging.error(f"News sentiment analysis failed: {e}")
            return 0.0
    
    def analyze_social_sentiment(self, asset: str) -> float:
        """Analyze social media sentiment for the given asset."""
        try:
            # This would integrate with social media APIs like Twitter, Reddit, etc.
            # For now, we'll simulate sentiment analysis
            # In production, you would:
            # 1. Fetch social media posts about the asset
            # 2. Use sentiment analysis models
            # 3. Aggregate sentiment scores
            
            # Simulate social sentiment
            sentiment_score = np.random.normal(0, 0.4)  # Simulated sentiment
            sentiment_score = max(-1, min(1, sentiment_score))  # Clamp to [-1, 1]
            
            return sentiment_score
            
        except Exception as e:
            bt.logging.error(f"Social sentiment analysis failed: {e}")
            return 0.0
    
    def get_fear_greed_index(self, asset: str) -> float:
        """Get Fear & Greed Index for cryptocurrency market."""
        try:
            # This would fetch from Fear & Greed Index API
            # For now, we'll simulate the index
            # In production, you would fetch from:
            # - Alternative.me Fear & Greed Index
            # - CNN Fear & Greed Index
            # - Custom sentiment indicators
            
            # Simulate Fear & Greed Index (0-100, normalized to -1 to 1)
            fear_greed = np.random.uniform(20, 80)  # Simulated index
            sentiment_score = (fear_greed - 50) / 50  # Normalize to [-1, 1]
            
            return sentiment_score
            
        except Exception as e:
            bt.logging.error(f"Fear & Greed Index failed: {e}")
            return 0.0
    
    def analyze_technical_sentiment(self, data: pd.DataFrame) -> float:
        """Analyze technical indicators for sentiment."""
        if data.empty or len(data) < 50:
            return 0.0
        
        try:
            prices = data['ReferenceRateUSD']
            
            # Calculate technical sentiment indicators
            sma_20 = prices.rolling(20).mean()
            sma_50 = prices.rolling(50).mean()
            
            # Price position relative to moving averages
            current_price = prices.iloc[-1]
            sma_20_current = sma_20.iloc[-1]
            sma_50_current = sma_50.iloc[-1]
            
            # Sentiment based on price position
            sentiment_score = 0.0
            
            # Above/below moving averages
            if current_price > sma_20_current:
                sentiment_score += 0.3
            else:
                sentiment_score -= 0.3
                
            if sma_20_current > sma_50_current:
                sentiment_score += 0.2
            else:
                sentiment_score -= 0.2
            
            # Recent momentum
            recent_return = (current_price / prices.iloc[-5] - 1) if len(prices) >= 5 else 0
            sentiment_score += recent_return * 2  # Scale momentum impact
            
            # Volatility sentiment (high volatility = uncertainty)
            volatility = prices.pct_change().rolling(20).std().iloc[-1]
            if volatility > 0.05:  # High volatility
                sentiment_score *= 0.8  # Reduce confidence in high volatility
            
            return max(-1, min(1, sentiment_score))
            
        except Exception as e:
            bt.logging.error(f"Technical sentiment analysis failed: {e}")
            return 0.0
    
    def calculate_composite_sentiment(self, asset: str, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate composite sentiment score from multiple sources."""
        sentiments = {}
        
        # News sentiment
        sentiments['news'] = self.analyze_news_sentiment(asset)
        
        # Social sentiment
        sentiments['social'] = self.analyze_social_sentiment(asset)
        
        # Fear & Greed Index
        sentiments['fear_greed'] = self.get_fear_greed_index(asset)
        
        # Technical sentiment
        sentiments['technical'] = self.analyze_technical_sentiment(data)
        
        # Calculate weighted composite sentiment
        composite_sentiment = 0.0
        for source, sentiment in sentiments.items():
            weight = self.sentiment_weights.get(source, 0.0)
            composite_sentiment += sentiment * weight
        
        sentiments['composite'] = composite_sentiment
        
        return sentiments
    
    def update_sentiment_history(self, asset: str, sentiments: Dict[str, float]):
        """Update sentiment history for trend analysis."""
        history_entry = {
            'timestamp': datetime.now(),
            'asset': asset,
            'sentiments': sentiments
        }
        
        self.sentiment_history.append(history_entry)
        
        # Keep only recent history
        if len(self.sentiment_history) > self.max_history:
            self.sentiment_history = self.sentiment_history[-self.max_history:]
    
    def analyze_sentiment_trend(self, asset: str) -> Dict[str, float]:
        """Analyze sentiment trend over time."""
        if len(self.sentiment_history) < 5:
            return {'trend': 0.0, 'momentum': 0.0, 'volatility': 0.0}
        
        # Get recent sentiment history for this asset
        recent_sentiments = [
            entry['sentiments']['composite'] 
            for entry in self.sentiment_history[-10:] 
            if entry['asset'] == asset
        ]
        
        if len(recent_sentiments) < 3:
            return {'trend': 0.0, 'momentum': 0.0, 'volatility': 0.0}
        
        # Calculate trend (slope of sentiment over time)
        x = np.arange(len(recent_sentiments))
        y = np.array(recent_sentiments)
        trend = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0.0
        
        # Calculate momentum (recent change)
        momentum = recent_sentiments[-1] - recent_sentiments[0] if len(recent_sentiments) > 1 else 0.0
        
        # Calculate volatility
        volatility = np.std(recent_sentiments) if len(recent_sentiments) > 1 else 0.0
        
        return {
            'trend': trend,
            'momentum': momentum,
            'volatility': volatility
        }


class SentimentMiner:
    """Sentiment-based prediction miner."""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.sentiment_impact_factor = 0.02  # How much sentiment affects price prediction
    
    def generate_sentiment_prediction(self, asset: str, data: pd.DataFrame) -> Tuple[float, float, float]:
        """Generate price prediction based on sentiment analysis."""
        if data.empty:
            latest_price = 50000.0
            return latest_price, latest_price * 0.95, latest_price * 1.05
        
        current_price = float(data['ReferenceRateUSD'].iloc[-1])
        
        # Calculate composite sentiment
        sentiments = self.sentiment_analyzer.calculate_composite_sentiment(asset, data)
        
        # Update sentiment history
        self.sentiment_analyzer.update_sentiment_history(asset, sentiments)
        
        # Analyze sentiment trend
        trend_analysis = self.sentiment_analyzer.analyze_sentiment_trend(asset)
        
        # Calculate sentiment-based price adjustment
        sentiment_adjustment = 0.0
        
        # Base sentiment impact
        composite_sentiment = sentiments['composite']
        sentiment_adjustment += composite_sentiment * self.sentiment_impact_factor
        
        # Trend-based adjustment
        trend = trend_analysis['trend']
        sentiment_adjustment += trend * 0.01
        
        # Momentum-based adjustment
        momentum = trend_analysis['momentum']
        sentiment_adjustment += momentum * 0.005
        
        # Volatility adjustment (high sentiment volatility = uncertainty)
        sentiment_volatility = trend_analysis['volatility']
        if sentiment_volatility > 0.5:  # High sentiment volatility
            sentiment_adjustment *= 0.5  # Reduce confidence
        
        # Calculate final prediction
        point_estimate = current_price * (1 + sentiment_adjustment)
        
        # Calculate confidence interval based on sentiment volatility
        base_volatility = data['ReferenceRateUSD'].pct_change().rolling(20).std().iloc[-1] if len(data) >= 20 else 0.02
        sentiment_uncertainty = sentiment_volatility * 0.01
        
        total_uncertainty = base_volatility + sentiment_uncertainty
        confidence_interval = point_estimate * total_uncertainty * 2.0  # 95% confidence
        
        lower_bound = point_estimate - confidence_interval
        upper_bound = point_estimate + confidence_interval
        
        # Ensure reasonable bounds
        lower_bound = max(lower_bound, point_estimate * 0.9)
        upper_bound = min(upper_bound, point_estimate * 1.1)
        
        return point_estimate, lower_bound, upper_bound


async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    """Sentiment-based forward function for price prediction."""
    start_time = time.perf_counter()
    
    # Get assets to predict
    assets = [asset.lower() for asset in synapse.assets] if synapse.assets else ["btc"]
    
    bt.logging.info(f"📰 Sentiment Miner: Analyzing {assets} at {synapse.timestamp}")
    
    predictions = {}
    intervals = {}
    
    # Initialize sentiment miner
    sentiment_miner = SentimentMiner()
    
    for asset in assets:
        try:
            # Get historical data (2 hours for sentiment analysis)
            end_time = to_datetime(synapse.timestamp)
            start_time_data = get_before(synapse.timestamp, hours=2, minutes=0, seconds=0)
            
            # Fetch data
            data = cm.get_CM_ReferenceRate(
                assets=[asset],
                start=to_str(start_time_data),
                end=to_str(end_time),
                frequency="1s"
            )
            
            if data.empty:
                bt.logging.warning(f"No data for {asset}, using fallback")
                latest_price = 50000.0
                predictions[asset] = latest_price
                intervals[asset] = [latest_price * 0.95, latest_price * 1.05]
                continue
            
            # Generate sentiment-based prediction
            point_estimate, lower_bound, upper_bound = sentiment_miner.generate_sentiment_prediction(asset, data)
            
            predictions[asset] = point_estimate
            intervals[asset] = [lower_bound, upper_bound]
            
            bt.logging.info(
                f"📰 {asset}: Sentiment Prediction=${point_estimate:.2f} | "
                f"Interval=[${lower_bound:.2f}, ${upper_bound:.2f}]"
            )
            
        except Exception as e:
            bt.logging.error(f"Sentiment prediction failed for {asset}: {e}")
            # Fallback to latest price
            if not data.empty:
                latest_price = float(data['ReferenceRateUSD'].iloc[-1])
                predictions[asset] = latest_price
                intervals[asset] = [latest_price * 0.95, latest_price * 1.05]
    
    # Set synapse results
    synapse.predictions = predictions
    synapse.intervals = intervals
    
    total_time = time.perf_counter() - start_time
    bt.logging.debug(f"⏱️ Sentiment Miner took: {total_time:.3f} seconds")
    
    return synapse
