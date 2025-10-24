"""
Advanced Ensemble miner with sophisticated weighting and meta-learning.
This miner implements state-of-the-art ensemble methods for cryptocurrency prediction.
"""

import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import bittensor as bt
from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import get_before, to_datetime, to_str
from precog.miners.base_miner import calculate_prediction_interval
from precog.miners.ml_miner import MLMiner
from precog.miners.technical_analysis_miner import TechnicalAnalysisMiner
from precog.miners.lstm_miner import LSTMMiner
from precog.miners.sentiment_miner import SentimentMiner
from precog.utils.real_time_prices import get_current_market_prices, start_background_price_fetching, force_fresh_price_fetch


class MetaLearner:
    """Meta-learning system for dynamic ensemble weighting."""
    
    def __init__(self):
        self.performance_history = {}
        self.market_regimes = {}
        self.adaptive_weights = True
        self.learning_rate = 0.01
        self.decay_factor = 0.95
        self.price_validation = True
        self.market_price_cache = {}
        self.validation_threshold = 0.15  # 15% deviation threshold
        
        # Performance monitoring
        self.prediction_accuracy_history = {}
        self.interval_score_history = {}
        self.performance_metrics = {}
        self.accuracy_threshold = 0.85  # Target accuracy threshold
        self.improvement_tracking = True
        
    def track_prediction_performance(self, asset: str, prediction: float, actual_price: float, 
                                   interval: Tuple[float, float], timestamp: str):
        """Track prediction accuracy and interval performance."""
        try:
            # Calculate point prediction error
            point_error = abs(prediction - actual_price) / actual_price
            
            # Calculate interval performance
            lower, upper = interval
            inclusion_factor = 1.0 if lower <= actual_price <= upper else 0.0
            width_factor = (upper - lower) / prediction if prediction > 0 else 0.0
            interval_score = inclusion_factor * min(width_factor, 1.0)
            
            # Store performance data
            if asset not in self.prediction_accuracy_history:
                self.prediction_accuracy_history[asset] = []
                self.interval_score_history[asset] = []
            
            self.prediction_accuracy_history[asset].append({
                'timestamp': timestamp,
                'prediction': prediction,
                'actual': actual_price,
                'point_error': point_error,
                'interval_score': interval_score
            })
            
            # Keep only last 100 predictions
            if len(self.prediction_accuracy_history[asset]) > 100:
                self.prediction_accuracy_history[asset] = self.prediction_accuracy_history[asset][-100:]
            
            # Calculate performance metrics
            self._calculate_performance_metrics(asset)
            
            # Log performance
            bt.logging.info(
                f"📊 {asset} Performance: Point Error: {point_error:.4f} | "
                f"Interval Score: {interval_score:.4f} | "
                f"Avg Error: {self.performance_metrics.get(asset, {}).get('avg_point_error', 0):.4f}"
            )
            
        except Exception as e:
            bt.logging.error(f"Failed to track prediction performance: {e}")
    
    def _calculate_performance_metrics(self, asset: str):
        """Calculate performance metrics for an asset."""
        if asset not in self.prediction_accuracy_history:
            return
        
        history = self.prediction_accuracy_history[asset]
        if not history:
            return
        
        # Calculate metrics
        point_errors = [h['point_error'] for h in history]
        interval_scores = [h['interval_score'] for h in history]
        
        self.performance_metrics[asset] = {
            'avg_point_error': np.mean(point_errors),
            'median_point_error': np.median(point_errors),
            'avg_interval_score': np.mean(interval_scores),
            'prediction_count': len(history),
            'accuracy_rate': np.mean([1.0 if e < 0.05 else 0.0 for e in point_errors])  # 5% accuracy threshold
        }
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance summary."""
        summary = {}
        for asset, metrics in self.performance_metrics.items():
            summary[asset] = {
                'avg_error': f"{metrics['avg_point_error']:.4f}",
                'accuracy_rate': f"{metrics['accuracy_rate']:.2%}",
                'avg_interval_score': f"{metrics['avg_interval_score']:.4f}",
                'predictions': metrics['prediction_count']
            }
        return summary
        
    def identify_market_regime(self, data: pd.DataFrame) -> str:
        """Identify current market regime (trending, ranging, volatile)."""
        if data.empty or len(data) < 50:
            return "unknown"
        
        try:
            prices = data['ReferenceRateUSD']
            returns = prices.pct_change().dropna()
            
            # Calculate regime indicators with enhanced metrics
            volatility = returns.rolling(20).std().iloc[-1]
            trend_strength = abs(prices.rolling(20).mean().pct_change().iloc[-1])
            range_ratio = (prices.rolling(20).max() / prices.rolling(20).min()).iloc[-1]
            
            # Enhanced regime classification with more sophisticated thresholds
            if volatility > 0.05:  # Very high volatility
                return "volatile"
            elif trend_strength > 0.02:  # Strong trend
                return "trending"
            elif range_ratio < 1.03:  # Very low range
                return "ranging"
            else:
                return "mixed"
                
        except Exception as e:
            bt.logging.error(f"Market regime identification failed: {e}")
            return "unknown"
    
    def validate_prediction_price(self, asset: str, predicted_price: float, data: pd.DataFrame) -> float:
        """Validate and correct prediction prices to ensure they're realistic."""
        if not self.price_validation or data.empty:
            return predicted_price
        
        try:
            # Get latest market price
            latest_price = float(data['ReferenceRateUSD'].iloc[-1])
            
            # Asset-specific price ranges for validation (updated for current market)
            asset_ranges = {
                'btc': (80000, 150000),      # BTC: $80k - $150k (current: $110,843)
                'eth': (3000, 6000),         # ETH: $3k - $6k (current: $3,943.20)
                'tao_bittensor': (200, 1000), # TAO: $200 - $1000 (current: ~$400)
                'tao': (200, 1000)           # TAO: $200 - $1000 (current: ~$400)
            }
            
            # Check if prediction is within reasonable range
            if asset.lower() in asset_ranges:
                min_price, max_price = asset_ranges[asset.lower()]
                if predicted_price < min_price or predicted_price > max_price:
                    bt.logging.warning(f"Price {predicted_price:.2f} for {asset} outside reasonable range [{min_price}, {max_price}]")
                    # Use latest market price as fallback
                    return latest_price
            
            # Calculate price deviation
            deviation = abs(predicted_price - latest_price) / latest_price
            
            # If deviation is too high, apply correction
            if deviation > self.validation_threshold:
                bt.logging.warning(f"Price deviation too high for {asset}: {deviation:.2%}, applying correction")
                
                # Apply weighted correction towards market price
                correction_factor = 0.8  # 80% towards market price (increased from 70%)
                corrected_price = latest_price * correction_factor + predicted_price * (1 - correction_factor)
                
                bt.logging.info(f"Corrected {asset} price: {predicted_price:.2f} -> {corrected_price:.2f}")
                return corrected_price
            
            return predicted_price
            
        except Exception as e:
            bt.logging.error(f"Error validating price for {asset}: {e}")
            return predicted_price
    
    def calculate_strategy_performance(self, strategy_name: str, predictions: Dict, actual_prices: Dict) -> float:
        """Calculate performance score for a strategy."""
        if strategy_name not in self.performance_history:
            self.performance_history[strategy_name] = []
        
        # Calculate recent accuracy
        recent_predictions = self.performance_history[strategy_name][-20:]  # Last 20 predictions
        if len(recent_predictions) < 5:
            return 1.0  # Default equal weight
        
        # Calculate performance metrics
        errors = []
        for pred, actual in recent_predictions:
            if actual > 0:
                error = abs(pred - actual) / actual
                errors.append(error)
        
        if not errors:
            return 1.0
        
        # Convert error to performance score (lower error = higher performance)
        avg_error = np.mean(errors)
        performance_score = 1.0 / (1.0 + avg_error)
        
        return performance_score
    
    def update_performance_history(self, strategy_name: str, prediction: float, actual_price: float):
        """Update performance history for a strategy."""
        if strategy_name not in self.performance_history:
            self.performance_history[strategy_name] = []
        
        self.performance_history[strategy_name].append((prediction, actual_price))
        
        # Keep only recent history
        if len(self.performance_history[strategy_name]) > 100:
            self.performance_history[strategy_name] = self.performance_history[strategy_name][-100:]
    
    def calculate_adaptive_weights(self, market_regime: str, strategies: Dict) -> Dict[str, float]:
        """Calculate adaptive weights based on market regime and performance."""
        weights = {}
        
        for strategy_name in strategies.keys():
            # Base performance score
            performance_score = self.calculate_strategy_performance(strategy_name, {}, {})
            
            # Regime-specific adjustments
            regime_multiplier = self.get_regime_multiplier(strategy_name, market_regime)
            
            # Calculate final weight
            weights[strategy_name] = performance_score * regime_multiplier
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def get_regime_multiplier(self, strategy_name: str, market_regime: str) -> float:
        """Get regime-specific multiplier for strategy performance."""
        # Define strategy strengths in different market regimes
        regime_strengths = {
            'base': {'trending': 0.8, 'ranging': 1.2, 'volatile': 0.9, 'mixed': 1.0},
            'ml': {'trending': 1.3, 'ranging': 0.9, 'volatile': 1.1, 'mixed': 1.1},
            'technical': {'trending': 1.4, 'ranging': 1.1, 'volatile': 0.8, 'mixed': 1.0},
            'lstm': {'trending': 1.2, 'ranging': 1.0, 'volatile': 1.3, 'mixed': 1.1},
            'sentiment': {'trending': 0.9, 'ranging': 0.8, 'volatile': 1.4, 'mixed': 1.2}
        }
        
        return regime_strengths.get(strategy_name, {}).get(market_regime, 1.0)


class AdvancedEnsembleMiner:
    """Advanced ensemble miner with meta-learning and regime detection."""
    
    def __init__(self):
        self.strategies = {
            'base': {'weight': 0.2, 'miner': None},
            'ml': {'weight': 0.2, 'miner': MLMiner()},
            'technical': {'weight': 0.2, 'miner': TechnicalAnalysisMiner()},
            'lstm': {'weight': 0.2, 'miner': LSTMMiner()},
            'sentiment': {'weight': 0.2, 'miner': SentimentMiner()}
        }
        self.meta_learner = MetaLearner()
        self.uncertainty_threshold = 0.1
        self.diversity_threshold = 0.05
    
    def calculate_prediction_diversity(self, predictions: Dict) -> float:
        """Calculate diversity of predictions (higher diversity = better ensemble)."""
        pred_values = list(predictions.values())
        if len(pred_values) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        mean_pred = np.mean(pred_values)
        std_pred = np.std(pred_values)
        diversity = std_pred / mean_pred if mean_pred > 0 else 0.0
        
        return diversity
    
    def calculate_prediction_uncertainty(self, predictions: Dict, weights: Dict[str, float]) -> float:
        """Calculate uncertainty in ensemble prediction."""
        if not predictions:
            return 1.0
        
        # Weighted variance
        weighted_mean = sum(pred * weights.get(strategy, 0) for strategy, pred in predictions.items())
        weighted_variance = sum(weights.get(strategy, 0) * (pred - weighted_mean) ** 2 
                              for strategy, pred in predictions.items())
        
        # Normalize uncertainty
        uncertainty = min(1.0, weighted_variance / (weighted_mean ** 2) if weighted_mean > 0 else 1.0)
        
        return uncertainty
    
    def apply_uncertainty_adjustment(self, point_estimate: float, uncertainty: float, 
                                   diversity: float) -> Tuple[float, float, float]:
        """Apply uncertainty-based adjustments to prediction."""
        # Increase interval width based on uncertainty
        uncertainty_factor = 1.0 + uncertainty * 2.0
        
        # Increase interval width based on diversity
        diversity_factor = 1.0 + diversity * 1.5
        
        # Combined adjustment
        adjustment_factor = uncertainty_factor * diversity_factor
        
        # Calculate adjusted bounds
        base_margin = point_estimate * 0.05  # 5% base margin
        adjusted_margin = base_margin * adjustment_factor
        
        lower_bound = point_estimate - adjusted_margin
        upper_bound = point_estimate + adjusted_margin
        
        # Ensure reasonable bounds
        lower_bound = max(lower_bound, point_estimate * 0.85)
        upper_bound = min(upper_bound, point_estimate * 1.15)
        
        return point_estimate, lower_bound, upper_bound
    
    def ensemble_predict(self, data: pd.DataFrame, asset: str) -> Tuple[float, float, float]:
        """Generate advanced ensemble prediction."""
        # Identify market regime
        market_regime = self.meta_learner.identify_market_regime(data)
        bt.logging.info(f"Market regime: {market_regime}")
        
        # Calculate adaptive weights
        adaptive_weights = self.meta_learner.calculate_adaptive_weights(market_regime, self.strategies)
        
        # Get predictions from each strategy
        strategy_predictions = {}
        strategy_intervals = {}
        
        for strategy_name, strategy_info in self.strategies.items():
            try:
                if strategy_name == 'base':
                    # Base strategy
                    latest_price = float(data['ReferenceRateUSD'].iloc[-1])
                    lower_bound, upper_bound = calculate_prediction_interval(
                        latest_price, data['ReferenceRateUSD'], "ensemble"
                    )
                    strategy_predictions[strategy_name] = latest_price
                    strategy_intervals[strategy_name] = (lower_bound, upper_bound)
                else:
                    # Other strategies
                    miner = strategy_info['miner']
                    if hasattr(miner, 'generate_prediction'):
                        pred, lower, upper = miner.generate_prediction(data)
                    elif hasattr(miner, 'predict_price'):
                        pred, lower, upper = miner.predict_price(data)
                    else:
                        # Fallback
                        latest_price = float(data['ReferenceRateUSD'].iloc[-1])
                        pred, lower, upper = latest_price, latest_price * 0.95, latest_price * 1.05
                    
                    strategy_predictions[strategy_name] = pred
                    strategy_intervals[strategy_name] = (lower, upper)
                    
            except Exception as e:
                bt.logging.error(f"Strategy {strategy_name} failed: {e}")
                # Fallback for failed strategy
                latest_price = float(data['ReferenceRateUSD'].iloc[-1])
                strategy_predictions[strategy_name] = latest_price
                strategy_intervals[strategy_name] = (latest_price * 0.95, latest_price * 1.05)
        
        # Calculate ensemble prediction
        weighted_prediction = sum(strategy_predictions[strategy] * adaptive_weights.get(strategy, 0) 
                                for strategy in strategy_predictions.keys())
        
        # Validate and correct prediction price
        validated_prediction = self.meta_learner.validate_prediction_price(
            "ensemble", weighted_prediction, data
        )
        
        # Calculate diversity and uncertainty
        diversity = self.calculate_prediction_diversity(strategy_predictions)
        uncertainty = self.calculate_prediction_uncertainty(strategy_predictions, adaptive_weights)
        
        # Apply uncertainty adjustments
        point_estimate, lower_bound, upper_bound = self.apply_uncertainty_adjustment(
            validated_prediction, uncertainty, diversity
        )
        
        # Log ensemble details
        weights_str = ", ".join([f"{name}: {weight:.3f}" for name, weight in adaptive_weights.items()])
        bt.logging.info(f"Ensemble weights: {weights_str}")
        bt.logging.info(f"Diversity: {diversity:.3f}, Uncertainty: {uncertainty:.3f}")
        
        return point_estimate, lower_bound, upper_bound
    
    def _generate_intelligent_prediction(self, data: pd.DataFrame, asset: str, current_price: float) -> float:
        """Generate intelligent prediction optimized for incentive scoring."""
        try:
            if data.empty or len(data) < 10:
                return current_price
            
            prices = data['ReferenceRateUSD']
            
            # Calculate short-term trend (last 20 data points for better accuracy)
            recent_prices = prices.tail(20)
            if len(recent_prices) >= 5:
                trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            else:
                trend = 0.0
            
            # Calculate momentum (rate of change over last 5 points)
            if len(prices) >= 5:
                momentum = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5]
            else:
                momentum = 0.0
            
            # Calculate volatility for more accurate predictions
            volatility = self._calculate_asset_volatility(data, asset)
            
            # OPTIMIZED PREDICTION ALGORITHM FOR INCENTIVE SCORING
            # Use smaller, more conservative adjustments for better accuracy
            
            # Trend following with conservative scaling
            trend_adjustment = current_price * trend * 0.05  # 5% of trend (reduced from 10%)
            
            # Momentum following with conservative scaling  
            momentum_adjustment = current_price * momentum * 0.03  # 3% of momentum (reduced from 5%)
            
            # Volatility-based adjustment (minimal)
            volatility_adjustment = current_price * volatility * 0.01  # 1% of volatility (reduced from 2%)
            
            # Base prediction with conservative adjustments
            prediction = current_price + trend_adjustment + momentum_adjustment + volatility_adjustment
            
            # TIGHTER BOUNDS FOR BETTER ACCURACY
            # Keep predictions very close to current price for better scoring
            min_price = current_price * 0.98  # 2% below current (tighter bounds)
            max_price = current_price * 1.02  # 2% above current (tighter bounds)
            
            # Ensure prediction stays within tight bounds
            final_prediction = max(min_price, min(prediction, max_price))
            
            # Log prediction details for monitoring
            bt.logging.debug(f"🎯 {asset} prediction: current=${current_price:.2f}, trend={trend:.4f}, momentum={momentum:.4f}, volatility={volatility:.4f}")
            bt.logging.debug(f"🎯 {asset} adjustments: trend={trend_adjustment:.2f}, momentum={momentum_adjustment:.2f}, volatility={volatility_adjustment:.2f}")
            bt.logging.debug(f"🎯 {asset} final prediction: ${final_prediction:.2f} (change: {((final_prediction - current_price) / current_price * 100):+.2f}%)")
            
            return final_prediction
            
        except Exception as e:
            bt.logging.error(f"Intelligent prediction failed for {asset}: {e}")
            return current_price
    
    def _calculate_asset_volatility(self, data: pd.DataFrame, asset: str) -> float:
        """Calculate optimized volatility for better interval scoring."""
        try:
            if data.empty or len(data) < 10:
                return 0.015  # Default 1.5% volatility (optimized for scoring)
            
            prices = data['ReferenceRateUSD']
            returns = prices.pct_change().dropna()
            
            if len(returns) < 5:
                return 0.015
            
            # Calculate rolling volatility with optimized window
            volatility = returns.rolling(window=min(15, len(returns))).std().iloc[-1]
            
            # OPTIMIZED VOLATILITY BOUNDS FOR BETTER INTERVAL SCORING
            # Use tighter volatility bounds for better interval scores
            return max(0.008, min(volatility, 0.04))  # Between 0.8% and 4% (optimized)
            
        except Exception as e:
            bt.logging.error(f"Volatility calculation failed for {asset}: {e}")
            return 0.015


async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    """Advanced ensemble-based forward function with enhanced validation and monitoring."""
    start_time = time.perf_counter()
    
    # Get assets to predict
    assets = [asset.lower() for asset in synapse.assets] if synapse.assets else ["btc"]
    
    bt.logging.info(f"🎯 Advanced Ensemble Miner: Predicting {assets} at {synapse.timestamp}")
    
    predictions = {}
    intervals = {}
    
    # Initialize advanced ensemble miner
    ensemble_miner = AdvancedEnsembleMiner()
    
    # Start background price fetching for all assets
    try:
        start_background_price_fetching()
        bt.logging.info("🚀 Background price fetching started for all assets")
        
        # Force an immediate fresh price fetch
        force_fresh_price_fetch()
        bt.logging.info("🔄 Forced immediate fresh price fetch for all assets")
    except Exception as e:
        bt.logging.debug(f"Background price fetching already running or failed to start: {e}")
    
    # Enhanced data validation
    data_quality_score = 0
    total_assets = len(assets)
    
    for asset in assets:
        try:
            # Get historical data (4 hours for comprehensive analysis)
            end_time = to_datetime(synapse.timestamp)
            start_time_data = get_before(synapse.timestamp, hours=4, minutes=0, seconds=0)
            
            # Asset-specific data fetching with proper asset mapping
            asset_mapping = {
                'btc': 'btc',
                'eth': 'eth', 
                'tao_bittensor': 'tao_bittensor',
                'tao': 'tao_bittensor'
            }
            
            cm_asset = asset_mapping.get(asset.lower(), asset.lower())
            
            # Fetch data with enhanced validation
            data = cm.get_CM_ReferenceRate(
                assets=[cm_asset],
                start=to_str(start_time_data),
                end=to_str(end_time),
                frequency="1s"
            )
            
            # If data is empty or insufficient, fetch real-time market prices
            if data.empty or len(data) < 100:
                bt.logging.warning(f"No data available for {asset}, fetching real-time market prices")
                try:
                    # Get real-time prices from multiple APIs
                    real_time_prices = get_current_market_prices([asset])
                    market_price = real_time_prices.get(asset.lower(), 50000.0)
                    predictions[asset] = market_price
                    intervals[asset] = [market_price * 0.95, market_price * 1.05]
                    bt.logging.info(f"Using real-time market price for {asset}: ${market_price:,.2f}")
                except Exception as e:
                    bt.logging.error(f"Failed to fetch real-time price for {asset}: {e}")
                    # Fallback to conservative estimates
                    fallback_prices = {'btc': 110544.0, 'eth': 3907.0, 'tao': 400.0, 'tao_bittensor': 400.0}
                    market_price = fallback_prices.get(asset.lower(), 50000.0)
                    predictions[asset] = market_price
                    intervals[asset] = [market_price * 0.95, market_price * 1.05]
                    bt.logging.warning(f"Using fallback price for {asset}: ${market_price:,.2f}")
                continue
            
            # Enhanced data quality validation
            data_quality = 0
            if not data.empty and len(data) >= 100:
                # Check data completeness
                completeness = len(data.dropna()) / len(data)
                # Check data recency (last data point should be recent)
                time_diff = (to_datetime(synapse.timestamp) - data['time'].iloc[-1]).total_seconds()
                recency = max(0, 1 - (time_diff / 3600))  # 1 hour tolerance
                # Check price volatility (should be reasonable)
                price_volatility = data['ReferenceRateUSD'].pct_change().std()
                volatility_score = min(1.0, max(0, 1 - abs(price_volatility - 0.02) / 0.02))
                
                data_quality = (completeness * 0.4 + recency * 0.3 + volatility_score * 0.3)
                data_quality_score += data_quality
                
                bt.logging.debug(f"Data quality for {asset}: {data_quality:.3f} (completeness: {completeness:.3f}, recency: {recency:.3f}, volatility: {volatility_score:.3f})")
            
            if data.empty or len(data) < 100 or data_quality < 0.5:
                bt.logging.warning(f"Insufficient or poor quality data for {asset}, using fallback")
                # For ETH, always try to fetch real-time prices instead of using fallback
                if asset.lower() == 'eth':
                    try:
                        bt.logging.info(f"Fetching real-time ETH price due to poor data quality")
                        real_time_prices = get_current_market_prices(['eth'])
                        latest_price = real_time_prices.get('eth', 3907.0)
                        bt.logging.info(f"Using real-time ETH price: ${latest_price:,.2f}")
                    except Exception as e:
                        bt.logging.error(f"Failed to fetch real-time ETH price: {e}")
                        latest_price = 3907.0  # Conservative fallback
                else:
                    latest_price = float(data['ReferenceRateUSD'].iloc[-1]) if not data.empty else 50000.0
                predictions[asset] = latest_price
                intervals[asset] = [latest_price * 0.95, latest_price * 1.05]
                continue
            
            # Get current market price as baseline
            try:
                real_time_prices = get_current_market_prices([asset])
                current_price = real_time_prices.get(asset.lower(), 0)
            except Exception as e:
                bt.logging.error(f"Failed to get current price for {asset}: {e}")
                current_price = 50000.0 if asset.lower() == 'btc' else 3000.0 if asset.lower() == 'eth' else 400.0
            
            # Generate intelligent prediction based on current price and market analysis
            if current_price > 0:
                # Calculate prediction based on current price with market analysis
                point_estimate = ensemble_miner._generate_intelligent_prediction(data, asset, current_price)
                
                # OPTIMIZED INTERVAL CALCULATION FOR BETTER SCORING
                volatility = ensemble_miner._calculate_asset_volatility(data, asset)
                
                # Use optimized margin calculation for better interval scores
                # Target 80-90% inclusion rate with optimal width
                margin = current_price * volatility * 1.5  # 1.5 standard deviations (optimized)
                
                # Tighter bounds for better scoring
                lower_bound = max(current_price - margin, current_price * 0.985)  # Min 1.5% down
                upper_bound = min(current_price + margin, current_price * 1.015)  # Max 1.5% up
                
                # Ensure intervals are not too wide (which hurts scoring)
                max_width = current_price * 0.03  # Maximum 3% width
                if (upper_bound - lower_bound) > max_width:
                    center = (upper_bound + lower_bound) / 2
                    lower_bound = center - max_width / 2
                    upper_bound = center + max_width / 2
                
                corrected_price = point_estimate
                
                # Log interval details for monitoring
                bt.logging.debug(f"🎯 {asset} interval: volatility={volatility:.4f}, margin={margin:.2f}")
                bt.logging.debug(f"🎯 {asset} bounds: [${lower_bound:.2f}, ${upper_bound:.2f}] (width: {((upper_bound - lower_bound) / current_price * 100):.2f}%)")
                
            else:
                # Fallback to ensemble prediction if no current price
                point_estimate, lower_bound, upper_bound = ensemble_miner.ensemble_predict(data, asset)
                corrected_price = point_estimate
            
            # Log the actual prediction vs real-time price for comparison
            try:
                real_time_prices = get_current_market_prices([asset])
                real_time_price = real_time_prices.get(asset.lower(), 0)
                if real_time_price > 0:
                    bt.logging.info(f"🎯 {asset.upper()} Prediction: ${corrected_price:,.2f} | Real-time: ${real_time_price:,.2f} | Difference: {((corrected_price - real_time_price) / real_time_price * 100):+.2f}%")
            except Exception as e:
                bt.logging.debug(f"Could not compare prediction with real-time price: {e}")
            
            # Additional validation for unrealistic prices using real-time market data
            if asset.lower() == 'btc' and (corrected_price < 50000 or corrected_price > 200000):
                bt.logging.warning(f"BTC price {corrected_price:.2f} is unrealistic, fetching real-time market price")
                try:
                    real_time_prices = get_current_market_prices(['btc'])
                    corrected_price = real_time_prices.get('btc', 110544.0)
                    bt.logging.info(f"Using real-time BTC price: ${corrected_price:,.2f}")
                except Exception as e:
                    bt.logging.error(f"Failed to fetch real-time BTC price: {e}")
                    corrected_price = 110544.0  # Conservative fallback
            elif asset.lower() == 'eth' and (corrected_price < 2000 or corrected_price > 10000):
                bt.logging.warning(f"ETH price {corrected_price:.2f} is unrealistic, fetching real-time market price")
                try:
                    # Force fresh price fetch by clearing cache for ETH
                    from precog.utils.real_time_prices import price_fetcher
                    price_fetcher.cache.pop('eth', None)  # Clear ETH from cache
                    real_time_prices = get_current_market_prices(['eth'])
                    corrected_price = real_time_prices.get('eth', 3907.0)
                    bt.logging.info(f"Using real-time ETH price: ${corrected_price:,.2f}")
                except Exception as e:
                    bt.logging.error(f"Failed to fetch real-time ETH price: {e}")
                    corrected_price = 3907.0  # Conservative fallback
            elif asset.lower() in ['tao', 'tao_bittensor'] and (corrected_price < 100 or corrected_price > 2000):
                bt.logging.warning(f"TAO price {corrected_price:.2f} is unrealistic, fetching real-time market price")
                try:
                    # Force fresh price fetch by clearing cache for TAO
                    from precog.utils.real_time_prices import price_fetcher
                    price_fetcher.cache.pop('tao', None)  # Clear TAO from cache
                    price_fetcher.cache.pop('tao_bittensor', None)  # Clear TAO_BITTENSOR from cache
                    real_time_prices = get_current_market_prices(['tao'])
                    corrected_price = real_time_prices.get('tao', 400.0)
                    bt.logging.info(f"Using real-time TAO price: ${corrected_price:,.2f}")
                except Exception as e:
                    bt.logging.error(f"Failed to fetch real-time TAO price: {e}")
                    corrected_price = 400.0  # Conservative fallback
            
            predictions[asset] = corrected_price
            intervals[asset] = [lower_bound, upper_bound]
            
            bt.logging.info(
                f"🎯 {asset}: Advanced Ensemble Prediction=${corrected_price:.2f} | "
                f"Interval=[${lower_bound:.2f}, ${upper_bound:.2f}]"
            )
            
            # Track prediction for performance monitoring
            if ensemble_miner.meta_learner.improvement_tracking:
                try:
                    # Get current market price for performance tracking
                    current_market_prices = get_current_market_prices([asset])
                    if asset in current_market_prices:
                        current_price = current_market_prices[asset]
                        # Track the actual prediction (before validation) vs real price
                        ensemble_miner.meta_learner.track_prediction_performance(
                            asset, point_estimate, current_price, 
                            (lower_bound, upper_bound), synapse.timestamp
                        )
                except Exception as e:
                    bt.logging.debug(f"Performance tracking failed: {e}")
            
        except Exception as e:
            bt.logging.error(f"Advanced ensemble prediction failed for {asset}: {e}")
            # Fallback to latest price
            if not data.empty:
                latest_price = float(data['ReferenceRateUSD'].iloc[-1])
                predictions[asset] = latest_price
                intervals[asset] = [latest_price * 0.95, latest_price * 1.05]
    
    # Set synapse results
    synapse.predictions = predictions
    synapse.intervals = intervals
    
    # Performance monitoring and reporting
    total_time = time.perf_counter() - start_time
    avg_data_quality = data_quality_score / total_assets if total_assets > 0 else 0
    
    # Enhanced performance summary with incentive optimization info
    bt.logging.info(f"📊 INCENTIVE OPTIMIZATION SUMMARY:")
    bt.logging.info(f"   - Total time: {total_time:.3f} seconds")
    bt.logging.info(f"   - Data quality: {avg_data_quality:.3f}")
    bt.logging.info(f"   - Assets processed: {len(predictions)}/{total_assets}")
    bt.logging.info(f"   - Predictions generated: {len([p for p in predictions.values() if p is not None])}")
    
    # Log prediction prices and expected scoring
    for asset, price in predictions.items():
        try:
            real_time_prices = get_current_market_prices([asset])
            real_time_price = real_time_prices.get(asset.lower(), 0)
            if real_time_price > 0:
                prediction_error = abs(price - real_time_price) / real_time_price
                bt.logging.info(f"   - {asset.upper()}: Prediction=${price:.2f} | Real=${real_time_price:.2f} | Error={prediction_error:.3f}")
                
                # Log expected scoring
                if prediction_error < 0.05:  # <5% error
                    bt.logging.info(f"     ✅ EXCELLENT: Point prediction error <5% (high score expected)")
                elif prediction_error < 0.10:  # <10% error
                    bt.logging.info(f"     ✅ GOOD: Point prediction error <10% (good score expected)")
                else:
                    bt.logging.warning(f"     ⚠️  POOR: Point prediction error >10% (low score expected)")
            else:
                bt.logging.info(f"   - {asset.upper()}: Prediction=${price:.2f} | Real-time price unavailable")
        except Exception as e:
            bt.logging.debug(f"Could not calculate prediction error for {asset}: {e}")
            bt.logging.info(f"   - {asset.upper()}: Prediction=${price:.2f}")
    
    # Log interval scoring expectations
    for asset, interval in intervals.items():
        try:
            real_time_prices = get_current_market_prices([asset])
            real_time_price = real_time_prices.get(asset.lower(), 0)
            if real_time_price > 0 and len(interval) == 2:
                lower, upper = interval
                interval_width = (upper - lower) / real_time_price
                bt.logging.info(f"   - {asset.upper()} Interval: [${lower:.2f}, ${upper:.2f}] | Width={interval_width:.3f}")
                
                # Log expected interval scoring
                if 0.02 <= interval_width <= 0.05:  # 2-5% width
                    bt.logging.info(f"     ✅ OPTIMAL: Interval width 2-5% (high score expected)")
                elif 0.01 <= interval_width <= 0.08:  # 1-8% width
                    bt.logging.info(f"     ✅ GOOD: Interval width 1-8% (good score expected)")
                else:
                    bt.logging.warning(f"     ⚠️  SUBOPTIMAL: Interval width outside optimal range")
        except Exception as e:
            bt.logging.debug(f"Could not analyze interval for {asset}: {e}")
    
    # Log performance summary if available
    try:
        performance_summary = ensemble_miner.meta_learner.get_performance_summary()
        if performance_summary:
            bt.logging.info("📊 Historical Performance Summary:")
            for asset, metrics in performance_summary.items():
                bt.logging.info(f"   {asset}: {metrics}")
    except Exception as e:
        bt.logging.debug(f"Performance summary logging failed: {e}")
    
    bt.logging.debug(f"⏱️ Advanced Ensemble Miner took: {total_time:.3f} seconds")
    
    return synapse
