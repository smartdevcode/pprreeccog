"""
Ensemble miner that combines multiple prediction strategies.
This miner implements a sophisticated ensemble approach for robust predictions.
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


class EnsembleMiner:
    """Advanced ensemble miner combining multiple prediction strategies."""
    
    def __init__(self):
        self.strategies = {
            'base': {'weight': 0.2, 'miner': None},
            'ml': {'weight': 0.4, 'miner': MLMiner()},
            'technical': {'weight': 0.4, 'miner': TechnicalAnalysisMiner()}
        }
        self.performance_history = {}
        self.adaptive_weights = True
    
    def calculate_strategy_performance(self, predictions: Dict, actual_prices: Dict) -> Dict[str, float]:
        """Calculate recent performance of each strategy."""
        performance = {}
        
        for strategy_name in self.strategies.keys():
            if strategy_name not in self.performance_history:
                self.performance_history[strategy_name] = []
            
            # Calculate recent accuracy (last 10 predictions)
            recent_predictions = self.performance_history[strategy_name][-10:]
            if len(recent_predictions) < 3:
                performance[strategy_name] = 1.0  # Default equal weight
                continue
            
            # Calculate mean absolute percentage error
            errors = []
            for pred, actual in recent_predictions:
                if actual > 0:
                    error = abs(pred - actual) / actual
                    errors.append(error)
            
            if errors:
                avg_error = np.mean(errors)
                # Convert error to performance score (lower error = higher performance)
                performance[strategy_name] = 1.0 / (1.0 + avg_error)
            else:
                performance[strategy_name] = 1.0
        
        return performance
    
    def update_adaptive_weights(self, performance: Dict[str, float]):
        """Update strategy weights based on recent performance."""
        if not self.adaptive_weights:
            return
        
        # Normalize performance scores
        total_performance = sum(performance.values())
        if total_performance == 0:
            return
        
        # Update weights with exponential smoothing
        alpha = 0.1  # Smoothing factor
        for strategy_name, perf in performance.items():
            normalized_perf = perf / total_performance
            current_weight = self.strategies[strategy_name]['weight']
            new_weight = (1 - alpha) * current_weight + alpha * normalized_perf
            self.strategies[strategy_name]['weight'] = new_weight
        
        # Normalize weights to sum to 1
        total_weight = sum(s['weight'] for s in self.strategies.values())
        for strategy in self.strategies.values():
            strategy['weight'] /= total_weight
    
    def get_base_prediction(self, data: pd.DataFrame) -> Tuple[float, float, float]:
        """Get base miner prediction (latest price with volatility-based interval)."""
        if data.empty:
            return 50000.0, 47500.0, 52500.0
        
        latest_price = float(data['ReferenceRateUSD'].iloc[-1])
        historical_prices = data['ReferenceRateUSD']
        
        # Use base miner interval calculation
        lower_bound, upper_bound = calculate_prediction_interval(
            latest_price, historical_prices, "ensemble"
        )
        
        return latest_price, lower_bound, upper_bound
    
    def get_ml_prediction(self, data: pd.DataFrame) -> Tuple[float, float, float]:
        """Get ML miner prediction."""
        try:
            ml_miner = self.strategies['ml']['miner']
            ml_miner.train_models(data)
            return ml_miner.predict_price(data)
        except Exception as e:
            bt.logging.error(f"ML prediction failed: {e}")
            # Fallback to base prediction
            return self.get_base_prediction(data)
    
    def get_technical_prediction(self, data: pd.DataFrame) -> Tuple[float, float, float]:
        """Get technical analysis prediction."""
        try:
            ta_miner = self.strategies['technical']['miner']
            return ta_miner.generate_prediction(data)
        except Exception as e:
            bt.logging.error(f"Technical analysis failed: {e}")
            # Fallback to base prediction
            return self.get_base_prediction(data)
    
    def ensemble_predict(self, data: pd.DataFrame) -> Tuple[float, float, float]:
        """Generate ensemble prediction combining all strategies."""
        predictions = {}
        intervals = {}
        
        # Get predictions from each strategy
        try:
            base_pred, base_lower, base_upper = self.get_base_prediction(data)
            predictions['base'] = (base_pred, base_lower, base_upper)
        except Exception as e:
            bt.logging.error(f"Base prediction failed: {e}")
            predictions['base'] = (50000.0, 47500.0, 52500.0)
        
        try:
            ml_pred, ml_lower, ml_upper = self.get_ml_prediction(data)
            predictions['ml'] = (ml_pred, ml_lower, ml_upper)
        except Exception as e:
            bt.logging.error(f"ML prediction failed: {e}")
            predictions['ml'] = (50000.0, 47500.0, 52500.0)
        
        try:
            ta_pred, ta_lower, ta_upper = self.get_technical_prediction(data)
            predictions['technical'] = (ta_pred, ta_lower, ta_upper)
        except Exception as e:
            bt.logging.error(f"Technical prediction failed: {e}")
            predictions['technical'] = (50000.0, 47500.0, 52500.0)
        
        # Weighted ensemble prediction
        weighted_prediction = 0.0
        weighted_lower = 0.0
        weighted_upper = 0.0
        total_weight = 0.0
        
        for strategy_name, (pred, lower, upper) in predictions.items():
            weight = self.strategies[strategy_name]['weight']
            weighted_prediction += pred * weight
            weighted_lower += lower * weight
            weighted_upper += upper * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            weighted_prediction /= total_weight
            weighted_lower /= total_weight
            weighted_upper /= total_weight
        
        # Calculate ensemble confidence interval
        # Use the weighted average of individual intervals
        ensemble_lower = weighted_lower
        ensemble_upper = weighted_upper
        
        # Add ensemble uncertainty based on prediction variance
        pred_values = [pred[0] for pred in predictions.values()]
        pred_variance = np.var(pred_values)
        uncertainty_margin = np.sqrt(pred_variance) * 0.5
        
        ensemble_lower -= uncertainty_margin
        ensemble_upper += uncertainty_margin
        
        # Ensure reasonable bounds
        ensemble_lower = max(ensemble_lower, weighted_prediction * 0.9)
        ensemble_upper = min(ensemble_upper, weighted_prediction * 1.1)
        
        return weighted_prediction, ensemble_lower, ensemble_upper
    
    def update_performance_history(self, asset: str, predictions: Dict, actual_price: float):
        """Update performance history for adaptive weighting."""
        for strategy_name, (pred, _, _) in predictions.items():
            if strategy_name not in self.performance_history:
                self.performance_history[strategy_name] = []
            
            self.performance_history[strategy_name].append((pred, actual_price))
            
            # Keep only last 50 predictions
            if len(self.performance_history[strategy_name]) > 50:
                self.performance_history[strategy_name] = self.performance_history[strategy_name][-50:]


async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    """Ensemble-based forward function."""
    start_time = time.perf_counter()
    
    # Get assets to predict
    assets = [asset.lower() for asset in synapse.assets] if synapse.assets else ["btc"]
    
    bt.logging.info(f"🎯 Ensemble Miner: Predicting {assets} at {synapse.timestamp}")
    
    predictions = {}
    intervals = {}
    ensemble_miner = EnsembleMiner()
    
    for asset in assets:
        try:
            # Get historical data (3 hours for comprehensive analysis)
            end_time = to_datetime(synapse.timestamp)
            start_time_data = get_before(synapse.timestamp, hours=3, minutes=0, seconds=0)
            
            # Fetch data
            data = cm.get_CM_ReferenceRate(
                assets=[asset],
                start=to_str(start_time_data),
                end=to_str(end_time),
                frequency="1s"
            )
            
            if data.empty or len(data) < 100:
                bt.logging.warning(f"Insufficient data for {asset}, using fallback")
                latest_price = float(data['ReferenceRateUSD'].iloc[-1]) if not data.empty else 50000.0
                predictions[asset] = latest_price
                intervals[asset] = [latest_price * 0.95, latest_price * 1.05]
                continue
            
            # Generate ensemble prediction
            point_estimate, lower_bound, upper_bound = ensemble_miner.ensemble_predict(data)
            
            predictions[asset] = point_estimate
            intervals[asset] = [lower_bound, upper_bound]
            
            # Log strategy weights
            weights_str = ", ".join([f"{name}: {s['weight']:.2f}" for name, s in ensemble_miner.strategies.items()])
            bt.logging.info(f"🎯 {asset}: Ensemble weights - {weights_str}")
            
            bt.logging.info(
                f"🎯 {asset}: Ensemble Prediction=${point_estimate:.2f} | "
                f"Interval=[${lower_bound:.2f}, ${upper_bound:.2f}]"
            )
            
        except Exception as e:
            bt.logging.error(f"Ensemble prediction failed for {asset}: {e}")
            # Fallback to latest price
            if not data.empty:
                latest_price = float(data['ReferenceRateUSD'].iloc[-1])
                predictions[asset] = latest_price
                intervals[asset] = [latest_price * 0.95, latest_price * 1.05]
    
    # Set synapse results
    synapse.predictions = predictions
    synapse.intervals = intervals
    
    total_time = time.perf_counter() - start_time
    bt.logging.debug(f"⏱️ Ensemble Miner took: {total_time:.3f} seconds")
    
    return synapse

