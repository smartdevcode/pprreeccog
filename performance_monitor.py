#!/usr/bin/env python3
"""
Performance monitoring script for Precog miner.
This script helps track prediction accuracy and performance metrics.
"""

import json
import time
import requests
from datetime import datetime, timedelta
import bittensor as bt
from typing import Dict, List
import pandas as pd
import numpy as np

class PerformanceMonitor:
    """Monitor miner performance and provide insights."""
    
    def __init__(self):
        self.prediction_history = []
        self.performance_metrics = {}
        self.monitoring_active = True
        
    def fetch_current_prices(self, assets: List[str]) -> Dict[str, float]:
        """Fetch current market prices for comparison."""
        try:
            # Use CoinGecko API for current prices
            asset_mapping = {
                'btc': 'bitcoin',
                'eth': 'ethereum', 
                'tao_bittensor': 'bittensor',
                'tao': 'bittensor'
            }
            
            coin_ids = [asset_mapping.get(asset.lower(), asset.lower()) for asset in assets]
            coin_ids_str = ','.join(coin_ids)
            
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_ids_str}&vs_currencies=usd"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            prices = {}
            for asset in assets:
                coin_id = asset_mapping.get(asset.lower(), asset.lower())
                if coin_id in data and 'usd' in data[coin_id]:
                    prices[asset.lower()] = float(data[coin_id]['usd'])
                    
            return prices
            
        except Exception as e:
            print(f"Failed to fetch current prices: {e}")
            return {}
    
    def calculate_performance_metrics(self, predictions: Dict, actual_prices: Dict) -> Dict:
        """Calculate performance metrics for predictions."""
        metrics = {}
        
        for asset, predicted_price in predictions.items():
            if asset in actual_prices:
                actual_price = actual_prices[asset]
                
                # Point prediction error
                point_error = abs(predicted_price - actual_price) / actual_price
                
                # Performance score (lower error = higher score)
                performance_score = 1.0 / (1.0 + point_error)
                
                metrics[asset] = {
                    'predicted_price': predicted_price,
                    'actual_price': actual_price,
                    'point_error': point_error,
                    'performance_score': performance_score,
                    'accuracy_percentage': max(0, (1 - point_error) * 100)
                }
        
        return metrics
    
    def log_performance_summary(self, metrics: Dict):
        """Log performance summary."""
        print("\n" + "="*60)
        print("📊 MINER PERFORMANCE SUMMARY")
        print("="*60)
        
        total_score = 0
        asset_count = 0
        
        for asset, metric in metrics.items():
            print(f"\n{asset.upper()}:")
            print(f"  Predicted: ${metric['predicted_price']:.2f}")
            print(f"  Actual:    ${metric['actual_price']:.2f}")
            print(f"  Error:     {metric['point_error']:.4f} ({metric['point_error']*100:.2f}%)")
            print(f"  Accuracy:  {metric['accuracy_percentage']:.1f}%")
            print(f"  Score:     {metric['performance_score']:.4f}")
            
            total_score += metric['performance_score']
            asset_count += 1
        
        if asset_count > 0:
            avg_score = total_score / asset_count
            print(f"\n🎯 OVERALL PERFORMANCE:")
            print(f"  Average Score: {avg_score:.4f}")
            print(f"  Performance Rating: {'Excellent' if avg_score > 0.9 else 'Good' if avg_score > 0.8 else 'Fair' if avg_score > 0.7 else 'Needs Improvement'}")
        
        print("="*60)
    
    def monitor_miner_logs(self):
        """Monitor miner logs for performance data."""
        print("🔍 Monitoring miner performance...")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while self.monitoring_active:
                # This would typically read from your miner logs
                # For now, we'll simulate monitoring
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped by user")
            self.monitoring_active = False
    
    def analyze_historical_performance(self, days: int = 7):
        """Analyze historical performance over specified days."""
        print(f"📈 Analyzing historical performance over last {days} days...")
        
        # This would typically read from your miner's performance logs
        # For demonstration, we'll show the structure
        
        print("Historical analysis would include:")
        print("- Average prediction accuracy by asset")
        print("- Performance trends over time")
        print("- Best/worst performing periods")
        print("- Recommendations for improvement")
    
    def generate_improvement_recommendations(self, metrics: Dict) -> List[str]:
        """Generate recommendations for improving performance."""
        recommendations = []
        
        for asset, metric in metrics.items():
            error = metric['point_error']
            
            if error > 0.10:  # >10% error
                recommendations.append(f"{asset.upper()}: High error rate ({error*100:.1f}%) - Consider model retraining")
            elif error > 0.05:  # >5% error
                recommendations.append(f"{asset.upper()}: Moderate error ({error*100:.1f}%) - Review prediction parameters")
            else:
                recommendations.append(f"{asset.upper()}: Good performance ({error*100:.1f}% error) - Maintain current approach")
        
        return recommendations


def main():
    """Main monitoring function."""
    monitor = PerformanceMonitor()
    
    print("🚀 Precog Miner Performance Monitor")
    print("="*50)
    
    # Example usage
    assets = ['btc', 'eth', 'tao_bittensor']
    
    # Fetch current prices
    print("📡 Fetching current market prices...")
    current_prices = monitor.fetch_current_prices(assets)
    
    if current_prices:
        print("✅ Current market prices:")
        for asset, price in current_prices.items():
            print(f"  {asset.upper()}: ${price:.2f}")
        
        # Example predictions (replace with actual miner predictions)
        example_predictions = {
            'btc': current_prices.get('btc', 50000) * 1.02,  # 2% higher
            'eth': current_prices.get('eth', 3000) * 0.98,   # 2% lower
            'tao_bittensor': current_prices.get('tao_bittensor', 400) * 1.01  # 1% higher
        }
        
        # Calculate performance metrics
        metrics = monitor.calculate_performance_metrics(example_predictions, current_prices)
        
        # Log performance summary
        monitor.log_performance_summary(metrics)
        
        # Generate recommendations
        recommendations = monitor.generate_improvement_recommendations(metrics)
        
        print("\n💡 IMPROVEMENT RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  • {rec}")
        
        print("\n🔄 To monitor real-time performance:")
        print("  1. Run your miner with the enhanced advanced_ensemble_miner.py")
        print("  2. Check the performance logs in your miner output")
        print("  3. Monitor incentive/emission scores over time")
        
    else:
        print("❌ Failed to fetch current prices")


if __name__ == "__main__":
    main()
