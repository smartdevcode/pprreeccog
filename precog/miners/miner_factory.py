"""
Miner Factory for creating and managing different miner strategies.
This factory provides a unified interface for all miner types.
"""

from typing import Dict, Any, Optional
import bittensor as bt
from precog.protocol import Challenge
from precog.utils.cm_data import CMData

# Import all miner strategies
from precog.miners.base_miner import forward as base_forward
from precog.miners.ml_miner import forward as ml_forward
from precog.miners.technical_analysis_miner import forward as ta_forward
from precog.miners.ensemble_miner import forward as ensemble_forward
from precog.miners.lstm_miner import forward as lstm_forward
from precog.miners.sentiment_miner import forward as sentiment_forward
from precog.miners.advanced_ensemble_miner import forward as advanced_ensemble_forward


class MinerFactory:
    """Factory for creating and managing miner strategies."""
    
    def __init__(self):
        self.available_strategies = {
            'base': {
                'name': 'Base Miner',
                'description': 'Simple price-based prediction with volatility intervals',
                'forward_function': base_forward,
                'complexity': 'low',
                'performance': 'medium'
            },
            'ml': {
                'name': 'ML Miner',
                'description': 'Machine learning with scikit-learn models',
                'forward_function': ml_forward,
                'complexity': 'medium',
                'performance': 'high'
            },
            'technical': {
                'name': 'Technical Analysis Miner',
                'description': 'Advanced technical indicators and chart patterns',
                'forward_function': ta_forward,
                'complexity': 'medium',
                'performance': 'high'
            },
            'ensemble': {
                'name': 'Ensemble Miner',
                'description': 'Combines multiple strategies with adaptive weighting',
                'forward_function': ensemble_forward,
                'complexity': 'high',
                'performance': 'very_high'
            },
            'lstm': {
                'name': 'LSTM Miner',
                'description': 'Deep learning with LSTM and Transformer models',
                'forward_function': lstm_forward,
                'complexity': 'very_high',
                'performance': 'very_high'
            },
            'sentiment': {
                'name': 'Sentiment Miner',
                'description': 'News and social media sentiment analysis',
                'forward_function': sentiment_forward,
                'complexity': 'high',
                'performance': 'high'
            },
            'advanced_ensemble': {
                'name': 'Advanced Ensemble Miner',
                'description': 'Meta-learning ensemble with regime detection',
                'forward_function': advanced_ensemble_forward,
                'complexity': 'very_high',
                'performance': 'maximum'
            }
        }
    
    def get_available_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available mining strategies."""
        return self.available_strategies
    
    def create_miner(self, strategy_name: str) -> Optional[callable]:
        """Create a miner instance for the specified strategy."""
        if strategy_name not in self.available_strategies:
            bt.logging.error(f"Unknown strategy: {strategy_name}")
            return None
        
        strategy_info = self.available_strategies[strategy_name]
        return strategy_info['forward_function']
    
    def get_strategy_info(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific strategy."""
        return self.available_strategies.get(strategy_name)
    
    def list_strategies_by_complexity(self, complexity: str) -> Dict[str, Dict[str, Any]]:
        """List strategies filtered by complexity level."""
        return {
            name: info for name, info in self.available_strategies.items()
            if info['complexity'] == complexity
        }
    
    def list_strategies_by_performance(self, performance: str) -> Dict[str, Dict[str, Any]]:
        """List strategies filtered by performance level."""
        return {
            name: info for name, info in self.available_strategies.items()
            if info['performance'] == performance
        }
    
    def get_recommended_strategy(self, requirements: Dict[str, Any]) -> str:
        """Get recommended strategy based on requirements."""
        complexity = requirements.get('complexity', 'medium')
        performance = requirements.get('performance', 'high')
        
        # Filter strategies by requirements
        suitable_strategies = []
        for name, info in self.available_strategies.items():
            if (info['complexity'] == complexity or 
                (complexity == 'high' and info['complexity'] in ['medium', 'high', 'very_high']) or
                (complexity == 'very_high' and info['complexity'] in ['high', 'very_high'])):
                
                if (info['performance'] == performance or
                    (performance == 'high' and info['performance'] in ['high', 'very_high', 'maximum']) or
                    (performance == 'very_high' and info['performance'] in ['very_high', 'maximum'])):
                    
                    suitable_strategies.append((name, info))
        
        if not suitable_strategies:
            # Fallback to best available
            return 'advanced_ensemble'
        
        # Return the first suitable strategy
        return suitable_strategies[0][0]


# Global miner factory instance
miner_factory = MinerFactory()


def get_miner_strategy(strategy_name: str) -> Optional[callable]:
    """Get miner strategy function by name."""
    return miner_factory.create_miner(strategy_name)


def list_available_strategies() -> Dict[str, Dict[str, Any]]:
    """List all available mining strategies."""
    return miner_factory.get_available_strategies()


def get_strategy_recommendation(requirements: Dict[str, Any]) -> str:
    """Get recommended strategy based on requirements."""
    return miner_factory.get_recommended_strategy(requirements)
