"""
Advanced ML-based miner using scikit-learn for Bitcoin price prediction.
This miner implements multiple ML models and ensemble methods.
"""

import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import bittensor as bt
from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import get_before, to_datetime, to_str


class MLMiner:
    """Machine Learning miner with ensemble prediction capabilities."""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0)
        }
        self.scalers = {}
        self.is_trained = False
        self.feature_columns = []
        
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract technical indicators and features from price data."""
        if data.empty or len(data) < 50:
            return pd.DataFrame()
            
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['ReferenceRateUSD'].pct_change()
        df['log_returns'] = np.log(df['ReferenceRateUSD'] / df['ReferenceRateUSD'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['ReferenceRateUSD'].rolling(window).mean()
            df[f'price_sma_{window}_ratio'] = df['ReferenceRateUSD'] / df[f'sma_{window}']
        
        # Volatility features
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        
        # Technical indicators
        df['rsi'] = self._calculate_rsi(df['ReferenceRateUSD'], 14)
        df['bollinger_upper'], df['bollinger_lower'] = self._calculate_bollinger_bands(df['ReferenceRateUSD'], 20)
        df['bollinger_position'] = (df['ReferenceRateUSD'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
        
        # Momentum features
        df['momentum_5'] = df['ReferenceRateUSD'] / df['ReferenceRateUSD'].shift(5) - 1
        df['momentum_10'] = df['ReferenceRateUSD'] / df['ReferenceRateUSD'].shift(10) - 1
        
        # Volume features (if available)
        if 'volume' in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'price_lag_{lag}'] = df['ReferenceRateUSD'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Time-based features
        df['hour'] = pd.to_datetime(df['time']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['time']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, lower
    
    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with features and targets."""
        df = self.extract_features(data)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if len(df) < 100:
            return np.array([]), np.array([])
        
        # Define feature columns
        feature_cols = [col for col in df.columns if col not in ['time', 'ReferenceRateUSD', 'asset']]
        self.feature_columns = feature_cols
        
        # Create target (future price)
        df['target'] = df['ReferenceRateUSD'].shift(-1)  # Next hour's price
        
        # Remove last row (no target)
        df = df[:-1]
        
        X = df[feature_cols].values
        y = df['target'].values
        
        return X, y
    
    def train_models(self, data: pd.DataFrame):
        """Train all ML models on historical data."""
        X, y = self.prepare_training_data(data)
        
        if len(X) == 0 or len(y) == 0:
            bt.logging.warning("Insufficient data for training")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # Train models
        for name, model in self.models.items():
            try:
                model.fit(X_train_scaled, y_train)
                bt.logging.info(f"Trained {name} model with {len(X_train)} samples")
            except Exception as e:
                bt.logging.error(f"Failed to train {name} model: {e}")
        
        self.is_trained = True
    
    def predict_price(self, data: pd.DataFrame) -> Tuple[float, float, float]:
        """Make price prediction using ensemble of models."""
        if not self.is_trained:
            # Fallback to latest price
            latest_price = float(data['ReferenceRateUSD'].iloc[-1])
            return latest_price, latest_price * 0.95, latest_price * 1.05
        
        # Extract features for latest data point
        df = self.extract_features(data)
        if df.empty or len(df) < 1:
            latest_price = float(data['ReferenceRateUSD'].iloc[-1])
            return latest_price, latest_price * 0.95, latest_price * 1.05
        
        # Get latest row
        latest_features = df[self.feature_columns].iloc[-1:].values
        
        # Scale features
        if 'main' in self.scalers:
            latest_features_scaled = self.scalers['main'].transform(latest_features)
        else:
            latest_features_scaled = latest_features
        
        # Make predictions
        predictions = []
        for name, model in self.models.items():
            try:
                pred = model.predict(latest_features_scaled)[0]
                predictions.append(pred)
            except Exception as e:
                bt.logging.error(f"Prediction failed for {name}: {e}")
                continue
        
        if not predictions:
            latest_price = float(data['ReferenceRateUSD'].iloc[-1])
            return latest_price, latest_price * 0.95, latest_price * 1.05
        
        # Ensemble prediction (average)
        point_estimate = np.mean(predictions)
        
        # Calculate confidence interval based on prediction variance
        pred_std = np.std(predictions)
        confidence_interval = 2.0 * pred_std  # 95% confidence
        
        lower_bound = point_estimate - confidence_interval
        upper_bound = point_estimate + confidence_interval
        
        return point_estimate, lower_bound, upper_bound


async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    """ML-based forward function for price prediction."""
    start_time = time.perf_counter()
    
    # Get assets to predict
    assets = [asset.lower() for asset in synapse.assets] if synapse.assets else ["btc"]
    
    bt.logging.info(f"🤖 ML Miner: Predicting {assets} at {synapse.timestamp}")
    
    predictions = {}
    intervals = {}
    
    # Initialize ML miner
    ml_miner = MLMiner()
    
    for asset in assets:
        try:
            # Get historical data (2 hours for training)
            end_time = to_datetime(synapse.timestamp)
            start_time_data = get_before(synapse.timestamp, hours=2, minutes=0, seconds=0)
            
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
            
            # Train models on historical data
            ml_miner.train_models(data)
            
            # Make prediction
            point_estimate, lower_bound, upper_bound = ml_miner.predict_price(data)
            
            predictions[asset] = point_estimate
            intervals[asset] = [lower_bound, upper_bound]
            
            bt.logging.info(
                f"🤖 {asset}: ML Prediction=${point_estimate:.2f} | "
                f"Interval=[${lower_bound:.2f}, ${upper_bound:.2f}]"
            )
            
        except Exception as e:
            bt.logging.error(f"ML prediction failed for {asset}: {e}")
            # Fallback to latest price
            if not data.empty:
                latest_price = float(data['ReferenceRateUSD'].iloc[-1])
                predictions[asset] = latest_price
                intervals[asset] = [latest_price * 0.95, latest_price * 1.05]
    
    # Set synapse results
    synapse.predictions = predictions
    synapse.intervals = intervals
    
    total_time = time.perf_counter() - start_time
    bt.logging.debug(f"⏱️ ML Miner took: {total_time:.3f} seconds")
    
    return synapse

