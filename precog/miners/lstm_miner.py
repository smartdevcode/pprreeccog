"""
Advanced LSTM-based miner for time series prediction.
This miner implements deep learning models specifically designed for cryptocurrency price prediction.
"""

import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import bittensor as bt
from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import get_before, to_datetime, to_str


class LSTMModel(nn.Module):
    """LSTM model for time series prediction."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply attention mechanism
        lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch, hidden)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)  # (batch, seq_len, hidden)
        
        # Use last time step
        last_output = attn_out[:, -1, :]
        
        # Apply dropout and final layer
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output


class TransformerModel(nn.Module):
    """Transformer model for time series prediction."""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = self._create_positional_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_len: int, d_model: int):
        """Create positional encoding for transformer."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        seq_len = x.size(1)
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        
        # Apply transformer
        x = self.dropout(x)
        transformer_out = self.transformer(x)
        
        # Use last time step
        last_output = transformer_out[:, -1, :]
        
        # Final prediction
        output = self.fc(last_output)
        
        return output


class LSTMMiner:
    """Advanced LSTM-based miner with multiple deep learning models."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = 60  # 60 time steps (1 hour at 1-minute intervals)
        
    def create_sequences(self, data: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i-seq_len:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive features for deep learning."""
        if data.empty or len(data) < 100:
            return pd.DataFrame()
            
        df = data.copy()
        
        # Price features
        df['returns'] = df['ReferenceRateUSD'].pct_change()
        df['log_returns'] = np.log(df['ReferenceRateUSD'] / df['ReferenceRateUSD'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Technical indicators
        df['sma_5'] = df['ReferenceRateUSD'].rolling(5).mean()
        df['sma_20'] = df['ReferenceRateUSD'].rolling(20).mean()
        df['ema_12'] = df['ReferenceRateUSD'].ewm(span=12).mean()
        df['ema_26'] = df['ReferenceRateUSD'].ewm(span=26).mean()
        
        # RSI
        delta = df['ReferenceRateUSD'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_middle = df['ReferenceRateUSD'].rolling(20).mean()
        bb_std = df['ReferenceRateUSD'].rolling(20).std()
        df['bb_upper'] = bb_middle + (bb_std * 2)
        df['bb_lower'] = bb_middle - (bb_std * 2)
        df['bb_position'] = (df['ReferenceRateUSD'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Momentum indicators
        df['momentum_5'] = df['ReferenceRateUSD'] / df['ReferenceRateUSD'].shift(5) - 1
        df['momentum_10'] = df['ReferenceRateUSD'] / df['ReferenceRateUSD'].shift(10) - 1
        
        # Volume features (if available)
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        else:
            df['volume_ratio'] = 1.0
        
        # Time features
        df['hour'] = pd.to_datetime(df['time']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['time']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for deep learning models."""
        df = self.extract_features(data)
        df = df.dropna()
        
        if len(df) < 200:
            return np.array([]), np.array([])
        
        # Select features
        feature_cols = [
            'ReferenceRateUSD', 'returns', 'log_returns', 'volatility',
            'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi',
            'bb_position', 'macd', 'macd_signal', 'macd_histogram',
            'momentum_5', 'momentum_10', 'volume_ratio',
            'hour', 'day_of_week', 'is_weekend'
        ]
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols].values
        
        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['features'] = scaler
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, self.sequence_length)
        
        return X_seq, y_seq
    
    def train_lstm_model(self, X: np.ndarray, y: np.ndarray):
        """Train LSTM model."""
        if len(X) == 0:
            return
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        input_size = X.shape[2]
        model = LSTMModel(input_size).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        model.train()
        for epoch in range(50):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            if epoch % 10 == 0:
                bt.logging.info(f"LSTM Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        self.models['lstm'] = model
    
    def train_transformer_model(self, X: np.ndarray, y: np.ndarray):
        """Train Transformer model."""
        if len(X) == 0:
            return
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        input_size = X.shape[2]
        model = TransformerModel(input_size).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        model.train()
        for epoch in range(50):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            if epoch % 10 == 0:
                bt.logging.info(f"Transformer Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        self.models['transformer'] = model
    
    def train_models(self, data: pd.DataFrame):
        """Train all deep learning models."""
        X, y = self.prepare_training_data(data)
        
        if len(X) == 0:
            bt.logging.warning("Insufficient data for LSTM training")
            return
        
        bt.logging.info(f"Training deep learning models with {len(X)} sequences")
        
        # Train LSTM
        self.train_lstm_model(X, y)
        
        # Train Transformer
        self.train_transformer_model(X, y)
        
        self.is_trained = True
    
    def predict_price(self, data: pd.DataFrame) -> Tuple[float, float, float]:
        """Make prediction using ensemble of deep learning models."""
        if not self.is_trained:
            latest_price = float(data['ReferenceRateUSD'].iloc[-1])
            return latest_price, latest_price * 0.95, latest_price * 1.05
        
        # Prepare features
        df = self.extract_features(data)
        if df.empty or len(df) < self.sequence_length:
            latest_price = float(data['ReferenceRateUSD'].iloc[-1])
            return latest_price, latest_price * 0.95, latest_price * 1.05
        
        # Get latest sequence
        feature_cols = [
            'ReferenceRateUSD', 'returns', 'log_returns', 'volatility',
            'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi',
            'bb_position', 'macd', 'macd_signal', 'macd_histogram',
            'momentum_5', 'momentum_10', 'volume_ratio',
            'hour', 'day_of_week', 'is_weekend'
        ]
        
        available_cols = [col for col in feature_cols if col in df.columns]
        X_latest = df[available_cols].iloc[-self.sequence_length:].values
        
        # Scale features
        if 'features' in self.scalers:
            X_scaled = self.scalers['features'].transform(X_latest)
        else:
            X_scaled = X_latest
        
        # Reshape for model input
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)
        
        # Make predictions
        predictions = []
        
        for model_name, model in self.models.items():
            try:
                model.eval()
                with torch.no_grad():
                    pred = model(X_tensor).cpu().numpy()[0][0]
                    predictions.append(pred)
            except Exception as e:
                bt.logging.error(f"Prediction failed for {model_name}: {e}")
                continue
        
        if not predictions:
            latest_price = float(data['ReferenceRateUSD'].iloc[-1])
            return latest_price, latest_price * 0.95, latest_price * 1.05
        
        # Ensemble prediction
        point_estimate = np.mean(predictions)
        
        # Calculate confidence interval
        pred_std = np.std(predictions)
        confidence_interval = 2.0 * pred_std
        
        lower_bound = point_estimate - confidence_interval
        upper_bound = point_estimate + confidence_interval
        
        # Ensure reasonable bounds
        lower_bound = max(lower_bound, point_estimate * 0.9)
        upper_bound = min(upper_bound, point_estimate * 1.1)
        
        return point_estimate, lower_bound, upper_bound


async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    """LSTM-based forward function for deep learning predictions."""
    start_time = time.perf_counter()
    
    # Get assets to predict
    assets = [asset.lower() for asset in synapse.assets] if synapse.assets else ["btc"]
    
    bt.logging.info(f"🧠 LSTM Miner: Predicting {assets} at {synapse.timestamp}")
    
    predictions = {}
    intervals = {}
    
    # Initialize LSTM miner
    lstm_miner = LSTMMiner()
    
    for asset in assets:
        try:
            # Get historical data (4 hours for deep learning)
            end_time = to_datetime(synapse.timestamp)
            start_time_data = get_before(synapse.timestamp, hours=4, minutes=0, seconds=0)
            
            # Fetch data
            data = cm.get_CM_ReferenceRate(
                assets=[asset],
                start=to_str(start_time_data),
                end=to_str(end_time),
                frequency="1s"
            )
            
            if data.empty or len(data) < 200:
                bt.logging.warning(f"Insufficient data for {asset}, using fallback")
                latest_price = float(data['ReferenceRateUSD'].iloc[-1]) if not data.empty else 50000.0
                predictions[asset] = latest_price
                intervals[asset] = [latest_price * 0.95, latest_price * 1.05]
                continue
            
            # Train models on historical data
            lstm_miner.train_models(data)
            
            # Make prediction
            point_estimate, lower_bound, upper_bound = lstm_miner.predict_price(data)
            
            predictions[asset] = point_estimate
            intervals[asset] = [lower_bound, upper_bound]
            
            bt.logging.info(
                f"🧠 {asset}: LSTM Prediction=${point_estimate:.2f} | "
                f"Interval=[${lower_bound:.2f}, ${upper_bound:.2f}]"
            )
            
        except Exception as e:
            bt.logging.error(f"LSTM prediction failed for {asset}: {e}")
            # Fallback to latest price
            if not data.empty:
                latest_price = float(data['ReferenceRateUSD'].iloc[-1])
                predictions[asset] = latest_price
                intervals[asset] = [latest_price * 0.95, latest_price * 1.05]
    
    # Set synapse results
    synapse.predictions = predictions
    synapse.intervals = intervals
    
    total_time = time.perf_counter() - start_time
    bt.logging.debug(f"⏱️ LSTM Miner took: {total_time:.3f} seconds")
    
    return synapse
