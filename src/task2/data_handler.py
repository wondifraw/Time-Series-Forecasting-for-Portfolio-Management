"""
Data Handler Module for Time Series Forecasting
Centralizes data loading, preprocessing, and splitting logic
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler

# Import from Task 1
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_loader import FinancialDataLoader

class TimeSeriesDataHandler:
    """Handles data operations for time series forecasting."""
    
    def __init__(self, symbols: list = None):
        """Initialize data handler."""
        self.symbols = symbols or ['TSLA']
        self.data_loader = FinancialDataLoader(self.symbols)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_processed_data(self, symbol: str = 'TSLA') -> Optional[pd.Series]:
        """Load data from processed CSV file."""
        try:
            data_path = f'../data/processed/{symbol}_cleaned.csv'
            if not os.path.exists(data_path):
                data_path = f'data/processed/{symbol}_cleaned.csv'
            
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            close_prices = data['Close'].dropna()
            
            print(f"✅ Loaded {symbol} data: {len(close_prices)} observations")
            print(f"📅 Date range: {close_prices.index[0].date()} to {close_prices.index[-1].date()}")
            return close_prices
            
        except FileNotFoundError:
            print(f"❌ {symbol} data not found. Please run main analysis first.")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None
    
    def split_data(self, data: pd.Series, test_size: float = 0.2) -> Tuple[pd.Series, pd.Series]:
        """Split data chronologically for time series validation."""
        if data is None or len(data) == 0:
            return None, None
            
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        return train_data, test_data
    
    def create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i, 0])
            y.append(data[i, 0])
        
        return np.array(X), np.array(y)
    
    def prepare_lstm_data(self, data: pd.Series, sequence_length: int = 60, 
                         test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for LSTM training with proper scaling and sequencing."""
        if data is None or len(data) == 0:
            return None, None, None, None
        
        # Convert to numpy array and reshape
        data_values = data.values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data_values)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data, sequence_length)
        
        # Split chronologically
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to original scale."""
        return self.scaler.inverse_transform(scaled_data.reshape(-1, 1)).flatten()