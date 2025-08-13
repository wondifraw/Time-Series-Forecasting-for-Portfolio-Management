"""
Configuration Management for Time Series Forecasting
Centralizes all configuration parameters and settings
"""

from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class ARIMAConfig:
    """Configuration for ARIMA model."""
    max_p: int = 5
    max_d: int = 2
    max_q: int = 5
    alpha: float = 0.05  # For stationarity test
    
@dataclass
class LSTMConfig:
    """Configuration for LSTM model."""
    sequence_length: int = 60
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    lstm_units: List[int] = None
    
    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [50, 50, 50]

@dataclass
class DataConfig:
    """Configuration for data handling."""
    test_size: float = 0.2
    symbols: List[str] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['TSLA']

@dataclass
class ModelConfig:
    """Master configuration for all models."""
    arima: ARIMAConfig = None
    lstm: LSTMConfig = None
    data: DataConfig = None
    
    def __post_init__(self):
        if self.arima is None:
            self.arima = ARIMAConfig()
        if self.lstm is None:
            self.lstm = LSTMConfig()
        if self.data is None:
            self.data = DataConfig()

class ConfigManager:
    """Manages configuration loading and validation."""
    
    @staticmethod
    def get_default_config() -> ModelConfig:
        """Get default configuration."""
        return ModelConfig()
    
    @staticmethod
    def validate_config(config: ModelConfig) -> bool:
        """Validate configuration parameters."""
        try:
            # Validate ARIMA config
            assert config.arima.max_p > 0, "max_p must be positive"
            assert config.arima.max_d >= 0, "max_d must be non-negative"
            assert config.arima.max_q > 0, "max_q must be positive"
            assert 0 < config.arima.alpha < 1, "alpha must be between 0 and 1"
            
            # Validate LSTM config
            assert config.lstm.sequence_length > 0, "sequence_length must be positive"
            assert config.lstm.epochs > 0, "epochs must be positive"
            assert config.lstm.batch_size > 0, "batch_size must be positive"
            assert config.lstm.learning_rate > 0, "learning_rate must be positive"
            assert 0 <= config.lstm.dropout_rate < 1, "dropout_rate must be between 0 and 1"
            
            # Validate data config
            assert 0 < config.data.test_size < 1, "test_size must be between 0 and 1"
            assert len(config.data.symbols) > 0, "symbols list cannot be empty"
            
            return True
            
        except AssertionError as e:
            print(f"❌ Configuration validation failed: {str(e)}")
            return False
    
    @staticmethod
    def update_config(config: ModelConfig, updates: Dict[str, Any]) -> ModelConfig:
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif '.' in key:
                # Handle nested attributes like 'arima.max_p'
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
        
        return config