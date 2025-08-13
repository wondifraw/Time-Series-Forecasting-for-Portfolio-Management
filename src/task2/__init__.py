"""
Task 2: Time Series Forecasting Models

Refactored modular implementation with ARIMA and LSTM models for stock price forecasting.
Provides clean interfaces, configuration management, and factory pattern.
"""

from .base_model import BaseForecaster
from .data_handler import TimeSeriesDataHandler
from .model_evaluator import ModelEvaluator
from .config import ModelConfig, ARIMAConfig, LSTMConfig, ConfigManager
from .model_factory import ModelFactory
from .arima_model import ARIMAForecaster
from .lstm_model import LSTMForecaster
from .model_comparison import ModelComparison
from .forecasting_pipeline import ForecastingPipeline
from .simple_pipeline import forecast, quick_forecast, compare_models

# Register models with factory
ModelFactory.register_model('ARIMA', ARIMAForecaster)
ModelFactory.register_model('LSTM', LSTMForecaster)

__all__ = [
    'BaseForecaster',
    'TimeSeriesDataHandler', 
    'ModelEvaluator',
    'ModelConfig',
    'ARIMAConfig',
    'LSTMConfig',
    'ConfigManager',
    'ModelFactory',
    'ARIMAForecaster',
    'LSTMForecaster', 
    'ModelComparison',
    'ForecastingPipeline',
    'forecast',
    'quick_forecast',
    'compare_models'
]