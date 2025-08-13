"""
Simple Pipeline Interface
Provides the most streamlined interface for forecasting
"""

from typing import Dict, Any, Optional, List
from .forecasting_pipeline import ForecastingPipeline
from .config import ConfigManager

def forecast(symbol: str = 'TSLA', 
            models: List[str] = None,
            test_size: float = 0.2,
            epochs: int = 50,
            plot: bool = True) -> Optional[Dict[str, Any]]:
    """
    Simple forecasting function with minimal configuration.
    
    Args:
        symbol: Stock symbol to analyze
        models: List of models to run ['ARIMA', 'LSTM']
        test_size: Fraction of data for testing
        epochs: LSTM training epochs
        plot: Whether to show plots
    
    Returns:
        Dictionary with results and comparison
    """
    # Create config
    config = ConfigManager.get_default_config()
    config.data.test_size = test_size
    config.lstm.epochs = epochs
    
    # Run pipeline
    pipeline = ForecastingPipeline(config=config)
    return pipeline.run(symbol=symbol, models=models, plot=plot)

def quick_forecast(symbol: str = 'TSLA') -> Optional[Dict[str, float]]:
    """
    Ultra-quick forecasting that returns just the metrics.
    
    Args:
        symbol: Stock symbol to analyze
    
    Returns:
        Dictionary with model metrics
    """
    results = forecast(symbol=symbol, plot=False)
    if not results:
        return None
    
    metrics = {}
    for model_name, result in results['results'].items():
        model_metrics = result['metrics']
        metrics[f"{model_name}_MAE"] = model_metrics['MAE']
        metrics[f"{model_name}_RMSE"] = model_metrics['RMSE']
        metrics[f"{model_name}_MAPE"] = model_metrics['MAPE']
    
    return metrics

def compare_models(symbol: str = 'TSLA') -> Optional[str]:
    """
    Compare models and return the best one.
    
    Args:
        symbol: Stock symbol to analyze
    
    Returns:
        Name of the best performing model
    """
    results = forecast(symbol=symbol, plot=False)
    if not results or not results['comparison']:
        return None
    
    ranking = results['comparison']['overall_ranking']
    return ranking.index[0]