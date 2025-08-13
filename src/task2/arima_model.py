"""
ARIMA Model Implementation for Time Series Forecasting
Refactored with modular design principles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

from .base_model import BaseForecaster
from .data_handler import TimeSeriesDataHandler
from .model_evaluator import ModelEvaluator
from .config import ModelConfig, ARIMAConfig

class ARIMAForecaster(BaseForecaster):
    """ARIMA model for time series forecasting with modular design."""
    
    def __init__(self, config: ModelConfig = None, symbols: list = None):
        """Initialize ARIMA forecaster."""
        super().__init__(symbols)
        self.config = config.arima if config else ARIMAConfig()
        self.data_handler = TimeSeriesDataHandler(self.symbols)
        self.fitted_model = None
        
    def load_data(self, symbol: str = 'TSLA'):
        """Load data using data handler."""
        return self.data_handler.load_processed_data(symbol)
    
    def prepare_data(self, data, test_size: float = None):
        """Split data using data handler."""
        if test_size is None:
            test_size = 0.2
        return self.data_handler.split_data(data, test_size)
    
    def check_stationarity(self, data, alpha: float = None):
        """Check if time series is stationary using ADF test."""
        if data is None or len(data) == 0:
            return False
        
        if alpha is None:
            alpha = self.config.alpha
            
        try:
            result = adfuller(data.dropna())
            p_value = result[1]
            is_stationary = p_value < alpha
            
            print(f"ADF Statistic: {result[0]:.4f}")
            print(f"p-value: {p_value:.4f}")
            print(f"Critical Values: {result[4]}")
            print(f"Stationary: {'Yes' if is_stationary else 'No'}")
            
            return is_stationary
        except Exception as e:
            print(f"Error in stationarity test: {str(e)}")
            return False
    
    def find_optimal_params(self, data):
        """Find optimal ARIMA parameters using grid search."""
        if data is None or len(data) == 0:
            return (1, 1, 1)
            
        best_aic = float('inf')
        best_params = (1, 1, 1)
        
        print("Searching for optimal ARIMA parameters...")
        
        for p in range(self.config.max_p + 1):
            for d in range(self.config.max_d + 1):
                for q in range(self.config.max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted_model = model.fit()
                        aic = fitted_model.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_params = (p, d, q)
                            
                    except Exception:
                        continue
        
        print(f"Best parameters: ARIMA{best_params} with AIC: {best_aic:.2f}")
        return best_params
    
    def fit(self, train_data, order=None, **kwargs):
        """Fit ARIMA model to training data."""
        if train_data is None or len(train_data) == 0:
            print("❌ No training data provided")
            return False
            
        if order is None:
            order = self.find_optimal_params(train_data)
        
        try:
            self.model = ARIMA(train_data, order=order)
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            print(f"✅ ARIMA{order} model fitted successfully")
            return True
        except Exception as e:
            print(f"❌ Error fitting ARIMA model: {str(e)}")
            return False
    
    def predict(self, steps, **kwargs):
        """Generate forecasts for specified number of steps."""
        if not self.is_fitted or self.fitted_model is None:
            print("❌ Model not fitted. Call fit() first.")
            return None
            
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            return forecast.values if hasattr(forecast, 'values') else forecast
        except Exception as e:
            print(f"❌ Error generating forecast: {str(e)}")
            return None
    
    def forecast_with_intervals(self, steps):
        """Generate forecasts with confidence intervals."""
        if not self.is_fitted or self.fitted_model is None:
            print("❌ Model not fitted. Call fit() first.")
            return None
            
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            conf_int = self.fitted_model.get_forecast(steps=steps).conf_int()
            
            return {
                'forecast': forecast,
                'conf_int': conf_int
            }
        except Exception as e:
            print(f"❌ Error generating forecast: {str(e)}")
            return None
    
    def evaluate(self, actual, predicted):
        """Evaluate model performance using standardized metrics."""
        return ModelEvaluator.calculate_metrics(actual, predicted)
    
    def plot_results(self, train_data, test_data, forecast_result):
        """Plot training data, test data, and forecasts."""
        if any(x is None for x in [train_data, test_data, forecast_result]):
            print("❌ Missing data for plotting")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Plot training data
        plt.plot(train_data.index, train_data.values, 
                label='Training Data', color='blue', alpha=0.7)
        
        # Plot test data
        plt.plot(test_data.index, test_data.values, 
                label='Actual', color='green', linewidth=2)
        
        # Plot forecast
        forecast_index = test_data.index[:len(forecast_result['forecast'])]
        plt.plot(forecast_index, forecast_result['forecast'], 
                label='ARIMA Forecast', color='red', linewidth=2)
        
        # Plot confidence intervals
        if 'conf_int' in forecast_result:
            plt.fill_between(forecast_index,
                           forecast_result['conf_int'].iloc[:, 0],
                           forecast_result['conf_int'].iloc[:, 1],
                           color='red', alpha=0.2, label='Confidence Interval')
        
        plt.title('ARIMA Model Forecast Results')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        """Run complete ARIMA analysis pipeline."""
        print("🔧 ARIMA FORECASTING ANALYSIS")
        print("=" * 40)
        
        # Load data
        data = self.load_data()
        if data is None:
            return None
        
        # Prepare data
        train_data, test_data = self.prepare_data(data)
        if train_data is None or test_data is None:
            return None
        
        print(f"\n📊 Data Split:")
        print(f"Training: {len(train_data)} observations")
        print(f"Testing: {len(test_data)} observations")
        
        # Check stationarity
        print(f"\n🔍 Stationarity Check:")
        is_stationary = self.check_stationarity(train_data)
        
        # Fit model
        print(f"\n⚙️ Model Fitting:")
        success = self.fit(train_data)
        if not success:
            return None
        
        # Generate forecasts
        print(f"\n📈 Generating Forecasts:")
        forecast_result = self.forecast(len(test_data))
        if forecast_result is None:
            return None
        
        # Evaluate performance
        print(f"\n📊 Model Evaluation:")
        metrics = self.evaluate(test_data, forecast_result['forecast'])
        if metrics:
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        
        # Plot results
        self.plot_results(train_data, test_data, forecast_result)
        
        return {
            'model': self.fitted_model,
            'forecast': forecast_result,
            'metrics': metrics,
            'train_data': train_data,
            'test_data': test_data
        }
    
    def save_model(self, filepath: str = None):
        """Save the fitted ARIMA model."""
        if filepath is None:
            os.makedirs('models', exist_ok=True)
            filepath = 'models/arima_model.pkl'
        self.model = self.fitted_model
        return super().save_model(filepath)
    
    def load_model(self, filepath: str = None):
        """Load a saved ARIMA model."""
        if filepath is None:
            filepath = 'models/arima_model.pkl'
        success = super().load_model(filepath)
        if success:
            self.fitted_model = self.model
        return success