"""
Simplified Future Market Forecaster
Direct implementation without complex imports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional
import pickle

class SimpleFutureForecaster:
    """Simplified forecaster using saved models."""
    
    def __init__(self, symbol: str = 'TSLA'):
        self.symbol = symbol
        self.arima_model = None
        self.lstm_model = None
        
    def load_data(self) -> Optional[pd.Series]:
        """Load processed data."""
        try:
            data_path = f'data/processed/{self.symbol}_cleaned.csv'
            if not os.path.exists(data_path):
                data_path = f'../data/processed/{self.symbol}_cleaned.csv'
            
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            return data['Close'].dropna()
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None
    
    def load_arima_model(self) -> bool:
        """Load ARIMA model."""
        try:
            model_path = 'models/arima_model.pkl'
            if not os.path.exists(model_path):
                model_path = '../models/arima_model.pkl'
            
            with open(model_path, 'rb') as f:
                self.arima_model = pickle.load(f)
            print("✅ ARIMA model loaded")
            return True
        except Exception as e:
            print(f"❌ ARIMA model not found: {str(e)}")
            return False
    
    def load_lstm_model(self) -> bool:
        """Load LSTM model."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            
            model_path = 'models/lstm_model.h5'
            if not os.path.exists(model_path):
                model_path = '../models/lstm_model.h5'
            
            self.lstm_model = load_model(model_path)
            print("✅ LSTM model loaded")
            return True
        except Exception as e:
            print(f"❌ LSTM model not available: {str(e)}")
            return False
    
    def generate_arima_forecast(self, data: pd.Series, months: int = 6) -> Dict:
        """Generate ARIMA forecast."""
        if self.arima_model is None:
            return None
        
        try:
            steps = months * 21  # Trading days
            
            # Fit model on full data
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(data, order=(1, 1, 1))  # Simple ARIMA
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=steps)
            conf_int = fitted_model.get_forecast(steps=steps).conf_int()
            
            # Create future dates
            last_date = data.index[-1]
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=steps,
                freq='B'
            )
            
            forecast_series = pd.Series(forecast, index=future_dates)
            conf_int_df = pd.DataFrame(conf_int, index=future_dates, columns=['lower', 'upper'])
            
            return {
                'forecast': forecast_series,
                'confidence_intervals': conf_int_df,
                'historical_data': data
            }
            
        except Exception as e:
            print(f"❌ ARIMA forecast error: {str(e)}")
            return None
    
    def analyze_trend(self, forecast: pd.Series) -> Dict:
        """Analyze forecast trend."""
        if len(forecast) < 2:
            return {'direction': 'No Data', 'total_change_pct': 0.0, 'volatility': 0.0}
        
        start_price = forecast.iloc[0]
        end_price = forecast.iloc[-1]
        total_change = (end_price - start_price) / start_price * 100
        
        if total_change > 10:
            direction = "Strong Upward"
        elif total_change > 2:
            direction = "Moderate Upward"
        elif total_change < -10:
            direction = "Strong Downward"
        elif total_change < -2:
            direction = "Moderate Downward"
        else:
            direction = "Sideways"
        
        volatility = forecast.pct_change().std() * np.sqrt(252) * 100
        
        return {
            'direction': direction,
            'total_change_pct': total_change,
            'volatility': volatility
        }
    
    def assess_risk(self, forecast: pd.Series) -> Dict:
        """Assess risk levels."""
        volatility = forecast.pct_change().std() * np.sqrt(252) * 100
        risk_level = "Low" if volatility < 20 else "Medium" if volatility < 40 else "High"
        
        # Calculate max drawdown
        peak = forecast.expanding().max()
        drawdown = (forecast - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        return {
            'volatility_pct': volatility,
            'risk_level': risk_level,
            'max_drawdown': max_drawdown
        }
    
    def plot_forecast(self, result: Dict, title: str = "Forecast"):
        """Plot forecast results."""
        if not result:
            return
        
        plt.figure(figsize=(14, 8))
        
        # Plot recent historical data
        historical = result['historical_data']
        recent_hist = historical.iloc[-126:] if len(historical) > 126 else historical
        plt.plot(recent_hist.index, recent_hist.values, 
                label='Historical', color='blue', linewidth=2)
        
        # Plot forecast
        forecast = result['forecast']
        plt.plot(forecast.index, forecast.values, 
                label='Forecast', color='red', linewidth=2, linestyle='--')
        
        # Plot confidence intervals if available
        if 'confidence_intervals' in result:
            conf_int = result['confidence_intervals']
            plt.fill_between(conf_int.index, conf_int['lower'], conf_int['upper'],
                           alpha=0.3, color='red', label='95% Confidence Interval')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def run_forecast_analysis(self, months: int = 6):
        """Run complete forecast analysis."""
        print(f"🔮 TESLA {months}-MONTH FORECAST ANALYSIS")
        print("=" * 50)
        
        # Load data
        data = self.load_data()
        if data is None:
            return None
        
        print(f"📊 Data loaded: {len(data)} observations")
        print(f"Current price: ${data.iloc[-1]:.2f}")
        
        # Try ARIMA forecast
        arima_loaded = self.load_arima_model()
        if arima_loaded:
            print("\n📈 Generating ARIMA forecast...")
            arima_result = self.generate_arima_forecast(data, months)
            
            if arima_result:
                trend_data = self.analyze_trend(arima_result['forecast'])
                risk_data = self.assess_risk(arima_result['forecast'])
                
                print(f"\n📊 ARIMA RESULTS:")
                print(f"Trend: {trend_data['direction']}")
                print(f"Expected Return: {trend_data['total_change_pct']:.1f}%")
                print(f"Risk Level: {risk_data['risk_level']}")
                print(f"Volatility: {risk_data['volatility_pct']:.1f}%")
                print(f"Max Drawdown: {risk_data['max_drawdown']:.1f}%")
                
                # Generate recommendation
                if trend_data['total_change_pct'] > 10 and risk_data['risk_level'] != 'High':
                    recommendation = "BUY - Positive outlook"
                elif trend_data['total_change_pct'] < -10:
                    recommendation = "SELL - Negative outlook"
                else:
                    recommendation = "HOLD - Neutral outlook"
                
                print(f"\n💡 RECOMMENDATION: {recommendation}")
                
                # Plot results
                self.plot_forecast(arima_result, f"TSLA - ARIMA {months}-Month Forecast")
                
                # Confidence interval analysis
                if 'confidence_intervals' in arima_result:
                    conf_int = arima_result['confidence_intervals']
                    forecast = arima_result['forecast']
                    
                    interval_width = conf_int['upper'] - conf_int['lower']
                    width_pct = interval_width / forecast * 100
                    
                    initial_width = width_pct.iloc[:21].mean()
                    final_width = width_pct.iloc[-21:].mean()
                    width_expansion = (final_width - initial_width) / initial_width * 100
                    
                    print(f"\n📊 CONFIDENCE INTERVAL ANALYSIS:")
                    print(f"Initial uncertainty: {initial_width:.1f}%")
                    print(f"Final uncertainty: {final_width:.1f}%")
                    print(f"Uncertainty growth: {width_expansion:.1f}%")
                    
                    if width_expansion < 20:
                        reliability = "High reliability"
                    elif width_expansion < 50:
                        reliability = "Moderate reliability"
                    else:
                        reliability = "Low long-term reliability"
                    
                    print(f"Forecast reliability: {reliability}")
                
                return {
                    'forecast_result': arima_result,
                    'trend_analysis': trend_data,
                    'risk_analysis': risk_data,
                    'recommendation': recommendation
                }
        
        print("❌ No models available for forecasting")
        return None