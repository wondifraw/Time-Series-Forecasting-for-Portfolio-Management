"""
Forecast Integration Module
Integrates forecasting results with portfolio optimization
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, Optional

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class ForecastIntegrator:
    """Integrate forecasting results with portfolio optimization."""
    
    def __init__(self):
        self.forecast_data = None
        
    def get_best_model_forecast(self) -> Optional[float]:
        """Get forecast from best performing model."""
        try:
            # Try to load ARIMA forecast (assuming it's the best model)
            from ..task3.simple_forecaster import SimpleFutureForecaster
            
            forecaster = SimpleFutureForecaster('TSLA')
            data = forecaster.load_data()
            
            if data is not None:
                # Generate simple forecast based on recent trend
                recent_returns = data.pct_change().dropna().tail(30)
                avg_return = recent_returns.mean()
                
                # Annualize the return
                annual_return = avg_return * 252
                
                print(f"Using TSLA forecast: {annual_return:.2%} annual return")
                return annual_return
            
        except Exception as e:
            print(f"Could not load forecast model: {e}")
            
        # Fallback: use historical average
        return self.get_historical_forecast()
    
    def get_historical_forecast(self) -> float:
        """Get historical average as forecast fallback."""
        try:
            file_path = 'data/processed/TSLA_returns.csv'
            if not os.path.exists(file_path):
                file_path = '../data/processed/TSLA_returns.csv'
            
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            avg_return = data['Daily_Return'].mean() * 252  # Annualized
            
            print(f"Using TSLA historical average: {avg_return:.2%} annual return")
            return avg_return
            
        except Exception as e:
            print(f"Error loading historical data: {e}")
            return 0.15  # Default 15% return
    
    def create_forecast_scenarios(self, base_forecast: float) -> Dict[str, float]:
        """Create forecast scenarios for sensitivity analysis."""
        return {
            'pessimistic': base_forecast * 0.7,
            'base': base_forecast,
            'optimistic': base_forecast * 1.3
        }
    
    def get_market_context(self) -> Dict:
        """Get market context for all assets."""
        context = {}
        
        symbols = ['TSLA', 'BND', 'SPY']
        
        for symbol in symbols:
            try:
                file_path = f'data/processed/{symbol}_returns.csv'
                if not os.path.exists(file_path):
                    file_path = f'../data/processed/{symbol}_returns.csv'
                
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                returns = data['Daily_Return']
                
                context[symbol] = {
                    'annual_return': returns.mean() * 252,
                    'annual_volatility': returns.std() * np.sqrt(252),
                    'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252))
                }
                
            except Exception as e:
                print(f"Warning: Could not load data for {symbol}: {e}")
                
        return context
    
    def print_market_context(self) -> None:
        """Print market context summary."""
        context = self.get_market_context()
        
        print("\nMARKET CONTEXT ANALYSIS")
        print("=" * 50)
        
        for symbol, metrics in context.items():
            print(f"\n{symbol}:")
            print(f"  Expected Return: {metrics['annual_return']:.2%}")
            print(f"  Volatility: {metrics['annual_volatility']:.2%}")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            
            # Asset classification
            if symbol == 'TSLA':
                print("  Type: High-growth equity (volatile)")
            elif symbol == 'BND':
                print("  Type: Bond ETF (stable income)")
            elif symbol == 'SPY':
                print("  Type: Market index (diversified equity)")
    
    def validate_forecast(self, forecast: float) -> bool:
        """Validate forecast reasonableness."""
        # Check if forecast is within reasonable bounds
        if forecast < -0.5 or forecast > 2.0:  # -50% to +200%
            print(f"Warning: Forecast {forecast:.2%} seems extreme")
            return False
        
        return True