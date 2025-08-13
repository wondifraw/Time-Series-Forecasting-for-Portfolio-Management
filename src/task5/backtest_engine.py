"""
Backtesting Engine for Portfolio Strategy Validation
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_loader import FinancialDataLoader
from data_preprocessor import FinancialDataPreprocessor
from risk_calculator import RiskCalculator
from task2.forecasting_pipeline import ForecastingPipeline
from task4.portfolio_optimizer import PortfolioOptimizer
from task4.forecast_integrator import ForecastIntegrator

class BacktestEngine:
    """Core backtesting engine for portfolio strategies."""
    
    def __init__(self, symbols: list = ['TSLA', 'BND', 'SPY'], 
                 start_date: str = '2024-01-01', end_date: str = '2024-12-31'):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.price_data = None
        self.returns_data = None
        
        # Initialize inherited components
        self.data_loader = FinancialDataLoader(symbols)
        self.preprocessor = FinancialDataPreprocessor()
        self.risk_calculator = RiskCalculator()
        self.forecasting_pipeline = ForecastingPipeline()
        
    def load_backtest_data(self) -> pd.DataFrame:
        """Load price data using Task 1 data loader."""
        # Use Task 1 data preprocessor to load cleaned data
        price_dict = {}
        
        for symbol in self.symbols:
            try:
                file_path = f'../data/processed/{symbol}_cleaned.csv'
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                price_dict[symbol] = data['Close']
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
                
        self.price_data = pd.DataFrame(price_dict)
        
        # Skip date filtering - use all available data
        pass
        
        # Calculate returns using Task 1 preprocessor or fallback
        try:
            # Convert DataFrame to dict format expected by preprocessor
            price_dict = {col: self.price_data[col] for col in self.price_data.columns}
            returns_dict = self.preprocessor.calculate_returns(price_dict)
            self.returns_data = pd.DataFrame(returns_dict)
        except:
            self.returns_data = self.price_data.pct_change().dropna()
        
        return self.price_data
    
    def get_optimal_weights(self) -> Dict[str, float]:
        """Get optimal portfolio weights using Task 2 forecasts and Task 4 optimization."""
        try:
            # Try to get TSLA forecast using Task 2 pipeline
            tsla_forecast = self.forecasting_pipeline.get_best_forecast('TSLA')
        except:
            # Fallback to Task 4 forecast integrator
            forecast_integrator = ForecastIntegrator()
            tsla_forecast = forecast_integrator.get_best_model_forecast()
        
        # Use Task 4 optimizer with forecast
        optimizer = PortfolioOptimizer(self.symbols)
        optimizer.load_returns_data()
        optimizer.calculate_expected_returns(forecast_return=tsla_forecast)
        optimizer.calculate_covariance_matrix()
        
        # Get max Sharpe portfolio
        max_sharpe = optimizer.optimize_max_sharpe()
        return max_sharpe['weights']
    
    def simulate_strategy(self, weights: Dict[str, float], 
                         rebalance_freq: str = 'M') -> pd.Series:
        """Simulate portfolio performance with given weights."""
        if self.price_data is None:
            self.load_backtest_data()
        
        # Convert weights to array
        weight_array = np.array([weights[symbol] for symbol in self.symbols])
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns_data * weight_array).sum(axis=1)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        return cumulative_returns
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics using Task 1 risk calculator."""
        # Convert cumulative returns to daily returns
        daily_returns = returns.pct_change().dropna()
        
        # Basic metrics
        total_return = returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_volatility = daily_returns.std() * np.sqrt(252)
        
        # Try to use Task 1 risk calculator, fallback to manual calculation
        try:
            # Convert to dict format expected by risk calculator
            returns_dict = {'portfolio': daily_returns}
            sharpe_results = self.risk_calculator.calculate_sharpe_ratio(returns_dict)
            sharpe_ratio = sharpe_results['portfolio']['sharpe_ratio']
        except:
            sharpe_ratio = (annual_return - 0.02) / annual_volatility if annual_volatility > 0 else 0
        
        # Calculate max drawdown
        peak = returns.expanding().max()
        drawdown = (returns - peak) / peak
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }