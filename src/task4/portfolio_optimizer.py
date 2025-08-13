"""
Portfolio Optimizer using Modern Portfolio Theory
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

class PortfolioOptimizer:
    """Portfolio optimization using Modern Portfolio Theory."""
    
    def __init__(self, symbols: list = ['TSLA', 'BND', 'SPY'], risk_free_rate: float = 0.02):
        self.symbols = symbols
        self.risk_free_rate = risk_free_rate
        self.returns_data = None
        self.expected_returns = None
        self.cov_matrix = None
        
    def load_returns_data(self) -> pd.DataFrame:
        """Load historical returns data."""
        returns_dict = {}
        
        for symbol in self.symbols:
            try:
                file_path = f'data/processed/{symbol}_returns.csv'
                if not os.path.exists(file_path):
                    file_path = f'../data/processed/{symbol}_returns.csv'
                
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                returns_dict[symbol] = data['Daily_Return']
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
                
        self.returns_data = pd.DataFrame(returns_dict)
        return self.returns_data
    
    def calculate_expected_returns(self, forecast_return: Optional[float] = None) -> np.ndarray:
        """Calculate expected returns vector."""
        if self.returns_data is None:
            self.load_returns_data()
            
        expected_returns = {}
        
        for symbol in self.symbols:
            if symbol == 'TSLA' and forecast_return is not None:
                # Use forecast for TSLA
                expected_returns[symbol] = forecast_return * 252  # Annualized
            else:
                # Use historical average for other assets
                expected_returns[symbol] = self.returns_data[symbol].mean() * 252
                
        self.expected_returns = np.array([expected_returns[symbol] for symbol in self.symbols])
        return self.expected_returns
    
    def calculate_covariance_matrix(self) -> np.ndarray:
        """Calculate covariance matrix."""
        if self.returns_data is None:
            self.load_returns_data()
            
        self.cov_matrix = self.returns_data.cov().values * 252  # Annualized
        return self.cov_matrix
    
    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float]:
        """Calculate portfolio return and volatility."""
        portfolio_return = np.sum(weights * self.expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return portfolio_return, portfolio_volatility
    
    def sharpe_ratio(self, weights: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        ret, vol = self.portfolio_performance(weights)
        return (ret - self.risk_free_rate) / vol
    
    def optimize_max_sharpe(self) -> Dict:
        """Find portfolio with maximum Sharpe ratio."""
        n_assets = len(self.symbols)
        
        def neg_sharpe(weights):
            return -self.sharpe_ratio(weights)
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(neg_sharpe, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
            ret, vol = self.portfolio_performance(weights)
            sharpe = self.sharpe_ratio(weights)
            
            return {
                'weights': dict(zip(self.symbols, weights)),
                'return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe
            }
        else:
            raise ValueError("Optimization failed")
    
    def optimize_min_volatility(self) -> Dict:
        """Find portfolio with minimum volatility."""
        n_assets = len(self.symbols)
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(portfolio_volatility, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
            ret, vol = self.portfolio_performance(weights)
            sharpe = self.sharpe_ratio(weights)
            
            return {
                'weights': dict(zip(self.symbols, weights)),
                'return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe
            }
        else:
            raise ValueError("Optimization failed")
    
    def optimize_for_return(self, target_return: float) -> Dict:
        """Find portfolio with minimum volatility for target return."""
        n_assets = len(self.symbols)
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(x * self.expected_returns) - target_return}
        ]
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(portfolio_volatility, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
            ret, vol = self.portfolio_performance(weights)
            sharpe = self.sharpe_ratio(weights)
            
            return {
                'weights': dict(zip(self.symbols, weights)),
                'return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe
            }
        else:
            return None