"""
Risk Calculation Module for Financial Time Series Analysis

This module handles risk metric calculations including VaR and Sharpe Ratio.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


class RiskCalculator:
    """
    Handles calculation of financial risk metrics including VaR and Sharpe Ratio.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the risk calculator.
        
        Args:
            risk_free_rate (float): Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_var(self, returns_data: Dict[str, pd.Series], 
                     confidence_level: float = 0.05) -> Dict[str, Dict]:
        """
        Calculate Value at Risk (VaR) using multiple methods.
        
        VaR represents the maximum expected loss over a specific time period at a given
        confidence level. This implementation uses three approaches:
        1. Historical VaR: Uses empirical distribution of returns
        2. Parametric VaR: Assumes normal distribution of returns
        3. Modified VaR: Uses Cornish-Fisher expansion for non-normal distributions
        
        Args:
            returns_data (Dict[str, pd.Series]): Daily returns for each symbol
            confidence_level (float): Confidence level for VaR (default: 5% = 95% confidence)
            
        Returns:
            Dict[str, Dict]: VaR calculations for each symbol containing:
                - historical_var: Empirical quantile-based VaR
                - parametric_var: Normal distribution assumption VaR
                - modified_var: Cornish-Fisher adjusted VaR for skewness/kurtosis
                - confidence_level: The confidence level used
        
        Note:
            VaR values are negative, representing potential losses
        """
        print(f"Calculating VaR at {(1-confidence_level)*100}% confidence level...")
        
        var_results = {}
        
        for symbol, returns in returns_data.items():
            # Edge case: Check for sufficient data points
            if len(returns) < 30:
                print(f"Warning: Insufficient data for reliable VaR calculation for {symbol}")
                continue
            
            # Edge case: Handle extreme market conditions (high volatility periods)
            volatility = returns.std()
            if volatility > 0.1:  # Daily volatility > 10%
                print(f"Warning: High volatility detected for {symbol} ({volatility*100:.1f}%)")
            
            # Historical VaR (empirical quantile) - most robust method
            var_historical = np.percentile(returns, confidence_level * 100)
            
            # Parametric VaR (assuming normal distribution)
            # Note: This assumes returns follow normal distribution, which may not hold during crises
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Edge case: Handle zero or negative standard deviation
            if std_return <= 0:
                print(f"Warning: Zero/negative volatility for {symbol}, using historical VaR only")
                var_parametric = var_historical
                var_modified = var_historical
            else:
                var_parametric = mean_return - stats.norm.ppf(1 - confidence_level) * std_return
                
                # Modified VaR (Cornish-Fisher expansion for non-normal distributions)
                # Accounts for skewness (asymmetry) and kurtosis (fat tails)
                skewness = returns.skew()
                kurtosis = returns.kurtosis()
                z_score = stats.norm.ppf(1 - confidence_level)
                
                # Edge case: Handle extreme skewness/kurtosis values
                if abs(skewness) > 5 or abs(kurtosis) > 10:
                    print(f"Warning: Extreme distribution shape for {symbol} (skew: {skewness:.2f}, kurt: {kurtosis:.2f})")
                
                # Cornish-Fisher adjustment for higher moments
                cf_adjustment = (z_score + 
                               (z_score**2 - 1) * skewness / 6 + 
                               (z_score**3 - 3*z_score) * kurtosis / 24 - 
                               (2*z_score**3 - 5*z_score) * skewness**2 / 36)
                
                var_modified = mean_return - cf_adjustment * std_return
            
            var_results[symbol] = {
                'historical_var': var_historical,
                'parametric_var': var_parametric,
                'modified_var': var_modified,
                'confidence_level': confidence_level
            }
            
            print(f"✓ {symbol} VaR calculated:")
            print(f"  Historical: {var_historical:.4f} ({var_historical*100:.2f}%)")
            print(f"  Parametric: {var_parametric:.4f} ({var_parametric*100:.2f}%)")
            print(f"  Modified: {var_modified:.4f} ({var_modified*100:.2f}%)")
            
            # Performance note: VaR calculation is O(n log n) due to percentile calculation
            # For large datasets, consider using approximate quantiles for better performance
        
        return var_results
    
    def calculate_sharpe_ratio(self, returns_data: Dict[str, pd.Series]) -> Dict[str, Dict]:
        """
        Calculate Sharpe Ratio and related performance metrics.
        
        Args:
            returns_data (Dict[str, pd.Series]): Daily returns for each symbol
            
        Returns:
            Dict[str, Dict]: Sharpe ratio and performance metrics for each symbol
        """
        print(f"Calculating Sharpe Ratios (Risk-free rate: {self.risk_free_rate*100}%)...")
        
        sharpe_results = {}
        
        for symbol, returns in returns_data.items():
            # Annualized metrics
            annual_return = returns.mean() * 252
            annual_volatility = returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility
            
            # Additional performance metrics
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # Sortino ratio (uses downside deviation instead of total volatility)
            sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else np.inf
            
            # Maximum drawdown calculation
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            sharpe_results[symbol] = {
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'downside_deviation': downside_deviation
            }
            
            print(f"✓ {symbol} Performance Metrics:")
            print(f"  Annual Return: {annual_return:.4f} ({annual_return*100:.2f}%)")
            print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
            print(f"  Max Drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
        
        return sharpe_results
    
    def calculate_portfolio_metrics(self, returns_data: Dict[str, pd.Series], 
                                  weights: Dict[str, float] = None) -> Dict:
        """
        Calculate portfolio-level risk metrics.
        
        Args:
            returns_data (Dict[str, pd.Series]): Daily returns for each symbol
            weights (Dict[str, float]): Portfolio weights (equal weight if None)
            
        Returns:
            Dict: Portfolio risk metrics
        """
        print("Calculating portfolio-level metrics...")
        
        # Default to equal weights if not provided
        if weights is None:
            n_assets = len(returns_data)
            weights = {symbol: 1/n_assets for symbol in returns_data.keys()}
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate portfolio returns
        weight_series = pd.Series(weights)
        portfolio_returns = (returns_df * weight_series).sum(axis=1)
        
        # Portfolio metrics
        portfolio_annual_return = portfolio_returns.mean() * 252
        portfolio_annual_vol = portfolio_returns.std() * np.sqrt(252)
        portfolio_sharpe = (portfolio_annual_return - self.risk_free_rate) / portfolio_annual_vol
        
        # Portfolio VaR
        portfolio_var_95 = np.percentile(portfolio_returns, 5)
        
        portfolio_metrics = {
            'annual_return': portfolio_annual_return,
            'annual_volatility': portfolio_annual_vol,
            'sharpe_ratio': portfolio_sharpe,
            'var_95': portfolio_var_95,
            'weights': weights
        }
        
        print(f"✓ Portfolio Metrics (Equal Weight):")
        print(f"  Annual Return: {portfolio_annual_return:.4f} ({portfolio_annual_return*100:.2f}%)")
        print(f"  Sharpe Ratio: {portfolio_sharpe:.4f}")
        
        return portfolio_metrics
    
    def calculate_beta(self, returns_data: Dict[str, pd.Series], 
                      market_symbol: str = 'SPY') -> Dict[str, float]:
        """
        Calculate beta coefficients relative to market (SPY).
        
        Args:
            returns_data (Dict[str, pd.Series]): Daily returns for each symbol
            market_symbol (str): Market benchmark symbol (default: 'SPY')
            
        Returns:
            Dict[str, float]: Beta coefficients for each symbol
        """
        if market_symbol not in returns_data:
            print(f"Warning: Market symbol {market_symbol} not found in data")
            return {}
        
        print(f"Calculating beta coefficients relative to {market_symbol}...")
        
        market_returns = returns_data[market_symbol]
        beta_results = {}
        
        for symbol, returns in returns_data.items():
            if symbol == market_symbol:
                beta_results[symbol] = 1.0  # Market beta is always 1
                continue
            
            # Calculate beta using covariance method
            covariance = np.cov(returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance > 0 else 0
            
            beta_results[symbol] = beta
            print(f"✓ {symbol} Beta: {beta:.4f}")
        
        return beta_results