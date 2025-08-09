"""
Exploratory Data Analysis Module for Financial Time Series

This module handles visualization and exploratory analysis of financial data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from statsmodels.tsa.stattools import adfuller


class EDAAnalyzer:
    """
    Handles exploratory data analysis and visualization of financial data.
    """
    
    def __init__(self):
        """Initialize the EDA analyzer."""
        self.figures = {}
    
    def create_comprehensive_plots(self, cleaned_data: Dict[str, pd.DataFrame], 
                                 returns_data: Dict[str, pd.Series]) -> None:
        """
        Create comprehensive visualization plots for financial data.
        
        Args:
            cleaned_data (Dict[str, pd.DataFrame]): Cleaned price data
            returns_data (Dict[str, pd.Series]): Return data
        """
        print("Generating comprehensive visualizations...")
        
        # Set up the plot layout
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Financial Data Exploratory Analysis', fontsize=16, fontweight='bold')
        
        # 1. Price trends over time
        self._plot_price_trends(axes[0, 0], cleaned_data)
        
        # 2. Daily returns over time
        self._plot_return_trends(axes[0, 1], returns_data)
        
        # 3. Return distribution histograms
        self._plot_return_distributions(axes[1, 0], returns_data)
        
        # 4. Rolling volatility (30-day annualized)
        self._plot_rolling_volatility(axes[1, 1], returns_data)
        
        # 5. Return box plots
        self._plot_return_boxplots(axes[2, 0], returns_data)
        
        # 6. Correlation heatmap
        self._plot_correlation_heatmap(axes[2, 1], returns_data)
        
        plt.tight_layout()
        plt.show()
        
        self.figures['comprehensive'] = fig
    
    def _plot_price_trends(self, ax, cleaned_data: Dict[str, pd.DataFrame]) -> None:
        """Plot closing price trends for all symbols."""
        for symbol, data in cleaned_data.items():
            ax.plot(data.index, data['Close'], label=symbol, linewidth=1.2)
        
        ax.set_title('Closing Price Trends', fontweight='bold')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_return_trends(self, ax, returns_data: Dict[str, pd.Series]) -> None:
        """Plot daily return trends for all symbols."""
        for symbol, returns in returns_data.items():
            ax.plot(returns.index, returns, label=symbol, alpha=0.7, linewidth=0.6)
        
        ax.set_title('Daily Returns Over Time', fontweight='bold')
        ax.set_ylabel('Daily Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_return_distributions(self, ax, returns_data: Dict[str, pd.Series]) -> None:
        """Plot return distribution histograms."""
        for symbol, returns in returns_data.items():
            ax.hist(returns, bins=50, alpha=0.6, label=symbol, density=True)
        
        ax.set_title('Return Distributions', fontweight='bold')
        ax.set_xlabel('Daily Return')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_rolling_volatility(self, ax, returns_data: Dict[str, pd.Series]) -> None:
        """Plot 30-day rolling volatility (annualized)."""
        for symbol, returns in returns_data.items():
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
            ax.plot(rolling_vol.index, rolling_vol, label=f'{symbol}', linewidth=1.2)
        
        ax.set_title('Rolling Volatility (30-day, Annualized)', fontweight='bold')
        ax.set_ylabel('Volatility')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_return_boxplots(self, ax, returns_data: Dict[str, pd.Series]) -> None:
        """Plot box plots for return distributions."""
        return_values = [returns.values for returns in returns_data.values()]
        symbols = list(returns_data.keys())
        
        ax.boxplot(return_values, labels=symbols)
        ax.set_title('Return Distribution Box Plots', fontweight='bold')
        ax.set_ylabel('Daily Return')
        ax.grid(True, alpha=0.3)
    
    def _plot_correlation_heatmap(self, ax, returns_data: Dict[str, pd.Series]) -> None:
        """Plot correlation heatmap of returns."""
        if len(returns_data) > 1:
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                       center=0, ax=ax, square=True, fmt='.3f')
            ax.set_title('Return Correlations', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Correlation requires multiple symbols', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def perform_stationarity_tests(self, cleaned_data: Dict[str, pd.DataFrame], 
                                 returns_data: Dict[str, pd.Series]) -> Dict[str, Dict]:
        """
        Perform Augmented Dickey-Fuller tests for stationarity.
        
        Args:
            cleaned_data (Dict[str, pd.DataFrame]): Price data
            returns_data (Dict[str, pd.Series]): Return data
            
        Returns:
            Dict[str, Dict]: ADF test results for each symbol
        """
        print("Performing stationarity tests (ADF)...")
        
        adf_results = {}
        
        for symbol in cleaned_data.keys():
            prices = cleaned_data[symbol]['Close'].dropna()
            returns = returns_data[symbol]
            
            # ADF test on prices
            adf_prices = adfuller(prices, autolag='AIC')
            
            # ADF test on returns
            adf_returns = adfuller(returns, autolag='AIC')
            
            adf_results[symbol] = {
                'prices': {
                    'adf_statistic': adf_prices[0],
                    'p_value': adf_prices[1],
                    'critical_values': adf_prices[4],
                    'is_stationary': adf_prices[1] < 0.05
                },
                'returns': {
                    'adf_statistic': adf_returns[0],
                    'p_value': adf_returns[1],
                    'critical_values': adf_returns[4],
                    'is_stationary': adf_returns[1] < 0.05
                }
            }
            
            print(f"✓ {symbol} ADF Tests:")
            print(f"  Prices: {'Stationary' if adf_prices[1] < 0.05 else 'Non-stationary'} (p={adf_prices[1]:.4f})")
            print(f"  Returns: {'Stationary' if adf_returns[1] < 0.05 else 'Non-stationary'} (p={adf_returns[1]:.4f})")
        
        return adf_results
    
    def analyze_volatility_patterns(self, returns_data: Dict[str, pd.Series]) -> Dict[str, Dict]:
        """
        Analyze volatility patterns and clustering.
        
        Args:
            returns_data (Dict[str, pd.Series]): Return data
            
        Returns:
            Dict[str, Dict]: Volatility analysis results
        """
        print("Analyzing volatility patterns...")
        
        volatility_analysis = {}
        
        for symbol, returns in returns_data.items():
            # Calculate rolling statistics
            rolling_mean = returns.rolling(window=30).mean()
            rolling_std = returns.rolling(window=30).std()
            
            # Volatility clustering analysis (squared returns)
            squared_returns = returns ** 2
            autocorr_vol = squared_returns.autocorr(lag=1)
            
            volatility_analysis[symbol] = {
                'mean_volatility': rolling_std.mean(),
                'volatility_of_volatility': rolling_std.std(),
                'volatility_clustering': autocorr_vol,
                'max_volatility_period': rolling_std.idxmax(),
                'min_volatility_period': rolling_std.idxmin()
            }
            
            print(f"✓ {symbol} volatility analysis completed")
        
        return volatility_analysis