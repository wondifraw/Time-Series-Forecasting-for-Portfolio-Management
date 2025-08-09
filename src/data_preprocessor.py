"""
Data Preprocessing Module for Financial Time Series Analysis

This module handles data cleaning, missing value treatment, and return calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from scipy import stats


class FinancialDataPreprocessor:
    """
    Handles preprocessing of financial data including cleaning and return calculations.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.cleaned_data = {}
        self.returns_data = {}
        self.statistics = {}
    
    def clean_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Clean financial data with advanced missing value handling and save to data/processed.
        
        Args:
            raw_data (Dict[str, pd.DataFrame]): Raw financial data
            
        Returns:
            Dict[str, pd.DataFrame]: Cleaned financial data
        """
        print("Cleaning financial data...")
        
        # Create data/processed directory if it doesn't exist
        import os
        processed_data_dir = os.path.join('data', 'processed')
        os.makedirs(processed_data_dir, exist_ok=True)
        
        for symbol, df in raw_data.items():
            cleaned_df = df.copy()
            
            # Check and report missing values
            missing_count = cleaned_df.isnull().sum().sum()
            if missing_count > 0:
                print(f"{symbol}: {missing_count} missing values found")
                
                # Forward fill for price continuity (logical for financial data)
                cleaned_df = cleaned_df.fillna(method='ffill')
                # Backward fill for any remaining NaN at start
                cleaned_df = cleaned_df.fillna(method='bfill')
            
            # Ensure numeric data types
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in cleaned_df.columns:
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            
            # Remove rows with all NaN values
            cleaned_df = cleaned_df.dropna(how='all')
            
            # Validate price data consistency (High >= Low, etc.)
            if len(cleaned_df) > 0:
                cleaned_df = self._validate_price_data(cleaned_df, symbol)
            
            # Save cleaned data to CSV
            csv_path = os.path.join(processed_data_dir, f"{symbol}_cleaned.csv")
            cleaned_df.to_csv(csv_path)
            print(f"✓ Saved cleaned {symbol} data to {csv_path}")
            
            self.cleaned_data[symbol] = cleaned_df
            print(f"✓ Cleaned data for {symbol}: {len(cleaned_df)} records")
        
        return self.cleaned_data
    
    def _validate_price_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate price data consistency (High >= Low, etc.).
        
        Args:
            df (pd.DataFrame): Price data
            symbol (str): Symbol name for logging
            
        Returns:
            pd.DataFrame: Validated price data
        """
        # Check for logical price relationships
        invalid_rows = (df['High'] < df['Low']) | (df['High'] < df['Close']) | (df['Low'] > df['Close'])
        
        if invalid_rows.any():
            print(f"Warning: {invalid_rows.sum()} invalid price relationships found in {symbol}")
            # Remove invalid rows
            df = df[~invalid_rows]
        
        return df
    
    def calculate_returns(self, cleaned_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Calculate daily percentage returns for all symbols and save to data/processed.
        
        Args:
            cleaned_data (Dict[str, pd.DataFrame]): Cleaned price data
            
        Returns:
            Dict[str, pd.Series]: Daily returns for each symbol
        """
        print("Calculating daily returns...")
        
        import os
        processed_data_dir = os.path.join('data', 'processed')
        os.makedirs(processed_data_dir, exist_ok=True)
        
        for symbol, df in cleaned_data.items():
            # Calculate daily percentage returns
            returns = df['Close'].pct_change().dropna()
            
            # Save returns to CSV
            csv_path = os.path.join(processed_data_dir, f"{symbol}_returns.csv")
            returns.to_csv(csv_path, header=['Daily_Return'])
            print(f"✓ Saved {symbol} returns to {csv_path}")
            
            self.returns_data[symbol] = returns
            print(f"✓ Calculated {len(returns)} return observations for {symbol}")
        
        return self.returns_data
    
    def calculate_statistics(self) -> Dict[str, Dict]:
        """
        Calculate comprehensive statistics for prices and returns.
        
        Returns:
            Dict[str, Dict]: Statistical measures for each symbol
        """
        print("Calculating statistical measures...")
        
        for symbol in self.cleaned_data.keys():
            prices = self.cleaned_data[symbol]['Close']
            returns = self.returns_data[symbol]
            
            # Price statistics
            price_stats = {
                'count': len(prices),
                'mean': prices.mean(),
                'std': prices.std(),
                'min': prices.min(),
                'max': prices.max(),
                'median': prices.median()
            }
            
            # Return statistics
            return_stats = {
                'mean': returns.mean(),
                'std': returns.std(),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'min': returns.min(),
                'max': returns.max(),
                'median': returns.median()
            }
            
            self.statistics[symbol] = {
                'price_stats': price_stats,
                'return_stats': return_stats
            }
            
            print(f"✓ Statistics calculated for {symbol}")
        
        return self.statistics
    
    def detect_outliers(self) -> Dict[str, Dict]:
        """
        Detect outliers in return data using statistical methods.
        
        Returns:
            Dict[str, Dict]: Outlier information for each symbol
        """
        print("Detecting outliers...")
        
        outlier_results = {}
        
        for symbol, returns in self.returns_data.items():
            # Z-score method (3 standard deviations)
            z_scores = np.abs(stats.zscore(returns))
            outliers_z = returns[z_scores > 3]
            
            # IQR method
            Q1 = returns.quantile(0.25)
            Q3 = returns.quantile(0.75)
            IQR = Q3 - Q1
            outliers_iqr = returns[(returns < Q1 - 1.5*IQR) | (returns > Q3 + 1.5*IQR)]
            
            outlier_results[symbol] = {
                'z_score_outliers': len(outliers_z),
                'iqr_outliers': len(outliers_iqr),
                'extreme_returns': {
                    'min': outliers_z.min() if len(outliers_z) > 0 else None,
                    'max': outliers_z.max() if len(outliers_z) > 0 else None
                }
            }
            
            print(f"✓ {symbol}: {len(outliers_z)} Z-score outliers, {len(outliers_iqr)} IQR outliers")
        
        return outlier_results