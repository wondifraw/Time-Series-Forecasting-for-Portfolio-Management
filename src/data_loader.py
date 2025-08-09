"""
Data Loader Module for Financial Time Series Analysis

This module handles loading and initial validation of financial data using YFinance.
"""

import pandas as pd
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class FinancialDataLoader:
    """
    Handles loading financial data from Yahoo Finance with error handling and validation.
    """
    
    def __init__(self, symbols: List[str], period: str = '5y'):
        """
        Initialize the data loader.
        
        Args:
            symbols (List[str]): List of stock symbols to load
            period (str): Time period for data (default: '5y')
        """
        self.symbols = symbols
        self.period = period
        self.raw_data = {}
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load financial data for all symbols using YFinance and save to data/raw.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with symbol as key and DataFrame as value
            
        Raises:
            ValueError: If no data is successfully loaded for any symbol
        """
        print(f"Loading financial data for {len(self.symbols)} symbols...")
        
        # Import yfinance with error handling
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required. Install with: pip install yfinance")
        
        # Create data/raw directory if it doesn't exist
        import os
        raw_data_dir = os.path.join('data', 'raw')
        os.makedirs(raw_data_dir, exist_ok=True)
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=self.period)
                
                if data.empty:
                    print(f"Warning: No data found for {symbol}")
                    continue
                
                # Validate required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in data.columns for col in required_cols):
                    print(f"Warning: Missing required columns for {symbol}")
                    continue
                
                # Save raw data to CSV
                csv_path = os.path.join(raw_data_dir, f"{symbol}_{self.period}.csv")
                data.to_csv(csv_path)
                print(f"✓ Saved {symbol} data to {csv_path}")
                
                self.raw_data[symbol] = data
                print(f"✓ Loaded {len(data)} records for {symbol}")
                
            except Exception as e:
                print(f"Error loading {symbol}: {str(e)}")
        
        if not self.raw_data:
            raise ValueError("No data successfully loaded for any symbol")
            
        return self.raw_data
    
    def get_data_info(self) -> Dict[str, Dict]:
        """
        Get basic information about loaded data.
        
        Returns:
            Dict[str, Dict]: Information about each symbol's data
        """
        info = {}
        for symbol, data in self.raw_data.items():
            info[symbol] = {
                'records': len(data),
                'date_range': f"{data.index.min().date()} to {data.index.max().date()}",
                'missing_values': data.isnull().sum().sum(),
                'columns': list(data.columns),
                'saved_to': csv_path
            }
        return info