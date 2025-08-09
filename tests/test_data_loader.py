"""
Tests for data_loader module
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import FinancialDataLoader
import pandas as pd


class TestFinancialDataLoader(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.symbols = ['AAPL']  # Use single symbol for faster testing
        self.period = '1mo'  # Short period for testing
        self.loader = FinancialDataLoader(self.symbols, self.period)
    
    def test_initialization(self):
        """Test loader initialization"""
        self.assertEqual(self.loader.symbols, self.symbols)
        self.assertEqual(self.loader.period, self.period)
        self.assertEqual(self.loader.raw_data, {})
    
    def test_load_data(self):
        """Test data loading functionality"""
        data = self.loader.load_data()
        
        # Check data is loaded
        self.assertIsInstance(data, dict)
        self.assertIn('AAPL', data)
        
        # Check data structure
        df = data['AAPL']
        self.assertIsInstance(df, pd.DataFrame)
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            self.assertIn(col, df.columns)
    
    def test_get_data_info(self):
        """Test data info functionality"""
        self.loader.load_data()
        info = self.loader.get_data_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn('AAPL', info)
        
        # Check info structure
        aapl_info = info['AAPL']
        self.assertIn('records', aapl_info)
        self.assertIn('date_range', aapl_info)
        self.assertIn('missing_values', aapl_info)
        self.assertIn('columns', aapl_info)


if __name__ == '__main__':
    unittest.main()