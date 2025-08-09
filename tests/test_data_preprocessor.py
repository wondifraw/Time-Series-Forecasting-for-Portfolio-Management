"""
Tests for data_preprocessor module
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessor import FinancialDataPreprocessor
import pandas as pd
import numpy as np


class TestFinancialDataPreprocessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = FinancialDataPreprocessor()
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        self.sample_data = {
            'TEST': pd.DataFrame({
                'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
                'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
                'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
            }, index=dates)
        }
    
    def test_clean_data(self):
        """Test data cleaning functionality"""
        cleaned = self.preprocessor.clean_data(self.sample_data)
        
        self.assertIsInstance(cleaned, dict)
        self.assertIn('TEST', cleaned)
        
        df = cleaned['TEST']
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10)
    
    def test_calculate_returns(self):
        """Test returns calculation"""
        cleaned_data = self.preprocessor.clean_data(self.sample_data)
        returns = self.preprocessor.calculate_returns(cleaned_data)
        
        self.assertIsInstance(returns, dict)
        self.assertIn('TEST', returns)
        
        test_returns = returns['TEST']
        self.assertIsInstance(test_returns, pd.Series)
        self.assertEqual(len(test_returns), 9)  # One less due to pct_change
    
    def test_calculate_statistics(self):
        """Test statistics calculation"""
        cleaned_data = self.preprocessor.clean_data(self.sample_data)
        self.preprocessor.calculate_returns(cleaned_data)
        stats = self.preprocessor.calculate_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('TEST', stats)
        
        test_stats = stats['TEST']
        self.assertIn('price_stats', test_stats)
        self.assertIn('return_stats', test_stats)


if __name__ == '__main__':
    unittest.main()