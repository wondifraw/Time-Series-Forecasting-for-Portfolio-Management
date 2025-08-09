"""
Tests for edge cases and extreme market conditions
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from risk_calculator import RiskCalculator
from data_preprocessor import FinancialDataPreprocessor
import pandas as pd
import numpy as np


class TestEdgeCases(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with extreme scenarios"""
        self.risk_calc = RiskCalculator()
        self.preprocessor = FinancialDataPreprocessor()
        
        # Create extreme market scenarios
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Scenario 1: Market crash (extreme negative returns)
        crash_returns = np.concatenate([
            np.random.normal(0.001, 0.01, 80),  # Normal period
            np.array([-0.2, -0.15, -0.1, -0.08, -0.05] * 4)  # Crash period
        ])
        
        # Scenario 2: High volatility (crypto-like behavior)
        high_vol_returns = np.random.normal(0.002, 0.08, 100)  # 8% daily volatility
        
        # Scenario 3: Zero volatility (constant price)
        zero_vol_returns = np.zeros(100)
        
        # Scenario 4: Extreme skewness
        skewed_returns = np.concatenate([
            np.random.normal(-0.001, 0.01, 95),  # Mostly small negative
            np.array([0.5, 0.3, 0.2, 0.1, 0.05])  # Few large positive
        ])
        
        self.extreme_scenarios = {
            'CRASH': pd.Series(crash_returns, index=dates),
            'HIGH_VOL': pd.Series(high_vol_returns, index=dates),
            'ZERO_VOL': pd.Series(zero_vol_returns, index=dates),
            'SKEWED': pd.Series(skewed_returns, index=dates)
        }
    
    def test_var_extreme_scenarios(self):
        """Test VaR calculation under extreme market conditions"""
        var_results = self.risk_calc.calculate_var(self.extreme_scenarios)
        
        # Test crash scenario
        crash_var = var_results['CRASH']['historical_var']
        self.assertLess(crash_var, -0.05)  # Should show significant loss potential
        
        # Test high volatility scenario
        high_vol_var = var_results['HIGH_VOL']['historical_var']
        self.assertLess(abs(high_vol_var), 0.5)  # Should be reasonable despite high vol
        
        # Test zero volatility scenario
        zero_vol_var = var_results['ZERO_VOL']['historical_var']
        self.assertAlmostEqual(zero_vol_var, 0, places=10)  # Should be near zero
    
    def test_insufficient_data(self):
        """Test handling of insufficient data"""
        # Create very small dataset
        small_data = {'SMALL': pd.Series([0.01, -0.02, 0.005], 
                                        index=pd.date_range('2023-01-01', periods=3))}
        
        # Should handle gracefully without crashing
        var_results = self.risk_calc.calculate_var(small_data)
        # May return empty dict or warning, but shouldn't crash
        self.assertIsInstance(var_results, dict)
    
    def test_extreme_price_data(self):
        """Test preprocessing with extreme price movements"""
        # Create data with extreme price jumps
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        extreme_data = {
            'EXTREME': pd.DataFrame({
                'Open': [100, 101, 1000, 1001, 10, 11, 12, 13, 14, 15],  # Extreme jumps
                'High': [101, 102, 1001, 1002, 11, 12, 13, 14, 15, 16],
                'Low': [99, 100, 999, 1000, 9, 10, 11, 12, 13, 14],
                'Close': [100.5, 101.5, 1000.5, 1001.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
                'Volume': [1000] * 10
            }, index=dates)
        }
        
        # Should handle extreme data without crashing
        cleaned = self.preprocessor.clean_data(extreme_data)
        self.assertIn('EXTREME', cleaned)
        
        returns = self.preprocessor.calculate_returns(cleaned)
        self.assertIn('EXTREME', returns)
        
        # Check for extreme returns
        extreme_returns = returns['EXTREME']
        max_return = extreme_returns.max()
        min_return = extreme_returns.min()
        
        # Should detect extreme movements
        self.assertGreater(abs(max_return), 0.5)  # Should have large positive return
        self.assertLess(min_return, -0.5)  # Should have large negative return
    
    def test_missing_data_patterns(self):
        """Test handling of various missing data patterns"""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        
        # Pattern 1: Missing values at start
        data_start_missing = pd.DataFrame({
            'Open': [np.nan, np.nan, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [np.nan, np.nan, 103, 104, 105, 106, 107, 108, 109, 110],
            'Low': [np.nan, np.nan, 101, 102, 103, 104, 105, 106, 107, 108],
            'Close': [np.nan, np.nan, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'Volume': [np.nan, np.nan, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }, index=dates)
        
        missing_data = {'START_MISSING': data_start_missing}
        
        # Should handle missing data gracefully
        cleaned = self.preprocessor.clean_data(missing_data)
        self.assertIn('START_MISSING', cleaned)
        
        # Should have filled missing values
        cleaned_df = cleaned['START_MISSING']
        self.assertFalse(cleaned_df.isnull().any().any())
    
    def test_correlation_edge_cases(self):
        """Test correlation calculation edge cases"""
        # Perfect correlation
        base_returns = np.random.normal(0.001, 0.02, 50)
        perfect_corr_data = {
            'ASSET1': pd.Series(base_returns),
            'ASSET2': pd.Series(base_returns)  # Identical returns
        }
        
        # Should handle perfect correlation
        returns_df = pd.DataFrame(perfect_corr_data)
        corr_matrix = returns_df.corr()
        
        # Should be exactly 1.0 for identical series
        self.assertAlmostEqual(corr_matrix.loc['ASSET1', 'ASSET2'], 1.0, places=10)


if __name__ == '__main__':
    unittest.main()