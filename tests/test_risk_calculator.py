"""
Tests for risk_calculator module
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from risk_calculator import RiskCalculator
import pandas as pd
import numpy as np


class TestRiskCalculator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.risk_calc = RiskCalculator(risk_free_rate=0.02)
        
        # Create sample returns data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.returns_data = {
            'TEST': pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
        }
    
    def test_initialization(self):
        """Test risk calculator initialization"""
        self.assertEqual(self.risk_calc.risk_free_rate, 0.02)
    
    def test_calculate_var(self):
        """Test VaR calculation"""
        var_results = self.risk_calc.calculate_var(self.returns_data)
        
        self.assertIsInstance(var_results, dict)
        self.assertIn('TEST', var_results)
        
        test_var = var_results['TEST']
        self.assertIn('historical_var', test_var)
        self.assertIn('parametric_var', test_var)
        self.assertIn('modified_var', test_var)
        
        # VaR should be negative (loss)
        self.assertLess(test_var['historical_var'], 0)
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        sharpe_results = self.risk_calc.calculate_sharpe_ratio(self.returns_data)
        
        self.assertIsInstance(sharpe_results, dict)
        self.assertIn('TEST', sharpe_results)
        
        test_sharpe = sharpe_results['TEST']
        self.assertIn('annual_return', test_sharpe)
        self.assertIn('annual_volatility', test_sharpe)
        self.assertIn('sharpe_ratio', test_sharpe)
        self.assertIn('max_drawdown', test_sharpe)
        
        # Check values are reasonable
        self.assertIsInstance(test_sharpe['sharpe_ratio'], float)
        self.assertLess(test_sharpe['max_drawdown'], 0)  # Drawdown should be negative
    
    def test_calculate_beta(self):
        """Test beta calculation"""
        # Add market data
        self.returns_data['SPY'] = pd.Series(np.random.normal(0.0008, 0.015, 100))
        
        beta_results = self.risk_calc.calculate_beta(self.returns_data, 'SPY')
        
        self.assertIsInstance(beta_results, dict)
        self.assertIn('TEST', beta_results)
        self.assertIn('SPY', beta_results)
        
        # Market beta should be 1.0
        self.assertEqual(beta_results['SPY'], 1.0)
        self.assertIsInstance(beta_results['TEST'], float)


if __name__ == '__main__':
    unittest.main()